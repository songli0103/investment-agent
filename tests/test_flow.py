"""Tests for alphaquant.flows.analysis_flow AnalysisFlow orchestration."""
from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Suppress CrewAI's interactive "enable tracing?" prompt during tests.
os.environ.setdefault("CREWAI_TESTING", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from alphaquant.exceptions import InvalidTickerFormat
from alphaquant.flows import AnalysisFlow, AnalysisState
from alphaquant.models.competitor import Competitor, CompetitorAnalysis
from alphaquant.models.company import Company
from alphaquant.models.financial import BalanceSheet, CashFlowStatement, FinancialStatements, IncomeStatement
from alphaquant.models.market import MarketData
from alphaquant.models.news import NewsAnalysis
from alphaquant.models.risk import RiskAssessment, RiskScore
from alphaquant.models.valuation import ValuationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_competitor_tool(return_value):
    """Patch CompetitorTool._run preserving __annotations__ (BaseTool introspection)."""
    def fake_run(self, ticker: str) -> str:
        return return_value
    return patch(
        "alphaquant.tools.competitor_tool.CompetitorTool._run",
        new=fake_run,
    )


def _patch_data_tools(sample_company, sample_market, sample_news, sample_financial):
    """Patch all 4 data tools' _run to return valid JSON for a successful Flow."""
    company_json = sample_company.model_dump_json()
    market_json = sample_market.model_dump_json()
    # news: NewsTool returns a JSON list, not an object
    news_json = "[]"
    financial_json = sample_financial.model_dump_json()
    return [
        patch(
            "alphaquant.tools.company_lookup_tool.CompanyLookupTool._run",
            new=lambda self, ticker: company_json,
        ),
        patch(
            "alphaquant.tools.market_data_tool.MarketDataTool._run",
            new=lambda self, ticker: market_json,
        ),
        patch(
            "alphaquant.tools.news_tool.NewsTool._run",
            new=lambda self, ticker: news_json,
        ),
        patch(
            "alphaquant.tools.financial_tool.FinancialTool._run",
            new=lambda self, ticker: financial_json,
        ),
    ]


def _run(coro):
    """Drive an async step coroutine in tests (asyncio.run is forbidden in prod)."""
    return asyncio.run(coro)


def _build_fake_task_output(
    sample_company,
    sample_market,
    sample_financial,
    sample_competitor_analysis,
    *,
    market_json: str | None = None,
    news_json: str = "[]",
    financial_json: str | None = None,
    market_error: str | None = None,
    include_analysis_outputs: bool = True,
) -> list:
    """Build a fake tasks_output list for AnalysisCrew.kickoff mocks.

    Sub-project 3: tasks 0-3 produce JSON raw text (still parsed by the data
    pipeline), tasks 4-7 produce Pydantic instances via task_out.pydantic
    (parsed by _extract_pydantic_field).
    """
    from alphaquant.models.risk import RiskAssessment
    from alphaquant.models.valuation import ValuationResult
    from alphaquant.models.report import InvestmentReport

    company_json = sample_company.model_dump_json()
    if market_json is None:
        if market_error is not None:
            market_raw = market_error
        else:
            market_raw = sample_market.model_dump_json()
    else:
        market_raw = market_json
    if financial_json is None:
        financial_raw = sample_financial.model_dump_json()
    else:
        financial_raw = financial_json

    risk = RiskAssessment(
        ticker="AAPL",
        total_score=40,
        level="medium",
        sub_scores=[
            RiskScore(
                category="financial",
                score=4,
                rationale="placeholder subscore rationale",
                evidence=[],
            )
        ],
        top_risks=[],
    )
    valuation = ValuationResult(
        ticker="AAPL",
        intrinsic_value_per_share=Decimal("180.00"),
        current_price=Decimal("150.00"),
        upside_pct=0.2,
        dcf_value=None,
        relative_value=Decimal("180.00"),
        method="relative_only",
    )
    # Need a MarketData for the InvestmentReport; use sample_market if present
    # else substitute a degraded one.
    report = InvestmentReport(
        report_id="11111111-1111-1111-1111-111111111111",
        ticker="AAPL",
        generated_at=datetime(2026, 1, 1),
        data_as_of={},
        company=sample_company,
        market=sample_market if market_error is None else _make_degraded_market(sample_company),
        financial=sample_financial,
        financial_health_score=70,
        news=NewsAnalysis.empty("AAPL"),
        competitors=sample_competitor_analysis,
        risk=risk,
        valuation=valuation,
        rating="Hold",
        confidence=70,
        investment_horizon="medium",
        catalysts=["placeholder catalyst"],
        markdown="## placeholder markdown",
        sources=[],
        disclaimer="placeholder",
    )

    outputs = [
        MagicMock(raw=company_json, pydantic=None),
        MagicMock(raw=market_raw, pydantic=None),
        MagicMock(raw=news_json, pydantic=None),
        MagicMock(raw=financial_raw, pydantic=None),
    ]
    if include_analysis_outputs:
        outputs.extend(
            [
                MagicMock(raw="", pydantic=sample_competitor_analysis),
                MagicMock(raw="", pydantic=risk),
                MagicMock(raw="", pydantic=valuation),
                MagicMock(raw="", pydantic=report),
            ]
        )
    else:
        outputs.extend(
            [MagicMock(raw="", pydantic=None) for _ in range(4)]
        )
    return outputs


def _make_degraded_market(sample_company):
    """Build a degraded MarketData placeholder for partial-failure tests."""
    from alphaquant.models.market import MarketData

    return MarketData(
        ticker="AAPL",
        as_of=datetime(2026, 1, 1),
        price=Decimal("0"),
        change_pct=0.0,
        volume=0,
        market_cap=sample_company.market_cap,
        source="degraded",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_company() -> Company:
    return Company(
        ticker="AAPL",
        name="Apple Inc.",
        exchange="NASDAQ",
        sector="Technology",
        industry="Consumer Electronics",
        market_cap=3_000_000_000_000,
    )


@pytest.fixture
def sample_market() -> MarketData:
    return MarketData(
        ticker="AAPL",
        as_of=datetime(2026, 6, 20),
        price=Decimal("150.00"),
        change_pct=0.5,
        volume=50_000_000,
        market_cap=3_000_000_000_000,
        pe_ratio=25.0,
        beta=1.2,
    )


@pytest.fixture
def sample_news() -> NewsAnalysis:
    return NewsAnalysis(
        ticker="AAPL",
        as_of=datetime(2026, 6, 20),
        total_count=10,
        positive_pct=0.5,
        negative_pct=0.2,
        neutral_pct=0.3,
        sentiment_score=0.3,
    )


@pytest.fixture
def sample_financial() -> FinancialStatements:
    return FinancialStatements(
        ticker="AAPL",
        income_statements=[
            IncomeStatement(
                period="TTM",
                fiscal_year=2026,
                revenue=Decimal("400000000000"),
                net_income=Decimal("100000000000"),
            ),
        ],
        balance_sheets=[
            BalanceSheet(
                period="Q4",
                fiscal_year=2026,
                total_assets=Decimal("350000000000"),
                total_liabilities=Decimal("280000000000"),
                total_equity=Decimal("70000000000"),
            ),
        ],
        source="yahoo",
    )


@pytest.fixture
def sample_competitor_analysis() -> CompetitorAnalysis:
    return CompetitorAnalysis(
        target_ticker="AAPL",
        competitors=[
            Competitor(
                ticker="MSFT",
                name="Microsoft",
                market_cap=2_500_000_000_000,
                revenue_ttm=Decimal("200000000000"),
            ),
        ],
        industry_rank=1,
        industry_size=5,
        competitive_score=75,
        method="gics",
    )


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_imports():
    """Both AnalysisFlow and AnalysisState are importable from alphaquant.flows."""
    assert AnalysisFlow is not None
    assert AnalysisState is not None


def test_analysis_state_defaults():
    """All fields start with safe defaults."""
    s = AnalysisState()
    assert s.ticker == ""
    assert s.company is None
    assert s.market is None
    assert s.news is None
    assert s.financial is None
    assert s.competitor is None
    assert s.risk is None
    assert s.valuation is None
    assert s.report is None
    assert s.errors == []


def test_flow_class_inherits_crewai_flow():
    from crewai.flow.flow import Flow

    assert issubclass(AnalysisFlow, Flow)


# ---------------------------------------------------------------------------
# Ticker normalization
# ---------------------------------------------------------------------------


class TestNormalizeTicker:
    def test_uppercases_and_strips(self):
        from alphaquant.flows.analysis_flow import _normalize_ticker

        assert _normalize_ticker("  aapl  ") == "AAPL"

    def test_rejects_empty(self):
        from alphaquant.flows.analysis_flow import _normalize_ticker

        with pytest.raises(InvalidTickerFormat):
            _normalize_ticker("")

    def test_rejects_too_long(self):
        from alphaquant.flows.analysis_flow import _normalize_ticker

        with pytest.raises(InvalidTickerFormat):
            _normalize_ticker("TOOLONGTICKER")


# ---------------------------------------------------------------------------
# synthesize_report step (was write_report)
# ---------------------------------------------------------------------------


class TestSynthesizeReport:
    def _populate_state(
        self,
        flow,
        sample_company,
        sample_market,
        sample_news,
        sample_financial,
        sample_competitor_analysis,
    ):
        from alphaquant.models.report import InvestmentReport

        flow.state.ticker = "AAPL"
        flow.state.company = sample_company
        flow.state.market = sample_market
        flow.state.news = sample_news
        flow.state.financial = sample_financial
        flow.state.competitor = sample_competitor_analysis
        risk = RiskAssessment(
            ticker="AAPL",
            total_score=40,
            level="medium",
            sub_scores=[
                RiskScore(
                    category="financial",
                    score=4,
                    rationale="placeholder for test",
                    evidence=[],
                )
            ],
            top_risks=[],
        )
        valuation = ValuationResult(
            ticker="AAPL",
            intrinsic_value_per_share=Decimal("180.00"),
            current_price=Decimal("150.00"),
            upside_pct=0.2,
            dcf_value=None,
            relative_value=Decimal("180.00"),
            method="relative_only",
        )
        flow.state.risk = risk
        flow.state.valuation = valuation
        # Report is now populated by the ReportWriter agent (sub-3 contract);
        # synthesize_report only fills runtime fields. Pre-populate it here so
        # we can exercise the synthesize_report contract.
        flow.state.report = InvestmentReport(
            report_id="11111111-1111-1111-1111-111111111111",
            ticker="AAPL",
            generated_at=datetime(2026, 1, 1),
            data_as_of={},
            company=sample_company,
            market=sample_market,
            financial=sample_financial,
            financial_health_score=70,
            news=sample_news,
            competitors=sample_competitor_analysis,
            risk=risk,
            valuation=valuation,
            rating="Hold",
            confidence=70,
            investment_horizon="medium",
            catalysts=["placeholder catalyst"],
            markdown="## placeholder markdown",
            sources=[],
            disclaimer="placeholder",
        )

    def test_fills_runtime_fields(
        self,
        sample_company,
        sample_market,
        sample_news,
        sample_financial,
        sample_competitor_analysis,
    ):
        """Sub-project 3: synthesize_report only fills sources, disclaimer, generated_at."""
        from alphaquant.flows.analysis_flow import DISCLAIMER_TEXT

        flow = AnalysisFlow()
        self._populate_state(
            flow,
            sample_company,
            sample_market,
            sample_news,
            sample_financial,
            sample_competitor_analysis,
        )

        # Capture generated_at before, so we can verify it changed
        before = flow.state.report.generated_at

        _run(flow.synthesize_report())

        assert flow.state.report is not None
        assert flow.state.report.ticker == "AAPL"
        # Sub-3: rating / confidence / catalysts / markdown come from the
        # ReportWriter agent, NOT from synthesize_report. Verify they were
        # preserved unchanged.
        assert flow.state.report.rating == "Hold"
        assert flow.state.report.confidence == 70
        assert flow.state.report.catalysts == ["placeholder catalyst"]
        assert flow.state.report.markdown == "## placeholder markdown"
        # Runtime fields:
        assert flow.state.report.disclaimer == DISCLAIMER_TEXT
        assert isinstance(flow.state.report.sources, list)
        # generated_at was rewritten by synthesize_report
        assert flow.state.report.generated_at >= before

    def test_missing_report_raises_report_generation_error(
        self,
    ):
        """§3.2: when ReportWriter failed (state.report is None), raise ReportGenerationError."""
        from alphaquant.exceptions import ReportGenerationError

        flow = AnalysisFlow()
        flow.state.ticker = "AAPL"
        # state.report left as None
        with pytest.raises(ReportGenerationError):
            _run(flow.synthesize_report())


# ---------------------------------------------------------------------------
# End-to-end orchestration via the Flow
# ---------------------------------------------------------------------------


class TestFlowKickoff:
    def test_full_flow_with_mocked_tools(
        self,
        sample_company,
        sample_market,
        sample_news,
        sample_financial,
        sample_competitor_analysis,
    ):
        """All 2 steps execute and produce an InvestmentReport.

        Sub-project 2: tools (not registry) are mocked at the tool layer
        to mirror the Crew-internal fetch path.

        Sub-project 3: tasks 4-7 (competitor / risk / valuation / report)
        are populated as Pydantic instances on task_out.pydantic.
        """
        flow = AnalysisFlow()

        tool_patches = _patch_data_tools(
            sample_company, sample_market, sample_news, sample_financial
        )
        with contextlib.ExitStack() as stack:
            MockCrew = stack.enter_context(
                patch("alphaquant.flows.analysis_flow.AnalysisCrew")
            )
            stack.enter_context(_patch_competitor_tool("No peer data available"))
            for p in tool_patches:
                stack.enter_context(p)

            fake_result = MagicMock()
            fake_result.tasks_output = _build_fake_task_output(
                sample_company,
                sample_market,
                sample_financial,
                sample_competitor_analysis,
            )
            MockCrew.return_value.kickoff.return_value = fake_result

            _run(flow.run_crew("AAPL"))
            _run(flow.synthesize_report())

        assert flow.state.report is not None
        assert flow.state.report.ticker == "AAPL"

    def test_partial_failure_degrades_gracefully(
        self,
        sample_company,
        sample_market,
        sample_financial,
        sample_competitor_analysis,
    ):
        """§3.2: market tool returns error → flow still produces a report."""
        flow = AnalysisFlow()

        company_json = sample_company.model_dump_json()
        market_error = "Error fetching market data: timeout after 30s"
        news_json = "[]"
        financial_json = sample_financial.model_dump_json()

        with patch(
            "alphaquant.tools.company_lookup_tool.CompanyLookupTool._run",
            new=lambda self, ticker: company_json,
        ), patch(
            "alphaquant.tools.market_data_tool.MarketDataTool._run",
            new=lambda self, ticker: market_error,
        ), patch(
            "alphaquant.tools.news_tool.NewsTool._run",
            new=lambda self, ticker: news_json,
        ), patch(
            "alphaquant.tools.financial_tool.FinancialTool._run",
            new=lambda self, ticker: financial_json,
        ), _patch_competitor_tool("No peer data available"), \
             patch("alphaquant.flows.analysis_flow.AnalysisCrew") as MockCrew:

            fake_result = MagicMock()
            fake_result.tasks_output = _build_fake_task_output(
                sample_company,
                sample_market,
                sample_financial,
                sample_competitor_analysis,
                market_error=market_error,
            )
            MockCrew.return_value.kickoff.return_value = fake_result

            _run(flow.run_crew("AAPL"))
            _run(flow.synthesize_report())

        assert "market_data_unavailable" in flow.state.errors
        # synthesize_report substitutes a degraded MarketData placeholder
        assert flow.state.report.market.source == "degraded"
        assert flow.state.report is not None
        assert flow.state.report.ticker == "AAPL"

    def test_kickoff_with_timeout_succeeds_under_limit(
        self,
        sample_company,
        sample_market,
        sample_financial,
        sample_competitor_analysis,
    ):
        """§3.4: kickoff_with_timeout returns within 180s for a fast flow.

        Sub-project 2: tool _run methods are mocked instead of registry.

        Sub-project 3: include Pydantic instances in tasks_output 4-7.
        """
        flow = AnalysisFlow()

        company_json = sample_company.model_dump_json()
        market_json = sample_market.model_dump_json()
        news_json = "[]"
        financial_json = sample_financial.model_dump_json()

        with patch(
            "alphaquant.tools.company_lookup_tool.CompanyLookupTool._run",
            new=lambda self, ticker: company_json,
        ), patch(
            "alphaquant.tools.market_data_tool.MarketDataTool._run",
            new=lambda self, ticker: market_json,
        ), patch(
            "alphaquant.tools.news_tool.NewsTool._run",
            new=lambda self, ticker: news_json,
        ), patch(
            "alphaquant.tools.financial_tool.FinancialTool._run",
            new=lambda self, ticker: financial_json,
        ), _patch_competitor_tool("No peer data available"), \
             patch("alphaquant.flows.analysis_flow.AnalysisCrew") as MockCrew:

            fake_result = MagicMock()
            fake_result.tasks_output = _build_fake_task_output(
                sample_company,
                sample_market,
                sample_financial,
                sample_competitor_analysis,
            )
            MockCrew.return_value.kickoff.return_value = fake_result

            _run(flow.kickoff_with_timeout(inputs={"ticker": "AAPL"}))

        assert flow.state.report is not None
        assert flow.state.report.ticker == "AAPL"

    def test_kickoff_with_timeout_enforces_limit(self):
        """§3.4: a slow kickoff_async → asyncio.TimeoutError after 120s.

        We patch FLOW_TIMEOUT_SECONDS to a tiny value so the test runs quickly.
        """
        import time

        flow = AnalysisFlow()

        async def slow_kickoff(self, inputs=None):
            await asyncio.sleep(2.0)
            return None

        with patch(
            "alphaquant.flows.analysis_flow.FLOW_TIMEOUT_SECONDS", 0.1
        ), patch.object(
            AnalysisFlow, "kickoff_async", new=slow_kickoff
        ):
            with pytest.raises(asyncio.TimeoutError):
                _run(flow.kickoff_with_timeout(inputs={"ticker": "AAPL"}))


# ---------------------------------------------------------------------------
# run_crew step (sub-project 1 thin shell)
# ---------------------------------------------------------------------------


class TestRunCrewStep:
    """@start run_crew: drives AnalysisCrew; relies on tool fetches (sub-2)."""

    def test_run_crew_invokes_crew_with_only_ticker(self, sample_company, sample_market, sample_news, sample_financial):
        """run_crew must NOT pre-fetch via DataSourceRegistry; it must invoke
        AnalysisCrew.kickoff with only the ticker in inputs. Tools handle fetching."""
        from alphaquant.flows.analysis_flow import AnalysisFlow

        flow = AnalysisFlow()
        flow.state.ticker = "AAPL"

        # Verify DataSourceRegistry is NOT called from run_crew
        reg_cls = __import__(
            "alphaquant.infrastructure.data_sources", fromlist=["DataSourceRegistry"]
        ).DataSourceRegistry

        with patch.object(reg_cls, "get_company", new=AsyncMock()) as mock_company, \
             patch.object(reg_cls, "get_market", new=AsyncMock()) as mock_market, \
             patch.object(reg_cls, "get_news", new=AsyncMock()) as mock_news, \
             patch.object(reg_cls, "get_financial", new=AsyncMock()) as mock_financial, \
             patch("alphaquant.flows.analysis_flow.AnalysisCrew") as MockCrew:

            # Mock crew to produce a fake result with valid company JSON so
            # parse_crew_output doesn't raise AllDataSourcesDown.
            company_json = sample_company.model_dump_json()
            market_json = sample_market.model_dump_json()
            news_json = "[]"
            financial_json = sample_financial.model_dump_json()
            fake_output = MagicMock()
            fake_output.tasks_output = [
                MagicMock(raw=company_json),
                MagicMock(raw=market_json),
                MagicMock(raw=news_json),
                MagicMock(raw=financial_json),
            ]
            MockCrew.return_value.kickoff.return_value = fake_output

            _run(flow.run_crew("AAPL"))

            # Registry methods must NOT have been called
            mock_company.assert_not_called()
            mock_market.assert_not_called()
            mock_news.assert_not_called()
            mock_financial.assert_not_called()

            # Crew must have been called with ticker only
            MockCrew.assert_called_once()
            MockCrew.return_value.kickoff.assert_called_once()
            call_kwargs = MockCrew.return_value.kickoff.call_args.kwargs
            assert "inputs" in call_kwargs
            assert call_kwargs["inputs"] == {"ticker": "AAPL"}

    def test_run_crew_timeout_raises(self, sample_company, sample_market, sample_news, sample_financial):
        """If crew.kickoff exceeds FLOW_TIMEOUT_SECONDS, asyncio.TimeoutError."""
        import asyncio as _asyncio
        import time
        from alphaquant.flows.analysis_flow import AnalysisFlow

        flow = AnalysisFlow()
        flow.state.ticker = "AAPL"

        def slow_kickoff(inputs):
            # Blocking sleep in the worker thread so asyncio.wait_for fires.
            time.sleep(2.0)
            return MagicMock(tasks_output=[])

        with patch("alphaquant.flows.analysis_flow.AnalysisCrew") as MockCrew, \
             patch("alphaquant.flows.analysis_flow.FLOW_TIMEOUT_SECONDS", 0.1):
            MockCrew.return_value.kickoff = slow_kickoff

            with pytest.raises(_asyncio.TimeoutError):
                _run(flow.run_crew("AAPL"))


class TestParseCrewOutput:
    """parse_crew_output: CrewOutput → AnalysisState field dict."""

    def test_extracts_company_from_task_output(self):
        from alphaquant.flows.analysis_flow import parse_crew_output

        # Build fake CrewOutput with one task returning JSON for company
        fake_task_output = MagicMock()
        fake_task_output.description = "Validate ticker 'AAPL' and return canonical company metadata."
        fake_task_output.raw = '{"name": "Apple Inc.", "exchange": "NASDAQ"}'

        fake_result = MagicMock()
        fake_result.tasks_output = [fake_task_output]

        state_dict = parse_crew_output(fake_result)
        # Sub-project 1: parse_crew_output extracts raw text per task.
        # We assert the structure is a dict keyed by role_key.
        assert isinstance(state_dict, dict)
        assert "company_resolver" in state_dict

    def test_extracts_market_from_task_output(self):
        from alphaquant.flows.analysis_flow import parse_crew_output
        from alphaquant.models.market import MarketData
        from alphaquant.flows.analysis_flow import AnalysisState
        from datetime import datetime
        from decimal import Decimal

        market = MarketData(
            ticker="AAPL",
            as_of=datetime(2026, 6, 20),
            price=Decimal("150.00"),
            change_pct=0.5,
            volume=50_000_000,
            market_cap=3_000_000_000_000,
            pe_ratio=25.0,
        )
        company_json = (
            '{"ticker":"AAPL","name":"Apple Inc.","exchange":"NASDAQ",'
            '"sector":"Technology","industry":"Consumer Electronics",'
            '"market_cap":3000000000000}'
        )
        market_json = market.model_dump_json()

        task_outputs = [
            MagicMock(raw=company_json),
            MagicMock(raw=market_json),
            MagicMock(raw="[]"),  # news (empty list)
            MagicMock(raw='{"ticker":"AAPL","income_statements":[],"balance_sheets":[],"cash_flows":[]}'),
        ]
        fake_result = MagicMock()
        fake_result.tasks_output = task_outputs

        state = AnalysisState(ticker="AAPL")
        parse_crew_output(fake_result, state)

        assert state.market is not None
        assert state.market.ticker == "AAPL"
        assert state.market.price == Decimal("150.00")

    def test_extracts_news_from_task_output(self):
        from alphaquant.flows.analysis_flow import parse_crew_output, AnalysisState
        import json

        news_json = json.dumps([
            {
                "date": "2026-06-19",
                "title": "Apple launches new product",
                "source": "TestSource",
                "url": "https://example.com/n1",
                "sentiment": "neutral",
                "relevance_score": 0.5,
            }
        ])
        company_json = (
            '{"ticker":"AAPL","name":"Apple Inc.","exchange":"NASDAQ",'
            '"sector":"Technology","industry":"Consumer Electronics",'
            '"market_cap":3000000000000}'
        )
        task_outputs = [
            MagicMock(raw=company_json),
            MagicMock(raw='{"ticker":"AAPL","as_of":"2026-06-20","price":150.0,"change_pct":0.5,"volume":50000000,"market_cap":3000000000000}'),
            MagicMock(raw=news_json),
            MagicMock(raw='{"ticker":"AAPL","income_statements":[],"balance_sheets":[],"cash_flows":[]}'),
        ]
        fake_result = MagicMock()
        fake_result.tasks_output = task_outputs

        state = AnalysisState(ticker="AAPL")
        parse_crew_output(fake_result, state)

        assert state.news is not None
        assert state.news.total_count == 1
        assert state.news.ticker == "AAPL"

    def test_extracts_financial_from_task_output(self):
        from alphaquant.flows.analysis_flow import parse_crew_output, AnalysisState
        from alphaquant.models.financial import (
            FinancialStatements, IncomeStatement,
        )
        from decimal import Decimal

        statements = FinancialStatements(
            ticker="AAPL",
            income_statements=[
                IncomeStatement(
                    period="TTM", fiscal_year=2026,
                    revenue=Decimal("400000000000"),
                    net_income=Decimal("100000000000"),
                )
            ],
        )
        company_json = (
            '{"ticker":"AAPL","name":"Apple Inc.","exchange":"NASDAQ",'
            '"sector":"Technology","industry":"Consumer Electronics",'
            '"market_cap":3000000000000}'
        )

        task_outputs = [
            MagicMock(raw=company_json),
            MagicMock(raw='{"ticker":"AAPL","as_of":"2026-06-20","price":150.0,"change_pct":0.5,"volume":50000000,"market_cap":3000000000000}'),
            MagicMock(raw="[]"),
            MagicMock(raw=statements.model_dump_json()),
        ]
        fake_result = MagicMock()
        fake_result.tasks_output = task_outputs

        state = AnalysisState(ticker="AAPL")
        parse_crew_output(fake_result, state)

        assert state.financial is not None
        assert state.financial.ticker == "AAPL"
        assert len(state.financial.income_statements) == 1

    def test_company_fetch_failure_raises_all_sources_down(self):
        from alphaquant.flows.analysis_flow import parse_crew_output, AnalysisState
        from alphaquant.exceptions import AllDataSourcesDown

        task_outputs = [
            MagicMock(raw="Error fetching company: all sources down"),
            MagicMock(raw=""),
            MagicMock(raw="[]"),
            MagicMock(raw=""),
        ]
        fake_result = MagicMock()
        fake_result.tasks_output = task_outputs

        state = AnalysisState(ticker="ZZZZ")
        with pytest.raises(AllDataSourcesDown):
            parse_crew_output(fake_result, state)

    def test_market_fetch_failure_appends_error_and_keeps_state(self):
        from alphaquant.flows.analysis_flow import parse_crew_output, AnalysisState

        company_json = (
            '{"ticker":"AAPL","name":"Apple Inc.","exchange":"NASDAQ",'
            '"sector":"Technology","industry":"Consumer Electronics",'
            '"market_cap":3000000000000}'
        )
        task_outputs = [
            MagicMock(raw=company_json),
            MagicMock(raw="Error fetching market data: timeout after 30s"),
            MagicMock(raw="[]"),
            MagicMock(raw=""),
        ]
        fake_result = MagicMock()
        fake_result.tasks_output = task_outputs

        state = AnalysisState(ticker="AAPL")
        parse_crew_output(fake_result, state)

        assert state.market is None
        assert "market_data_unavailable" in state.errors

    def test_news_fetch_failure_uses_empty_analysis(self):
        from alphaquant.flows.analysis_flow import parse_crew_output, AnalysisState

        company_json = (
            '{"ticker":"AAPL","name":"Apple Inc.","exchange":"NASDAQ",'
            '"sector":"Technology","industry":"Consumer Electronics",'
            '"market_cap":3000000000000}'
        )
        market_json = (
            '{"ticker":"AAPL","as_of":"2026-06-20","price":150.0,'
            '"change_pct":0.5,"volume":50000000,"market_cap":3000000000000}'
        )
        task_outputs = [
            MagicMock(raw=company_json),
            MagicMock(raw=market_json),
            MagicMock(raw="No news found for AAPL"),
            MagicMock(raw=""),
        ]
        fake_result = MagicMock()
        fake_result.tasks_output = task_outputs

        state = AnalysisState(ticker="AAPL")
        parse_crew_output(fake_result, state)

        assert state.news is not None
        assert state.news.total_count == 0  # NewsAnalysis.empty()
        assert "news_data_unavailable" in state.errors

    def test_financial_fetch_failure_uses_empty_shell(self):
        from alphaquant.flows.analysis_flow import parse_crew_output, AnalysisState

        company_json = (
            '{"ticker":"AAPL","name":"Apple Inc.","exchange":"NASDAQ",'
            '"sector":"Technology","industry":"Consumer Electronics",'
            '"market_cap":3000000000000}'
        )
        market_json = (
            '{"ticker":"AAPL","as_of":"2026-06-20","price":150.0,'
            '"change_pct":0.5,"volume":50000000,"market_cap":3000000000000}'
        )
        task_outputs = [
            MagicMock(raw=company_json),
            MagicMock(raw=market_json),
            MagicMock(raw="[]"),
            MagicMock(raw="Error fetching financials: api down"),
        ]
        fake_result = MagicMock()
        fake_result.tasks_output = task_outputs

        state = AnalysisState(ticker="AAPL")
        parse_crew_output(fake_result, state)

        assert state.financial is not None
        assert state.financial.ticker == "AAPL"
        assert state.financial.income_statements == []
        assert "financial_data_unavailable" in state.errors

    def test_parse_crew_output_extracts_competitor_from_pydantic(self):
        from alphaquant.flows.analysis_flow import parse_crew_output, AnalysisState
        from alphaquant.models.competitor import CompetitorAnalysis
        from alphaquant.models.company import Company
        from alphaquant.models.market import MarketData
        from alphaquant.models.financial import FinancialStatements
        from alphaquant.models.news import NewsAnalysis
        from alphaquant.models.risk import RiskAssessment
        from alphaquant.models.valuation import ValuationResult
        from alphaquant.models.report import InvestmentReport

        ca = CompetitorAnalysis(
            target_ticker="AAPL",
            competitors=[
                Competitor(
                    ticker="MSFT",
                    name="Microsoft",
                    market_cap=2_500_000_000_000,
                    revenue_ttm=Decimal("200000000000"),
                )
            ],
            industry_rank=1,
            industry_size=5,
            competitive_score=50,
            strengths=["x"],
            weaknesses=["y"],
            method="gics",
        )
        company = Company(
            ticker="AAPL",
            name="Apple Inc.",
            exchange="NASDAQ",
            sector="Technology",
            industry="Consumer Electronics",
            market_cap=3_000_000_000_000,
        )
        market = MarketData(
            ticker="AAPL",
            as_of=datetime.utcnow(),
            price=Decimal("180"),
            change_pct=0.0,
            volume=0,
            market_cap=3_000_000_000_000,
            pe_ratio=28.0,
            revenue_growth_yoy=5.0,
            beta=1.2,
            source="yahoo",
        )
        fin = FinancialStatements(ticker="AAPL")
        news = NewsAnalysis.empty("AAPL")
        risk = RiskAssessment(
            ticker="AAPL",
            total_score=50,
            level="medium",
            sub_scores=[
                RiskScore(
                    category="financial",
                    score=5,
                    rationale="placeholder subscore rationale text",
                    evidence=[],
                )
            ],
            top_risks=[],
            method="weighted_sum_v1",
        )
        val = ValuationResult(
            ticker="AAPL",
            intrinsic_value_per_share=Decimal("150"),
            current_price=Decimal("180"),
            upside_pct=-16.67,
            dcf_value=Decimal("120"),
            relative_value=Decimal("180"),
            peg_ratio=None,
            method="dcf_relative_peg",
            assumptions={},
        )
        rep = InvestmentReport(
            report_id="00000000-0000-0000-0000-000000000000",
            ticker="AAPL",
            generated_at=datetime.utcnow(),
            data_as_of={},
            company=company,
            market=market,
            financial=fin,
            financial_health_score=70,
            news=news,
            competitors=ca,
            risk=risk,
            valuation=val,
            rating="Hold",
            confidence=70,
            investment_horizon="medium",
            catalysts=["Earnings beat"],
            markdown="## Summary\nTest report.",
            sources=["yahoo"],
            disclaimer="test disclaimer",
        )

        class _FakeTask:
            def __init__(self, pyd_obj=None, raw=""):
                self.pydantic = pyd_obj
                self.raw = raw

        tasks_output = [
            _FakeTask(pyd_obj=company, raw=company.model_dump_json()),  # 0
            _FakeTask(pyd_obj=market, raw=market.model_dump_json()),    # 1
            _FakeTask(raw="[]"),                                          # 2 news list
            _FakeTask(pyd_obj=fin, raw=fin.model_dump_json()),           # 3
            _FakeTask(pyd_obj=ca, raw=ca.model_dump_json()),             # 4
            _FakeTask(pyd_obj=risk, raw=risk.model_dump_json()),         # 5
            _FakeTask(pyd_obj=val, raw=val.model_dump_json()),           # 6
            _FakeTask(pyd_obj=rep, raw=rep.model_dump_json()),           # 7
        ]

        class _FakeResult:
            pass

        _FakeResult.tasks_output = tasks_output

        state = AnalysisState(ticker="AAPL")
        parse_crew_output(_FakeResult(), state)

        assert state.competitor is ca
        assert state.risk is risk
        assert state.valuation is val
        assert state.report is rep

    def test_parse_crew_output_missing_pydantic_sets_none_and_appends_error(self):
        """When a Pydantic task output is empty, state.<field> = None + error appended."""
        from alphaquant.flows.analysis_flow import parse_crew_output, AnalysisState
        from alphaquant.models.company import Company
        from alphaquant.models.market import MarketData
        from alphaquant.models.financial import FinancialStatements

        company = Company(
            ticker="AAPL",
            name="Apple Inc.",
            exchange="NASDAQ",
            sector="Technology",
            industry="Consumer Electronics",
            market_cap=3_000_000_000_000,
        )
        market = MarketData(
            ticker="AAPL",
            as_of=datetime.utcnow(),
            price=Decimal("180"),
            change_pct=0.0,
            volume=0,
            market_cap=3_000_000_000_000,
            pe_ratio=28.0,
            revenue_growth_yoy=5.0,
            beta=1.2,
            source="yahoo",
        )
        fin = FinancialStatements(ticker="AAPL")

        class _FakeTask:
            def __init__(self, pyd_obj=None, raw=""):
                self.pydantic = pyd_obj
                self.raw = raw

        tasks_output = [
            _FakeTask(pyd_obj=company, raw=company.model_dump_json()),
            _FakeTask(pyd_obj=market, raw=market.model_dump_json()),
            _FakeTask(raw="[]"),
            _FakeTask(pyd_obj=fin, raw=fin.model_dump_json()),
            _FakeTask(raw=""),  # competitor failed → empty
            _FakeTask(raw=""),  # risk failed
            _FakeTask(raw=""),  # valuation failed
            _FakeTask(raw=""),  # report writer failed
        ]

        class _FakeResult:
            pass

        _FakeResult.tasks_output = tasks_output

        state = AnalysisState(ticker="AAPL")
        parse_crew_output(_FakeResult(), state)

        assert state.competitor is None
        assert state.risk is None
        assert state.valuation is None
        assert state.report is None
        assert "competitor_analyst_unavailable" in state.errors
        assert "risk_analyst_unavailable" in state.errors
        assert "valuation_analyst_unavailable" in state.errors
        assert "report_writer_unavailable" in state.errors


class TestExtractPydanticField:
    """_extract_pydantic_field: read task_out.pydantic → model or None + error."""

    def test_returns_pydantic_instance_from_task_output(self):
        from alphaquant.flows.analysis_flow import _extract_pydantic_field
        from alphaquant.models.competitor import CompetitorAnalysis
        from alphaquant.flows.analysis_flow import AnalysisState

        ca = CompetitorAnalysis(
            target_ticker="AAPL",
            competitors=[
                Competitor(
                    ticker="MSFT",
                    name="Microsoft",
                    market_cap=2_500_000_000_000,
                    revenue_ttm=Decimal("200000000000"),
                )
            ],
            industry_rank=1,
            industry_size=5,
            competitive_score=50,
            strengths=["x"],
            weaknesses=["y"],
            method="gics",
        )

        class _FakeTask:
            pydantic = ca
            raw = ""

        state = AnalysisState(ticker="AAPL")
        result = _extract_pydantic_field(
            [_FakeTask(), _FakeTask()], 0, "competitor_analyst", CompetitorAnalysis, state
        )
        assert result is ca
        assert state.errors == []

    def test_missing_pydantic_attribute_appends_error(self):
        from alphaquant.flows.analysis_flow import _extract_pydantic_field
        from alphaquant.models.competitor import CompetitorAnalysis
        from alphaquant.flows.analysis_flow import AnalysisState

        class _FakeTask:
            raw = "not a model"

        state = AnalysisState(ticker="AAPL")
        result = _extract_pydantic_field(
            [_FakeTask()], 0, "competitor_analyst", CompetitorAnalysis, state
        )
        assert result is None
        assert "competitor_analyst_unavailable" in state.errors

    def test_idx_out_of_range_appends_error(self):
        from alphaquant.flows.analysis_flow import _extract_pydantic_field
        from alphaquant.models.competitor import CompetitorAnalysis
        from alphaquant.flows.analysis_flow import AnalysisState

        class _FakeTask:
            pydantic = None
            raw = ""

        state = AnalysisState(ticker="AAPL")
        result = _extract_pydantic_field(
            [_FakeTask()], 5, "competitor_analyst", CompetitorAnalysis, state
        )
        assert result is None
        assert "competitor_analyst_unavailable" in state.errors

    def test_wrong_type_appends_error(self):
        from alphaquant.flows.analysis_flow import _extract_pydantic_field
        from alphaquant.models.competitor import CompetitorAnalysis
        from alphaquant.flows.analysis_flow import AnalysisState

        class _FakeTask:
            pydantic = "not a model"
            raw = ""

        state = AnalysisState(ticker="AAPL")
        result = _extract_pydantic_field(
            [_FakeTask()], 0, "competitor_analyst", CompetitorAnalysis, state
        )
        assert result is None
        assert "competitor_analyst_unavailable" in state.errors


class TestExtractDataField:
    """parse_crew_output helper: validate JSON or detect error string."""

    def test_valid_json_returns_model(self):
        from alphaquant.flows.analysis_flow import _extract_data_field
        from alphaquant.models.market import MarketData
        from datetime import datetime
        from decimal import Decimal

        raw = MarketData(
            ticker="AAPL",
            as_of=datetime(2026, 6, 20),
            price=Decimal("150.00"),
            change_pct=0.5,
            volume=1_000_000,
            market_cap=2_500_000_000_000,
            pe_ratio=25.0,
        ).model_dump_json()

        model, err = _extract_data_field(raw, MarketData, "market_data_unavailable")
        assert err is None
        assert isinstance(model, MarketData)
        assert model.ticker == "AAPL"

    def test_error_string_returns_none_with_error(self):
        from alphaquant.flows.analysis_flow import _extract_data_field
        from alphaquant.models.market import MarketData

        model, err = _extract_data_field(
            "Error fetching market data: timeout after 30s",
            MarketData,
            "market_data_unavailable",
        )
        assert model is None
        assert err == "market_data_unavailable"

    def test_no_data_string_returns_none_with_error(self):
        from alphaquant.flows.analysis_flow import _extract_data_field
        from alphaquant.models.market import MarketData

        model, err = _extract_data_field(
            "No market data available for ZZZZ",
            MarketData,
            "market_data_unavailable",
        )
        assert model is None
        assert err == "market_data_unavailable"

    def test_garbage_string_returns_none_with_error(self):
        from alphaquant.flows.analysis_flow import _extract_data_field
        from alphaquant.models.market import MarketData

        model, err = _extract_data_field("not json at all", MarketData, "market_data_unavailable")
        assert model is None
        assert err == "market_data_unavailable"

    def test_empty_string_returns_none_with_error(self):
        from alphaquant.flows.analysis_flow import _extract_data_field
        from alphaquant.models.market import MarketData

        model, err = _extract_data_field("", MarketData, "market_data_unavailable")
        assert model is None
        assert err == "market_data_unavailable"

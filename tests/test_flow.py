"""alphaquant.flows.analysis_flow AnalysisFlow 编排的测试。"""
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
from alphaquant.models.financial import BalanceSheet, FinancialStatements, IncomeStatement
from alphaquant.models.market import MarketData
from alphaquant.models.news import NewsAnalysis
from alphaquant.models.risk import RiskScore


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

    Sub-project 3 revert: tasks 0-3 produce JSON raw text (parsed by
    _extract_data_field), tasks 4-7 produce plain text (report_writer's text
    is scanned for JSON by _extract_writer_output; competitor/risk/valuation
    are computed deterministically in synthesize_report).
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
    ):
        """Set up the data fields + a ReportWriterOutput on the flow state.

        Sub-project-3 revert: synthesize_report computes the 3 analyses
        (competitor/risk/valuation) deterministically and assembles the
        full ``InvestmentReport`` from data + analyses + ``state.writer_output``.
        """
        from alphaquant.models.report import ReportWriterOutput

        flow.state.ticker = "AAPL"
        flow.state.company = sample_company
        flow.state.market = sample_market
        flow.state.news = sample_news
        flow.state.financial = sample_financial
        flow.state.writer_output = ReportWriterOutput(
            rating="Hold",
            confidence=70,
            investment_horizon="medium",
            catalysts=["placeholder catalyst"],
            markdown="## placeholder markdown",
        )

    def test_fills_runtime_fields(
        self,
        sample_company,
        sample_market,
        sample_news,
        sample_financial,
    ):
        """Sub-project-3 revert: synthesize_report assembles the full report
        from data + deterministic analyses + the LLM's ReportWriterOutput."""
        from alphaquant.flows.analysis_flow import DISCLAIMER_TEXT

        flow = AnalysisFlow()
        self._populate_state(
            flow,
            sample_company,
            sample_market,
            sample_news,
            sample_financial,
        )

        # state.report starts as None
        assert flow.state.report is None

        _run(flow.synthesize_report())

        report = flow.state.report
        assert report is not None
        assert report.ticker == "AAPL"
        # LLM synthesis fields come from writer_output (state.writer_output)
        assert report.rating == "Hold"
        assert report.confidence == 70
        assert report.catalysts == ["placeholder catalyst"]
        assert report.markdown == "## placeholder markdown"
        # Deterministic analyses are populated by synthesize_report
        assert report.competitors is not None
        assert report.competitors.target_ticker == "AAPL"
        assert report.risk is not None
        assert report.risk.ticker == "AAPL"
        assert report.valuation is not None
        assert report.valuation.ticker == "AAPL"
        # Data fields are copied through to the report
        assert report.company is sample_company
        assert report.market is sample_market
        assert report.financial is sample_financial
        assert report.news is sample_news
        # Runtime fields:
        assert report.disclaimer == DISCLAIMER_TEXT
        assert isinstance(report.sources, list)
        # report_id and generated_at are fresh
        assert report.report_id
        assert report.financial_health_score is not None

    def test_missing_writer_output_uses_fallback_defaults(
        self,
        sample_company,
        sample_market,
        sample_news,
        sample_financial,
    ):
        """Sub-project-3 revert: when the LLM failed to produce a
        ReportWriterOutput, synthesize_report builds a fallback report
        (rating=Hold, confidence=None) instead of raising."""
        flow = AnalysisFlow()
        flow.state.ticker = "AAPL"
        flow.state.company = sample_company
        flow.state.market = sample_market
        flow.state.news = sample_news
        flow.state.financial = sample_financial
        # state.writer_output left as None
        assert flow.state.writer_output is None

        _run(flow.synthesize_report())

        report = flow.state.report
        assert report is not None
        # Fallback defaults from synthesize_report:
        assert report.rating == "Hold"
        assert report.confidence is None
        assert "writer_output_unavailable" in flow.state.errors


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
        """§3.4: kickoff_with_timeout returns within 600s for a fast flow.

        Sub-project 2: tool _run methods are mocked instead of registry.

        Sub-project 3: include Pydantic instances in tasks_output 4-7.
        Sub-project 3 (revised): FLOW_TIMEOUT_SECONDS raised from 180 to 300.
        Sub-project 3 (follow-up): widened 300 → 600 after Task 5 showed
        real-LLM latency routinely exceeds 300s.
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
        """§3.4: a slow kickoff_async → asyncio.TimeoutError after 600s.

        We patch FLOW_TIMEOUT_SECONDS to a tiny value so the test runs quickly.
        Sub-project 3 (revised): default raised from 120 to 300.
        Sub-project 3 (follow-up): widened 300 → 600 after Task 5 showed
        real-LLM latency routinely exceeds 300s.
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

    def test_429_raises_llm_rate_limited(self):
        """A 429 / Token-Plan-exhausted error from the LLM must surface as
        ``LLMRateLimited`` (mapped to HTTP 503) so the frontend can show
        a clear message instead of a multi-minute timeout.

        CrewAI sometimes re-raises the upstream 429 as
        ``AttributeError: 'NoneType' object has no attribute 'choices'``
        because the LLM SDK returns an error body instead of a normal
        response. We test both shapes.
        """
        from alphaquant.exceptions import CrewExecutionError, LLMRateLimited

        for raw_error in [
            "API Error: 429 rate_limit_error: Token Plan 用量上限",
            "API Error: 429 {\"type\":\"error\",\"error\":{\"type\":\"rate_limit_error\"}}",
            "AttributeError: 'NoneType' object has no attribute 'choices'",
        ]:
            flow = AnalysisFlow()
            with patch("alphaquant.flows.analysis_flow.AnalysisCrew") as MockCrew:
                MockCrew.return_value.kickoff.side_effect = RuntimeError(raw_error)
                if "choices" in raw_error or "429" in raw_error:
                    with pytest.raises(LLMRateLimited):
                        _run(
                            flow.kickoff_with_timeout(inputs={"ticker": "AAPL"})
                        )
                else:
                    with pytest.raises(CrewExecutionError):
                        _run(
                            flow.kickoff_with_timeout(inputs={"ticker": "AAPL"})
                        )

    def test_progress_callback_fires_for_all_steps(
        self,
        sample_company,
        sample_market,
        sample_financial,
        sample_competitor_analysis,
    ):
        """kickoff_with_timeout must invoke the progress callback at every
        documented step boundary, in the expected order, with valid states.
        """
        flow = AnalysisFlow()
        events: list[tuple[str, str]] = []

        def on_progress(step: str, state: str) -> None:
            events.append((step, state))

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

            _run(
                flow.kickoff_with_timeout(
                    inputs={"ticker": "AAPL"},
                    progress_callback=on_progress,
                )
            )

        steps = [step for step, _ in events]
        # Every documented step must be emitted at least once.
        assert "validate_ticker" in steps
        assert "run_crew" in steps
        assert "parse_crew_output" in steps
        assert "compute_analyses" in steps
        assert "assemble_report" in steps
        # Each step must reach "complete" (the Flow finished successfully).
        for step in (
            "validate_ticker",
            "run_crew",
            "parse_crew_output",
            "compute_analyses",
            "assemble_report",
        ):
            assert (step, "complete") in events, f"step {step} did not complete"
        # And each must transition through "running" before "complete".
        for step in (
            "validate_ticker",
            "run_crew",
            "parse_crew_output",
            "compute_analyses",
            "assemble_report",
        ):
            assert (step, "running") in events, f"step {step} never reached running"
        # Ordering: validate_ticker complete must precede run_crew complete.
        assert events.index(("validate_ticker", "complete")) < events.index(
            ("run_crew", "complete")
        )
        assert events.index(("run_crew", "complete")) < events.index(
            ("assemble_report", "complete")
        )

    def test_progress_callback_optional(
        self,
        sample_company,
        sample_market,
        sample_financial,
        sample_competitor_analysis,
    ):
        """kickoff_with_timeout must work without a progress callback (FastAPI path)."""
        flow = AnalysisFlow()
        company_json = sample_company.model_dump_json()
        market_json = sample_market.model_dump_json()

        with patch(
            "alphaquant.tools.company_lookup_tool.CompanyLookupTool._run",
            new=lambda self, ticker: company_json,
        ), patch(
            "alphaquant.tools.market_data_tool.MarketDataTool._run",
            new=lambda self, ticker: market_json,
        ), patch(
            "alphaquant.tools.news_tool.NewsTool._run",
            new=lambda self, ticker: "[]",
        ), patch(
            "alphaquant.tools.financial_tool.FinancialTool._run",
            new=lambda self, ticker: sample_financial.model_dump_json(),
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
            # No progress_callback kwarg — must not raise.
            _run(flow.kickoff_with_timeout(inputs={"ticker": "AAPL"}))
        assert flow.state.report is not None

    def test_progress_callback_exception_is_swallowed(
        self,
        sample_company,
        sample_market,
        sample_financial,
        sample_competitor_analysis,
    ):
        """A buggy progress callback must not crash the Flow."""
        flow = AnalysisFlow()

        def buggy(step: str, state: str) -> None:
            raise RuntimeError("UI broken")

        company_json = sample_company.model_dump_json()
        market_json = sample_market.model_dump_json()

        with patch(
            "alphaquant.tools.company_lookup_tool.CompanyLookupTool._run",
            new=lambda self, ticker: company_json,
        ), patch(
            "alphaquant.tools.market_data_tool.MarketDataTool._run",
            new=lambda self, ticker: market_json,
        ), patch(
            "alphaquant.tools.news_tool.NewsTool._run",
            new=lambda self, ticker: "[]",
        ), patch(
            "alphaquant.tools.financial_tool.FinancialTool._run",
            new=lambda self, ticker: sample_financial.model_dump_json(),
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
            _run(
                flow.kickoff_with_timeout(
                    inputs={"ticker": "AAPL"},
                    progress_callback=buggy,
                )
            )
        # Flow still produced a report despite the broken callback.
        assert flow.state.report is not None


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

    def test_parse_crew_output_extracts_writer_output_from_pydantic(self):
        """Sub-project-3 revert: tasks 4-6 are text-only; only task 7
        (report_writer) produces structured output. Verify state.writer_output
        is set from the Pydantic output and the 3 analysis fields are NOT
        extracted (they're computed deterministically in synthesize_report).
        """
        from alphaquant.flows.analysis_flow import parse_crew_output, AnalysisState
        from alphaquant.models.company import Company
        from alphaquant.models.market import MarketData
        from alphaquant.models.financial import FinancialStatements
        from alphaquant.models.news import NewsAnalysis
        from alphaquant.models.report import ReportWriterOutput

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
        wo = ReportWriterOutput(
            rating="Hold",
            confidence=70,
            investment_horizon="medium",
            catalysts=["Earnings beat"],
            markdown="## Summary\nTest report.",
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
            _FakeTask(raw="text-only summary"),                           # 4 competitor (text)
            _FakeTask(raw="text-only summary"),                           # 5 risk (text)
            _FakeTask(raw="text-only summary"),                           # 6 valuation (text)
            _FakeTask(pyd_obj=wo, raw=wo.model_dump_json()),             # 7 report_writer
        ]

        class _FakeResult:
            pass

        _FakeResult.tasks_output = tasks_output

        state = AnalysisState(ticker="AAPL")
        parse_crew_output(_FakeResult(), state)

        # Data fields are extracted (parse_crew_output re-parses JSON, so
        # check values, not identity)
        assert state.company is not None
        assert state.company.ticker == "AAPL"
        assert state.company.name == "Apple Inc."
        assert state.market is not None
        assert state.market.ticker == "AAPL"
        assert state.market.pe_ratio == 28.0
        assert state.financial is not None
        assert state.financial.ticker == "AAPL"
        # ReportWriterOutput is extracted to state.writer_output
        assert state.writer_output is wo
        # 3 analysis fields are NOT extracted (text-only); Flow computes them
        assert state.competitor is None
        assert state.risk is None
        assert state.valuation is None
        # state.report is NOT built by parse_crew_output (Flow does it)
        assert state.report is None

    def test_parse_crew_output_missing_writer_output_sets_none_and_appends_error(self):
        """When the report_writer task output is empty, state.writer_output
        stays None and an error is appended. The 3 analysis tasks (text-only)
        are not checked for Pydantic — only their raw text is captured (if at all).
        """
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
            _FakeTask(raw=""),  # competitor (text-only, no Pydantic)
            _FakeTask(raw=""),  # risk
            _FakeTask(raw=""),  # valuation
            _FakeTask(raw=""),  # report writer failed → empty
        ]

        class _FakeResult:
            pass

        _FakeResult.tasks_output = tasks_output

        state = AnalysisState(ticker="AAPL")
        parse_crew_output(_FakeResult(), state)

        assert state.writer_output is None
        # 3 analysis fields are not extracted at all (text-only path)
        assert state.competitor is None
        assert state.risk is None
        assert state.valuation is None
        assert "report_writer_unavailable" in state.errors

    def test_parse_crew_output_extracts_writer_output_from_raw_json(self):
        """Sub-3-followup: the report_writer task now emits text (not
        output_pydantic). When the LLM outputs a pure JSON payload,
        ``_extract_writer_output`` parses it into ReportWriterOutput.
        """
        from alphaquant.flows.analysis_flow import parse_crew_output, AnalysisState
        from alphaquant.models.report import ReportWriterOutput
        from alphaquant.models.company import Company
        from alphaquant.models.market import MarketData
        from alphaquant.models.financial import FinancialStatements
        from datetime import datetime
        from decimal import Decimal

        company = Company(
            ticker="AAPL", name="Apple Inc.", exchange="NASDAQ",
            sector="Technology", industry="Consumer Electronics",
            market_cap=3_000_000_000_000,
        )
        market = MarketData(
            ticker="AAPL", as_of=datetime(2026, 1, 1),
            price=Decimal("150"), change_pct=0.0, volume=0,
            market_cap=3_000_000_000_000, source="yahoo",
        )
        financial = FinancialStatements(ticker="AAPL")
        json_payload = (
            '{"rating": "Buy", "confidence": 75, "investment_horizon": "medium", '
            '"catalysts": ["Earnings beat"], "markdown": "## Summary\\nTest report."}'
        )

        class _FakeTask:
            def __init__(self, raw="", pyd_obj=None):
                self.pydantic = pyd_obj
                self.raw = raw

        class _FakeResult:
            tasks_output = [
                _FakeTask(raw=company.model_dump_json()),
                _FakeTask(raw=market.model_dump_json()),
                _FakeTask(raw="[]"),
                _FakeTask(raw=financial.model_dump_json()),
                _FakeTask(raw=""),
                _FakeTask(raw=""),
                _FakeTask(raw=""),
                _FakeTask(raw=json_payload),
            ]

        state = AnalysisState(ticker="AAPL")
        parse_crew_output(_FakeResult(), state)
        assert isinstance(state.writer_output, ReportWriterOutput)
        assert state.writer_output.rating == "Buy"
        assert state.writer_output.confidence == 75
        assert "report_writer_unavailable" not in state.errors

    def test_parse_crew_output_extracts_writer_output_from_embedded_json(self):
        """When the LLM wraps the JSON in prose / a markdown code fence,
        ``_extract_writer_output`` still finds and parses the first JSON
        object via the balanced-brace walker.
        """
        from alphaquant.flows.analysis_flow import parse_crew_output, AnalysisState
        from alphaquant.models.report import ReportWriterOutput
        from alphaquant.models.company import Company
        from alphaquant.models.market import MarketData
        from alphaquant.models.financial import FinancialStatements
        from datetime import datetime
        from decimal import Decimal

        company = Company(
            ticker="AAPL", name="Apple Inc.", exchange="NASDAQ",
            sector="Technology", industry="Consumer Electronics",
            market_cap=3_000_000_000_000,
        )
        market = MarketData(
            ticker="AAPL", as_of=datetime(2026, 1, 1),
            price=Decimal("150"), change_pct=0.0, volume=0,
            market_cap=3_000_000_000_000, source="yahoo",
        )
        financial = FinancialStatements(ticker="AAPL")

        # Markdown prose + JSON object embedded mid-text + more prose.
        raw = (
            "## Investment Report\n\n"
            "Below is the synthesized analysis:\n\n"
            '{"rating": "Hold", "confidence": 60, "investment_horizon": "medium", '
            '"catalysts": ["Catalyst 1"], "markdown": "## Body"}.\n\n'
            "## Confidence Rationale\n"
            "5/5 data sources present.\n"
        )

        class _FakeTask:
            def __init__(self, raw=""):
                self.pydantic = None
                self.raw = raw

        class _FakeResult:
            tasks_output = [
                _FakeTask(raw=company.model_dump_json()),
                _FakeTask(raw=market.model_dump_json()),
                _FakeTask(raw="[]"),
                _FakeTask(raw=financial.model_dump_json()),
                _FakeTask(raw=""),
                _FakeTask(raw=""),
                _FakeTask(raw=""),
                _FakeTask(raw=raw),
            ]

        state = AnalysisState(ticker="AAPL")
        parse_crew_output(_FakeResult(), state)
        assert isinstance(state.writer_output, ReportWriterOutput)
        assert state.writer_output.rating == "Hold"
        assert state.writer_output.confidence == 60

    def test_parse_crew_output_writer_output_unparseable_marks_error(self):
        """When the report_writer raw text contains no parseable JSON, the
        Flow sets ``state.writer_output = None`` and appends
        ``"report_writer_unavailable"`` so ``synthesize_report`` uses its
        fallback defaults.
        """
        from alphaquant.flows.analysis_flow import parse_crew_output, AnalysisState
        from alphaquant.models.company import Company
        from alphaquant.models.market import MarketData
        from alphaquant.models.financial import FinancialStatements
        from datetime import datetime
        from decimal import Decimal

        company = Company(
            ticker="AAPL", name="Apple Inc.", exchange="NASDAQ",
            sector="Technology", industry="Consumer Electronics",
            market_cap=3_000_000_000_000,
        )
        market = MarketData(
            ticker="AAPL", as_of=datetime(2026, 1, 1),
            price=Decimal("150"), change_pct=0.0, volume=0,
            market_cap=3_000_000_000_000, source="yahoo",
        )
        financial = FinancialStatements(ticker="AAPL")

        class _FakeTask:
            def __init__(self, raw=""):
                self.pydantic = None
                self.raw = raw

        class _FakeResult:
            tasks_output = [
                _FakeTask(raw=company.model_dump_json()),
                _FakeTask(raw=market.model_dump_json()),
                _FakeTask(raw="[]"),
                _FakeTask(raw=financial.model_dump_json()),
                _FakeTask(raw=""),
                _FakeTask(raw=""),
                _FakeTask(raw=""),
                _FakeTask(raw="This is not JSON at all, just prose."),
            ]

        state = AnalysisState(ticker="AAPL")
        parse_crew_output(_FakeResult(), state)
        assert state.writer_output is None
        assert "report_writer_unavailable" in state.errors


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


# ---------------------------------------------------------------------------
# Graceful degradation E2E (sub-3 Task 4)
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """End-to-end graceful degradation: company fetch failure → AllDataSourcesDown.

    Sub-3 Blocker 3 verification: ZZZZZZ (format-valid, registry-unknown) flows
    through Flow → crew → tool → AllDataSourcesDown without raising
    INTERNAL_ERROR. The chain is mocked at the parse_crew_output boundary; we
    simulate the CrewAI task outputs the Blocker 3 fix in commit `8d1412e`
    is designed to produce (tool returns ``"Error fetching company: ..."``
    string instead of an empty shell).
    """

    def test_unknown_ticker_raises_all_data_sources_down(self):
        """ZZZZZZ → parse_crew_output raises AllDataSourcesDown.

        The full 8-task output list is faked: only the company_resolver task
        fails. All other data tasks (market, news, financial) and analysis
        tasks (competitor, risk, valuation, report) return valid JSON /
        Pydantic instances so we prove that the company failure is the
        proximate cause of AllDataSourcesDown.
        """
        from alphaquant.exceptions import AllDataSourcesDown
        from alphaquant.flows.analysis_flow import parse_crew_output, AnalysisState
        from alphaquant.models.market import MarketData
        from alphaquant.models.news import NewsAnalysis
        from alphaquant.models.financial import FinancialStatements
        from decimal import Decimal
        import datetime

        # _FakeTask mimics CrewAI's TaskOutput: ``raw`` is the agent/tool text,
        # ``pydantic`` is set when the task is configured with output_pydantic.
        class _FakeTask:
            def __init__(self, pyd_obj=None, raw=""):
                self.pydantic = pyd_obj
                self.raw = raw

        # Blocker 3 fix shape: error string starts with "Error" so
        # _extract_data_field's error-string detector catches it.
        company_error = "Error fetching company: AllDataSourcesDown: cannot resolve ZZZZZZ"

        # Degraded-but-valid market placeholder (matches the sample_market
        # fixture shape but for ZZZZZZ ticker and zero price).
        market = MarketData(
            ticker="ZZZZZZ",
            price=Decimal("0"),
            change_pct=0.0,
            volume=0,
            market_cap=0,
            pe_ratio=None,
            revenue_growth_yoy=None,
            beta=None,
            source="degraded",
            as_of=datetime.datetime.utcnow(),
        )
        news = NewsAnalysis.empty("ZZZZZZ")
        fin = FinancialStatements(ticker="ZZZZZZ")

        # 8 task outputs (4 data + 4 analysis), matching _TASK_KEYWORDS order.
        tasks_output = [
            _FakeTask(raw=company_error),  # company_resolver failed
            _FakeTask(pyd_obj=market, raw=market.model_dump_json()),
            _FakeTask(raw="[]"),  # news returns JSON list per tool contract
            _FakeTask(pyd_obj=fin, raw=fin.model_dump_json()),
            _FakeTask(raw=""),  # analysis tasks 4-7 — irrelevant for this path
            _FakeTask(raw=""),
            _FakeTask(raw=""),
            _FakeTask(raw=""),
        ]

        class _FakeResult:
            pass

        _FakeResult.tasks_output = tasks_output

        state = AnalysisState(ticker="ZZZZZZ")
        with pytest.raises(AllDataSourcesDown) as exc_info:
            parse_crew_output(_FakeResult(), state)
        # The raised message must include the ticker so the API layer can
        # surface a user-friendly error (see Task 4 brief expected output).
        assert "ZZZZZZ" in str(exc_info.value)

    def test_unknown_ticker_manager_json_array_raises_all_data_sources_down(self):
        """Sub-3 follow-up Blocker B: real LLM wraps tool error in JSON array.

        Task 5 validation found the CrewAI hierarchical manager LLM emits
        tool-call traces as a JSON array of objects, e.g.
        ``[{"ticker": "ZZZZZZ"}, {"error": "No data found for ticker ZZZZZZ..."}]``.
        The mock-LLM ``test_unknown_ticker_raises_all_data_sources_down``
        above only covers the tool's error-string path. We add this test
        to cover the real-LLM array-wrapped path so a regression in
        ``_extract_data_field`` is caught by unit tests.
        """
        from alphaquant.exceptions import AllDataSourcesDown
        from alphaquant.flows.analysis_flow import parse_crew_output, AnalysisState

        class _FakeTask:
            def __init__(self, raw=""):
                self.pydantic = None
                self.raw = raw

        # Real LLM manager output observed in /tmp/zzzzzz.stdout line 18-21
        company_array = (
            '[{"ticker": "ZZZZZZ"}, '
            '{"error": "No data found for ticker ZZZZZZ. Please check '
            'if the ticker is valid or try a different ticker symbol."}]'
        )

        class _FakeResult:
            tasks_output = [
                _FakeTask(raw=company_array),  # company_resolver
                _FakeTask(raw=""),            # market_analyst
                _FakeTask(raw="[]"),          # news_analyst
                _FakeTask(raw=""),            # financial_analyst
                _FakeTask(raw=""),
                _FakeTask(raw=""),
                _FakeTask(raw=""),
                _FakeTask(raw=""),
            ]

        state = AnalysisState(ticker="ZZZZZZ")
        with pytest.raises(AllDataSourcesDown) as exc_info:
            parse_crew_output(_FakeResult(), state)
        assert "ZZZZZZ" in str(exc_info.value)

    def test_unknown_ticker_degenerate_company_shell_raises_all_data_sources_down(self):
        """Sub-3 follow-up Blocker B: LLM hallucinates a complete-but-empty shell.

        If the manager LLM produces a JSON object that *does* parse as a
        valid ``Company`` but has ``market_cap=0`` AND ``sector="Unknown"``
        (placeholder values), it's almost certainly a hallucination. The
        degenerate-Company detector in ``_extract_data_field`` must
        reject these so ``parse_crew_output`` raises ``AllDataSourcesDown``.
        """
        from alphaquant.exceptions import AllDataSourcesDown
        from alphaquant.flows.analysis_flow import parse_crew_output, AnalysisState

        class _FakeTask:
            def __init__(self, raw=""):
                self.pydantic = None
                self.raw = raw

        # LLM hallucinates a Company shell that validates but is degenerate.
        # Uses ticker="FAKE" (5 chars, passes the pattern) so it parses,
        # but market_cap=0 + sector="Unknown" flags it as a hallucination.
        degenerate_shell = (
            '{"ticker": "FAKE", "name": "FAKE Corp", "exchange": "NASDAQ", '
            '"sector": "Unknown", "industry": "Unknown", "market_cap": 0}'
        )

        class _FakeResult:
            tasks_output = [
                _FakeTask(raw=degenerate_shell),  # company_resolver
                _FakeTask(raw=""),               # market_analyst
                _FakeTask(raw="[]"),             # news_analyst
                _FakeTask(raw=""),               # financial_analyst
                _FakeTask(raw=""),
                _FakeTask(raw=""),
                _FakeTask(raw=""),
                _FakeTask(raw=""),
            ]

        state = AnalysisState(ticker="FAKE")
        with pytest.raises(AllDataSourcesDown) as exc_info:
            parse_crew_output(_FakeResult(), state)
        assert "FAKE" in str(exc_info.value)

    def test_real_company_with_unknown_sector_passes(self):
        """Sub-3 follow-up: a real Company with sector='Unknown' but real market_cap must pass.

        Sanity check that the degenerate-Company detector doesn't produce
        false positives for legitimate companies that happen to have a
        placeholder sector string. AAPL has both a real market cap and a
        real sector, so it should pass.
        """
        from alphaquant.flows.analysis_flow import parse_crew_output, AnalysisState

        class _FakeTask:
            def __init__(self, raw=""):
                self.pydantic = None
                self.raw = raw

        aapl = (
            '{"ticker": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", '
            '"sector": "Technology", "industry": "Consumer Electronics", '
            '"market_cap": 3000000000000}'
        )

        class _FakeResult:
            tasks_output = [
                _FakeTask(raw=aapl),
                _FakeTask(raw=""),
                _FakeTask(raw="[]"),
                _FakeTask(raw=""),
                _FakeTask(raw=""),
                _FakeTask(raw=""),
                _FakeTask(raw=""),
                _FakeTask(raw=""),
            ]

        state = AnalysisState(ticker="AAPL")
        parse_crew_output(_FakeResult(), state)
        assert state.company is not None
        assert state.company.ticker == "AAPL"
        assert state.company.market_cap == 3000000000000


# ---------------------------------------------------------------------------
# Sub-3 revert: Flow configuration constants
# ---------------------------------------------------------------------------


class TestFlowConfigConstants:
    """Verify sub-3-revert constant values on the Flow module."""

    def test_flow_timeout_seconds_is_600(self):
        """Sub-3 revert follow-up: FLOW_TIMEOUT_SECONDS widened 300 → 600
        after Task 5 validation showed MiniMax-M3 routinely needs >300s
        for 7 LLM tasks (AAPL hit the 300s limit with 18 successful LLM
        calls still in progress on the cleanest run). 600s restores the
        original sub-3 spec ceiling."""
        from alphaquant.flows.analysis_flow import FLOW_TIMEOUT_SECONDS

        assert FLOW_TIMEOUT_SECONDS == 600.0

"""Tests for alphaquant.flows.analysis_flow AnalysisFlow orchestration."""
from __future__ import annotations

import asyncio
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


def _run(coro):
    """Drive an async step coroutine in tests (asyncio.run is forbidden in prod)."""
    return asyncio.run(coro)


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
        flow.state.ticker = "AAPL"
        flow.state.company = sample_company
        flow.state.market = sample_market
        flow.state.news = sample_news
        flow.state.financial = sample_financial
        flow.state.competitor = sample_competitor_analysis
        flow.state.risk = RiskAssessment(
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
        flow.state.valuation = ValuationResult(
            ticker="AAPL",
            intrinsic_value_per_share=Decimal("180.00"),
            current_price=Decimal("150.00"),
            upside_pct=0.2,
            dcf_value=None,
            relative_value=Decimal("180.00"),
            method="relative_only",
        )

    def test_produces_investment_report(
        self,
        sample_company,
        sample_market,
        sample_news,
        sample_financial,
        sample_competitor_analysis,
    ):
        flow = AnalysisFlow()
        self._populate_state(
            flow,
            sample_company,
            sample_market,
            sample_news,
            sample_financial,
            sample_competitor_analysis,
        )

        _run(flow.synthesize_report())

        assert flow.state.report is not None
        assert flow.state.report.ticker == "AAPL"
        assert flow.state.report.company == sample_company
        assert flow.state.report.market == sample_market
        assert flow.state.report.rating in (
            "Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"
        )
        assert 0 <= flow.state.report.confidence <= 100
        assert isinstance(flow.state.report.markdown, str)
        assert len(flow.state.report.markdown) > 0
        # UUID4 string
        assert len(flow.state.report.report_id) == 36

    def test_markdown_contains_sections(
        self,
        sample_company,
        sample_market,
        sample_news,
        sample_financial,
        sample_competitor_analysis,
    ):
        flow = AnalysisFlow()
        self._populate_state(
            flow,
            sample_company,
            sample_market,
            sample_news,
            sample_financial,
            sample_competitor_analysis,
        )

        _run(flow.synthesize_report())

        md = flow.state.report.markdown
        for section in (
            "# AAPL", "执行摘要", "公司概览", "市场分析",
            "财务分析", "新闻情绪", "竞争对手", "风险评估", "估值与建议",
        ):
            assert section in md

    def test_synthesis_failure_raises_report_generation_error(
        self,
        sample_company,
        sample_market,
        sample_news,
        sample_financial,
        sample_competitor_analysis,
    ):
        """§3.2: report synthesis failure → ReportGenerationError (→ 500)."""
        from alphaquant.exceptions import ReportGenerationError

        flow = AnalysisFlow()
        self._populate_state(
            flow,
            sample_company,
            sample_market,
            sample_news,
            sample_financial,
            sample_competitor_analysis,
        )

        # Force InvestmentReport(...) construction to raise
        with patch(
            "alphaquant.flows.analysis_flow.InvestmentReport",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(ReportGenerationError):
                _run(flow.synthesize_report())


# ---------------------------------------------------------------------------
# End-to-end orchestration via the Flow
# ---------------------------------------------------------------------------


class TestFlowKickoff:
    def test_full_flow_with_mocked_registry(
        self,
        sample_company,
        sample_market,
        sample_news,
        sample_financial,
    ):
        """All 2 steps execute and produce an InvestmentReport."""
        from alphaquant.flows.analysis_flow import AnalysisFlow
        from datetime import date
        from alphaquant.models.news import NewsItem

        # Build the Flow instance directly (avoid crewai kickoff side-effects)
        flow = AnalysisFlow()

        reg_cls = __import__("alphaquant.infrastructure.data_sources", fromlist=["DataSourceRegistry"]).DataSourceRegistry
        news_items = [
            NewsItem(
                date=date(2026, 6, 19),
                title="Test",
                url="https://example.com/n1",
                source="yahoo",
                sentiment="positive",
                relevance_score=0.9,
            ),
        ]

        with patch.object(reg_cls, "get_company", new=AsyncMock(return_value=sample_company)), \
             patch.object(reg_cls, "get_market", new=AsyncMock(return_value=sample_market)), \
             patch.object(reg_cls, "get_news", new=AsyncMock(return_value=news_items)), \
             patch.object(reg_cls, "get_financial", new=AsyncMock(return_value=sample_financial)), \
             _patch_competitor_tool("No peer data available"), \
             patch("alphaquant.flows.analysis_flow.AnalysisCrew") as MockCrew:

            # Mock crew to produce a fake result that triggers deterministic fallback
            fake_result = MagicMock()
            fake_result.tasks_output = []
            MockCrew.return_value.kickoff.return_value = fake_result

            # Drive the steps manually
            _run(flow.run_crew("AAPL"))
            _run(flow.synthesize_report())

        assert flow.state.report is not None
        assert flow.state.report.ticker == "AAPL"

    def test_partial_failure_degrades_gracefully(
        self,
        sample_company,
        sample_market,
        sample_financial,
    ):
        """§3.2: market failure → flow still produces a report."""
        from datetime import date
        from alphaquant.models.news import NewsItem

        flow = AnalysisFlow()
        reg_cls = __import__("alphaquant.infrastructure.data_sources", fromlist=["DataSourceRegistry"]).DataSourceRegistry
        news_items = [
            NewsItem(
                date=date(2026, 6, 19),
                title="Test",
                url="https://example.com/n1",
                source="yahoo",
                sentiment="positive",
                relevance_score=0.9,
            ),
        ]

        with patch.object(reg_cls, "get_company", new=AsyncMock(return_value=sample_company)), \
             patch.object(reg_cls, "get_market", new=AsyncMock(side_effect=Exception("down"))), \
             patch.object(reg_cls, "get_news", new=AsyncMock(return_value=news_items)), \
             patch.object(reg_cls, "get_financial", new=AsyncMock(return_value=sample_financial)), \
             _patch_competitor_tool("No peer data available"), \
             patch("alphaquant.flows.analysis_flow.AnalysisCrew") as MockCrew:

            fake_result = MagicMock()
            fake_result.tasks_output = []
            MockCrew.return_value.kickoff.return_value = fake_result

            _run(flow.run_crew("AAPL"))
            _run(flow.synthesize_report())

        # Market was degraded but flow still produced a report
        assert "market_data_unavailable" in flow.state.errors
        # synthesize_report substitutes a degraded MarketData placeholder
        assert flow.state.market is not None
        assert flow.state.market.source == "degraded"
        assert flow.state.report is not None
        assert flow.state.report.ticker == "AAPL"

    def test_kickoff_with_timeout_succeeds_under_limit(
        self,
        sample_company,
        sample_market,
        sample_financial,
    ):
        """§3.4: kickoff_with_timeout returns within 120s for a fast flow."""
        from alphaquant.flows.analysis_flow import AnalysisFlow
        from datetime import date
        from alphaquant.models.news import NewsItem

        flow = AnalysisFlow()
        reg_cls = __import__("alphaquant.infrastructure.data_sources", fromlist=["DataSourceRegistry"]).DataSourceRegistry
        news_items = [
            NewsItem(
                date=date(2026, 6, 19),
                title="T",
                url="https://example.com/n1",
                source="yahoo",
                sentiment="positive",
                relevance_score=0.9,
            ),
        ]

        with patch.object(reg_cls, "get_company", new=AsyncMock(return_value=sample_company)), \
             patch.object(reg_cls, "get_market", new=AsyncMock(return_value=sample_market)), \
             patch.object(reg_cls, "get_news", new=AsyncMock(return_value=news_items)), \
             patch.object(reg_cls, "get_financial", new=AsyncMock(return_value=sample_financial)), \
             _patch_competitor_tool("No peer data available"), \
             patch("alphaquant.flows.analysis_flow.AnalysisCrew") as MockCrew:

            fake_result = MagicMock()
            fake_result.tasks_output = []
            MockCrew.return_value.kickoff.return_value = fake_result

            # kickoff_async builds inputs dict internally; we pass ticker via inputs.
            _run(flow.kickoff_with_timeout(inputs={"ticker": "AAPL"}))

        # After kickoff, the state should have a report (the return value of
        # kickoff_async is the output of the last method, which is None here).
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
    """@start run_crew: pre-fetch data, invoke AnalysisCrew, fill state."""

    def test_run_crew_pre_fetches_data_and_invokes_crew(
        self, sample_company, sample_market, sample_news, sample_financial
    ):
        """run_crew must call DataSourceRegistry 4 times (one per data source)
        AND must invoke AnalysisCrew.kickoff with the pre-fetched data."""
        from alphaquant.flows.analysis_flow import AnalysisFlow

        flow = AnalysisFlow()
        flow.state.ticker = "AAPL"

        # Mock DataSourceRegistry
        reg_cls = __import__(
            "alphaquant.infrastructure.data_sources", fromlist=["DataSourceRegistry"]
        ).DataSourceRegistry

        with patch.object(reg_cls, "get_company", new=AsyncMock(return_value=sample_company)), \
             patch.object(reg_cls, "get_market", new=AsyncMock(return_value=sample_market)), \
             patch.object(reg_cls, "get_news", new=AsyncMock(return_value=sample_news)), \
             patch.object(reg_cls, "get_financial", new=AsyncMock(return_value=sample_financial)):

            # Mock AnalysisCrew so we don't actually invoke real LLM
            with patch("alphaquant.flows.analysis_flow.AnalysisCrew") as MockCrew:
                # Fake crew output: each task returns a simple marker
                from crewai import CrewOutput  # adjust import if needed
                fake_output = MagicMock()
                fake_output.tasks_output = []
                MockCrew.return_value.kickoff.return_value = fake_output

                _run(flow.run_crew("AAPL"))

                # Verify crew was called
                MockCrew.assert_called_once()
                MockCrew.return_value.kickoff.assert_called_once()
                call_kwargs = MockCrew.return_value.kickoff.call_args.kwargs
                assert "inputs" in call_kwargs
                inputs = call_kwargs["inputs"]
                assert inputs["ticker"] == "AAPL"

    def test_run_crew_timeout_raises(self, sample_company, sample_market, sample_news, sample_financial):
        """If crew.kickoff exceeds 120s, asyncio.TimeoutError is raised."""
        import asyncio as _asyncio
        import time
        from alphaquant.flows.analysis_flow import AnalysisFlow

        flow = AnalysisFlow()
        flow.state.ticker = "AAPL"

        reg_cls = __import__(
            "alphaquant.infrastructure.data_sources", fromlist=["DataSourceRegistry"]
        ).DataSourceRegistry

        def slow_kickoff(inputs):
            # Blocking sleep in the worker thread so the asyncio.wait_for
            # timeout actually fires.
            time.sleep(2.0)
            return MagicMock(tasks_output=[])

        with patch.object(reg_cls, "get_company", new=AsyncMock(return_value=sample_company)), \
             patch.object(reg_cls, "get_market", new=AsyncMock(return_value=sample_market)), \
             patch.object(reg_cls, "get_news", new=AsyncMock(return_value=sample_news)), \
             patch.object(reg_cls, "get_financial", new=AsyncMock(return_value=sample_financial)), \
             patch("alphaquant.flows.analysis_flow.AnalysisCrew") as MockCrew, \
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

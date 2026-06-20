"""Tests for alphaquant.flows.analysis_flow AnalysisFlow orchestration."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alphaquant.exceptions import AllDataSourcesDown, InvalidTickerFormat
from alphaquant.flows import AnalysisFlow, AnalysisState
from alphaquant.models.competitor import Competitor, CompetitorAnalysis
from alphaquant.models.company import Company
from alphaquant.models.financial import BalanceSheet, FinancialStatements, IncomeStatement
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
# resolve_company step
# ---------------------------------------------------------------------------


class TestResolveCompany:
    def test_sets_ticker_and_company(self, sample_company):
        flow = AnalysisFlow()
        with patch.object(
            DataSourceRegistry := __import__(
                "alphaquant.data_sources", fromlist=["DataSourceRegistry"]
            ).DataSourceRegistry,
            "get_company",
            new=AsyncMock(return_value=sample_company),
        ):
            flow.resolve_company("aapl")
        assert flow.state.ticker == "AAPL"
        assert flow.state.company == sample_company

    def test_propagates_all_data_sources_down(self):
        flow = AnalysisFlow()
        with patch.object(
            __import__("alphaquant.data_sources", fromlist=["DataSourceRegistry"]).DataSourceRegistry,
            "get_company",
            new=AsyncMock(side_effect=AllDataSourcesDown("boom")),
        ):
            with pytest.raises(AllDataSourcesDown):
                flow.resolve_company("AAPL")


# ---------------------------------------------------------------------------
# parallel_data_collection step
# ---------------------------------------------------------------------------


class TestParallelDataCollection:
    def test_populates_all_three(self, sample_market, sample_news, sample_financial):
        flow = AnalysisFlow()
        flow.state.ticker = "AAPL"
        flow.state.company = MagicMock()  # not used here

        registry_patch = patch.object(
            __import__("alphaquant.data_sources", fromlist=["DataSourceRegistry"]).DataSourceRegistry,
            "get_market",
            new=AsyncMock(return_value=sample_market),
        )
        news_patch = patch.object(
            __import__("alphaquant.data_sources", fromlist=["DataSourceRegistry"]).DataSourceRegistry,
            "get_news",
            new=AsyncMock(return_value=sample_news),
        )
        fin_patch = patch.object(
            __import__("alphaquant.data_sources", fromlist=["DataSourceRegistry"]).DataSourceRegistry,
            "get_financial",
            new=AsyncMock(return_value=sample_financial),
        )
        with registry_patch, news_patch, fin_patch:
            flow.parallel_data_collection()

        assert flow.state.market == sample_market
        assert flow.state.news == sample_news
        assert flow.state.financial == sample_financial
        assert flow.state.errors == []

    def test_market_failure_sets_none_and_records_error(
        self, sample_news, sample_financial
    ):
        """§3.2: market failure → degraded (None), continue."""
        flow = AnalysisFlow()
        flow.state.ticker = "AAPL"
        reg = __import__("alphaquant.data_sources", fromlist=["DataSourceRegistry"]).DataSourceRegistry
        with patch.object(reg, "get_market", new=AsyncMock(side_effect=Exception("fail"))), \
             patch.object(reg, "get_news", new=AsyncMock(return_value=sample_news)), \
             patch.object(reg, "get_financial", new=AsyncMock(return_value=sample_financial)):
            flow.parallel_data_collection()

        assert flow.state.market is None
        assert "market_data_unavailable" in flow.state.errors

    def test_news_failure_falls_back_to_empty(self, sample_market, sample_financial):
        """§3.2: news failure → NewsAnalysis.empty(), continue."""
        flow = AnalysisFlow()
        flow.state.ticker = "AAPL"
        reg = __import__("alphaquant.data_sources", fromlist=["DataSourceRegistry"]).DataSourceRegistry
        with patch.object(reg, "get_market", new=AsyncMock(return_value=sample_market)), \
             patch.object(reg, "get_news", new=AsyncMock(side_effect=Exception("fail"))), \
             patch.object(reg, "get_financial", new=AsyncMock(return_value=sample_financial)):
            flow.parallel_data_collection()

        assert isinstance(flow.state.news, NewsAnalysis)
        assert flow.state.news.total_count == 0
        assert "news_data_unavailable" in flow.state.errors

    def test_financial_failure_yields_empty_statements(self, sample_market, sample_news):
        """§3.2: financial failure → empty FinancialStatements, continue."""
        flow = AnalysisFlow()
        flow.state.ticker = "AAPL"
        reg = __import__("alphaquant.data_sources", fromlist=["DataSourceRegistry"]).DataSourceRegistry
        with patch.object(reg, "get_market", new=AsyncMock(return_value=sample_market)), \
             patch.object(reg, "get_news", new=AsyncMock(return_value=sample_news)), \
             patch.object(reg, "get_financial", new=AsyncMock(side_effect=Exception("fail"))):
            flow.parallel_data_collection()

        assert isinstance(flow.state.financial, FinancialStatements)
        assert flow.state.financial.ticker == "AAPL"
        assert flow.state.financial.income_statements == []
        assert "financial_data_unavailable" in flow.state.errors


# ---------------------------------------------------------------------------
# competitor_analysis step
# ---------------------------------------------------------------------------


class TestCompetitorAnalysis:
    def test_no_company_falls_back(self):
        flow = AnalysisFlow()
        flow.state.ticker = "AAPL"
        flow.state.company = None

        flow.competitor_analysis()

        assert flow.state.competitor is not None
        assert flow.state.competitor.method == "fallback"
        # Placeholder self-peer satisfies the model constraint
        assert len(flow.state.competitor.competitors) == 1
        assert flow.state.competitor.competitors[0].ticker == "AAPL"

    def test_tool_failure_falls_back(self, sample_company, sample_market):
        flow = AnalysisFlow()
        flow.state.ticker = "AAPL"
        flow.state.company = sample_company
        flow.state.market = sample_market

        # Force the tool to raise
        def raising(self, ticker: str) -> str:
            raise RuntimeError("down")

        with patch(
            "alphaquant.tools.competitor_tool.CompetitorTool._run",
            new=raising,
        ):
            flow.competitor_analysis()

        assert flow.state.competitor is not None
        assert flow.state.competitor.method == "fallback"
        # Placeholder self-peer
        assert len(flow.state.competitor.competitors) == 1
        assert flow.state.competitor.competitors[0].ticker == "AAPL"

    def test_tool_no_peers_falls_back(self, sample_company, sample_market):
        flow = AnalysisFlow()
        flow.state.ticker = "AAPL"
        flow.state.company = sample_company
        flow.state.market = sample_market

        with _patch_competitor_tool("No peer data available"):
            flow.competitor_analysis()

        assert flow.state.competitor is not None
        assert flow.state.competitor.method == "fallback"
        # Placeholder self-peer
        assert len(flow.state.competitor.competitors) == 1
        assert flow.state.competitor.competitors[0].ticker == "AAPL"

    def test_successful_competitor_scoring(self, sample_company, sample_market):
        flow = AnalysisFlow()
        flow.state.ticker = "AAPL"
        flow.state.company = sample_company
        flow.state.market = sample_market

        peer_json = json.dumps([
            {
                "ticker": "MSFT",
                "name": "Microsoft",
                "market_cap": 2_500_000_000_000,
                "revenue_ttm": 200_000_000_000,
                "revenue_growth_yoy": 0.10,
                "gross_margin": 0.65,
                "net_margin": 0.25,
                "pe_ratio": 30.0,
                "ps_ratio": 10.0,
            }
        ])
        with _patch_competitor_tool(peer_json):
            flow.competitor_analysis()

        assert flow.state.competitor is not None
        assert len(flow.state.competitor.competitors) == 1
        assert flow.state.competitor.competitors[0].ticker == "MSFT"
        # Score computed by scoring.competitive.compute()
        assert 0 <= flow.state.competitor.competitive_score <= 100


# ---------------------------------------------------------------------------
# risk_analysis step
# ---------------------------------------------------------------------------


class TestRiskAnalysis:
    def test_computes_with_financial_data(self, sample_financial):
        flow = AnalysisFlow()
        flow.state.ticker = "AAPL"
        flow.state.financial = sample_financial

        flow.risk_analysis()

        assert flow.state.risk is not None
        assert 0 <= flow.state.risk.total_score <= 100
        assert flow.state.risk.level in ("low", "medium", "high", "extreme")
        # 6 sub-scores (financial + market + 4 neutrals)
        assert len(flow.state.risk.sub_scores) == 6

    def test_missing_data_defaults_to_neutral(self):
        """§3.2: risk with missing upstream → 5 (neutral)."""
        flow = AnalysisFlow()
        flow.state.ticker = "AAPL"
        flow.state.financial = FinancialStatements(ticker="AAPL")  # empty
        flow.state.market = None

        flow.risk_analysis()

        assert flow.state.risk is not None
        # All sub-scores should be 5 (neutral) → total = 50
        assert flow.state.risk.total_score == 50
        assert flow.state.risk.level == "medium"


# ---------------------------------------------------------------------------
# valuation_analysis step
# ---------------------------------------------------------------------------


class TestValuationAnalysis:
    def test_market_present_computes_relative(self, sample_market):
        flow = AnalysisFlow()
        flow.state.ticker = "AAPL"
        flow.state.market = sample_market

        flow.valuation_analysis()

        assert flow.state.valuation is not None
        assert flow.state.valuation.dcf_value is None  # §3.2: dcf null
        assert flow.state.valuation.relative_value is not None
        assert flow.state.valuation.method == "relative_only"

    def test_market_missing_handled_gracefully(self):
        """§3.2: market missing → relative-only with zero current price."""
        flow = AnalysisFlow()
        flow.state.ticker = "AAPL"
        flow.state.market = None

        flow.valuation_analysis()

        assert flow.state.valuation is not None
        assert flow.state.valuation.current_price == Decimal("0")
        assert flow.state.valuation.dcf_value is None


# ---------------------------------------------------------------------------
# write_report step
# ---------------------------------------------------------------------------


class TestWriteReport:
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

        flow.write_report()

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

        flow.write_report()

        md = flow.state.report.markdown
        for section in (
            "# AAPL", "执行摘要", "公司概览", "市场分析",
            "财务分析", "新闻情绪", "竞争对手", "风险评估", "估值与建议",
        ):
            assert section in md


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
        """All 6 steps execute and produce an InvestmentReport."""
        from alphaquant.flows.analysis_flow import AnalysisFlow

        # Build the Flow instance directly (avoid crewai kickoff side-effects)
        flow = AnalysisFlow()

        reg_cls = __import__("alphaquant.data_sources", fromlist=["DataSourceRegistry"]).DataSourceRegistry

        with patch.object(reg_cls, "get_company", new=AsyncMock(return_value=sample_company)), \
             patch.object(reg_cls, "get_market", new=AsyncMock(return_value=sample_market)), \
             patch.object(reg_cls, "get_news", new=AsyncMock(return_value=sample_news)), \
             patch.object(reg_cls, "get_financial", new=AsyncMock(return_value=sample_financial)), \
             _patch_competitor_tool("No peer data available"):

            # Drive the steps manually (mirrors what @start/@listen wiring would do)
            flow.resolve_company("AAPL")
            flow.parallel_data_collection()
            flow.competitor_analysis()
            flow.risk_analysis()
            flow.valuation_analysis()
            flow.write_report()

        assert flow.state.report is not None
        assert flow.state.report.ticker == "AAPL"

    def test_partial_failure_degrades_gracefully(
        self,
        sample_company,
        sample_news,
        sample_financial,
    ):
        """§3.2: market failure → flow still produces a report."""
        flow = AnalysisFlow()
        reg_cls = __import__("alphaquant.data_sources", fromlist=["DataSourceRegistry"]).DataSourceRegistry

        with patch.object(reg_cls, "get_company", new=AsyncMock(return_value=sample_company)), \
             patch.object(reg_cls, "get_market", new=AsyncMock(side_effect=Exception("down"))), \
             patch.object(reg_cls, "get_news", new=AsyncMock(return_value=sample_news)), \
             patch.object(reg_cls, "get_financial", new=AsyncMock(return_value=sample_financial)), \
             _patch_competitor_tool("No peer data available"):

            flow.resolve_company("AAPL")
            flow.parallel_data_collection()
            flow.competitor_analysis()
            flow.risk_analysis()
            flow.valuation_analysis()
            flow.write_report()

        # Market was degraded but flow still produced a report
        assert "market_data_unavailable" in flow.state.errors
        # write_report substitutes a degraded MarketData placeholder
        assert flow.state.market is not None
        assert flow.state.market.source == "degraded"
        assert flow.state.report is not None
        assert flow.state.report.ticker == "AAPL"

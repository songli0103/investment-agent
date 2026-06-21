"""Tests for alphaquant.tools CrewAI wrappers."""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from unittest.mock import patch

import pytest

from alphaquant.tools.competitor_tool import CompetitorTool
from alphaquant.tools.company_lookup_tool import CompanyLookupTool
from alphaquant.tools.dcf_tool import DCFTool
from alphaquant.tools.financial_tool import FinancialTool
from alphaquant.tools.market_data_tool import MarketDataInput, MarketDataTool
from alphaquant.tools.news_tool import NewsTool


# ---------------------------------------------------------------------------
# Smoke: imports + tool attributes
# ---------------------------------------------------------------------------

def test_tools_importable():
    """All six tool classes import from alphaquant.tools."""
    from alphaquant.tools import (  # noqa: F401
        CompetitorTool,
        CompanyLookupTool,  # NEW sub-2
        DCFTool,
        FinancialTool,
        MarketDataTool,
        NewsTool,
    )


class TestToolMetadata:
    @pytest.mark.parametrize(
        "cls,expected_name",
        [
            (MarketDataTool, "market_data_lookup"),
            (NewsTool, "news_lookup"),
            (FinancialTool, "financial_statements_lookup"),
            (CompetitorTool, "competitor_lookup"),
            (DCFTool, "dcf_assumptions"),
            (CompanyLookupTool, "company_lookup"),  # NEW sub-2
        ],
    )
    def test_name(self, cls, expected_name):
        assert cls().name == expected_name

    @pytest.mark.parametrize(
        "cls",
        [MarketDataTool, NewsTool, FinancialTool, CompetitorTool, DCFTool, CompanyLookupTool],  # added
    )
    def test_description_is_nonempty(self, cls):
        assert isinstance(cls().description, str)
        assert len(cls().description) > 0

    @pytest.mark.parametrize(
        "cls",
        [MarketDataTool, NewsTool, FinancialTool, CompetitorTool, DCFTool, CompanyLookupTool],  # added
    )
    def test_has_run(self, cls):
        assert callable(getattr(cls(), "_run", None))


# ---------------------------------------------------------------------------
# MarketDataInput Pydantic model
# ---------------------------------------------------------------------------

class TestMarketDataInput:
    def test_ticker_required(self):
        with pytest.raises(Exception):
            MarketDataInput()

    def test_valid_ticker(self):
        m = MarketDataInput(ticker="AAPL")
        assert m.ticker == "AAPL"


# ---------------------------------------------------------------------------
# DCFTool: static text, no I/O
# ---------------------------------------------------------------------------

class TestDCFTool:
    def test_returns_default_assumptions(self):
        result = DCFTool()._run("AAPL")
        assert "Growth rate" in result
        assert "WACC" in result
        assert "Terminal growth" in result

    def test_ignores_ticker(self):
        a = DCFTool()._run("AAPL")
        b = DCFTool()._run("ZZZZ")
        assert a == b


# ---------------------------------------------------------------------------
# MarketDataTool: wraps DataSourceRegistry.get_market
# ---------------------------------------------------------------------------

class TestMarketDataTool:
    def test_returns_json_on_market_data(self):
        from alphaquant.models.market import MarketData

        market = MarketData(
            ticker="AAPL",
            as_of=datetime(2026, 1, 1),
            price=Decimal("150.00"),
            change_pct=0.01,
            volume=1_000_000,
            market_cap=2_500_000_000_000,
            pe_ratio=30.0,
        )

        class FakeRegistry:
            async def get_market(self, ticker):
                return market

        with patch("alphaquant.tools.market_data_tool.DataSourceRegistry", FakeRegistry):
            result = MarketDataTool()._run("AAPL")

        assert "AAPL" in result
        assert "150" in result  # price serialized

    def test_returns_message_when_no_data(self):
        class FakeRegistry:
            async def get_market(self, ticker):
                return None

        with patch("alphaquant.tools.market_data_tool.DataSourceRegistry", FakeRegistry):
            result = MarketDataTool()._run("ZZZZ")

        assert "No market data available for ZZZZ" in result

    def test_returns_error_message_on_exception(self):
        class FakeRegistry:
            async def get_market(self, ticker):
                raise RuntimeError("boom")

        with patch("alphaquant.tools.market_data_tool.DataSourceRegistry", FakeRegistry):
            result = MarketDataTool()._run("AAPL")

        assert "Error fetching market data" in result
        assert "boom" in result


# ---------------------------------------------------------------------------
# NewsTool: wraps DataSourceRegistry.get_news
# ---------------------------------------------------------------------------

class TestNewsTool:
    def _make_news(self, n: int):
        from alphaquant.models.news import NewsItem

        return [
            NewsItem(
                date=date(2026, 1, i + 1),
                title=f"Headline {i}",
                url=f"https://example.com/{i}",
                source="TestSource",
                sentiment="neutral",
                topic="other",
                relevance_score=0.5,
            )
            for i in range(n)
        ]

    def test_returns_json_list(self):
        news = self._make_news(3)

        class FakeRegistry:
            async def get_news(self, ticker, days=30):
                return news

        with patch("alphaquant.tools.news_tool.DataSourceRegistry", FakeRegistry):
            result = NewsTool()._run("AAPL")

        assert "Headline 0" in result
        assert "Headline 2" in result
        assert "TestSource" in result

    def test_returns_message_when_empty(self):
        class FakeRegistry:
            async def get_news(self, ticker, days=30):
                return []

        with patch("alphaquant.tools.news_tool.DataSourceRegistry", FakeRegistry):
            result = NewsTool()._run("ZZZZ")

        assert "No news found for ZZZZ" in result

    def test_caps_at_20_items(self):
        news = self._make_news(25)

        class FakeRegistry:
            async def get_news(self, ticker, days=30):
                return news

        with patch("alphaquant.tools.news_tool.DataSourceRegistry", FakeRegistry):
            result = NewsTool()._run("AAPL")

        # Headlines 0..19 present, 20..24 not present
        assert "Headline 19" in result
        assert "Headline 20" not in result

    def test_returns_error_message_on_exception(self):
        class FakeRegistry:
            async def get_news(self, ticker, days=30):
                raise RuntimeError("net down")

        with patch("alphaquant.tools.news_tool.DataSourceRegistry", FakeRegistry):
            result = NewsTool()._run("AAPL")

        assert "Error fetching news" in result
        assert "net down" in result


# ---------------------------------------------------------------------------
# FinancialTool: wraps DataSourceRegistry.get_financial
# ---------------------------------------------------------------------------

class TestFinancialTool:
    def test_returns_json_on_statements(self):
        from alphaquant.models.financial import (
            BalanceSheet,
            CashFlowStatement,
            FinancialStatements,
            IncomeStatement,
        )

        statements = FinancialStatements(
            ticker="AAPL",
            income_statements=[
                IncomeStatement(
                    period="TTM",
                    fiscal_year=2024,
                    revenue=Decimal("100000"),
                    gross_profit=Decimal("50000"),
                    net_income=Decimal("20000"),
                )
            ],
            balance_sheets=[
                BalanceSheet(
                    period="FY",
                    fiscal_year=2024,
                    total_assets=Decimal("300000"),
                    total_liabilities=Decimal("150000"),
                    total_equity=Decimal("150000"),
                )
            ],
            cash_flows=[
                CashFlowStatement(
                    period="FY",
                    fiscal_year=2024,
                    operating_cash_flow=Decimal("25000"),
                    investing_cash_flow=Decimal("-5000"),
                    financing_cash_flow=Decimal("-10000"),
                    free_cash_flow=Decimal("20000"),
                )
            ],
        )

        class FakeRegistry:
            async def get_financial(self, ticker):
                return statements

        with patch("alphaquant.tools.financial_tool.DataSourceRegistry", FakeRegistry):
            result = FinancialTool()._run("AAPL")

        assert "AAPL" in result

    def test_returns_message_when_no_statements(self):
        class FakeRegistry:
            async def get_financial(self, ticker):
                return None

        with patch("alphaquant.tools.financial_tool.DataSourceRegistry", FakeRegistry):
            result = FinancialTool()._run("ZZZZ")

        assert "No financial data available for ZZZZ" in result

    def test_returns_error_message_on_exception(self):
        class FakeRegistry:
            async def get_financial(self, ticker):
                raise RuntimeError("api down")

        with patch("alphaquant.tools.financial_tool.DataSourceRegistry", FakeRegistry):
            result = FinancialTool()._run("AAPL")

        assert "Error fetching financials" in result


# ---------------------------------------------------------------------------
# CompanyLookupTool: wraps DataSourceRegistry.get_company
# ---------------------------------------------------------------------------

class TestCompanyLookupTool:
    def test_returns_json_on_company_data(self):
        from alphaquant.models.company import Company

        company = Company(
            ticker="AAPL",
            name="Apple Inc.",
            exchange="NASDAQ",
            sector="Technology",
            industry="Consumer Electronics",
            market_cap=3_000_000_000_000,
        )

        class FakeRegistry:
            async def get_company(self, ticker):
                return company

        with patch("alphaquant.tools.company_lookup_tool.DataSourceRegistry", FakeRegistry):
            result = CompanyLookupTool()._run("AAPL")

        assert "Apple Inc." in result
        assert "NASDAQ" in result
        assert "Technology" in result

    def test_returns_error_message_on_alldatasourcesdown(self):
        from alphaquant.exceptions import AllDataSourcesDown

        class FakeRegistry:
            async def get_company(self, ticker):
                raise AllDataSourcesDown("all sources down for ZZZZ")

        with patch("alphaquant.tools.company_lookup_tool.DataSourceRegistry", FakeRegistry):
            result = CompanyLookupTool()._run("ZZZZ")

        # AllDataSourcesDown must be caught and returned as error string,
        # NOT propagated (agents receive strings, not exceptions)
        assert "Error fetching company" in result
        assert "all sources down" in result

    def test_returns_error_message_on_generic_exception(self):
        class FakeRegistry:
            async def get_company(self, ticker):
                raise RuntimeError("net down")

        with patch("alphaquant.tools.company_lookup_tool.DataSourceRegistry", FakeRegistry):
            result = CompanyLookupTool()._run("AAPL")

        assert "Error fetching company" in result
        assert "net down" in result

    def test_timeout_returns_error_message(self):
        """If get_company exceeds 30s, tool returns timeout error string."""
        import asyncio

        class FakeRegistry:
            async def get_company(self, ticker):
                await asyncio.sleep(60)  # exceeds 30s timeout
                return None

        with patch("alphaquant.tools.company_lookup_tool.DataSourceRegistry", FakeRegistry), \
             patch("alphaquant.tools.company_lookup_tool.TOOL_TIMEOUT_SECONDS", 0.1):
            result = CompanyLookupTool()._run("AAPL")

        assert "Error fetching company" in result
        assert "timeout" in result.lower() or "TimeoutError" in result


# ---------------------------------------------------------------------------
# CompetitorTool: uses YahooFinanceSource directly
# ---------------------------------------------------------------------------

class TestCompetitorTool:
    def test_no_company_info_returns_message(self):
        class FakeSrc:
            async def get_company_info(self, ticker):
                return None

            async def get_market_data(self, ticker):
                return None

        with patch("alphaquant.infrastructure.data_sources.yahoo.YahooFinanceSource", FakeSrc):
            result = CompetitorTool()._run("ZZZZ")

        assert "No company info for ZZZZ" in result

    def test_unknown_sector_falls_back_to_spy(self):
        from alphaquant.models.company import Company
        from alphaquant.models.market import MarketData

        company = Company(
            ticker="XYZ",
            name="XYZ Corp",
            exchange="NASDAQ",
            sector="Unknown",
            industry="Unknown",
            market_cap=1_000_000,
        )
        market = MarketData(
            ticker="SPY",
            as_of=datetime(2026, 1, 1),
            price=Decimal("500"),
            change_pct=0.0,
            volume=1_000_000,
            market_cap=500_000_000_000,
            pe_ratio=25.0,
        )

        class FakeSrc:
            async def get_company_info(self, ticker):
                return company

            async def get_market_data(self, ticker):
                if ticker == "SPY":
                    return market
                return None

        with patch("alphaquant.infrastructure.data_sources.yahoo.YahooFinanceSource", FakeSrc):
            result = CompetitorTool()._run("XYZ")

        # Should mention SPY (the fallback peer)
        assert "SPY" in result

    def test_known_sector_returns_peers(self):
        from alphaquant.models.company import Company
        from alphaquant.models.market import MarketData

        company = Company(
            ticker="AAPL",
            name="Apple",
            exchange="NASDAQ",
            sector="Technology",
            industry="Consumer Electronics",
            market_cap=2_500_000_000_000,
        )

        def make_market(t):
            return MarketData(
                ticker=t,
                as_of=datetime(2026, 1, 1),
                price=Decimal("100"),
                change_pct=0.0,
                volume=1_000_000,
                market_cap=1_000_000_000,
                pe_ratio=20.0,
            )

        class FakeSrc:
            async def get_company_info(self, ticker):
                return company

            async def get_market_data(self, ticker):
                return make_market(ticker)

        with patch("alphaquant.infrastructure.data_sources.yahoo.YahooFinanceSource", FakeSrc):
            result = CompetitorTool()._run("AAPL")

        # Technology peers: MSFT, GOOGL, META — AAPL itself filtered out
        assert "MSFT" in result
        assert "GOOGL" in result
        assert "META" in result

    def test_no_peer_data_returns_message(self):
        from alphaquant.models.company import Company

        company = Company(
            ticker="XYZ",
            name="XYZ Corp",
            exchange="NASDAQ",
            sector="Technology",
            industry="Unknown",
            market_cap=1_000_000,
        )

        class FakeSrc:
            async def get_company_info(self, ticker):
                return company

            async def get_market_data(self, ticker):
                return None  # all peers fail

        with patch("alphaquant.infrastructure.data_sources.yahoo.YahooFinanceSource", FakeSrc):
            result = CompetitorTool()._run("XYZ")

        assert "No peer data available" in result

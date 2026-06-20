"""Data source adapters with fallback registry."""
from __future__ import annotations

import asyncio

from alphaquant.data_sources.alpha_vantage import AlphaVantageSource
from alphaquant.data_sources.base import DataSourceInterface
from alphaquant.data_sources.finnhub import FinnhubSource
from alphaquant.data_sources.news import NewsAPISource
from alphaquant.data_sources.sec_edgar import SECEdgarSource
from alphaquant.data_sources.yahoo import YahooFinanceSource
from alphaquant.exceptions import AllDataSourcesDown, PartialDataFailure
from alphaquant.models.company import Company
from alphaquant.models.financial import FinancialStatements
from alphaquant.models.market import MarketData
from alphaquant.models.news import NewsItem
from alphaquant.observability import get_logger

log = get_logger("alphaquant.data_sources")


class DataSourceRegistry:
    """Registry of all data sources with fallback chains."""

    def __init__(self) -> None:
        self.sources: list[DataSourceInterface] = [
            YahooFinanceSource(),
            AlphaVantageSource(),
            FinnhubSource(),
            SECEdgarSource(),
            NewsAPISource(),
        ]

    async def _try_chain(self, method_name: str, ticker: str, **kwargs):
        """Try each source in order; return first non-None result."""
        errors: list[str] = []
        for src in self.sources:
            method = getattr(src, method_name, None)
            if method is None:
                continue
            try:
                result = await method(ticker, **kwargs)
                if result is not None:
                    log.info("data_source_hit", source=src.name, method=method_name)
                    return result
            except Exception as e:
                errors.append(f"{src.name}: {e}")
                continue
        log.warning("data_source_all_failed", method=method_name, ticker=ticker, errors=errors)
        return None

    async def get_company(self, ticker: str) -> Company:
        result = await self._try_chain("get_company_info", ticker)
        if result is None:
            raise AllDataSourcesDown(f"No data source could resolve {ticker}")
        return result

    async def get_market(self, ticker: str) -> MarketData | None:
        """Market data is non-critical. Returns None on full failure."""
        try:
            return await self._try_chain("get_market_data", ticker)
        except Exception:
            return None

    async def get_financial(self, ticker: str) -> FinancialStatements | None:
        try:
            return await self._try_chain("get_financials", ticker)
        except Exception:
            return None

    async def get_news(self, ticker: str, days: int = 30) -> list[NewsItem]:
        """Returns empty list on failure (never None)."""
        try:
            result = await self._try_chain("get_news", ticker, days=days)
            return result or []
        except Exception:
            return []


__all__ = [
    "AlphaVantageSource",
    "DataSourceInterface",
    "DataSourceRegistry",
    "FinnhubSource",
    "NewsAPISource",
    "SECEdgarSource",
    "YahooFinanceSource",
]
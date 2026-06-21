"""Abstract base for all data source adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod

from alphaquant.models.company import Company
from alphaquant.models.financial import FinancialStatements
from alphaquant.models.market import MarketData
from alphaquant.models.news import NewsItem


class DataSourceInterface(ABC):
    """All data source adapters must implement this interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g., 'yahoo', 'alpha_vantage')."""
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Quick health check (does NOT make a full query)."""
        ...

    @abstractmethod
    async def get_company_info(self, ticker: str) -> Company | None:
        """Return Company if found, None if not found (not an error)."""
        ...

    @abstractmethod
    async def get_market_data(self, ticker: str) -> MarketData | None:
        """Return MarketData if found, None if not found."""
        ...

    @abstractmethod
    async def get_financials(self, ticker: str) -> FinancialStatements | None:
        """Return FinancialStatements if found, None if not found."""
        ...

    @abstractmethod
    async def get_news(self, ticker: str, days: int = 30) -> list[NewsItem] | None:
        """Return list of NewsItem, empty list if none, None if source unavailable."""
        ...
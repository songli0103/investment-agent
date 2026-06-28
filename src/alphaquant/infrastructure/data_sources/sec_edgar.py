"""SEC EDGAR adapter for raw filings."""
from __future__ import annotations

import re

import httpx

from alphaquant.infrastructure.data_sources.base import DataSourceInterface
from alphaquant.models.company import Company
from alphaquant.models.financial import FinancialStatements
from alphaquant.models.market import MarketData
from alphaquant.models.news import NewsItem


class SECEdgarSource(DataSourceInterface):
    """SEC EDGAR. Free, no API key. Provides 10-K/10-Q raw filings."""

    BASE_URL = "https://data.sec.gov"
    HEADERS = {"User-Agent": "AlphaQuant research@example.com"}

    @property
    def name(self) -> str:
        return "sec_edgar"

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.BASE_URL}/submissions/CIK0000320193.json", headers=self.HEADERS)
                return r.status_code == 200
        except Exception:
            return False

    async def _ticker_to_cik(self, ticker: str) -> str | None:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    "https://www.sec.gov/cgi-bin/browse-edgar",
                    params={"action": "getcompany", "CIK": ticker, "type": "10-K", "output": "atom"},
                    headers=self.HEADERS,
                )
                match = re.search(r"CIK=(\d+)", r.text)
                return match.group(1).zfill(10) if match else None
        except Exception:
            return None

    async def get_company_info(self, ticker: str) -> Company | None:
        cik = await self._ticker_to_cik(ticker)
        if not cik:
            return None
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    f"{self.BASE_URL}/submissions/CIK{cik}.json", headers=self.HEADERS
                )
                r.raise_for_status()
                data = r.json()
        except Exception:
            return None
        return Company(
            ticker=ticker,
            name=data.get("name", ticker),
            exchange="NASDAQ",
            sector="Unknown",
            industry="Unknown",
            market_cap=0,
            description=data.get("description", "")[:500] if data.get("description") else None,
        )

    async def get_market_data(self, ticker: str) -> MarketData | None:
        return None  # SEC EDGAR does not provide market data

    async def get_financials(self, ticker: str) -> FinancialStatements | None:
        # For MVP, defer to Yahoo/Alpha Vantage. SEC provides XBRL but parsing is complex.
        return None

    async def get_news(self, ticker: str, days: int = 30) -> list[NewsItem] | None:
        return None
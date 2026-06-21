"""Finnhub adapter."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from alphaquant.infrastructure.config import get_settings
from alphaquant.infrastructure.data_sources.base import DataSourceInterface
from alphaquant.models.company import Company
from alphaquant.models.financial import FinancialStatements
from alphaquant.models.market import MarketData
from alphaquant.models.news import NewsItem


class FinnhubSource(DataSourceInterface):
    """Finnhub data source. Free tier: 60 calls/min."""

    def __init__(self) -> None:
        self._api_key = get_settings().finnhub_api_key
        self._client: object | None = None

    @property
    def name(self) -> str:
        return "finnhub"

    def _get_client(self):
        if self._client is None and self._api_key:
            import finnhub

            self._client = finnhub.Client(api_key=self._api_key)
        return self._client

    async def is_available(self) -> bool:
        return self._api_key is not None and len(self._api_key) > 0

    async def get_company_info(self, ticker: str) -> Company | None:
        def _fetch() -> Company | None:
            client = self._get_client()
            if not client:
                return None
            try:
                profile = client.company_profile2(symbol=ticker)
            except Exception:
                return None
            if not profile or not profile.get("ticker"):
                return None
            return Company(
                ticker=ticker,
                name=profile.get("name", ticker),
                exchange=profile.get("exchange", "NASDAQ"),
                sector="Unknown",
                industry=profile.get("finnhubIndustry", "Unknown"),
                market_cap=0,
            )

        return await asyncio.to_thread(_fetch)

    async def get_market_data(self, ticker: str) -> MarketData | None:
        def _fetch() -> MarketData | None:
            client = self._get_client()
            if not client:
                return None
            try:
                quote = client.quote(ticker)
            except Exception:
                return None
            if not quote or quote.get("c") == 0:
                return None
            return MarketData(
                ticker=ticker,
                as_of=datetime.utcnow(),
                price=float(quote["c"]),
                change_pct=float(quote.get("dp", 0) or 0),
                volume=0,
                market_cap=0,
                source="finnhub",
            )

        return await asyncio.to_thread(_fetch)

    async def get_financials(self, ticker: str) -> FinancialStatements | None:
        return None  # Finnhub financial statements require paid tier

    async def get_news(self, ticker: str, days: int = 30) -> list[NewsItem] | None:
        def _fetch() -> list[NewsItem]:
            client = self._get_client()
            if not client:
                return []
            try:
                frm = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
                to = datetime.utcnow().strftime("%Y-%m-%d")
                news = client.company_news(ticker, _from=frm, to=to)
            except Exception:
                return []
            items: list[NewsItem] = []
            for n in (news or [])[:20]:
                ts = datetime.utcfromtimestamp(n.get("datetime", 0))
                items.append(
                    NewsItem(
                        date=ts.date(),
                        title=n.get("headline", ""),
                        summary=n.get("summary"),
                        url=n.get("url", "https://finnhub.io"),
                        source=n.get("source", "Finnhub"),
                        sentiment="neutral",
                        topic="other",
                        relevance_score=0.5,
                    )
                )
            return items

        return await asyncio.to_thread(_fetch)

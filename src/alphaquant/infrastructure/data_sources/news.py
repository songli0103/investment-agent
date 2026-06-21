"""News API adapter (newsapi.org)."""
from __future__ import annotations

from datetime import datetime, timedelta

import httpx

from alphaquant.infrastructure.config import get_settings
from alphaquant.infrastructure.data_sources.base import DataSourceInterface
from alphaquant.models.company import Company
from alphaquant.models.financial import FinancialStatements
from alphaquant.models.market import MarketData
from alphaquant.models.news import NewsItem


class NewsAPISource(DataSourceInterface):
    """newsapi.org free tier: 100 req/day, headlines only."""

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self) -> None:
        self._api_key = get_settings().news_api_key

    @property
    def name(self) -> str:
        return "newsapi"

    async def is_available(self) -> bool:
        return self._api_key is not None and len(self._api_key) > 0

    async def get_company_info(self, ticker: str) -> Company | None:
        return None

    async def get_market_data(self, ticker: str) -> MarketData | None:
        return None

    async def get_financials(self, ticker: str) -> FinancialStatements | None:
        return None

    async def get_news(self, ticker: str, days: int = 30) -> list[NewsItem] | None:
        if not await self.is_available():
            return None
        frm = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    self.BASE_URL,
                    params={
                        "q": ticker,
                        "from": frm,
                        "sortBy": "relevancy",
                        "pageSize": 20,
                        "apiKey": self._api_key,
                    },
                )
                r.raise_for_status()
                data = r.json()
        except Exception:
            return []
        items: list[NewsItem] = []
        for n in data.get("articles", []):
            try:
                pub = datetime.fromisoformat(n["publishedAt"].replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                continue
            items.append(
                NewsItem(
                    date=pub.date(),
                    title=n.get("title", ""),
                    summary=n.get("description"),
                    url=n.get("url", "https://newsapi.org"),
                    source=n.get("source", {}).get("name", "NewsAPI"),
                    sentiment="neutral",
                    topic="other",
                    relevance_score=0.5,
                )
            )
        return items
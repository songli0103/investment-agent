"""Alpha Vantage adapter."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

from alphaquant.infrastructure.config import get_settings
from alphaquant.infrastructure.data_sources.base import DataSourceInterface
from alphaquant.models.company import Company
from alphaquant.models.financial import FinancialStatements
from alphaquant.models.market import MarketData
from alphaquant.models.news import NewsItem
from alphaquant.exceptions import PartialDataFailure


class AlphaVantageSource(DataSourceInterface):
    """Alpha Vantage data source. Free tier: 25 calls/day."""

    def __init__(self) -> None:
        self._api_key = get_settings().alpha_vantage_api_key
        self._base = "https://www.alphavantage.co/query"

    @property
    def name(self) -> str:
        return "alpha_vantage"

    async def is_available(self) -> bool:
        return self._api_key is not None and len(self._api_key) > 0

    async def _get(self, params: dict[str, Any]) -> dict[str, Any] | None:
        import httpx

        if not await self.is_available():
            return None
        params["apikey"] = self._api_key
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(self._base, params=params)
                r.raise_for_status()
                data = r.json()
                if "Note" in data or "Information" in data:
                    raise PartialDataFailure(f"Alpha Vantage rate limit: {data}")
                return data
        except Exception:
            return None

    async def get_company_info(self, ticker: str) -> Company | None:
        data = await self._get({"function": "OVERVIEW", "symbol": ticker})
        if not data or not data.get("Symbol"):
            return None
        return Company(
            ticker=ticker,
            name=data.get("Name", ticker),
            exchange=data.get("Exchange", "NASDAQ"),
            sector=data.get("Sector", "Unknown"),
            industry=data.get("Industry", "Unknown"),
            market_cap=int(float(data.get("MarketCapitalization", 0) or 0)),
            employees=int(data["FullTimeEmployees"]) if data.get("FullTimeEmployees") else None,
            description=data.get("Description"),
        )

    async def get_market_data(self, ticker: str) -> MarketData | None:
        data = await self._get({"function": "GLOBAL_QUOTE", "symbol": ticker})
        if not data or "Global Quote" not in data:
            return None
        q = data["Global Quote"]
        return MarketData(
            ticker=ticker,
            as_of=datetime.utcnow(),
            price=float(q.get("05. price", 0) or 0),
            change_pct=float(q.get("10. change percent", "0").rstrip("%") or 0),
            volume=int(q.get("06. volume", 0) or 0),
            market_cap=0,
            source="alpha_vantage",
        )

    async def get_financials(self, ticker: str) -> FinancialStatements | None:
        data = await self._get({"function": "INCOME_STATEMENT", "symbol": ticker})
        if not data or "annualReports" not in data:
            return None
        from alphaquant.models.financial import (
            BalanceSheet,
            CashFlowStatement,
            IncomeStatement,
        )

        stmts = FinancialStatements(ticker=ticker, source="alpha_vantage")
        for rep in data.get("annualReports", [])[:4]:
            stmts.income_statements.append(
                IncomeStatement(
                    period="FY",
                    fiscal_year=int(rep.get("fiscalDateEnding", "2020")[:4]),
                    revenue=float(rep.get("totalRevenue", 0) or 0),
                    net_income=float(rep.get("netIncome", 0) or 0),
                )
            )
        bs_data = await self._get({"function": "BALANCE_SHEET", "symbol": ticker})
        if bs_data and "annualReports" in bs_data:
            for rep in bs_data.get("annualReports", [])[:4]:
                stmts.balance_sheets.append(
                    BalanceSheet(
                        period="FY",
                        fiscal_year=int(rep.get("fiscalDateEnding", "2020")[:4]),
                        total_assets=float(rep.get("totalAssets", 0) or 0),
                        total_liabilities=float(rep.get("totalLiabilities", 0) or 0),
                        total_equity=float(rep.get("totalShareholderEquity", 0) or 0),
                    )
                )
        return stmts

    async def get_news(self, ticker: str, days: int = 30) -> list[NewsItem] | None:
        data = await self._get(
            {"function": "NEWS_SENTIMENT", "tickers": ticker, "limit": 50}
        )
        if not data or "feed" not in data:
            return []
        items: list[NewsItem] = []
        cutoff = datetime.utcnow() - timedelta(days=days)
        for n in data["feed"][:20]:
            try:
                ts = datetime.fromisoformat(n["time_published"].replace("Z", "+00:00"))
                ts = ts.replace(tzinfo=None)
                if ts < cutoff:
                    continue
                score = float(n.get("overall_sentiment_score", 0) or 0)
                sentiment = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
                items.append(
                    NewsItem(
                        date=ts.date(),
                        title=n.get("title", ""),
                        summary=n.get("summary"),
                        url=n.get("url", "https://alphavantage.co"),
                        source=n.get("source", "Alpha Vantage"),
                        sentiment=sentiment,
                        topic="other",
                        relevance_score=float(n.get("relevance_score", 0.5) or 0.5),
                    )
                )
            except Exception:
                continue
        return items

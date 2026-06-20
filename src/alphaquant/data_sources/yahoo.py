"""Yahoo Finance adapter using yfinance library."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import yfinance as yf

from alphaquant.data_sources.base import DataSourceInterface
from alphaquant.models.company import Company
from alphaquant.models.financial import (
    BalanceSheet,
    CashFlowStatement,
    FinancialStatements,
    IncomeStatement,
)
from alphaquant.models.market import MarketData
from alphaquant.models.news import NewsItem


class YahooFinanceSource(DataSourceInterface):
    """Yahoo Finance data source. MVP primary source."""

    @property
    def name(self) -> str:
        return "yahoo"

    async def is_available(self) -> bool:
        try:
            await asyncio.to_thread(lambda: yf.Ticker("AAPL").info)
            return True
        except Exception:
            return False

    async def get_company_info(self, ticker: str) -> Company | None:
        def _fetch() -> Company | None:
            try:
                info = yf.Ticker(ticker).info or {}
            except Exception:
                return None
            if not info or "shortName" not in info:
                return None
            return Company(
                ticker=ticker,
                name=info.get("shortName") or info.get("longName", ticker),
                exchange=info.get("exchange", "NASDAQ"),
                sector=info.get("sector", "Unknown"),
                industry=info.get("industry", "Unknown"),
                market_cap=int(info.get("marketCap", 0) or 0),
                employees=info.get("fullTimeEmployees"),
                description=info.get("longBusinessSummary"),
            )

        return await asyncio.to_thread(_fetch)

    async def get_market_data(self, ticker: str) -> MarketData | None:
        def _fetch() -> MarketData | None:
            try:
                info = yf.Ticker(ticker).info or {}
            except Exception:
                return None
            if not info or "currentPrice" not in info and "regularMarketPrice" not in info:
                return None
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
            return MarketData(
                ticker=ticker,
                as_of=datetime.utcnow(),
                price=price,
                change_pct=float(info.get("regularMarketChangePercent", 0) or 0),
                volume=int(info.get("volume", 0) or 0),
                market_cap=int(info.get("marketCap", 0) or 0),
                pe_ratio=info.get("trailingPE"),
                forward_pe=info.get("forwardPE"),
                pb_ratio=info.get("priceToBook"),
                ps_ratio=info.get("priceToSalesTrailing12Months"),
                eps=info.get("trailingEps"),
                revenue_growth_yoy=(
                    float(info["revenueGrowth"]) * 100 if info.get("revenueGrowth") else None
                ),
                high_52w=info.get("fiftyTwoWeekHigh"),
                low_52w=info.get("fiftyTwoWeekLow"),
                dividend_yield=(
                    float(info["dividendYield"]) * 100 if info.get("dividendYield") else None
                ),
                beta=info.get("beta"),
                source="yahoo",
            )

        return await asyncio.to_thread(_fetch)

    async def get_financials(self, ticker: str) -> FinancialStatements | None:
        def _fetch() -> FinancialStatements | None:
            try:
                t = yf.Ticker(ticker)
                inc = t.income_stmt
                bs = t.balance_sheet
                cf = t.cashflow
            except Exception:
                return None
            if inc is None or inc.empty:
                return None
            statements = FinancialStatements(ticker=ticker, source="yahoo")

            for col in inc.columns[:4]:
                row = inc[col]
                statements.income_statements.append(
                    IncomeStatement(
                        period="FY",
                        fiscal_year=col.year,
                        revenue=float(row.get("Total Revenue", 0) or 0),
                        cogs=row.get("Cost Of Revenue"),
                        gross_profit=row.get("Gross Profit"),
                        operating_income=row.get("Operating Income"),
                        net_income=float(row.get("Net Income", 0) or 0),
                        eps=row.get("Diluted EPS"),
                    )
                )

            if bs is not None and not bs.empty:
                for col in bs.columns[:4]:
                    row = bs[col]
                    statements.balance_sheets.append(
                        BalanceSheet(
                            period="FY",
                            fiscal_year=col.year,
                            total_assets=float(row.get("Total Assets", 0) or 0),
                            total_liabilities=float(row.get("Total Liab", 0) or 0),
                            total_equity=float(row.get("Total Equity", 0) or 0),
                            cash_and_equivalents=row.get("Cash"),
                            short_term_debt=row.get("Short Term Debt"),
                            long_term_debt=row.get("Long Term Debt"),
                        )
                    )

            if cf is not None and not cf.empty:
                for col in cf.columns[:4]:
                    row = cf[col]
                    statements.cash_flows.append(
                        CashFlowStatement(
                            period="FY",
                            fiscal_year=col.year,
                            operating_cash_flow=float(row.get("Operating Cash Flow", 0) or 0),
                            investing_cash_flow=row.get("Investing Cash Flow"),
                            financing_cash_flow=row.get("Financing Cash Flow"),
                            free_cash_flow=row.get("Free Cash Flow"),
                            capex=row.get("Capital Expenditures"),
                        )
                    )

            return statements

        return await asyncio.to_thread(_fetch)

    async def get_news(self, ticker: str, days: int = 30) -> list[NewsItem] | None:
        def _fetch() -> list[NewsItem]:
            try:
                news = yf.Ticker(ticker).news or []
            except Exception:
                return []
            cutoff = datetime.utcnow() - timedelta(days=days)
            items = []
            for n in news[:20]:
                pub = n.get("providerPublishTime")
                pub_dt = datetime.utcfromtimestamp(pub) if pub else datetime.utcnow()
                if pub_dt < cutoff:
                    continue
                items.append(
                    NewsItem(
                        date=pub_dt.date(),
                        title=n.get("title", ""),
                        summary=None,
                        url=n.get("link", "https://finance.yahoo.com"),
                        source=n.get("publisher", "Yahoo Finance"),
                        sentiment="neutral",
                        topic="other",
                        relevance_score=0.5,
                    )
                )
            return items

        return await asyncio.to_thread(_fetch)
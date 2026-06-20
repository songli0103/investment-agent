"""CrewAI tool for competitor identification."""
from __future__ import annotations

from crewai.tools import BaseTool

from alphaquant.data_sources import DataSourceRegistry
from alphaquant.models.competitor import Competitor


class CompetitorTool(BaseTool):
    name: str = "competitor_lookup"
    description: str = "Identify top 3 competitors for a given ticker in the same GICS industry. Returns peer company data."

    def _run(self, ticker: str) -> str:
        import asyncio
        from alphaquant.data_sources.yahoo import YahooFinanceSource

        def _get_company():
            src = YahooFinanceSource()
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(src.get_company_info(ticker))
            finally:
                loop.close()

        company = _get_company()
        if not company:
            return f"No company info for {ticker}"

        # Static fallback: hardcoded peer lists for common sectors (MVP shortcut)
        PEER_MAP: dict[str, list[str]] = {
            "Consumer Cyclical": ["WMT", "AMZN", "HD", "MCD", "NKE"],
            "Technology": ["MSFT", "GOOGL", "META", "ORCL", "CRM"],
            "Communication Services": ["GOOGL", "META", "NFLX", "DIS"],
            "Financial Services": ["JPM", "BAC", "GS", "MS", "WFC"],
            "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
            "Automotive": ["TM", "F", "GM", "STLA", "RIVN"],
        }
        peer_tickers = PEER_MAP.get(company.sector, ["SPY"])[:3]

        peers: list[Competitor] = []
        for pt in peer_tickers:
            if pt == ticker:
                continue

            def _fetch(pt=pt):
                src = YahooFinanceSource()
                loop = asyncio.new_event_loop()
                try:
                    m = loop.run_until_complete(src.get_market_data(pt))
                    return m
                finally:
                    loop.close()

            m = _fetch()
            if m:
                peers.append(
                    Competitor(
                        ticker=pt,
                        name=pt,
                        market_cap=m.market_cap,
                        revenue_ttm=m.market_cap,  # fallback; replace when financials available
                        gross_margin=None,
                        net_margin=None,
                        pe_ratio=m.pe_ratio,
                        ps_ratio=m.ps_ratio,
                    )
                )
        if not peers:
            return "No peer data available"
        import json

        return json.dumps([p.model_dump(mode="json") for p in peers], indent=2)


__all__ = ["CompetitorTool"]

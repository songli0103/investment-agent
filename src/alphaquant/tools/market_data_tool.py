"""CrewAI tool wrapper for market data."""
from __future__ import annotations

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from alphaquant.data_sources import DataSourceRegistry


class MarketDataInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. 'AAPL'")


class MarketDataTool(BaseTool):
    name: str = "market_data_lookup"
    description: str = "Look up real-time market data for a US stock ticker (price, P/E, market cap, volume, 52-week range, beta)."

    def _run(self, ticker: str) -> str:
        import asyncio

        registry = DataSourceRegistry()
        try:
            loop = asyncio.new_event_loop()
            market = loop.run_until_complete(registry.get_market(ticker))
            loop.close()
        except Exception as e:
            return f"Error fetching market data: {e}"
        if not market:
            return f"No market data available for {ticker}"
        return market.model_dump_json(indent=2)


__all__ = ["MarketDataTool", "MarketDataInput"]

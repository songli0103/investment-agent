"""CrewAI 市场数据工具包装器。"""
from __future__ import annotations

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from alphaquant.infrastructure.data_sources import DataSourceRegistry


TOOL_TIMEOUT_SECONDS = 30.0


class MarketDataInput(BaseModel):
    ticker: str = Field(..., description="股票代码,例如 'AAPL'")


class MarketDataTool(BaseTool):
    name: str = "market_data_lookup"
    description: str = "查询美股 ticker 的实时市场数据(价格、P/E、市值、成交量、52 周区间、beta)。"

    def _run(self, ticker: str) -> str:
        import asyncio

        registry = DataSourceRegistry()
        try:
            loop = asyncio.new_event_loop()
            try:
                market = loop.run_until_complete(
                    asyncio.wait_for(
                        registry.get_market(ticker),
                        timeout=TOOL_TIMEOUT_SECONDS,
                    )
                )
            finally:
                loop.close()
        except asyncio.TimeoutError:
            return f"Error fetching market data: timeout after {TOOL_TIMEOUT_SECONDS}s"
        except Exception as e:
            # Sub-3 Blocker 3: include exception type for diagnosability;
            # parse_crew_output's error-string detector still catches this prefix.
            return f"Error fetching market data: {type(e).__name__}: {e}"
        if not market:
            return f"No market data available for {ticker}"
        return market.model_dump_json(indent=2)


__all__ = ["MarketDataTool", "MarketDataInput", "TOOL_TIMEOUT_SECONDS"]
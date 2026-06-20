"""MarketAnalyst Agent."""
from __future__ import annotations

from crewai import Agent

from alphaquant.llm import get_llm
from alphaquant.tools.market_data_tool import MarketDataTool


def build_market_analyst_agent() -> Agent:
    return Agent(
        role="Senior Market Analyst",
        goal=(
            "Analyze market data for a US stock: price, valuation multiples, "
            "and generate a trend commentary."
        ),
        backstory=(
            "You are a Wall Street veteran market analyst. You interpret price, "
            "P/E, P/B, beta, and 52-week range to assess valuation and momentum. "
            "You always cite the data points you reference."
        ),
        tools=[MarketDataTool()],
        llm=get_llm(temperature=0.3),
        allow_delegation=False,
        verbose=True,
    )


__all__ = ["build_market_analyst_agent"]

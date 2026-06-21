"""MarketAnalyst Agent."""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM

from alphaquant.tools.market_data_tool import MarketDataTool


def build_market_analyst_agent(llm: LLM) -> Agent:
    return Agent(
        role="Market Data Specialist",
        goal=(
            "Fetch real-time market data (price, P/E, market cap, 52-week range, "
            "beta, growth metrics) for a US stock ticker. Report data verbatim - "
            "do not interpret or summarize."
        ),
        backstory=(
            "You are a quantitative data fetcher. You call market_data_lookup "
            "exactly once with the ticker and return its JSON output as-is."
        ),
        tools=[MarketDataTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_market_analyst_agent"]
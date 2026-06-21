"""CompetitorAnalyst Agent."""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM

from alphaquant.tools.competitor_tool import CompetitorTool


def build_competitor_analyst_agent(llm: LLM) -> Agent:
    return Agent(
        role="Competitive Landscape Analyst",
        goal=(
            "Identify and rank competitors for a US stock ticker. "
            "Return peer tickers, market caps, growth, margins."
        ),
        backstory=(
            "You are a sell-side equity analyst. You call competitor_lookup "
            "with the ticker, then summarize the peer set with industry rank."
        ),
        tools=[CompetitorTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_competitor_analyst_agent"]
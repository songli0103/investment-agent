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
            "You are a sell-side equity analyst. You MUST call competitor_lookup with the "
            "ticker, then summarize the competitive landscape in plain text. Do NOT "
            "produce structured Pydantic output; the Flow computes the structured "
            "CompetitorAnalysis from data. Your text is used as context for the report "
            "writer. Cover: peer tickers and names, market caps, growth, margins, and a "
            "short qualitative take on strengths and weaknesses."
        ),
        tools=[CompetitorTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_competitor_analyst_agent"]
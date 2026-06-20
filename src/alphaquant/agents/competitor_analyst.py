"""CompetitorAnalyst Agent."""
from __future__ import annotations

from crewai import Agent

from alphaquant.llm import get_llm
from alphaquant.tools.competitor_tool import CompetitorTool


def build_competitor_analyst_agent() -> Agent:
    return Agent(
        role="Industry Research Analyst",
        goal=(
            "Identify 3-5 publicly traded competitors in the same GICS sub-industry, "
            "compare them on size, growth, and profitability."
        ),
        backstory=(
            "You are an industry research expert. You map public companies to GICS "
            "sub-industries and select the most relevant peers for comparison."
        ),
        tools=[CompetitorTool()],
        llm=get_llm(temperature=0.3),
        allow_delegation=False,
        verbose=True,
    )


__all__ = ["build_competitor_analyst_agent"]

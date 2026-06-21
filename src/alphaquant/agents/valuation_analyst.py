"""ValuationAnalyst Agent."""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM

from alphaquant.tools.dcf_tool import DCFTool


def build_valuation_analyst_agent(llm: LLM) -> Agent:
    return Agent(
        role="Sell-side Valuation Modeler",
        goal=(
            "Estimate intrinsic value using DCF, relative valuation (P/E, P/B, P/S), "
            "and PEG. Provide a value range (low-high) with explicit assumptions."
        ),
        backstory=(
            "You are a sell-side equity research modeler. You build DCF models with "
            "explicit assumptions (growth, WACC, terminal). You cross-check with peer "
            "multiples. You never give a single point estimate-always a range +/-15%."
        ),
        tools=[DCFTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_valuation_analyst_agent"]
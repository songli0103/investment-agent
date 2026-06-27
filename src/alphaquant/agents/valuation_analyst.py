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
            "You are a sell-side equity valuation modeler. You call the DCF tool with "
            "explicit assumptions (growth rate, WACC, terminal growth), then summarize "
            "the valuation analysis (DCF + relative) in plain text. Do NOT produce "
            "structured Pydantic output; the Flow computes the structured ValuationResult "
            "from data. Your text is used as context for the report writer. Cover: "
            "intrinsic value per share, current price, upside %, DCF value, relative "
            "value, PEG, method used, and the key assumptions you applied."
        ),
        tools=[DCFTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_valuation_analyst_agent"]
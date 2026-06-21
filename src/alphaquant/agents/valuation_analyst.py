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
            "explicit assumptions (growth rate, WACC, terminal growth), then output a "
            "Pydantic ValuationResult. All fields required: ticker, intrinsic_value_per_share, "
            "current_price, upside_pct (%), dcf_value, relative_value, peg_ratio (nullable), "
            "method (one of the allowed Literal values), assumptions (dict of inputs you used). "
            "intrinsic_value_per_share is your blended estimate. If DCF is unavailable, use "
            "relative-only and explain in assumptions. dcf_value may be null only if FCF<=0."
        ),
        tools=[DCFTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_valuation_analyst_agent"]
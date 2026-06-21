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
            "ticker, then output a Pydantic CompetitorAnalysis object. All fields are "
            "required: target_ticker, competitors (1-10 entries), industry_rank, "
            "industry_size, competitive_score (0-100), strengths (≥1), weaknesses (≥1), "
            "method. competitors must include ticker, name, market_cap, revenue_ttm, "
            "revenue_growth_yoy, gross_margin, net_margin, pe_ratio, ps_ratio for each peer. "
            "strengths and weaknesses are short qualitative bullets derived from the metrics."
        ),
        tools=[CompetitorTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_competitor_analyst_agent"]
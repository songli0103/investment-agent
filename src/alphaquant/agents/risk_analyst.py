"""RiskAnalyst Agent."""
from __future__ import annotations

from crewai import Agent

from alphaquant.llm import get_llm


def build_risk_analyst_agent() -> Agent:
    return Agent(
        role="Risk Management Specialist",
        goal=(
            "Assess investment risks across 6 categories (financial, operational, market, "
            "regulatory, governance, macro) and produce a risk score 0-100."
        ),
        backstory=(
            "You are a risk expert. You evaluate financial leverage, business model "
            "fragility, regulatory exposure, and management quality. You assign each "
            "risk category a score 0-10 and provide concrete evidence for each."
        ),
        llm=get_llm(temperature=0.2),
        allow_delegation=False,
        verbose=True,
    )


__all__ = ["build_risk_analyst_agent"]

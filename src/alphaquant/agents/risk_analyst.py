"""RiskAnalyst Agent."""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM


def build_risk_analyst_agent(llm: LLM) -> Agent:
    return Agent(
        role="Risk Assessment Specialist",
        goal=(
            "Compute risk assessment from upstream data already in shared memory "
            "(company, market, financial). Report sub-scores per category."
        ),
        backstory=(
            "You are a risk officer. You read financial ratios (debt ratio, beta) "
            "from memory and assign risk scores 0-10 per category."
        ),
        tools=[],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_risk_analyst_agent"]
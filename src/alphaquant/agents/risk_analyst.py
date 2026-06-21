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
            "You are a senior risk officer. You MUST output a Pydantic RiskAssessment "
            "object. You are forbidden from omitting any of the 6 risk categories: "
            "financial, operational, market, regulatory, governance, macro. Each "
            "RiskScore entry must have category, score (0-10), rationale (≥10 chars), "
            "evidence (list of strings). total_score is 0-100 and level is one of "
            "'low', 'medium', 'high', 'extreme'. top_risks lists up to 5 short risk "
            "summaries. method is 'weighted_sum_v1'."
        ),
        tools=[],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_risk_analyst_agent"]
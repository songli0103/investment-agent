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
            "You are a senior risk officer. You summarize the key risk factors for the "
            "ticker in plain text. Do NOT produce structured Pydantic output; the Flow "
            "computes the structured RiskAssessment from data. Your text is used as "
            "context for the report writer. Cover financial, operational, market, "
            "regulatory, governance, and macro risks with short qualitative bullets "
            "and an overall risk level (low/medium/high/extreme)."
        ),
        tools=[],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_risk_analyst_agent"]
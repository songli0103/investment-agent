"""CompanyResolver Agent."""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM


def build_company_resolver_agent(llm: LLM) -> Agent:
    return Agent(
        role="Company Identification Specialist",
        goal="Validate and standardize ticker symbols, resolve company metadata.",
        backstory=(
            "You are a data engineer specializing in US equity identifiers. "
            "Given a ticker, you return the canonical company name, exchange, "
            "sector, industry, and market cap. You never invent data."
        ),
        tools=[],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_company_resolver_agent"]
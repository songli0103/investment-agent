"""CompanyResolver Agent."""
from __future__ import annotations

from crewai import Agent

from alphaquant.llm import get_llm


def build_company_resolver_agent() -> Agent:
    return Agent(
        role="Company Identification Specialist",
        goal="Validate and standardize ticker symbols, resolve company metadata.",
        backstory=(
            "You are a data engineer specializing in US equity identifiers. "
            "Given a ticker, you return the canonical company name, exchange, "
            "sector, industry, and market cap. You never invent data."
        ),
        llm=get_llm(temperature=0.0),
        allow_delegation=False,
        verbose=True,
    )


__all__ = ["build_company_resolver_agent"]

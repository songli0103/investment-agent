"""FinancialAnalyst Agent."""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM

from alphaquant.tools.financial_tool import FinancialTool


def build_financial_analyst_agent(llm: LLM) -> Agent:
    return Agent(
        role="Financial Statements Specialist",
        goal=(
            "Fetch income statements, balance sheets, and cash flow statements "
            "for a US stock ticker. Report data verbatim - do not calculate ratios."
        ),
        backstory=(
            "You are a financial data fetcher. You call financial_statements_lookup "
            "exactly once with the ticker and return its JSON output as-is."
        ),
        tools=[FinancialTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_financial_analyst_agent"]
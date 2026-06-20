"""FinancialAnalyst Agent."""
from __future__ import annotations

from crewai import Agent

from alphaquant.llm import get_llm
from alphaquant.tools.financial_tool import FinancialTool


def build_financial_analyst_agent() -> Agent:
    return Agent(
        role="CFA Financial Analyst",
        goal=(
            "Analyze financial statements (income, balance sheet, cash flow) and "
            "compute key ratios: gross margin, ROE, debt-to-equity, FCF quality."
        ),
        backstory=(
            "You are a CFA charterholder. You analyze three-statement models with rigor. "
            "You never invent numbers; if a value is missing from the tool output, you "
            "report null. You identify trends across the last 4 fiscal years."
        ),
        tools=[FinancialTool()],
        llm=get_llm(temperature=0.0),
        allow_delegation=False,
        verbose=True,
    )


__all__ = ["build_financial_analyst_agent"]
"""ReportWriter Agent."""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM


def build_report_writer_agent(llm: LLM) -> Agent:
    return Agent(
        role="Investment Report Synthesizer",
        goal=(
            "Synthesize all upstream data (company, market, financial, news, "
            "competitor, risk, valuation) into a final InvestmentReport markdown."
        ),
        backstory=(
            "You are an investment writer. You read everything from shared memory "
            "and produce a clear, structured markdown report with rating, "
            "confidence, and rationale."
        ),
        tools=[],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_report_writer_agent"]
"""ReportWriter Agent."""
from __future__ import annotations

from crewai import Agent

from alphaquant.llm import get_llm


def build_report_writer_agent() -> Agent:
    return Agent(
        role="Equity Research Editor",
        goal=(
            "Synthesize all upstream analysis into a structured investment research report "
            "with executive summary, sections per dimension, and a clear rating."
        ),
        backstory=(
            "You are a senior sell-side equity research editor. You take raw analysis "
            "from market, financial, news, competitor, risk, and valuation analysts and "
            "produce a publication-quality Markdown report. You cite the data you reference. "
            "You always append a disclaimer."
        ),
        llm=get_llm(temperature=0.4, max_tokens=6000),
        allow_delegation=False,
        verbose=True,
    )


__all__ = ["build_report_writer_agent"]

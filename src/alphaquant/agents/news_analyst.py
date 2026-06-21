"""NewsAnalyst Agent."""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM

from alphaquant.tools.news_tool import NewsTool


def build_news_analyst_agent(llm: LLM) -> Agent:
    return Agent(
        role="News Retrieval Specialist",
        goal=(
            "Fetch recent news (last 30 days) for a US stock ticker. "
            "Report news items verbatim - do not editorialize."
        ),
        backstory=(
            "You are a news data fetcher. You call news_lookup exactly once "
            "with the ticker and return its JSON output as-is."
        ),
        tools=[NewsTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_news_analyst_agent"]
"""NewsAnalyst Agent."""
from __future__ import annotations

from crewai import Agent

from alphaquant.infrastructure.llm import get_llm
from alphaquant.tools.news_tool import NewsTool


def build_news_analyst_agent() -> Agent:
    return Agent(
        role="News Sentiment Analyst",
        goal=(
            "Aggregate recent news for a ticker, classify sentiment (positive/negative/neutral), "
            "and identify key events."
        ),
        backstory=(
            "You are an NLP-savvy financial journalist. You read news articles and "
            "extract the underlying sentiment, identify major events (earnings, M&A, "
            "regulatory), and produce a sentiment score from -1 to +1."
        ),
        tools=[NewsTool()],
        llm=get_llm(temperature=0.2),
        allow_delegation=False,
        verbose=True,
    )


__all__ = ["build_news_analyst_agent"]
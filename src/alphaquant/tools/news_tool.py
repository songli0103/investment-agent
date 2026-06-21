"""CrewAI tool for news."""
from __future__ import annotations

from crewai.tools import BaseTool

from alphaquant.infrastructure.data_sources import DataSourceRegistry


class NewsTool(BaseTool):
    name: str = "news_lookup"
    description: str = "Fetch recent news (last 30 days) for a US stock ticker. Returns news items with titles, sources, dates."

    def _run(self, ticker: str) -> str:
        import asyncio

        registry = DataSourceRegistry()
        try:
            loop = asyncio.new_event_loop()
            news = loop.run_until_complete(registry.get_news(ticker, days=30))
            loop.close()
        except Exception as e:
            return f"Error fetching news: {e}"
        if not news:
            return f"No news found for {ticker}"
        import json
        from datetime import date

        return json.dumps(
            [
                {
                    "date": n.date.isoformat() if isinstance(n.date, date) else str(n.date),
                    "title": n.title,
                    "source": n.source,
                    "url": str(n.url),
                }
                for n in news[:20]
            ],
            indent=2,
        )


__all__ = ["NewsTool"]

"""CrewAI 新闻工具。"""
from __future__ import annotations

from crewai.tools import BaseTool

from alphaquant.infrastructure.data_sources import DataSourceRegistry


TOOL_TIMEOUT_SECONDS = 30.0


class NewsTool(BaseTool):
    name: str = "news_lookup"
    description: str = "获取美股 ticker 的近期新闻(最近 30 天)。返回包含标题、来源、日期的新闻条目。"

    def _run(self, ticker: str) -> str:
        import asyncio

        registry = DataSourceRegistry()
        try:
            loop = asyncio.new_event_loop()
            try:
                news = loop.run_until_complete(
                    asyncio.wait_for(
                        registry.get_news(ticker, days=30),
                        timeout=TOOL_TIMEOUT_SECONDS,
                    )
                )
            finally:
                loop.close()
        except asyncio.TimeoutError:
            return f"Error fetching news: timeout after {TOOL_TIMEOUT_SECONDS}s"
        except Exception as e:
            # Sub-3 Blocker 3: include exception type for diagnosability;
            # parse_crew_output's error-string detector still catches this prefix.
            return f"Error fetching news: {type(e).__name__}: {e}"
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


__all__ = ["NewsTool", "TOOL_TIMEOUT_SECONDS"]
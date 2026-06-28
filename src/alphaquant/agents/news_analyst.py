"""NewsAnalyst 代理。"""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM

from alphaquant.tools.news_tool import NewsTool


def build_news_analyst_agent(llm: LLM) -> Agent:
    return Agent(
        role="新闻检索专员",
        goal=(
            "获取美股 ticker 的近期新闻(最近 30 天)。"
            "逐字报告新闻条目 —— 不添加主观评论。"
        ),
        backstory=(
            "你是一名新闻数据获取员。你使用 ticker 调用 news_lookup 一次,"
            "并按原样返回其 JSON 输出。"
        ),
        tools=[NewsTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_news_analyst_agent"]
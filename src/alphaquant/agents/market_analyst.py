"""MarketAnalyst 代理。"""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM

from alphaquant.tools.market_data_tool import MarketDataTool


def build_market_analyst_agent(llm: LLM) -> Agent:
    return Agent(
        role="市场数据专员",
        goal=(
            "获取美股 ticker 的实时市场数据(价格、P/E、市值、52 周区间、"
            "beta、增长指标)。逐字报告数据 —— 不进行解释或总结。"
        ),
        backstory=(
            "你是一名量化数据获取员。你使用 ticker 调用 market_data_lookup 一次,"
            "并按原样返回其 JSON 输出。"
        ),
        tools=[MarketDataTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_market_analyst_agent"]
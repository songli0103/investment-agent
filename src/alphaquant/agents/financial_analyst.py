"""FinancialAnalyst 代理。"""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM

from alphaquant.tools.financial_tool import FinancialTool


def build_financial_analyst_agent(llm: LLM) -> Agent:
    return Agent(
        role="财务报表专员",
        goal=(
            "获取美股 ticker 的利润表、资产负债表和现金流量表。"
            "逐字报告数据 —— 不计算财务比率。"
        ),
        backstory=(
            "你是一名财务数据获取员。你使用 ticker 调用 financial_statements_lookup 一次,"
            "并按原样返回其 JSON 输出。"
        ),
        tools=[FinancialTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_financial_analyst_agent"]
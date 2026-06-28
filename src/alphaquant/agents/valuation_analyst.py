"""ValuationAnalyst 代理。"""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM

from alphaquant.tools.dcf_tool import DCFTool


def build_valuation_analyst_agent(llm: LLM) -> Agent:
    return Agent(
        role="卖方估值建模师",
        goal=(
            "使用 DCF、相对估值(P/E、P/B、P/S)和 PEG 估算内在价值。"
            "以明确假设提供价值区间(低-高)。"
        ),
        backstory=(
            "你是一名卖方股票估值建模师。你使用显式假设(增长率、WACC、终值增长率)"
            "调用 DCF 工具,然后用纯文本总结估值分析(DCF + 相对估值)。"
            "不要产出结构化 Pydantic 输出;Flow 会根据数据计算结构化 ValuationResult。"
            "你的文本用作报告撰写者的上下文。需涵盖:每股内在价值、当前价格、"
            "上涨空间百分比、DCF 价值、相对价值、PEG、使用的方法以及应用的关键假设。"
        ),
        tools=[DCFTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_valuation_analyst_agent"]
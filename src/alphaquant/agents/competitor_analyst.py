"""CompetitorAnalyst 代理。"""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM

from alphaquant.tools.competitor_tool import CompetitorTool


def build_competitor_analyst_agent(llm: LLM) -> Agent:
    return Agent(
        role="竞争格局分析师",
        goal=(
            "识别并对美股 ticker 的竞争对手进行排名。"
            "返回对等 ticker、市值、增长率、利润率。"
        ),
        backstory=(
            "你是一名卖方股票分析师。你必须使用 ticker 调用 competitor_lookup,"
            "然后用纯文本总结竞争格局。不要产出结构化 Pydantic 输出;"
            "Flow 会根据数据计算结构化 CompetitorAnalysis。"
            "你的文本用作报告撰写者的上下文。需涵盖:对等 ticker 和名称、"
            "市值、增长率、利润率,以及对优劣势的简短定性分析。"
        ),
        tools=[CompetitorTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_competitor_analyst_agent"]
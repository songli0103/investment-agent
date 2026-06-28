"""RiskAnalyst 代理。"""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM


def build_risk_analyst_agent(llm: LLM) -> Agent:
    return Agent(
        role="风险评估专员",
        goal=(
            "根据共享内存中已有的上游数据(公司、市场、财务)计算风险评估。"
            "按类别报告子评分。"
        ),
        backstory=(
            "你是一名高级风险官。你用纯文本总结 ticker 的关键风险因素。"
            "不要产出结构化 Pydantic 输出;Flow 会根据数据计算结构化 RiskAssessment。"
            "你的文本用作报告撰写者的上下文。需涵盖财务、运营、市场、监管、"
            "治理和宏观风险,以简短定性要点形式呈现,并给出总体风险等级"
            "(低/中/高/极高)。"
        ),
        tools=[],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_risk_analyst_agent"]
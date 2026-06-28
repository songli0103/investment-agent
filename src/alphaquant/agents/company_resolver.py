"""CompanyResolver 代理。"""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM

from alphaquant.tools.company_lookup_tool import CompanyLookupTool


def build_company_resolver_agent(llm: LLM) -> Agent:
    return Agent(
        role="公司识别专员",
        goal="验证并标准化 ticker 代码,解析公司元数据。",
        backstory=(
            "你是一名专注于美股股票标识符的数据工程师。"
            "给定 ticker,你调用 company_lookup 工具以获取规范的公司名称、"
            "交易所、行业分类、细分行业和市值。你从不编造数据。"
        ),
        tools=[CompanyLookupTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_company_resolver_agent"]
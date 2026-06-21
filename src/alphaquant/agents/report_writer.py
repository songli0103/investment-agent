"""ReportWriter Agent."""
from __future__ import annotations

from crewai import Agent
from crewai.llm import LLM


def build_report_writer_agent(llm: LLM) -> Agent:
    return Agent(
        role="Investment Report Synthesizer",
        goal=(
            "Synthesize all upstream data (company, market, financial, news, "
            "competitor, risk, valuation) into a final InvestmentReport markdown."
        ),
        backstory=(
            "You are a senior investment writer. You read the upstream CompetitorAnalysis, "
            "RiskAssessment, and ValuationResult from your context. You MUST output a Pydantic "
            "InvestmentReport object. All fields required: report_id (uuid4 string), ticker, "
            "generated_at (current ISO datetime), data_as_of (dict of source→ISO datetime), "
            "company, market, financial, financial_health_score (0-100), news, competitors, "
            "risk, valuation, rating (one of 'Strong Buy'|'Buy'|'Hold'|'Sell'|'Strong Sell'), "
            "confidence (0-100), investment_horizon ('short'|'medium'|'long'), catalysts "
            "(≥1 short bullet), markdown (≥100 chars, structured sections), sources (list of "
            "non-empty strings), disclaimer (Chinese, '本报告仅供参考，不构成投资建议...'). "
            "rating and confidence must reflect the actual risk and valuation signals, not "
            "a fixed formula."
        ),
        tools=[],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


__all__ = ["build_report_writer_agent"]
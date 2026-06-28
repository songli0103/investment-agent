"""最终的投资报告模型。"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

from alphaquant.models.company import Company
from alphaquant.models.competitor import CompetitorAnalysis
from alphaquant.models.financial import FinancialStatements
from alphaquant.models.market import MarketData
from alphaquant.models.news import NewsAnalysis
from alphaquant.models.risk import RiskAssessment
from alphaquant.models.valuation import ValuationResult


class ReportWriterOutput(BaseModel):
    """LLM 产出的 InvestmentReport 子集。子项目 3 回退:结构化分析字段
    (竞争、风险、估值)由 flow 确定性地计算;LLM 只产出下面的合成字段。
    Flow 通过将此与数据字段以及确定性分析相结合来组装完整的
    ``InvestmentReport``。
    """

    rating: Literal["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
    confidence: int | None = Field(None, ge=0, le=100)
    investment_horizon: Literal["short", "medium", "long"] = "medium"
    catalysts: list[str] = Field(default_factory=list)
    markdown: str = Field(..., min_length=1)

    @field_validator("rating", mode="before")
    @classmethod
    def _coerce_rating(cls, v: Any) -> Any:
        """LLM 守卫:将未知的 rating 值强制转换为 'Hold',以避免 flow 崩溃。
        参见 ``CompetitorAnalysis._coerce_method`` 了解已建立的模式。
        """
        allowed = {"Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"}
        return v if v in allowed else "Hold"


class InvestmentReport(BaseModel):
    """顶层投资研究报告输出。"""

    report_id: str = Field(..., description="UUID4")
    ticker: str
    generated_at: datetime
    data_as_of: dict[str, datetime] = Field(default_factory=dict)
    company: Company
    market: MarketData
    financial: FinancialStatements
    financial_health_score: int = Field(..., ge=0, le=100)
    news: NewsAnalysis
    competitors: CompetitorAnalysis
    risk: RiskAssessment
    valuation: ValuationResult
    rating: Literal["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
    confidence: int | None = Field(None, ge=0, le=100)
    investment_horizon: Literal["short", "medium", "long"] = "medium"
    catalysts: list[str] = Field(default_factory=list)
    markdown: str = Field(..., min_length=1)
    sources: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "本报告由 AI 自动生成，仅供参考，不构成任何投资建议。"
        "投资有风险，决策需谨慎。"
    )

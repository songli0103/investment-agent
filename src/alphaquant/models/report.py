"""Final investment report model."""
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
    """LLM-produced subset of InvestmentReport. Sub-project-3 revert: the
    structured analysis fields (competitors, risk, valuation) are computed
    deterministically by the flow; the LLM only produces the synthesis fields
    below. The flow assembles the full ``InvestmentReport`` by combining this
    with data fields and the deterministic analyses.
    """

    rating: Literal["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
    confidence: int | None = Field(None, ge=0, le=100)
    investment_horizon: Literal["short", "medium", "long"] = "medium"
    catalysts: list[str] = Field(default_factory=list)
    markdown: str = Field(..., min_length=1)

    @field_validator("rating", mode="before")
    @classmethod
    def _coerce_rating(cls, v: Any) -> Any:
        """LLM guard: coerce unknown rating values to 'Hold' so the flow
        does not crash. See ``CompetitorAnalysis._coerce_method`` for the
        established pattern."""
        allowed = {"Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"}
        return v if v in allowed else "Hold"


class InvestmentReport(BaseModel):
    """Top-level investment research report output."""

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

"""Risk assessment models."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class RiskScore(BaseModel):
    category: Literal["financial", "operational", "market", "regulatory", "governance", "macro"]
    score: int = Field(..., ge=0, le=10)
    rationale: str = Field(..., min_length=10)
    evidence: list[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    ticker: str
    total_score: int = Field(..., ge=0, le=100)
    level: Literal["low", "medium", "high", "extreme"]
    sub_scores: list[RiskScore] = Field(..., min_length=1)
    top_risks: list[str] = Field(..., max_length=5)
    method: str = "weighted_sum_v1"

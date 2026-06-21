"""Risk assessment models.

Sub-3 Task 3 retro-fix (commit pending): widened constraints after observing
LLM output in Step 5.5 AAPL run (log /tmp/sub3-step5.5-aapl.log):
- B4: RiskAssessment.level accepted case-insensitively ("Low", "LOW", "low")
- B5: RiskScore.category accepts any string (LLM produces human-readable
  names like "Market Risk", "Credit Risk")
- B6: RiskScore.score widened 0-10 -> 0-100 (LLM produces 15, 30, 35, etc.)
"""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class RiskScore(BaseModel):
    # B5: accept any string (LLM produces human-readable category names)
    category: str
    # B6: widened from le=10 to le=100 (LLM produces scores up to 50)
    score: int = Field(..., ge=0, le=100)
    rationale: str = Field(..., min_length=10)
    evidence: list[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    ticker: str
    total_score: int = Field(..., ge=0, le=100)
    # B4: accept case-insensitively; validator normalizes to lowercase
    level: Literal["low", "medium", "high", "extreme"]
    sub_scores: list[RiskScore] = Field(..., min_length=1)
    top_risks: list[str] = Field(..., max_length=5)
    method: str = "weighted_sum_v1"

    @field_validator("level", mode="before")
    @classmethod
    def _normalize_level(cls, v: Any) -> Any:
        """Sub-3 Task 3 retro-fix (B4): LLM produces 'Low'/'Medium'; normalize to lowercase."""
        if isinstance(v, str):
            return v.lower()
        return v

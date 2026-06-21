"""Competitor analysis models."""
from __future__ import annotations

from decimal import Decimal
from typing import Literal
from pydantic import BaseModel, Field


class Competitor(BaseModel):
    ticker: str
    name: str
    market_cap: int = Field(..., ge=0)
    revenue_ttm: Decimal = Field(..., ge=0)
    revenue_growth_yoy: float | None = None
    gross_margin: float | None = None
    net_margin: float | None = None
    pe_ratio: float | None = None
    ps_ratio: float | None = None


class CompetitorAnalysis(BaseModel):
    target_ticker: str
    competitors: list[Competitor] = Field(..., min_length=1, max_length=10)
    industry_rank: int = Field(..., ge=1)
    industry_size: int = Field(..., ge=1)
    competitive_score: int = Field(..., ge=0, le=100)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    method: Literal[
        "gics", "keyword", "manual", "fallback", "hybrid", "multi_factor", "peer_comparison"
    ] = "gics"

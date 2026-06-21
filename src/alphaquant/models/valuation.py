"""Valuation models."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal
from pydantic import BaseModel, Field


class ValuationResult(BaseModel):
    ticker: str
    intrinsic_value_per_share: Decimal | None = None
    current_price: Decimal = Field(..., ge=0)
    upside_pct: float
    dcf_value: Decimal | None = None
    relative_value: Decimal | None = None
    peg_ratio: float | None = None
    method: Literal[
        "dcf_relative_peg", "relative_only", "blended", "dcf_only", "relative", "dcf_relative_blended"
    ] = "dcf_relative_peg"
    assumptions: dict[str, Any] = Field(default_factory=dict)

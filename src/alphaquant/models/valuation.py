"""Valuation models."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


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

    @field_validator("method", mode="before")
    @classmethod
    def _coerce_method(cls, v: Any) -> Any:
        """LLM guard: valuation agent may return non-JSON or an unexpected
        method tag. Coerce any unknown value to the safe default
        'dcf_relative_peg' so the flow does not crash."""
        allowed = {
            "dcf_relative_peg", "relative_only", "blended",
            "dcf_only", "relative", "dcf_relative_blended",
        }
        return v if v in allowed else "dcf_relative_peg"

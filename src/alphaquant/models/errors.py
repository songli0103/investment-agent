"""API error response model."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    code: Literal[
        "TICKER_NOT_FOUND",
        "INVALID_TICKER_FORMAT",
        "ALL_DATA_SOURCES_DOWN",
        "PARTIAL_DATA_FAILURE",
        "RATE_LIMIT_EXCEEDED",
        "INTERNAL_ERROR",
    ]
    message: str
    details: dict[str, Any] = Field(default_factory=dict)

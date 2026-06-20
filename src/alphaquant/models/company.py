"""Company identification model."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class Company(BaseModel):
    """Standardized company metadata."""

    ticker: str = Field(..., pattern=r"^[A-Z]{1,5}(\.[A-Z])?$", description="US stock ticker")
    name: str = Field(..., min_length=1)
    name_cn: str | None = None
    exchange: Literal["NASDAQ", "NYSE", "NYSE Arca", "OTC"]
    sector: str
    industry: str
    country: Literal["US"] = "US"
    currency: Literal["USD"] = "USD"
    market_cap: int = Field(..., ge=0)
    employees: int | None = Field(None, ge=0)
    description: str | None = None

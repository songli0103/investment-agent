"""Market data model."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal
from pydantic import BaseModel, Field


class MarketData(BaseModel):
    """Real-time and historical market data for a ticker."""

    ticker: str
    as_of: datetime
    price: Decimal = Field(..., ge=0)
    change_pct: float
    volume: int = Field(..., ge=0)
    market_cap: int = Field(..., ge=0)
    pe_ratio: float | None = None
    forward_pe: float | None = None
    pb_ratio: float | None = None
    ps_ratio: float | None = None
    eps: Decimal | None = None
    revenue_growth_yoy: float | None = None
    high_52w: Decimal | None = None
    low_52w: Decimal | None = None
    dividend_yield: float | None = None
    beta: float | None = None
    source: Literal["yahoo", "alpha_vantage", "finnhub", "degraded"] = "yahoo"

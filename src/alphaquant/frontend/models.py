"""Frontend data models."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ReportRecord:
    """Persisted investment report row."""

    id: int
    ticker: str
    generated_at: datetime
    rating: str
    confidence: int
    market_price: float | None
    report_json: str
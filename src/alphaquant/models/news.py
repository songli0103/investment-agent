"""News analysis models."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal
from pydantic import BaseModel, Field, HttpUrl


class NewsItem(BaseModel):
    date: date
    title: str = Field(..., min_length=1)
    summary: str | None = None
    url: HttpUrl
    source: str = Field(..., min_length=1)
    sentiment: Literal["positive", "neutral", "negative"]
    topic: Literal["product", "regulatory", "personnel", "financial", "market", "other"] = "other"
    relevance_score: float = Field(..., ge=0, le=1)


class NewsAnalysis(BaseModel):
    ticker: str
    as_of: datetime
    window_days: int = 30
    total_count: int = Field(..., ge=0)
    positive_pct: float = Field(..., ge=0, le=1)
    negative_pct: float = Field(..., ge=0, le=1)
    neutral_pct: float = Field(..., ge=0, le=1)
    sentiment_score: float = Field(..., ge=-1, le=1)
    key_events: list[NewsItem] = Field(default_factory=list)
    source: str = "unavailable"

    @classmethod
    def empty(cls, ticker: str) -> "NewsAnalysis":
        """Empty result when news data unavailable."""
        return cls(
            ticker=ticker,
            as_of=datetime.utcnow(),
            total_count=0,
            positive_pct=0.0,
            negative_pct=0.0,
            neutral_pct=1.0,
            sentiment_score=0.0,
        )

"""API I/O schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field

from alphaquant.models.report import InvestmentReport


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, pattern=r"^[A-Za-z]{1,5}(\.[A-Za-z])?$")
    options: dict[str, bool] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    report_id: str
    status: str = "completed"
    report: InvestmentReport


class HealthResponse(BaseModel):
    status: str
    version: str
    data_sources: dict[str, str]

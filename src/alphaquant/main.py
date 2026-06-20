"""AlphaQuant entry points: run_analysis + FastAPI app."""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI

from alphaquant.api.routes import router
from alphaquant.exceptions import (
    AllDataSourcesDown,
    InvalidTickerFormat,
    TickerNotFound,
)
from alphaquant.flows import AnalysisFlow
from alphaquant.models.report import InvestmentReport

VERSION = "1.0.0"

app = FastAPI(
    title="AlphaQuant",
    description="AI Investment Research Analyst",
    version=VERSION,
)
app.include_router(router, prefix="/api/v1")


async def run_analysis_async(ticker: str) -> InvestmentReport:
    """Run the full analysis flow. Async entry for FastAPI."""
    flow = AnalysisFlow()
    flow.kickoff(inputs={"ticker": ticker})
    if flow.state.report is None:
        raise AllDataSourcesDown(f"Flow produced no report for {ticker}")
    return flow.state.report


def run_analysis(ticker: str) -> InvestmentReport:
    """Run the full analysis flow. Sync entry for CLI."""
    return asyncio.run(run_analysis_async(ticker))

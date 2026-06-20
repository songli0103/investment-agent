"""Shared analysis core.

This module is the single source of truth for "given a ticker, produce an
InvestmentReport". Both the FastAPI route (asynchronously) and the CLI
(synchronously) delegate here.

Importing this module does NOT pull in the FastAPI app, so it is safe to
import from anywhere without circular-import risk.
"""
from __future__ import annotations

import asyncio

from alphaquant.exceptions import AllDataSourcesDown
from alphaquant.flows import AnalysisFlow
from alphaquant.models.report import InvestmentReport
from alphaquant.observability import get_logger

log = get_logger("alphaquant.core")


async def run_analysis_async(ticker: str) -> InvestmentReport:
    """Run the full analysis flow. Async entry for FastAPI.

    Uses ``kickoff_with_timeout`` (spec §3.4: 120s whole-Flow timeout) to
    avoid blocking the FastAPI event loop with a synchronous CrewAI call.
    """
    log.info("analysis_started", ticker=ticker)
    flow = AnalysisFlow()
    await flow.kickoff_with_timeout({"ticker": ticker})
    if flow.state.report is None:
        log.error("analysis_no_report", ticker=ticker)
        raise AllDataSourcesDown(f"Flow produced no report for {ticker}")
    log.info(
        "analysis_completed",
        ticker=ticker,
        report_id=flow.state.report.report_id,
        rating=flow.state.report.rating,
    )
    return flow.state.report


def run_analysis(ticker: str) -> InvestmentReport:
    """Run the full analysis flow. Sync entry for CLI."""
    return asyncio.run(run_analysis_async(ticker))

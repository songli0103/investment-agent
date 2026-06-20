"""FastAPI routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from alphaquant.api.rate_limiter import rate_limit_analyze
from alphaquant.api.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse
from alphaquant.exceptions import (
    AllDataSourcesDown,
    InvalidTickerFormat,
    TickerNotFound,
)
from alphaquant.core import run_analysis_async

router = APIRouter()

VERSION = "1.0.0"


@router.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(rate_limit_analyze)])
async def analyze(req: AnalyzeRequest):
    """Run the full analysis flow. Delegates to the shared core in main.py.

    The shared ``run_analysis_async`` owns the Flow lifecycle, the 120s
    timeout (§3.4), and the exception semantics. This layer only translates
    domain exceptions to HTTP status codes per spec §5.2.
    """
    try:
        report = await run_analysis_async(req.ticker)
    except InvalidTickerFormat as e:
        raise HTTPException(400, detail={"code": "INVALID_TICKER_FORMAT", "message": str(e)})
    except TickerNotFound as e:
        raise HTTPException(404, detail={"code": "TICKER_NOT_FOUND", "message": str(e)})
    except AllDataSourcesDown as e:
        raise HTTPException(503, detail={"code": "ALL_DATA_SOURCES_DOWN", "message": str(e)})
    return AnalyzeResponse(report_id=report.report_id, report=report)


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version=VERSION,
        data_sources={"yahoo": "ok", "alpha_vantage": "ok", "finnhub": "ok"},
    )

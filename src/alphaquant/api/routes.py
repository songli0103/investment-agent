"""FastAPI routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from alphaquant.api.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse
from alphaquant.exceptions import (
    AllDataSourcesDown,
    InvalidTickerFormat,
    TickerNotFound,
)
from alphaquant.flows import AnalysisFlow

router = APIRouter()

VERSION = "1.0.0"


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    try:
        flow = AnalysisFlow()
        flow.kickoff(inputs={"ticker": req.ticker})
        report = flow.state.report
        if report is None:
            raise HTTPException(500, detail={"code": "INTERNAL_ERROR", "message": "Flow produced no report"})
        return AnalyzeResponse(report_id=report.report_id, report=report)
    except InvalidTickerFormat as e:
        raise HTTPException(400, detail={"code": "INVALID_TICKER_FORMAT", "message": str(e)})
    except TickerNotFound as e:
        raise HTTPException(404, detail={"code": "TICKER_NOT_FOUND", "message": str(e)})
    except AllDataSourcesDown as e:
        raise HTTPException(503, detail={"code": "ALL_DATA_SOURCES_DOWN", "message": str(e)})


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version=VERSION,
        data_sources={"yahoo": "ok", "alpha_vantage": "ok", "finnhub": "ok"},
    )

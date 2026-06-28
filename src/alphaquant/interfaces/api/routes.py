"""FastAPI 路由。"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from alphaquant.interfaces.api.rate_limiter import rate_limit_analyze
from alphaquant.interfaces.api.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse
from alphaquant.exceptions import (
    AllDataSourcesDown,
    CrewExecutionError,
    InvalidTickerFormat,
    LLMRateLimited,
    ReportGenerationError,
    TickerNotFound,
)
from alphaquant.core import run_analysis_async

router = APIRouter()

VERSION = "1.0.0"


@router.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(rate_limit_analyze)])
async def analyze(req: AnalyzeRequest):
    """运行完整的分析流程。委托给 main.py 中的共享核心。

    共享的 ``run_analysis_async`` 负责 Flow 生命周期、120 秒超时(§3.4)
    和异常语义。本层只按规范 §5.2 将领域异常转换为 HTTP 状态码。
    """
    try:
        report = await run_analysis_async(req.ticker)
    except InvalidTickerFormat as e:
        raise HTTPException(400, detail={"code": "INVALID_TICKER_FORMAT", "message": str(e)})
    except TickerNotFound as e:
        raise HTTPException(404, detail={"code": "TICKER_NOT_FOUND", "message": str(e)})
    except AllDataSourcesDown as e:
        raise HTTPException(503, detail={"code": "ALL_DATA_SOURCES_DOWN", "message": str(e)})
    except LLMRateLimited as e:
        # 429 / Token Plan 已用完:瞬时错误,用户可重试。
        raise HTTPException(503, detail={"code": "LLM_RATE_LIMITED", "message": str(e)})
    except CrewExecutionError as e:
        # 非 429 的内部 CrewAI / LLM 失败。
        raise HTTPException(500, detail={"code": "CREW_EXECUTION_ERROR", "message": str(e)})
    except ReportGenerationError as e:
        raise HTTPException(500, detail={"code": "REPORT_GENERATION_ERROR", "message": str(e)})
    except asyncio.TimeoutError:
        raise HTTPException(504, detail={"code": "GATEWAY_TIMEOUT", "message": "流程超过 600 秒预算"})
    return AnalyzeResponse(report_id=report.report_id, report=report)


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version=VERSION,
        data_sources={"yahoo": "ok", "alpha_vantage": "ok", "finnhub": "ok"},
    )

"""共享的分析核心。

本模块是"给定 ticker,产出 InvestmentReport"的单一事实来源。
FastAPI 路由(异步)和 CLI(同步)都委托于此。

导入本模块不会拉入 FastAPI 应用,因此可以安全地从任何位置导入,
没有循环导入的风险。
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable

from alphaquant.exceptions import AllDataSourcesDown
from alphaquant.flows import AnalysisFlow
from alphaquant.models.report import InvestmentReport
from alphaquant.observability import get_logger

log = get_logger("alphaquant.core")

# 进度回调签名:``(step_id, state)``,其中 ``step_id`` 是
# ``AnalysisFlow.kickoff_with_timeout`` 文档中列出的键之一,
# ``state`` 是 ``"running"`` / ``"complete"`` / ``"failed"`` 之一。
# 由 Streamlit 前端用于渲染实时进度指示器。
ProgressCallback = Callable[[str, str], None]


async def run_analysis_async(
    ticker: str,
    progress_callback: ProgressCallback | None = None,
) -> InvestmentReport:
    """运行完整的分析流程。FastAPI 的异步入口。

    使用 ``kickoff_with_timeout``(规范 §3.4:整个 Flow 超时)以避免
    同步 CrewAI 调用阻塞 FastAPI 事件循环。

    ``progress_callback`` 会在 Flow 内部的主要步骤边界被调用;
    调用方(例如 Streamlit 前端)用它来渲染实时进度指示器。
    可选 —— FastAPI 路由不传递。
    """
    log.info("analysis_started", ticker=ticker)
    flow = AnalysisFlow()
    await flow.kickoff_with_timeout(
        {"ticker": ticker}, progress_callback=progress_callback
    )
    if flow.state.report is None:
        log.error("analysis_no_report", ticker=ticker)
        raise AllDataSourcesDown(f"Flow 未为 {ticker} 产出报告")
    log.info(
        "analysis_completed",
        ticker=ticker,
        report_id=flow.state.report.report_id,
        rating=flow.state.report.rating,
    )
    return flow.state.report


def run_analysis(ticker: str) -> InvestmentReport:
    """运行完整的分析流程。CLI 的同步入口。"""
    return asyncio.run(run_analysis_async(ticker))

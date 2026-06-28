"""Analyze 页面:输入 ticker,运行完整分析,渲染报告。"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import ClassVar

import streamlit as st

from alphaquant.exceptions import (
    AllDataSourcesDown,
    CrewExecutionError,
    InvalidTickerFormat,
    LLMRateLimited,
    TickerNotFound,
)
from alphaquant.infrastructure.crew_events import (
    format_event_line,
    get_default_bridge,
)
from alphaquant.interfaces.frontend.components.charts import (
    render_risk_radar,
    render_sentiment_bar,
)
from alphaquant.interfaces.frontend.components.metrics_panel import render_metrics_panel
from alphaquant.interfaces.frontend.components.rating_card import render_rating_card
from alphaquant.infrastructure.persistence import DB
from alphaquant.main import run_analysis_async
from alphaquant.models.report import InvestmentReport
from alphaquant.observability import get_logger

log = get_logger("alphaquant.frontend.analyze")

# Step IDs must match the keys documented on
# ``AnalysisFlow.kickoff_with_timeout`` (and emitted via
# ``Flow._emit_progress``). The labels are the user-facing Chinese strings
# rendered in the Streamlit status container.
STEP_LABELS: dict[str, str] = {
    "validate_ticker": "验证 ticker 格式",
    "run_crew": "运行 8-agent 分析 crew (含 4 个数据工具 + 3 个分析 + report writer)",
    "parse_crew_output": "解析 crew 输出 (company / market / news / financial / writer_output)",
    "compute_analyses": "计算 3 项结构化分析 (competitor / risk / valuation)",
    "assemble_report": "拼装最终投资报告",
    "save": "保存到历史数据库",
}
STEP_ORDER: tuple[str, ...] = tuple(STEP_LABELS.keys())
MAX_LOG_LINES = 80

# Friendly state → glyph mapping. Keeps the render function readable.
_STATE_GLYPHS: ClassVar[dict[str, str]] = {
    "pending": "[ ]",
    "running": "[..]",
    "complete": "[OK]",
    "failed": "[!!]",
}


def _normalize_ticker(raw: str) -> str:
    """Validate and normalize ticker input per the flow's contract.

    Mirrors ``alphaquant.flows.analysis_flow._normalize_ticker``:
    strip whitespace, uppercase, require non-empty and len <= 6.
    """
    if raw is None:
        raise InvalidTickerFormat("")
    t = raw.strip().upper()
    if not t or len(t) > 6:
        raise InvalidTickerFormat(raw)
    return t


class _LogCapture(logging.Handler):
    """Thread-safe logging handler that buffers formatted records.

    CrewAI and LiteLLM use the standard ``logging`` module, so attaching
    this handler to the root logger captures their output alongside any
    other Python ``logging`` calls. ``structlog`` (used by AlphaQuant's own
    modules) does not route through this handler — but the progress
    callback writes its own human-readable lines into the same buffer.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self._lock = threading.Lock()
        self._lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        try:
            msg = self.format(record)
            with self._lock:
                self._lines.append(msg)
                if len(self._lines) > MAX_LOG_LINES * 2:
                    # Keep a rolling window; the page re-renders only the
                    # tail so the head is fair game to drop.
                    self._lines = self._lines[-MAX_LOG_LINES:]
        except Exception:
            # Never let a logging handler crash the producer.
            pass

    def tail(self, n: int = MAX_LOG_LINES) -> list[str]:
        with self._lock:
            return list(self._lines[-n:])

    def append(self, line: str) -> None:
        """Append a line to the buffer under the lock.

        Used by the page itself to inject flow-transition markers
        (``[flow] >>> validate_ticker`` etc.) into the same rolling
        buffer that the logging handler writes into.
        """
        with self._lock:
            self._lines.append(line)
            if len(self._lines) > MAX_LOG_LINES * 2:
                self._lines = self._lines[-MAX_LOG_LINES:]

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()


def _format_log_lines(lines: list[str]) -> str:
    if not lines:
        return "等待 crew 日志..."
    return "\n".join(lines)


def _count_llm_completions(log_lines: list[str]) -> int:
    """Count LLM call completion events in the captured log buffer.

    LiteLLM logs ``Wrapper: Completed Call`` once per successful LLM call.
    CrewAI emits this for both data agents and analysis agents, so the
    count grows as the crew progresses. Used by the live sub-step
    indicator so the user sees forward motion during the long
    ``run_crew`` step instead of a static "..." label.
    """
    count = 0
    for line in log_lines:
        if "Wrapper: Completed Call" in line or "completed LLM call" in line:
            count += 1
    return count


def _render_progress_text(
    step_states: dict[str, str],
    log_lines: list[str],
    *,
    bridge_snapshot=None,
) -> tuple[str, str]:
    """Build the (steps_md, log_md) strings to render in the status block.

    Returns a pair of plain strings so callers can hand them to ``st.markdown``
    / ``st.code`` placeholders. Pure function — no Streamlit calls here.

    ``bridge_snapshot`` (optional) is a ``CrewEventSnapshot`` from the
    ``CrewEventBridge``; when provided, the ``run_crew`` line surfaces the
    currently active agent + tool + task + LLM call count so the user sees
    real-time sub-step progress during the long crew execution.
    """
    step_lines: list[str] = []
    for step_id in STEP_ORDER:
        label = STEP_LABELS[step_id]
        glyph = _STATE_GLYPHS.get(step_states.get(step_id, "pending"), "[ ]")
        if step_states.get(step_id) == "complete":
            step_lines.append(f"- {glyph} **{label}**")
        elif step_states.get(step_id) == "running":
            extra = ""
            if step_id == "run_crew":
                # Sub-step indicator: show live LLM call count + the
                # currently running agent / tool from the CrewEventBridge.
                llm_count = _count_llm_completions(log_lines)
                extras: list[str] = []
                if llm_count:
                    extras.append(f"已完成 {llm_count} 个 LLM 调用")
                if bridge_snapshot is not None:
                    if bridge_snapshot.current_agent:
                        extras.append(f"agent: {bridge_snapshot.current_agent}")
                    if bridge_snapshot.current_tool:
                        extras.append(f"tool: {bridge_snapshot.current_tool}")
                    elif bridge_snapshot.current_task:
                        extras.append(
                            f"task: {_truncate(bridge_snapshot.current_task, 50)}"
                        )
                if extras:
                    extra = "  (" + ", ".join(extras) + ")"
            step_lines.append(f"- {glyph} {label} ...{extra}")
        elif step_states.get(step_id) == "failed":
            step_lines.append(f"- {glyph} {label} (失败)")
        else:
            step_lines.append(f"- {glyph} {label}")
    return ("\n".join(step_lines), _format_log_lines(log_lines))


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _format_event_log(events: list[dict], tail: int = 60) -> str:
    """Format the CrewEventBridge event list as a multi-line log string.

    Shown in a ``st.code`` block so the user sees every agent / tool
    transition the bridge captured during the run. The latest ``tail``
    events are returned in chronological order.
    """
    if not events:
        return ""
    lines = [format_event_line(e) for e in events[-tail:]]
    return "\n".join(lines)


def _merge_event_log(crew_log_md: str, events: list[dict]) -> str:
    """Append the CrewEventBridge event log below the LiteLLM log lines.

    The progress block shows two log sources: the standard ``logging``
    buffer (LiteLLM "Wrapper: Completed Call" etc.) and the bridge's
    per-agent events. Concatenate them so the user sees a single
    chronological-ish stream.
    """
    event_log = _format_event_log(events, tail=40)
    if not event_log:
        return crew_log_md
    if not crew_log_md or crew_log_md == "等待 crew 日志...":
        return event_log
    return f"{crew_log_md}\n\n--- agent events ---\n{event_log}"


db = DB()
db.init()


def _load_report_from_json(report_json: str) -> InvestmentReport | None:
    """Re-hydrate a stored report JSON; returns None on any parse error."""
    try:
        return InvestmentReport.model_validate_json(report_json)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("load_report_failed", error=str(exc))
        return None


# Page-load fallback: if a previous run left a report in the DB, auto-load
# it into session state so a browser refresh doesn't blank the page.
if "current_report" not in st.session_state:
    latest = db.get_latest_report()
    st.session_state["current_report"] = (
        _load_report_from_json(latest.report_json) if latest else None
    )

with st.form(key="analyze_form", clear_on_submit=False):
    ticker_input = st.text_input("股票代码", placeholder="例如 AAPL")
    submitted = st.form_submit_button("分析", type="primary")

if submitted:
    try:
        ticker = _normalize_ticker(ticker_input)
    except InvalidTickerFormat:
        st.error(
            "无效的 ticker 格式。应为类似 'AAPL' 或 'BRK.B' 的形式"
            "(仅字母,最多 6 个字符)。"
        )
        st.stop()

    # 新一次运行会使之前显示的报告失效(用户刚刚提交了新分析)。
    # 清除它以避免加载 UI 停留在过时的数据之上。
    st.session_state["current_report"] = None

    # 由进度回调捕获的可变容器。定义在外层作用域中,以便回调可以
    # 在多次调用之间修改它们。
    step_states: dict[str, str] = {sid: "pending" for sid in STEP_ORDER}
    log_capture = _LogCapture()
    log_capture.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    # 挂钩标准 logging 模块,使 CrewAI / LiteLLM 日志输出流入我们的
    # 缓冲区。在下面的 ``finally`` 块中恢复。
    root_logger = logging.getLogger()
    previous_root_level = root_logger.level
    if root_logger.level > logging.INFO or root_logger.level == logging.NOTSET:
        root_logger.setLevel(logging.INFO)
    root_logger.addHandler(log_capture)

    # 获取共享的 CrewEventBridge,以便轮询线程可以在 run_crew 行展示
    # 当前 agent / tool。Flow 在入口处重置它;我们只需要一个用于快照
    # 的引用。
    crew_bridge = get_default_bridge()

    # 我们还将高层 Flow 转换作为纯文本展示在日志缓冲区中
    # (Flow 本身使用 structlog,不会经过 Python 的 logging 模块
    # —— 所以我们在此镜像关键事件)。
    def _mirror_event(step_id: str, state: str) -> None:
        if state == "running":
            log_capture.append(f"[flow] >>> {step_id} (运行中)")
        elif state == "complete":
            log_capture.append(f"[flow] <<< {step_id} (完成)")
        elif state == "failed":
            log_capture.append(f"[flow] !!! {step_id} (失败)")

    def progress_cb(step_id: str, state: str) -> None:
        if step_id in step_states:
            step_states[step_id] = state
        _mirror_event(step_id, state)
        snap = crew_bridge.snapshot()
        steps_md, log_md = _render_progress_text(
            step_states, log_capture.tail(MAX_LOG_LINES), bridge_snapshot=snap
        )
        progress_ph.markdown(steps_md)
        log_md_with_events = _merge_event_log(log_md, snap.events)
        log_ph.code(log_md_with_events, language="log")

    status = st.status("运行分析中...", expanded=True)
    # 在 status 容器内创建占位符,以便步骤和日志缓冲区在可折叠块内渲染。
    # progress_cb 持有引用并从主异步函数更新它们。
    with status:
        progress_ph = st.empty()
        log_ph = st.empty()
        steps_md, log_md = _render_progress_text(step_states, [])
        progress_ph.markdown(steps_md)
        log_ph.code(log_md, language="log")

    # 轮询线程注意事项:Streamlit 的 ``st.empty()`` 占位符只能从主脚本线程
    # (ScriptRunContext) 更新。在这里派生 ``threading.Thread`` 会产生
    # "missing ScriptRunContext" 警告,且更新会被静默丢弃 —— 已在
    # 此修复之前的日志噪音中得到验证。Flow 步骤转换(validate_ticker /
    # run_crew / parse_crew_output / compute_analyses / assemble_report /
    # save)从主线程上的 ``progress_cb`` 更新占位符,这是有效的。
    # 对于 crew 内部 agent/tool 细节,我们在 ``asyncio.run`` 返回之后,
    # 报告下方的可折叠部分中渲染完整的 ``CrewEventBridge`` 事件日志。
    try:
        report = asyncio.run(
            run_analysis_async(ticker, progress_callback=progress_cb)
        )
    except InvalidTickerFormat as exc:
        st.error(f"无效的 ticker 格式:{exc.ticker!r}。")
    except TickerNotFound as exc:
        st.error(
            f"无法通过任何数据源解析 ticker {exc.ticker!r}。"
            "请仔细检查代码并重试。"
        )
    except AllDataSourcesDown:
        st.error(
            "目前所有数据源都不可用。"
            "请等待几分钟后重试。"
        )
    except LLMRateLimited as exc:
        # 429 / Token Plan 已用完:准确告诉用户问题所在以及如何恢复,
        # 而不是让 UI 在"运行"状态停留数分钟。
        st.error(
            f"LLM 配额已用完（HTTP 429 / Token Plan）。请等待几分钟后重试，"
            f"或升级 Token Plan 套餐。\n\n详情: {exc}"
        )
        status.update(label="LLM 配额已用完", state="error")
    except CrewExecutionError as exc:
        st.error(
            f"分析 crew 执行失败: {exc}\n\n"
            f"请刷新页面重试，或查看服务器日志。"
        )
        status.update(label="分析失败", state="error")
    except Exception:
        log.exception("analyze_unexpected_error", ticker=ticker)
        st.error(
            "运行分析时发生意外错误。"
            "请重试,如果问题仍然存在,请查看服务器日志。"
        )
        status.update(label="分析失败", state="error")
    else:
        # Persist + mark the save step.
        db.insert_report(ticker, report)
        st.session_state["current_report"] = report
        step_states["save"] = "complete"
        snap = crew_bridge.snapshot()
        steps_md, log_md = _render_progress_text(
            step_states, log_capture.tail(MAX_LOG_LINES), bridge_snapshot=snap
        )
        with status:
            progress_ph.markdown(steps_md)
            log_md_with_events = _merge_event_log(log_md, snap.events)
            log_ph.code(log_md_with_events, language="log")
        status.update(label="分析完成", state="complete")

        render_rating_card(report)
        render_metrics_panel(report)

        charts_left, charts_right = st.columns(2)
        with charts_left:
            st.plotly_chart(
                render_risk_radar(report.risk),
                key="risk_radar_after_analysis",
                width="stretch",
            )
        with charts_right:
            st.plotly_chart(
                render_sentiment_bar(report.news),
                key="sentiment_bar_after_analysis",
                width="stretch",
            )

        st.subheader("竞争对手")
        competitor_rows = [
            {
                "股票代码": c.ticker,
                "名称": c.name,
                "市值": c.market_cap,
                "TTM 营收": float(c.revenue_ttm),
                "P/E": c.pe_ratio,
            }
            for c in report.competitors.competitors
        ]
        st.table(competitor_rows)

        st.subheader("完整报告")
        st.markdown(report.markdown)

        # 执行后 crew 详细日志。在运行期间由 CrewEventBridge 捕获;
        # 在此处渲染,因为 bridge 处理器从 CrewAI 工作线程触发,无法在
        # ``asyncio.run`` 阻塞主线程时直接更新 Streamlit UI。
        bridge_events = crew_bridge.snapshot().events
        if bridge_events:
            with st.expander(
                f"Crew 执行日志 ({len(bridge_events)} 个事件)", expanded=False
            ):
                st.caption(
                    "每个 agent / tool 的开始 / 完成事件，按时间顺序展示。"
                )
                st.code(
                    _format_event_log(bridge_events, tail=200),
                    language="log",
                )
    finally:
        root_logger.removeHandler(log_capture)
        root_logger.setLevel(previous_root_level)


# ---- 最近分析面板(每次页面加载时显示) ----
st.divider()
with st.expander("最近分析", expanded=False):
    recent = db.get_latest_reports(limit=5)
    if not recent:
        st.info("还没有先前的分析。请在上方运行一次。")
    else:
        header_cols = st.columns([1, 1, 2, 1, 1])
        header_cols[0].markdown("**股票代码**")
        header_cols[1].markdown("**评级**")
        header_cols[2].markdown("**生成时间**")
        header_cols[3].markdown("**置信度**")
        header_cols[4].markdown("**操作**")
        for record in recent:
            row_cols = st.columns([1, 1, 2, 1, 1])
            row_cols[0].write(record.ticker)
            row_cols[1].write(record.rating)
            row_cols[2].write(record.generated_at.strftime("%Y-%m-%d %H:%M:%S"))
            row_cols[3].write("—" if record.confidence is None else f"{record.confidence}")
            if row_cols[4].button("加载", key=f"load_{record.id}"):
                loaded = _load_report_from_json(record.report_json)
                if loaded is not None:
                    st.session_state["current_report"] = loaded
                    st.rerun()
                else:
                    st.error(f"加载报告 #{record.id} 失败。")


# ---- 渲染当前报告(来自 session 或页面加载回退) ----
current_report = st.session_state.get("current_report")
if current_report is not None:
    st.divider()
    st.subheader(f"📊 {current_report.ticker} — 投资研究报告")
    st.caption(
        f"生成时间 {current_report.generated_at.isoformat(timespec='seconds')}"
    )
    render_rating_card(current_report)
    render_metrics_panel(current_report)

    charts_left, charts_right = st.columns(2)
    with charts_left:
        st.plotly_chart(
            render_risk_radar(current_report.risk),
            key="risk_radar_session",
            width="stretch",
        )
    with charts_right:
        st.plotly_chart(
            render_sentiment_bar(current_report.news),
            key="sentiment_bar_session",
            width="stretch",
        )

    st.subheader("竞争对手")
    competitor_rows = [
        {
            "股票代码": c.ticker,
            "名称": c.name,
            "市值": c.market_cap,
            "TTM 营收": float(c.revenue_ttm),
            "P/E": c.pe_ratio,
        }
        for c in current_report.competitors.competitors
    ]
    st.table(competitor_rows)

    st.subheader("完整报告")
    st.markdown(current_report.markdown)

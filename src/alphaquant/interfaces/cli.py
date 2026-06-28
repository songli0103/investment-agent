"""CLI 入口:`python -m alphaquant AAPL`。

常用选项:
  --debug           打印 CrewAI task/tool 事件到 stderr,实时观察 crew 在做什么
  --format markdown 输出 markdown 报告而不是 JSON
  --pretty          美化 JSON
  --output FILE     写入文件而不是 stdout
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from alphaquant.exceptions import (
    AllDataSourcesDown,
    InvalidTickerFormat,
    TickerNotFound,
)
from alphaquant.core import run_analysis


# 防止 --debug 被多次触发时重复订阅。``crewai_event_bus.on(...)`` 是
# 装饰器工厂,每次调用都注册一个新 handler —— 没有幂等保护。
_debug_handlers_installed = False


def _install_debug_handlers() -> None:
    """订阅 CrewAI 事件总线,把 task/tool 事件流式打印到 stderr。

    只在 ``--debug`` 时调用,做两件事:
    - 把每个 task_started / task_completed / tool_started / tool_finished
      实时打印,用户能看到 "现在是哪个 agent 调哪个 tool" 的真实时序
    - 用 ``flush=True`` 立即输出,不会被 Python 的行缓冲吞掉

    跟 ``infrastructure/crew_events.py`` 的区别:
    - ``crew_events.py`` 是线程安全的快照服务,给 Streamlit 轮询
    - 这里只打印,不需要快照、不需要跨线程同步
    """
    global _debug_handlers_installed
    if _debug_handlers_installed:
        return

    from crewai.events import (
        TaskCompletedEvent,
        TaskFailedEvent,
        TaskStartedEvent,
        ToolUsageFinishedEvent,
        ToolUsageStartedEvent,
        crewai_event_bus,
    )

    def _ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    def on_task_start(source, event):
        context = getattr(event, "context", None) or ""
        task = getattr(event, "task", None)
        desc = getattr(task, "description", None) if task else None
        label = (context or desc or "task").strip()[:80] or "task"
        print(f"[{_ts()}] ▶ {label}", file=sys.stderr, flush=True)

    def on_task_done(source, event):
        print(f"[{_ts()}] ✓ task 完成", file=sys.stderr, flush=True)

    def on_task_failed(source, event):
        print(f"[{_ts()}] ✗ task 失败", file=sys.stderr, flush=True)

    def on_tool_start(source, event):
        tool_args = getattr(event, "tool_args", None) or {}
        if isinstance(tool_args, dict):
            args_repr = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
        else:
            args_repr = str(tool_args)
        print(
            f"[{_ts()}]   ↪ {event.agent_role} → {event.tool_name}({args_repr})",
            file=sys.stderr,
            flush=True,
        )

    def on_tool_done(source, event):
        print(
            f"[{_ts()}]   ✓ {event.tool_name} 完成",
            file=sys.stderr,
            flush=True,
        )

    crewai_event_bus.on(TaskStartedEvent)(on_task_start)
    crewai_event_bus.on(TaskCompletedEvent)(on_task_done)
    crewai_event_bus.on(TaskFailedEvent)(on_task_failed)
    crewai_event_bus.on(ToolUsageStartedEvent)(on_tool_start)
    crewai_event_bus.on(ToolUsageFinishedEvent)(on_tool_done)
    _debug_handlers_installed = True


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="alphaquant",
        description="AI 投资研究分析师",
    )
    parser.add_argument("ticker", help="美股 ticker(例如 AAPL)")
    parser.add_argument(
        "--format", choices=["json", "markdown"], default="json", help="输出格式"
    )
    parser.add_argument("--pretty", action="store_true", help="美化打印 JSON")
    parser.add_argument("--output", type=str, help="写入文件而不是 stdout")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="实时打印 CrewAI task/tool 事件到 stderr,用于观察 crew 执行",
    )
    args = parser.parse_args()

    if args.debug:
        _install_debug_handlers()
        print(
            f"[debug] CrewAI 事件订阅已安装;开始分析 {args.ticker}",
            file=sys.stderr,
            flush=True,
        )

    try:
        report = run_analysis(args.ticker)
    except InvalidTickerFormat as e:
        print(json.dumps({"code": "INVALID_TICKER_FORMAT", "message": str(e)}), file=sys.stderr)
        return 2
    except TickerNotFound as e:
        print(json.dumps({"code": "TICKER_NOT_FOUND", "message": str(e)}), file=sys.stderr)
        return 3
    except AllDataSourcesDown as e:
        print(json.dumps({"code": "ALL_DATA_SOURCES_DOWN", "message": str(e)}), file=sys.stderr)
        return 4
    except Exception as e:
        print(json.dumps({"code": "INTERNAL_ERROR", "message": str(e)}), file=sys.stderr)
        return 1

    if args.debug:
        print(
            f"[debug] 分析完成,生成报告 (rating={report.rating}, confidence={report.confidence})",
            file=sys.stderr,
            flush=True,
        )

    if args.format == "json":
        output = report.model_dump_json(indent=2 if args.pretty else None)
    else:
        output = report.markdown

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"报告已写入 {args.output}", file=sys.stderr)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

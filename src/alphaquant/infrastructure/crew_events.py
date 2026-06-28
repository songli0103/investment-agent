"""从 CrewAI 事件总线到 Streamlit UI 的线程安全桥接。

CrewAI 0.203.2 在 crew 执行期间会触发一系列丰富的事件
(TaskStartedEvent、TaskCompletedEvent、ToolUsageStartedEvent、
ToolUsageFinishedEvent、……)。``Analyze`` 页面需要实时展示*当前正在
运行的代理*和*工具调用*,以便用户在漫长的 ``run_crew`` 步骤中看到
前进的进度。

本模块(进程范围内)一次性订阅这些事件,并将它们的最新值存储在
线程安全的 ``CrewEventBridge`` 实例中。UI 的轮询线程每秒读取一次
``bridge.snapshot()`` 并渲染结果。

设计说明:
- 模块级单个 ``_BRIDGE`` 是默认实例。Flow 也可以实例化自己的桥接
  并显式传递它(用于测试隔离或并发运行多个 crew 时)。
- 桥接器只调用 ``.on(...)`` 一次注册处理器;后续 ``install()`` 调用
  是 no-op,因此跨运行复用桥接是安全的。
- 所有公共状态在锁保护下读取;``snapshot()`` 返回当前值的浅拷贝,
  以便 UI 永远不会看到撕裂写入。
- 滚动事件日志以 ``_MAX_EVENTS`` 为上限,以限制内存。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from crewai.events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
    crewai_event_bus,
)

# 每个桥接的事件日志的滚动上限。UI 仅显示尾部(最后约 80 行),
# 因此任何更早的内容都可以丢弃。
_MAX_EVENTS = 200


@dataclass
class CrewEventSnapshot:
    """供 UI 读取的桥接状态只读视图。"""

    current_task: str | None = None
    current_agent: str | None = None
    current_tool: str | None = None
    current_tool_args: str | None = None
    completed_tasks: int = 0
    completed_tools: int = 0
    # 滚动事件日志,最旧优先(桥接器缓冲区的副本)。
    events: list[dict[str, Any]] = field(default_factory=list)


class CrewEventBridge:
    """订阅 CrewAI 事件总线并公开线程安全的快照。

    桥接器从 CrewAI 的工作线程修改内部状态,但从 Streamlit 轮询线程
    调用 ``snapshot()`` 是安全的。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_task: str | None = None
        self._current_agent: str | None = None
        self._current_tool: str | None = None
        self._current_tool_args: str | None = None
        self._completed_tasks = 0
        self._completed_tools = 0
        self._events: list[dict[str, Any]] = []
        self._installed = False

    # -- 订阅 ------------------------------------------------------------

    def install(self) -> None:
        """订阅 CrewAI 事件总线。幂等。

        处理器作为绑定方法注册到此实例。由于 CrewAI 的
        ``crewai_event_bus.on(...)`` 返回装饰器,调用 ``install()`` 两次
        会重复注册,导致重复事件。``_installed`` 标志可以防止这种情况。
        """
        if self._installed:
            return
        crewai_event_bus.on(TaskStartedEvent)(self._on_task_started)
        crewai_event_bus.on(TaskCompletedEvent)(self._on_task_completed)
        crewai_event_bus.on(TaskFailedEvent)(self._on_task_failed)
        crewai_event_bus.on(ToolUsageStartedEvent)(self._on_tool_started)
        crewai_event_bus.on(ToolUsageFinishedEvent)(self._on_tool_finished)
        self._installed = True

    def reset(self) -> None:
        """清除所有每次运行的状态。在每次 ``crew.kickoff()`` 之前调用。

        不会取消订阅 —— 处理器在多次运行之间保持连接状态,因此桥接
        可以被复用。
        """
        with self._lock:
            self._current_task = None
            self._current_agent = None
            self._current_tool = None
            self._current_tool_args = None
            self._completed_tasks = 0
            self._completed_tools = 0
            self._events = []

    # -- 快照 ------------------------------------------------------------

    def snapshot(self) -> CrewEventSnapshot:
        """返回当前状态的线程安全副本,供 UI 使用。"""
        with self._lock:
            return CrewEventSnapshot(
                current_task=self._current_task,
                current_agent=self._current_agent,
                current_tool=self._current_tool,
                current_tool_args=self._current_tool_args,
                completed_tasks=self._completed_tasks,
                completed_tools=self._completed_tools,
                events=list(self._events),
            )

    # -- 事件处理器 -------------------------------------------------------

    def _append(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)
            if len(self._events) > _MAX_EVENTS:
                # 删除头部;UI 仅显示尾部。
                del self._events[: len(self._events) - _MAX_EVENTS]

    def _on_task_started(self, source: Any, event: TaskStartedEvent) -> None:
        # ``event.context`` 是代理针对任务的自然语言意图
        # (例如 "Validate ticker 'AAPL' and return canonical metadata.")。
        # ``event.task`` 是 Task 对象;如果 context 为空,则提取其
        # ``description`` 作为回退。
        context = getattr(event, "context", None) or ""
        task = getattr(event, "task", None)
        desc = getattr(task, "description", None) if task else None
        label = (context or desc or "task").strip() or "task"
        # 截断长描述,以保持 UI 行宽度合理。
        if len(label) > 80:
            label = label[:77] + "..."
        with self._lock:
            self._current_task = label
            self._current_agent = None
            self._current_tool = None
            self._current_tool_args = None
        self._append({
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "type": "task_started",
            "label": label,
        })

    def _on_task_completed(self, source: Any, event: TaskCompletedEvent) -> None:
        with self._lock:
            self._completed_tasks += 1
            self._current_task = None
            self._current_agent = None
            self._current_tool = None
            self._current_tool_args = None
        self._append({
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "type": "task_completed",
        })

    def _on_task_failed(self, source: Any, event: TaskFailedEvent) -> None:
        with self._lock:
            self._current_task = None
            self._current_agent = None
            self._current_tool = None
            self._current_tool_args = None
        self._append({
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "type": "task_failed",
        })

    def _on_tool_started(self, source: Any, event: ToolUsageStartedEvent) -> None:
        agent_role = getattr(event, "agent_role", None) or "agent"
        tool_name = getattr(event, "tool_name", None) or "tool"
        tool_args = getattr(event, "tool_args", None)
        args_repr: str | None = None
        if isinstance(tool_args, dict):
            args_repr = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
        elif isinstance(tool_args, str):
            args_repr = tool_args
        with self._lock:
            self._current_agent = agent_role
            self._current_tool = tool_name
            self._current_tool_args = args_repr
        self._append({
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "type": "tool_started",
            "agent": agent_role,
            "tool": tool_name,
            "args": args_repr,
        })

    def _on_tool_finished(self, source: Any, event: ToolUsageFinishedEvent) -> None:
        agent_role = getattr(event, "agent_role", None) or "agent"
        tool_name = getattr(event, "tool_name", None) or "tool"
        with self._lock:
            self._completed_tools += 1
            # 这里不清除 current_agent —— 代理可能仍在运行不是工具调用的
            # 后续步骤。下一个 task_started 或 tool_started 事件会覆盖它。
            if self._current_tool == tool_name:
                self._current_tool = None
                self._current_tool_args = None
        self._append({
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "type": "tool_finished",
            "agent": agent_role,
            "tool": tool_name,
        })


# 模块级默认桥接器。Flow 使用它;测试可以构造自己的 ``CrewEventBridge``
# 进行隔离。
_default_bridge: CrewEventBridge | None = None
_default_lock = threading.Lock()


def get_default_bridge() -> CrewEventBridge:
    """返回进程范围的默认 ``CrewEventBridge``(惰性安装)。

    第一次调用安装处理器;后续调用返回同一实例。这将事件总线的副作用
    限制在一个地方,即使有多个 Flow 连续运行也是如此。
    """
    global _default_bridge
    with _default_lock:
        if _default_bridge is None:
            _default_bridge = CrewEventBridge()
        _default_bridge.install()
        return _default_bridge


def format_event_line(event: dict[str, Any]) -> str:
    """将桥接事件格式化为单行人类可读的日志。

    由 Streamlit 页面使用,在 ``run_crew`` 状态块内渲染滚动事件日志。
    纯函数 —— 可从 UI 线程安全调用。
    """
    ts = event.get("ts", "")
    etype = event.get("type", "")
    if etype == "task_started":
        return f"[{ts}] ▶ {event.get('label', 'task')}"
    if etype == "task_completed":
        return f"[{ts}] ✓ task 完成"
    if etype == "task_failed":
        return f"[{ts}] ✗ task 失败"
    if etype == "tool_started":
        args = event.get("args") or ""
        suffix = f" ({args})" if args else ""
        return f"[{ts}]   ↪ {event.get('agent', 'agent')} → {event.get('tool', 'tool')}{suffix}"
    if etype == "tool_finished":
        return f"[{ts}]   ✓ {event.get('agent', 'agent')} → {event.get('tool', 'tool')}"
    return f"[{ts}] {event}"


__all__ = [
    "CrewEventBridge",
    "CrewEventSnapshot",
    "get_default_bridge",
    "format_event_line",
]

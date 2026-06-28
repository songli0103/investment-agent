"""CrewEventBridge(实时 CrewAI 事件捕获)的测试。"""
from __future__ import annotations

import threading
from datetime import datetime

import pytest

from crewai.events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.tasks.task_output import TaskOutput
from alphaquant.infrastructure.crew_events import (
    CrewEventBridge,
    format_event_line,
)


# ---------------------------------------------------------------------------
# Handlers run in isolation
# ---------------------------------------------------------------------------


def _make_task_started_event(context: str = "Test task") -> TaskStartedEvent:
    return TaskStartedEvent(context=context)


def _make_task_completed_event() -> TaskCompletedEvent:
    """Construct a minimal TaskOutput so the event validates."""
    return TaskCompletedEvent(
        output=TaskOutput(
            description="test",
            raw="ok",
            agent="agent",
        )
    )


def _make_task_failed_event() -> TaskFailedEvent:
    return TaskFailedEvent(error="boom")


def _make_tool_started_event(
    agent_role: str = "Agent A",
    tool_name: str = "ToolX",
    tool_args: dict | str | None = None,
) -> ToolUsageStartedEvent:
    return ToolUsageStartedEvent(
        agent_role=agent_role,
        tool_name=tool_name,
        tool_args=tool_args if tool_args is not None else {},
    )


def _make_tool_finished_event(
    agent_role: str = "Agent A",
    tool_name: str = "ToolX",
) -> ToolUsageFinishedEvent:
    return ToolUsageFinishedEvent(
        agent_role=agent_role,
        tool_name=tool_name,
        started_at=datetime(2026, 6, 27),
        finished_at=datetime(2026, 6, 27),
        output="ok",
    )


# ---------------------------------------------------------------------------
# Snapshot semantics
# ---------------------------------------------------------------------------


def test_initial_snapshot_is_empty():
    """A fresh bridge has no current state and no events."""
    bridge = CrewEventBridge()
    snap = bridge.snapshot()
    assert snap.current_task is None
    assert snap.current_agent is None
    assert snap.current_tool is None
    assert snap.completed_tasks == 0
    assert snap.completed_tools == 0
    assert snap.events == []


def test_task_started_sets_current_task():
    bridge = CrewEventBridge()
    bridge._on_task_started(None, _make_task_started_event("Validate ticker AAPL"))
    snap = bridge.snapshot()
    assert snap.current_task == "Validate ticker AAPL"
    assert snap.completed_tasks == 0
    assert len(snap.events) == 1
    assert snap.events[0]["type"] == "task_started"
    assert snap.events[0]["label"] == "Validate ticker AAPL"


def test_task_completed_clears_current_task_and_increments_count():
    bridge = CrewEventBridge()
    bridge._on_task_started(None, _make_task_started_event("Do thing"))
    bridge._on_task_completed(None, _make_task_completed_event())
    snap = bridge.snapshot()
    assert snap.current_task is None
    assert snap.completed_tasks == 1
    assert any(e["type"] == "task_completed" for e in snap.events)


def test_task_failed_clears_current_task():
    bridge = CrewEventBridge()
    bridge._on_task_started(None, _make_task_started_event("Do thing"))
    bridge._on_task_failed(None, _make_task_failed_event())
    snap = bridge.snapshot()
    assert snap.current_task is None
    assert any(e["type"] == "task_failed" for e in snap.events)


def test_long_task_label_is_truncated():
    bridge = CrewEventBridge()
    long_text = "x" * 200
    bridge._on_task_started(None, _make_task_started_event(long_text))
    snap = bridge.snapshot()
    # The bridge truncates labels > 80 chars; ellipsis form.
    assert snap.current_task is not None
    assert len(snap.current_task) <= 80
    assert snap.current_task.endswith("...")


def test_tool_started_sets_agent_and_tool():
    bridge = CrewEventBridge()
    bridge._on_tool_started(
        None,
        _make_tool_started_event(
            agent_role="Company Resolver",
            tool_name="CompanyLookupTool",
            tool_args={"ticker": "AAPL"},
        ),
    )
    snap = bridge.snapshot()
    assert snap.current_agent == "Company Resolver"
    assert snap.current_tool == "CompanyLookupTool"
    assert snap.current_tool_args == "ticker='AAPL'"


def test_tool_started_with_string_args():
    bridge = CrewEventBridge()
    bridge._on_tool_started(
        None,
        _make_tool_started_event(tool_args="ticker=AAPL"),
    )
    snap = bridge.snapshot()
    assert snap.current_tool_args == "ticker=AAPL"


def test_tool_finished_clears_current_tool_but_keeps_agent():
    """A tool finishing doesn't clear current_agent — the agent may still
    be running its next reasoning step (which isn't a tool call)."""
    bridge = CrewEventBridge()
    bridge._on_tool_started(
        None, _make_tool_started_event(agent_role="A", tool_name="T")
    )
    bridge._on_tool_finished(
        None, _make_tool_finished_event(agent_role="A", tool_name="T")
    )
    snap = bridge.snapshot()
    assert snap.current_agent == "A"
    assert snap.current_tool is None
    assert snap.completed_tools == 1


def test_tool_finished_with_different_tool_keeps_current_tool():
    """Defensive: a finish event for an older tool shouldn't clear the
    newer current_tool value."""
    bridge = CrewEventBridge()
    bridge._on_tool_started(None, _make_tool_started_event(tool_name="T1"))
    bridge._on_tool_started(None, _make_tool_started_event(tool_name="T2"))
    bridge._on_tool_finished(
        None, _make_tool_finished_event(tool_name="T1")  # stale finish
    )
    snap = bridge.snapshot()
    assert snap.current_tool == "T2"


# ---------------------------------------------------------------------------
# Reset and rolling buffer
# ---------------------------------------------------------------------------


def test_reset_clears_all_state():
    bridge = CrewEventBridge()
    bridge._on_task_started(None, _make_task_started_event("T"))
    bridge._on_tool_started(
        None, _make_tool_started_event(agent_role="A", tool_name="X")
    )
    bridge._on_task_completed(None, _make_task_completed_event())
    bridge.reset()
    snap = bridge.snapshot()
    assert snap.current_task is None
    assert snap.current_agent is None
    assert snap.current_tool is None
    assert snap.completed_tasks == 0
    assert snap.completed_tools == 0
    assert snap.events == []


def test_events_buffer_caps_at_max():
    """The rolling event log caps at 200 entries to bound memory."""
    bridge = CrewEventBridge()
    for i in range(250):
        bridge._on_task_started(None, _make_task_started_event(f"Task {i}"))
    snap = bridge.snapshot()
    # Oldest 50 events dropped, 200 most recent kept.
    assert len(snap.events) == 200
    # The oldest surviving event is Task 50 (0..49 dropped).
    assert snap.events[0]["label"] == "Task 50"
    assert snap.events[-1]["label"] == "Task 249"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_snapshot_is_safe_under_concurrent_writes():
    """snapshot() must always return a consistent view, even if writers
    are running concurrently from multiple threads."""
    bridge = CrewEventBridge()
    stop = threading.Event()

    def writer() -> None:
        i = 0
        while not stop.is_set():
            bridge._on_task_started(None, _make_task_started_event(f"T{i}"))
            bridge._on_task_completed(None, _make_task_completed_event())
            i += 1

    threads = [threading.Thread(target=writer, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    # Hammer snapshot() while writers run.
    for _ in range(200):
        snap = bridge.snapshot()
        # Invariant: completed_tasks and events of type task_completed
        # should be consistent (no torn writes visible).
        completed = sum(1 for e in snap.events if e["type"] == "task_completed")
        # completed_tasks is incremented under the same lock that
        # appends events, so they should match.
        assert snap.completed_tasks == completed, (
            f"completed_tasks={snap.completed_tasks} but {completed} task_completed events"
        )
    stop.set()
    for t in threads:
        t.join(timeout=2.0)


# ---------------------------------------------------------------------------
# format_event_line
# ---------------------------------------------------------------------------


def test_format_event_line_task_started():
    line = format_event_line({
        "ts": "2026-06-27T12:00:00Z",
        "type": "task_started",
        "label": "Validate AAPL",
    })
    assert "▶" in line
    assert "Validate AAPL" in line


def test_format_event_line_tool_started_with_args():
    line = format_event_line({
        "ts": "2026-06-27T12:00:00Z",
        "type": "tool_started",
        "agent": "Resolver",
        "tool": "Lookup",
        "args": "ticker='AAPL'",
    })
    assert "Resolver" in line
    assert "Lookup" in line
    assert "ticker='AAPL'" in line


def test_format_event_line_tool_started_without_args():
    line = format_event_line({
        "ts": "2026-06-27T12:00:00Z",
        "type": "tool_started",
        "agent": "A",
        "tool": "T",
        "args": None,
    })
    assert "A" in line
    assert "T" in line
    # No trailing parens when args is None.
    assert "()" not in line


def test_format_event_line_task_completed():
    line = format_event_line({
        "ts": "2026-06-27T12:00:00Z",
        "type": "task_completed",
    })
    assert "✓" in line
    assert "task 完成" in line


def test_format_event_line_unknown_type_falls_through():
    """Unknown event types render as a raw repr, not a crash."""
    line = format_event_line({"ts": "t", "type": "weird", "foo": "bar"})
    assert "t" in line

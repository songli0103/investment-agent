"""Shared test fixtures and helpers for the AlphaQuant test suite."""
from __future__ import annotations

from typing import Any

from crewai.llm import LLM as _CrewLLM


class _FakeLLM(_CrewLLM):
    """Stub LLM that records calls and returns deterministic text. Used in unit tests."""

    def __init__(self) -> None:
        super().__init__(model="fake/model", api_key="fake")
        self.calls: list[dict[str, Any]] = []

    def call(self, messages, *args, **kwargs):  # type: ignore[override]
        from pydantic import BaseModel
        self.calls.append({"messages": messages, "kwargs": kwargs})
        # If a response_format (Pydantic schema) is requested, return a fake JSON string
        # that satisfies the schema. This avoids hitting the network.
        response_format = kwargs.get("response_format")
        if response_format is not None and isinstance(response_format, type) and issubclass(response_format, BaseModel):
            try:
                instance = response_format.model_construct()
                return instance.model_dump_json()
            except Exception:
                return "{}"
        return "fake llm response"

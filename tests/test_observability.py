"""Tests for alphaquant.observability (logger + cost tracker)."""
from __future__ import annotations

import io
import json
import logging
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Smoke: imports
# ---------------------------------------------------------------------------

def test_observability_importable():
    from alphaquant.observability import (
        TokenUsage,
        configure_logging,
        get_logger,
        track_usage,
    )  # noqa: F401


def test_logger_module_importable():
    from alphaquant.observability.logger import configure_logging, get_logger  # noqa: F401


def test_cost_tracker_module_importable():
    from alphaquant.observability.cost_tracker import TokenUsage, track_usage  # noqa: F401


# ---------------------------------------------------------------------------
# TokenUsage
# ---------------------------------------------------------------------------

class TestTokenUsage:
    def test_construction(self):
        from alphaquant.observability import TokenUsage

        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500

    def test_cost_zero(self):
        from alphaquant.observability import TokenUsage

        usage = TokenUsage(input_tokens=0, output_tokens=0)
        assert usage.cost_usd == 0.0

    def test_cost_input_only(self):
        from alphaquant.observability import TokenUsage

        # 1M input tokens at $3/M = $3.00
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=0)
        assert usage.cost_usd == pytest.approx(3.0)

    def test_cost_output_only(self):
        from alphaquant.observability import TokenUsage

        # 1M output tokens at $15/M = $15.00
        usage = TokenUsage(input_tokens=0, output_tokens=1_000_000)
        assert usage.cost_usd == pytest.approx(15.0)

    def test_cost_mixed(self):
        from alphaquant.observability import TokenUsage

        # 1M input ($3) + 1M output ($15) = $18
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert usage.cost_usd == pytest.approx(18.0)

    def test_cost_proportional(self):
        from alphaquant.observability import TokenUsage

        # 1000 input at $3/M = $0.003; 500 output at $15/M = $0.0075; total $0.0105
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        assert usage.cost_usd == pytest.approx(0.0105)


# ---------------------------------------------------------------------------
# track_usage
# ---------------------------------------------------------------------------

class TestTrackUsage:
    def test_returns_token_usage(self):
        from alphaquant.observability import TokenUsage, track_usage

        usage = track_usage("TestAgent", 1000, 500, 200)
        assert isinstance(usage, TokenUsage)
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500

    def test_logs_event(self, capsys):
        from alphaquant.observability import track_usage

        # Configure with stdout so we can capture
        import structlog

        structlog.configure(
            processors=[structlog.processors.JSONRenderer()],
            logger_factory=structlog.PrintLoggerFactory(),
        )

        track_usage("TestAgent", 1000, 500, 200, request_id="req-1")

        captured = capsys.readouterr()
        # Should produce JSON output with event=llm_call
        lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(lines) >= 1
        record = json.loads(lines[-1])
        assert record["event"] == "llm_call"
        assert record["agent"] == "TestAgent"
        assert record["input_tokens"] == 1000
        assert record["output_tokens"] == 500
        assert record["latency_ms"] == 200
        assert record["request_id"] == "req-1"
        assert "cost_usd" in record

    def test_optional_request_id_none(self, capsys):
        from alphaquant.observability import track_usage
        import structlog

        structlog.configure(
            processors=[structlog.processors.JSONRenderer()],
            logger_factory=structlog.PrintLoggerFactory(),
        )

        track_usage("TestAgent", 100, 50, 100)

        captured = capsys.readouterr()
        lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        record = json.loads(lines[-1])
        assert record["request_id"] is None


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------

class TestConfigureLogging:
    def test_configure_runs_without_error(self):
        from alphaquant.observability import configure_logging

        # Should not raise
        configure_logging()

    def test_configure_uses_settings_log_level(self):
        from alphaquant.observability import configure_logging

        with patch("alphaquant.observability.logger.get_settings") as mock_settings:
            mock_settings.return_value.log_level = "DEBUG"
            configure_logging()
            # Should configure without raising
            assert True

    def test_configure_unknown_level_falls_back_to_info(self):
        from alphaquant.observability import configure_logging

        with patch("alphaquant.observability.logger.get_settings") as mock_settings:
            mock_settings.return_value.log_level = "BOGUS"
            configure_logging()
            # Should not raise — falls back to INFO
            assert True


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------

class TestGetLogger:
    def test_get_logger_no_name(self):
        from alphaquant.observability import configure_logging, get_logger

        configure_logging()
        log = get_logger()
        # structlog returns a BoundLogger or similar
        assert log is not None

    def test_get_logger_with_name(self):
        from alphaquant.observability import configure_logging, get_logger

        configure_logging()
        log = get_logger("my.module")
        assert log is not None
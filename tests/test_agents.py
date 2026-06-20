"""Tests for alphaquant.agents CrewAI agent builders."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from alphaquant.tools.dcf_tool import DCFTool


# ---------------------------------------------------------------------------
# Smoke: imports
# ---------------------------------------------------------------------------

def test_valuation_analyst_importable():
    from alphaquant.agents.valuation_analyst import build_valuation_analyst_agent  # noqa: F401


def test_report_writer_importable():
    from alphaquant.agents.report_writer import build_report_writer_agent  # noqa: F401


# ---------------------------------------------------------------------------
# ValuationAnalyst
# ---------------------------------------------------------------------------

class TestValuationAnalyst:
    def test_role(self):
        from alphaquant.agents.valuation_analyst import build_valuation_analyst_agent

        with patch("alphaquant.agents.valuation_analyst.get_llm") as mock_llm:
            agent = build_valuation_analyst_agent()

        assert agent.role == "Sell-side Valuation Modeler"

    def test_has_goal_and_backstory(self):
        from alphaquant.agents.valuation_analyst import build_valuation_analyst_agent

        with patch("alphaquant.agents.valuation_analyst.get_llm") as mock_llm:
            agent = build_valuation_analyst_agent()

        assert isinstance(agent.goal, str) and len(agent.goal) > 0
        assert isinstance(agent.backstory, str) and len(agent.backstory) > 0
        # Mentions DCF and the range-not-point discipline
        assert "DCF" in agent.goal
        assert "range" in agent.backstory.lower() or "range" in agent.goal.lower()

    def test_uses_dcf_tool(self):
        from alphaquant.agents.valuation_analyst import build_valuation_analyst_agent

        with patch("alphaquant.agents.valuation_analyst.get_llm") as mock_llm:
            agent = build_valuation_analyst_agent()

        tool_names = [t.name for t in agent.tools]
        assert "dcf_assumptions" in tool_names
        # And the tool is the actual DCFTool class
        assert any(isinstance(t, DCFTool) for t in agent.tools)

    def test_calls_llm_with_low_temperature(self):
        from alphaquant.agents.valuation_analyst import build_valuation_analyst_agent

        with patch("alphaquant.agents.valuation_analyst.get_llm") as mock_llm:
            build_valuation_analyst_agent()

        mock_llm.assert_called_once()
        kwargs = mock_llm.call_args.kwargs
        assert kwargs.get("temperature") == pytest.approx(0.1)

    def test_no_delegation_and_verbose(self):
        from alphaquant.agents.valuation_analyst import build_valuation_analyst_agent

        with patch("alphaquant.agents.valuation_analyst.get_llm") as mock_llm:
            agent = build_valuation_analyst_agent()

        assert agent.allow_delegation is False
        assert agent.verbose is True


# ---------------------------------------------------------------------------
# ReportWriter
# ---------------------------------------------------------------------------

class TestReportWriter:
    def test_role(self):
        from alphaquant.agents.report_writer import build_report_writer_agent

        with patch("alphaquant.agents.report_writer.get_llm") as mock_llm:
            agent = build_report_writer_agent()

        assert agent.role == "Equity Research Editor"

    def test_has_goal_and_backstory(self):
        from alphaquant.agents.report_writer import build_report_writer_agent

        with patch("alphaquant.agents.report_writer.get_llm") as mock_llm:
            agent = build_report_writer_agent()

        assert isinstance(agent.goal, str) and len(agent.goal) > 0
        assert isinstance(agent.backstory, str) and len(agent.backstory) > 0
        # Should reference synthesizing prior analysis
        assert "synthes" in agent.goal.lower() or "synthes" in agent.backstory.lower()

    def test_has_no_tools(self):
        from alphaquant.agents.report_writer import build_report_writer_agent

        with patch("alphaquant.agents.report_writer.get_llm") as mock_llm:
            agent = build_report_writer_agent()

        # ReportWriter synthesizes upstream analysis; no data-lookup tools
        assert agent.tools == []

    def test_calls_llm_with_high_tokens_and_creativity(self):
        from alphaquant.agents.report_writer import build_report_writer_agent

        with patch("alphaquant.agents.report_writer.get_llm") as mock_llm:
            build_report_writer_agent()

        mock_llm.assert_called_once()
        kwargs = mock_llm.call_args.kwargs
        # Editorial tone needs more output and slightly higher temperature
        assert kwargs.get("temperature") == pytest.approx(0.4)
        assert kwargs.get("max_tokens") == 6000

    def test_no_delegation_and_verbose(self):
        from alphaquant.agents.report_writer import build_report_writer_agent

        with patch("alphaquant.agents.report_writer.get_llm") as mock_llm:
            agent = build_report_writer_agent()

        assert agent.allow_delegation is False
        assert agent.verbose is True

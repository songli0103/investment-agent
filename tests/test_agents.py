"""Tests for alphaquant.agents builder functions."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from crewai.llm import LLM

from alphaquant.agents.company_resolver import build_company_resolver_agent
from alphaquant.agents.competitor_analyst import build_competitor_analyst_agent
from alphaquant.agents.financial_analyst import build_financial_analyst_agent
from alphaquant.agents.market_analyst import build_market_analyst_agent
from alphaquant.agents.news_analyst import build_news_analyst_agent
from alphaquant.agents.report_writer import build_report_writer_agent
from alphaquant.agents.risk_analyst import build_risk_analyst_agent
from alphaquant.agents.valuation_analyst import build_valuation_analyst_agent
from alphaquant.tools.competitor_tool import CompetitorTool
from alphaquant.tools.company_lookup_tool import CompanyLookupTool
from alphaquant.tools.dcf_tool import DCFTool
from alphaquant.tools.financial_tool import FinancialTool
from alphaquant.tools.market_data_tool import MarketDataTool
from alphaquant.tools.news_tool import NewsTool


class _FakeLLM(LLM):
    """LLM subclass that bypasses real model init so Agent construction succeeds.

    CrewAI 0.203.2 Agent.__init__ calls ``create_llm(llm)`` which short-circuits
    for LLM/BaseLLM instances and returns the value as-is. A plain ``MagicMock``
    falls through to attribute extraction and is replaced with a real LLM,
    breaking identity tests. Subclassing LLM preserves identity.
    """

    def __init__(self) -> None:  # noqa: D401 - bypass real init
        # Skip ``LLM.__init__`` which requires a valid model + API key.
        # CrewAI only reads attributes (model, temperature, stop, ...) during
        # agent executor setup, so populating them after the fact is sufficient.
        pass


@pytest.fixture
def fake_llm() -> LLM:
    llm = _FakeLLM()
    llm.model = "gpt-4o-mini"
    llm.temperature = 0.2
    llm.stop = []
    return llm


class TestAgentBuilders:
    """Each builder accepts an LLM and returns a configured Agent."""

    def test_company_resolver_has_company_lookup_tool(self, fake_llm):
        from crewai import Agent
        agent = build_company_resolver_agent(fake_llm)
        assert isinstance(agent, Agent)
        assert len(agent.tools) == 1
        assert isinstance(agent.tools[0], CompanyLookupTool)

    def test_market_analyst_has_market_data_tool(self, fake_llm):
        agent = build_market_analyst_agent(fake_llm)
        assert len(agent.tools) == 1
        assert isinstance(agent.tools[0], MarketDataTool)

    def test_news_analyst_has_news_tool(self, fake_llm):
        agent = build_news_analyst_agent(fake_llm)
        assert len(agent.tools) == 1
        assert isinstance(agent.tools[0], NewsTool)

    def test_financial_analyst_has_financial_tool(self, fake_llm):
        agent = build_financial_analyst_agent(fake_llm)
        assert len(agent.tools) == 1
        assert isinstance(agent.tools[0], FinancialTool)

    def test_competitor_analyst_has_competitor_tool(self, fake_llm):
        agent = build_competitor_analyst_agent(fake_llm)
        assert len(agent.tools) == 1
        assert isinstance(agent.tools[0], CompetitorTool)

    def test_risk_analyst_has_no_tools(self, fake_llm):
        agent = build_risk_analyst_agent(fake_llm)
        assert agent.tools == []

    def test_valuation_analyst_has_dcf_tool(self, fake_llm):
        agent = build_valuation_analyst_agent(fake_llm)
        assert len(agent.tools) == 1
        assert isinstance(agent.tools[0], DCFTool)

    def test_report_writer_has_no_tools(self, fake_llm):
        agent = build_report_writer_agent(fake_llm)
        assert agent.tools == []

    @pytest.mark.parametrize(
        "builder_name",
        [
            "build_company_resolver_agent",
            "build_market_analyst_agent",
            "build_news_analyst_agent",
            "build_financial_analyst_agent",
            "build_competitor_analyst_agent",
            "build_risk_analyst_agent",
            "build_valuation_analyst_agent",
            "build_report_writer_agent",
        ],
    )
    def test_builder_uses_passed_llm(self, fake_llm, builder_name):
        """Each agent uses the LLM passed in (not a fresh get_llm() call)."""
        from alphaquant import agents

        builder = getattr(agents, builder_name)
        agent = builder(fake_llm)
        assert agent.llm is fake_llm

    @pytest.mark.parametrize(
        "builder_name",
        [
            "build_company_resolver_agent",
            "build_market_analyst_agent",
            "build_news_analyst_agent",
            "build_financial_analyst_agent",
            "build_competitor_analyst_agent",
            "build_risk_analyst_agent",
            "build_valuation_analyst_agent",
            "build_report_writer_agent",
        ],
    )
    def test_builder_verbose_is_false(self, fake_llm, builder_name):
        """Verbose=False in production to avoid log spam; can be toggled later."""
        from alphaquant import agents

        builder = getattr(agents, builder_name)
        agent = builder(fake_llm)
        assert agent.verbose is False
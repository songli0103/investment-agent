"""Tests for alphaquant.crews.AnalysisCrew."""
from __future__ import annotations

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
from alphaquant.tools.dcf_tool import DCFTool
from alphaquant.tools.financial_tool import FinancialTool
from alphaquant.tools.market_data_tool import MarketDataTool
from alphaquant.tools.news_tool import NewsTool


class _FakeLLM(LLM):
    """LLM subclass that bypasses real model init so Agent/Crew construction succeeds.

    CrewAI 0.203.2 calls ``create_llm(llm)`` on the manager_llm; a plain
    ``MagicMock`` falls through to attribute extraction and gets replaced with
    a real LLM, breaking identity tests. Subclassing LLM preserves identity.
    """

    def __init__(self) -> None:  # noqa: D401 - bypass real init
        pass


@pytest.fixture
def fake_llm() -> LLM:
    llm = _FakeLLM()
    llm.model = "gpt-4o-mini"
    llm.temperature = 0.2
    llm.stop = []
    return llm


class TestAnalysisCrew:
    def test_imports(self):
        from alphaquant.crews import AnalysisCrew

        assert AnalysisCrew is not None

    def test_instantiates_without_error(self, monkeypatch, fake_llm):
        """AnalysisCrew() should not raise; get_llm is mocked."""
        monkeypatch.setattr(
            "alphaquant.crews.analysis_crew.get_llm", lambda **kw: fake_llm
        )
        from alphaquant.crews import AnalysisCrew

        crew = AnalysisCrew()
        assert crew is not None

    def test_all_8_agents_built(self, monkeypatch, fake_llm):
        monkeypatch.setattr(
            "alphaquant.crews.analysis_crew.get_llm", lambda **kw: fake_llm
        )
        from alphaquant.crews import AnalysisCrew

        crew = AnalysisCrew()
        assert len(crew.agents) == 8

    def test_agents_are_crewai_agents(self, monkeypatch, fake_llm):
        from crewai import Agent

        monkeypatch.setattr(
            "alphaquant.crews.analysis_crew.get_llm", lambda **kw: fake_llm
        )
        from alphaquant.crews import AnalysisCrew

        crew = AnalysisCrew()
        for agent in crew.agents:
            assert isinstance(agent, Agent)

    def test_all_8_tasks_built(self, monkeypatch, fake_llm):
        from crewai import Task

        monkeypatch.setattr(
            "alphaquant.crews.analysis_crew.get_llm", lambda **kw: fake_llm
        )
        from alphaquant.crews import AnalysisCrew

        crew = AnalysisCrew()
        assert len(crew.tasks) == 8
        for task in crew.tasks:
            assert isinstance(task, Task)

    def test_process_is_hierarchical(self, monkeypatch, fake_llm):
        from crewai import Process

        monkeypatch.setattr(
            "alphaquant.crews.analysis_crew.get_llm", lambda **kw: fake_llm
        )
        from alphaquant.crews import AnalysisCrew

        crew = AnalysisCrew()
        assert crew.crew.process == Process.hierarchical

    def test_manager_llm_configured(self, monkeypatch, fake_llm):
        monkeypatch.setattr(
            "alphaquant.crews.analysis_crew.get_llm", lambda **kw: fake_llm
        )
        from alphaquant.crews import AnalysisCrew

        crew = AnalysisCrew()
        assert crew.crew.manager_llm is fake_llm

    def test_memory_disabled_for_sub1(self, monkeypatch, fake_llm):
        """Sub-project 1: memory=False. Sub-project 4 enables it."""
        monkeypatch.setattr(
            "alphaquant.crews.analysis_crew.get_llm", lambda **kw: fake_llm
        )
        from alphaquant.crews import AnalysisCrew

        crew = AnalysisCrew()
        assert crew.crew.memory is False

    def test_tools_mapping(self, monkeypatch, fake_llm):
        """Each agent's tools match the spec table."""
        expected_tools = {
            build_company_resolver_agent: [],
            build_market_analyst_agent: [MarketDataTool],
            build_news_analyst_agent: [NewsTool],
            build_financial_analyst_agent: [FinancialTool],
            build_competitor_analyst_agent: [CompetitorTool],
            build_risk_analyst_agent: [],
            build_valuation_analyst_agent: [DCFTool],
            build_report_writer_agent: [],
        }
        monkeypatch.setattr(
            "alphaquant.crews.analysis_crew.get_llm", lambda **kw: fake_llm
        )
        from alphaquant.crews import AnalysisCrew

        crew = AnalysisCrew()

        # Each agent's builder is identifiable via its role.
        role_to_builder = {
            "Company Identification Specialist": build_company_resolver_agent,
            "Market Data Specialist": build_market_analyst_agent,
            "News Retrieval Specialist": build_news_analyst_agent,
            "Financial Statements Specialist": build_financial_analyst_agent,
            "Competitive Landscape Analyst": build_competitor_analyst_agent,
            "Risk Assessment Specialist": build_risk_analyst_agent,
            "Sell-side Valuation Modeler": build_valuation_analyst_agent,
            "Investment Report Synthesizer": build_report_writer_agent,
        }
        for agent in crew.agents:
            builder = role_to_builder[agent.role]
            expected_classes = expected_tools[builder]
            assert len(agent.tools) == len(expected_classes), (
                f"{agent.role}: expected {len(expected_classes)} tool(s), "
                f"got {len(agent.tools)}"
            )
            for tool, expected_cls in zip(agent.tools, expected_classes):
                assert isinstance(tool, expected_cls)
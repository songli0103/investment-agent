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
from alphaquant.tools.company_lookup_tool import CompanyLookupTool  # NEW sub-2
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
            build_company_resolver_agent: [CompanyLookupTool],  # sub-2: was []
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

    def test_data_tasks_have_async_execution(self, monkeypatch, fake_llm):
        """Sub-project 3: 7 tasks (data + analysis) run in parallel; report writer (idx 7) is serial."""
        monkeypatch.setattr(
            "alphaquant.crews.analysis_crew.get_llm", lambda **kw: fake_llm
        )
        from alphaquant.crews import AnalysisCrew

        crew = AnalysisCrew()

        # First 7 tasks (data + analysis) are async per _ASYNC_TASK_INDICES.
        parallel_tasks = crew.tasks[:7]
        for task in parallel_tasks:
            assert task.async_execution is True, (
                f"Task '{task.description[:40]}...' should run in parallel"
            )

        # Report writer (idx 7) is serial.
        report_task = crew.tasks[7]
        assert report_task.async_execution in (False, None), (
            f"Task '{report_task.description[:40]}...' should run sequentially"
        )

    def test_task_templates_uses_3_tuple_with_pydantic_model(self):
        """_TASK_TEMPLATES entries must be (key, description, pydantic_model_or_None)."""
        from alphaquant.crews.analysis_crew import _TASK_TEMPLATES
        from alphaquant.models.competitor import CompetitorAnalysis
        from alphaquant.models.risk import RiskAssessment
        from alphaquant.models.valuation import ValuationResult
        from alphaquant.models.report import InvestmentReport

        assert len(_TASK_TEMPLATES) == 8
        for entry in _TASK_TEMPLATES:
            assert len(entry) == 3, f"expected 3-tuple, got {len(entry)}-tuple: {entry!r}"

        keys = [t[0] for t in _TASK_TEMPLATES]
        assert keys == [
            "company_resolver", "market_analyst", "news_analyst", "financial_analyst",
            "competitor_analyst", "risk_analyst", "valuation_analyst", "report_writer",
        ]

        # 4 data tasks: no Pydantic output (tool JSON only)
        assert _TASK_TEMPLATES[0][2] is None
        assert _TASK_TEMPLATES[1][2] is None
        assert _TASK_TEMPLATES[2][2] is None
        assert _TASK_TEMPLATES[3][2] is None
        # 3 analysis tasks + report writer: Pydantic
        assert _TASK_TEMPLATES[4][2] is CompetitorAnalysis
        assert _TASK_TEMPLATES[5][2] is RiskAssessment
        assert _TASK_TEMPLATES[6][2] is ValuationResult
        assert _TASK_TEMPLATES[7][2] is InvestmentReport

    def test_async_task_indices_cover_data_and_analysis_not_report(self):
        """_ASYNC_TASK_INDICES must cover 0-6 (data + analysis), not 7 (report writer)."""
        from alphaquant.crews.analysis_crew import AnalysisCrew as _AC

        # Build crew with a fake LLM to avoid network calls
        from unittest.mock import patch
        from tests.conftest import _FakeLLM
        fake = _FakeLLM()
        with patch("alphaquant.crews.analysis_crew.get_llm", return_value=fake):
            crew = _AC()
        async_indices = getattr(crew, "_ASYNC_TASK_INDICES", None)
        assert async_indices is not None, "_ASYNC_TASK_INDICES must be a class-level constant"
        assert async_indices == {0, 1, 2, 3, 4, 5, 6}
        assert 7 not in async_indices  # report writer is serial

    def test_report_writer_task_has_context_with_analysis_tasks(self):
        """Report writer (idx 7) must receive task 4/5/6 as context."""
        from alphaquant.crews.analysis_crew import AnalysisCrew as _AC
        from unittest.mock import patch
        from tests.conftest import _FakeLLM
        fake = _FakeLLM()
        with patch("alphaquant.crews.analysis_crew.get_llm", return_value=fake):
            crew = _AC()
        report_task = crew.tasks[7]
        ctx = getattr(report_task, "context", None) or []
        assert len(ctx) == 3
        # context should reference the same task objects as tasks 4/5/6
        assert crew.tasks[4] in ctx
        assert crew.tasks[5] in ctx
        assert crew.tasks[6] in ctx

    def test_competitor_task_has_output_pydantic(self):
        from alphaquant.crews.analysis_crew import AnalysisCrew as _AC
        from alphaquant.models.competitor import CompetitorAnalysis
        from unittest.mock import patch
        from tests.conftest import _FakeLLM
        fake = _FakeLLM()
        with patch("alphaquant.crews.analysis_crew.get_llm", return_value=fake):
            crew = _AC()
        assert getattr(crew.tasks[4], "output_pydantic", None) is CompetitorAnalysis

    def test_risk_task_has_output_pydantic(self):
        from alphaquant.crews.analysis_crew import AnalysisCrew as _AC
        from alphaquant.models.risk import RiskAssessment
        from unittest.mock import patch
        from tests.conftest import _FakeLLM
        fake = _FakeLLM()
        with patch("alphaquant.crews.analysis_crew.get_llm", return_value=fake):
            crew = _AC()
        assert getattr(crew.tasks[5], "output_pydantic", None) is RiskAssessment

    def test_valuation_task_has_output_pydantic(self):
        from alphaquant.crews.analysis_crew import AnalysisCrew as _AC
        from alphaquant.models.valuation import ValuationResult
        from unittest.mock import patch
        from tests.conftest import _FakeLLM
        fake = _FakeLLM()
        with patch("alphaquant.crews.analysis_crew.get_llm", return_value=fake):
            crew = _AC()
        assert getattr(crew.tasks[6], "output_pydantic", None) is ValuationResult

    def test_report_writer_task_has_output_pydantic(self):
        from alphaquant.crews.analysis_crew import AnalysisCrew as _AC
        from alphaquant.models.report import InvestmentReport
        from unittest.mock import patch
        from tests.conftest import _FakeLLM
        fake = _FakeLLM()
        with patch("alphaquant.crews.analysis_crew.get_llm", return_value=fake):
            crew = _AC()
        assert getattr(crew.tasks[7], "output_pydantic", None) is InvestmentReport
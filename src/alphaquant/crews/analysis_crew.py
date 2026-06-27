"""AnalysisCrew: 8-agent CrewAI Crew orchestrating the analysis pipeline.

Sub-project 1: Crew is built with Process.hierarchical and a shared
manager_llm, but ``memory=False`` and agents are 'thin' wrappers calling
the existing data tools. Sub-project 4 will enable ``memory`` and
``allow_delegation=True``.

Sub-project 2: The 4 data agents (CompanyResolver, MarketAnalyst,
NewsAnalyst, FinancialAnalyst) fetch their own data via the data tools
(``CompanyLookupTool`` + the 3 existing data tools) inside the Crew.
The Flow (``flows/analysis_flow.py``) is now pure orchestration: it
passes only ``{"ticker": ...}`` to ``crew.kickoff`` and calls
``parse_crew_output()`` on the result to populate ``AnalysisState``.
"""
from __future__ import annotations

from typing import Any

from crewai import Agent, Crew, Process, Task

from alphaquant.agents.company_resolver import build_company_resolver_agent
from alphaquant.agents.competitor_analyst import build_competitor_analyst_agent
from alphaquant.agents.financial_analyst import build_financial_analyst_agent
from alphaquant.agents.market_analyst import build_market_analyst_agent
from alphaquant.agents.news_analyst import build_news_analyst_agent
from alphaquant.agents.report_writer import build_report_writer_agent
from alphaquant.agents.risk_analyst import build_risk_analyst_agent
from alphaquant.agents.valuation_analyst import build_valuation_analyst_agent
from alphaquant.infrastructure.llm import get_llm
from alphaquant.models.report import ReportWriterOutput
from pydantic import BaseModel


# Sub-project 3 (then reverted): the 3 analysis tasks (idx 4-6) originally
# had output_pydantic set so the LLM produced structured Pydantic output. In
# production, the LLM was emitting structurally invalid output (wrong field
# names, conversational text) that caused the CrewAI converter to retry-loop
# until the 180s flow timeout. We reverted those tasks to text-only — the
# Flow now computes competitor/risk/valuation deterministically. The
# report_writer (idx 7) produces ``ReportWriterOutput`` (a slim subset of
# ``InvestmentReport``); the Flow assembles the full ``InvestmentReport``.
_TASK_TEMPLATES: list[tuple[str, str, type[BaseModel] | None]] = [
    (
        "company_resolver",
        "Validate ticker '{ticker}' and return canonical company metadata.",
        None,
    ),
    (
        "market_analyst",
        "Fetch market data for '{ticker}'.",
        None,
    ),
    (
        "news_analyst",
        "Fetch recent news for '{ticker}'.",
        None,
    ),
    (
        "financial_analyst",
        "Fetch financial statements for '{ticker}'.",
        None,
    ),
    (
        "competitor_analyst",
        "Summarize the competitive landscape for '{ticker}' in plain text. "
        "Do NOT produce structured Pydantic output; the Flow computes the "
        "structured CompetitorAnalysis from data. Your text is used as "
        "context for the report writer.",
        None,
    ),
    (
        "risk_analyst",
        "Summarize the key risk factors for '{ticker}' in plain text. "
        "Do NOT produce structured Pydantic output; the Flow computes the "
        "structured RiskAssessment from data. Your text is used as context "
        "for the report writer.",
        None,
    ),
    (
        "valuation_analyst",
        "Summarize the valuation analysis (DCF + relative) for '{ticker}' in "
        "plain text. Do NOT produce structured Pydantic output; the Flow "
        "computes the structured ValuationResult from data. Your text is "
        "used as context for the report writer.",
        None,
    ),
    (
        "report_writer",
        "Synthesize the final markdown report and rating for '{ticker}'.",
        ReportWriterOutput,
    ),
]


class AnalysisCrew:
    """Wraps the 8 CrewAI agents in a hierarchical Crew.

    Sub-project 1 keeps the crew as a structural shell: it can be invoked
    end-to-end, but its outputs are normalized by ``parse_crew_output``
    in the calling Flow. Sub-project 3 will let agents do real reasoning;
    sub-project 4 will enable memory and peer delegation.
    """

    # Indices of tasks that run in parallel via async_execution=True.
    # Data (0-3) and analysis (4-6) are independent → parallel.
    # Report writer (7) depends on analysis outputs → serial.
    _ASYNC_TASK_INDICES: set[int] = {0, 1, 2, 3, 4, 5, 6}

    def __init__(self) -> None:
        self._llm = get_llm(temperature=0.1)
        self.agents: list[Agent] = self._build_agents()
        self.tasks: list[Task] = self._build_tasks()
        self.crew: Crew = self._build_crew()

    def _build_agents(self) -> list[Agent]:
        return [
            build_company_resolver_agent(self._llm),
            build_market_analyst_agent(self._llm),
            build_news_analyst_agent(self._llm),
            build_financial_analyst_agent(self._llm),
            build_competitor_analyst_agent(self._llm),
            build_risk_analyst_agent(self._llm),
            build_valuation_analyst_agent(self._llm),
            build_report_writer_agent(self._llm),
        ]

    def _build_tasks(self) -> list[Task]:
        tasks: list[Task] = []
        for idx, (role_key, description, pydantic_model) in enumerate(_TASK_TEMPLATES):
            agent = self.agents[idx]
            task_kwargs: dict[str, Any] = {
                "description": description,
                "expected_output": pydantic_model.__name__ if pydantic_model else "raw text",
                "agent": agent,
                "async_execution": idx in self._ASYNC_TASK_INDICES,
            }
            if pydantic_model is not None:
                task_kwargs["output_pydantic"] = pydantic_model
            # Report writer (idx 7) consumes the 3 analysis tasks' Pydantic outputs
            if idx == 7:
                task_kwargs["context"] = [tasks[4], tasks[5], tasks[6]]
            tasks.append(Task(**task_kwargs))
        return tasks

    def _build_crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_llm=self._llm,
            memory=False,
            verbose=False,
        )

    def kickoff(self, inputs: dict[str, Any]):
        """Synchronous entry point — wraps Crew.kickoff().

        The Flow layer is responsible for invoking this inside
        ``asyncio.to_thread`` so the event loop is not blocked.
        """
        return self.crew.kickoff(inputs=inputs)


__all__ = ["AnalysisCrew"]
"""AnalysisCrew: 8-agent CrewAI Crew orchestrating the analysis pipeline.

Sub-project 1 (this file): Crew is built with Process.hierarchical and a
shared manager_llm, but ``memory=False`` and agents are 'thin' wrappers
calling the existing data tools. Sub-project 4 will enable ``memory`` and
``allow_delegation=True``.

The Flow (``flows/analysis_flow.py``) pre-fetches all raw data
(company / market / news / financial) via ``DataSourceRegistry`` and passes
it in ``inputs``. After ``kickoff`` returns, ``parse_crew_output()``
extracts the agent outputs into ``AnalysisState``.
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


_TASK_TEMPLATES: list[tuple[str, str, str]] = [
    # (role_key, description_template, expected_output)
    (
        "company_resolver",
        "Validate ticker '{ticker}' and return canonical company metadata.",
        "JSON with company name, exchange, sector, industry, market cap",
    ),
    (
        "market_analyst",
        "Fetch market data for '{ticker}'.",
        "JSON with price, P/E, market cap, 52w range, beta, growth metrics",
    ),
    (
        "news_analyst",
        "Fetch recent news for '{ticker}'.",
        "JSON array of news items (date, title, source, url)",
    ),
    (
        "financial_analyst",
        "Fetch financial statements for '{ticker}'.",
        "JSON with income statements, balance sheets, cash flows",
    ),
    (
        "competitor_analyst",
        "Identify competitors and compute competitive score for '{ticker}'.",
        "JSON with peer tickers, market caps, growth, margins, rank",
    ),
    (
        "risk_analyst",
        "Compute risk assessment for '{ticker}' from upstream data.",
        "JSON with sub-scores per category and total",
    ),
    (
        "valuation_analyst",
        "Compute valuation (DCF + relative) for '{ticker}'.",
        "JSON with intrinsic value, DCF value, relative value, upside",
    ),
    (
        "report_writer",
        "Synthesize InvestmentReport markdown for '{ticker}'.",
        "Markdown report with rating, confidence, sections",
    ),
]


class AnalysisCrew:
    """Wraps the 8 CrewAI agents in a hierarchical Crew.

    Sub-project 1 keeps the crew as a structural shell: it can be invoked
    end-to-end, but its outputs are normalized by ``parse_crew_output``
    in the calling Flow. Sub-project 3 will let agents do real reasoning;
    sub-project 4 will enable memory and peer delegation.
    """

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
        # Sub-project 2: first 4 (data) tasks run in parallel via
        # Task(async_execution=True). Manager LLM schedules them concurrently
        # in the hierarchical process. Remaining 4 (analysis) tasks stay
        # sequential (default CrewAI behavior in hierarchical mode).
        _ASYNC_TASK_INDICES = {0, 1, 2, 3}

        tasks: list[Task] = []
        for idx, (role_key, description, expected) in enumerate(_TASK_TEMPLATES):
            agent = self.agents[idx]
            tasks.append(
                Task(
                    description=description,
                    expected_output=expected,
                    agent=agent,
                    async_execution=(idx in _ASYNC_TASK_INDICES),
                )
            )
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
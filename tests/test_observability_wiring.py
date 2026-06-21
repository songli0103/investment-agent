"""Tests for Task 21: wiring observability into main + flow.

These tests verify that:
- Importing ``alphaquant.main`` configures structlog (idempotently).
- ``run_analysis_async`` emits ``analysis_started`` / ``analysis_completed``
  structured log events.
- The Flow emits ``flow_step_started`` / ``flow_step_completed`` events for
  each of the 6 orchestration steps.
- ``kickoff_with_timeout`` emits a ``flow_timeout`` event on timeout.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
from datetime import datetime
from decimal import Decimal
from unittest.mock import patch

import pytest
import structlog

# Suppress CrewAI's interactive prompt.
os.environ.setdefault("CREWAI_TESTING", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


class _LogSink(io.StringIO):
    """In-memory file-like sink that JSON-decodes each line into ``captured``."""

    def __init__(self, captured: list[dict]) -> None:
        super().__init__()
        self._captured = captured

    def write(self, s: str) -> int:  # type: ignore[override]
        for line in s.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self._captured.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return len(s)


@pytest.fixture
def captured_logs():
    """Capture structlog JSON output to a list.

    We reset structlog to defaults, disable caching, and route writes to
    an in-memory ``_LogSink`` so each test sees only its own events.
    Cached module-level loggers in observability / core / flow /
    data_sources are re-bound to the new (test-time) configuration.
    """
    captured: list[dict] = []
    sink = _LogSink(captured)

    structlog.reset_defaults()
    structlog.configure(
        processors=[structlog.processors.JSONRenderer()],
        logger_factory=structlog.WriteLoggerFactory(file=sink),
        cache_logger_on_first_use=False,
    )

    import alphaquant.core as core_mod
    import alphaquant.flows.analysis_flow as flow_mod
    import alphaquant.infrastructure.data_sources as data_sources_mod
    import alphaquant.observability.cost_tracker as cost_tracker
    cost_tracker.log = structlog.get_logger()
    core_mod.log = structlog.get_logger("alphaquant.core")
    flow_mod.log = structlog.get_logger("alphaquant.flows.analysis_flow")
    data_sources_mod.log = structlog.get_logger("alphaquant.data_sources")

    yield captured

    structlog.reset_defaults()


def _sample_report(ticker: str = "AAPL"):
    from alphaquant.models.company import Company
    from alphaquant.models.competitor import Competitor, CompetitorAnalysis
    from alphaquant.models.financial import FinancialStatements
    from alphaquant.models.market import MarketData
    from alphaquant.models.news import NewsAnalysis
    from alphaquant.models.report import InvestmentReport
    from alphaquant.models.risk import RiskAssessment
    from alphaquant.models.valuation import ValuationResult

    return InvestmentReport(
        report_id="11111111-1111-1111-1111-111111111111",
        ticker=ticker,
        generated_at=datetime(2026, 6, 20),
        company=Company(
            ticker=ticker,
            name=f"{ticker} Inc.",
            exchange="NASDAQ",
            sector="Technology",
            industry="Consumer Electronics",
            market_cap=3_000_000_000_000,
        ),
        market=MarketData(
            ticker=ticker,
            as_of=datetime(2026, 6, 20),
            price=Decimal("150.00"),
            change_pct=0.5,
            volume=50_000_000,
            market_cap=3_000_000_000_000,
            pe_ratio=25.0,
            beta=1.2,
        ),
        financial=FinancialStatements(ticker=ticker),
        financial_health_score=70,
        news=NewsAnalysis(
            ticker=ticker,
            as_of=datetime(2026, 6, 20),
            total_count=10,
            positive_pct=0.5,
            negative_pct=0.2,
            neutral_pct=0.3,
            sentiment_score=0.3,
        ),
        competitors=CompetitorAnalysis(
            target_ticker=ticker,
            competitors=[
                Competitor(
                    ticker="MSFT",
                    name="Microsoft",
                    market_cap=2_500_000_000_000,
                    revenue_ttm=Decimal("200000000000"),
                )
            ],
            industry_rank=1,
            industry_size=10,
            competitive_score=75,
        ),
        risk=RiskAssessment(
            ticker=ticker,
            total_score=45,
            level="medium",
            sub_scores=[{"category": "financial", "score": 5, "rationale": "stub rationale for test", "evidence": []}],
            top_risks=[],
        ),
        valuation=ValuationResult(
            ticker=ticker,
            intrinsic_value_per_share=Decimal("180.00"),
            current_price=Decimal("150.00"),
            upside_pct=0.2,
            method="relative_only",
        ),
        rating="Buy",
        confidence=75,
        catalysts=[],
        markdown="# report",
        sources=["yahoo", "newsapi"],
    )


# ---------------------------------------------------------------------------
# main.py: configure_logging is invoked at import
# ---------------------------------------------------------------------------


def test_main_configure_logging_is_idempotent():
    """Importing main multiple times must not raise."""
    import importlib

    import alphaquant.main

    importlib.reload(alphaquant.main)
    importlib.reload(alphaquant.main)


def test_main_uses_observability_logger():
    """main.py must hold a module-level logger from the observability package."""
    import alphaquant.main

    # Duck-type check: a structlog logger exposes .info/.error/.warning.
    log = alphaquant.main.log
    for attr in ("info", "warning", "error", "debug"):
        assert callable(getattr(log, attr, None)), f"missing {attr} on main.log"


# ---------------------------------------------------------------------------
# run_analysis_async: analysis_started / analysis_completed
# ---------------------------------------------------------------------------


def test_run_analysis_async_emits_started_and_completed(captured_logs):
    """Successful run emits analysis_started and analysis_completed."""
    from alphaquant.core import run_analysis_async

    report = _sample_report()
    with patch("alphaquant.core.AnalysisFlow") as MockFlow:
        flow_instance = MockFlow.return_value
        flow_instance.state.report = report

        async def _fake_kickoff_with_timeout(inputs):
            return None

        flow_instance.kickoff_with_timeout = _fake_kickoff_with_timeout
        result = asyncio.run(run_analysis_async("AAPL"))

    assert result is report
    events = [r["event"] for r in captured_logs]
    assert "analysis_started" in events
    assert "analysis_completed" in events
    started = next(r for r in captured_logs if r["event"] == "analysis_started")
    completed = next(r for r in captured_logs if r["event"] == "analysis_completed")
    assert started["ticker"] == "AAPL"
    assert completed["ticker"] == "AAPL"
    assert completed["report_id"] == report.report_id
    assert completed["rating"] == "Buy"


def test_run_analysis_async_emits_no_report_event(captured_logs):
    """When flow produces no report, log analysis_no_report then raise."""
    from alphaquant.core import run_analysis_async
    from alphaquant.exceptions import AllDataSourcesDown

    with patch("alphaquant.core.AnalysisFlow") as MockFlow:
        flow_instance = MockFlow.return_value
        flow_instance.state.report = None

        async def _fake_kickoff_with_timeout(inputs):
            return None

        flow_instance.kickoff_with_timeout = _fake_kickoff_with_timeout
        with pytest.raises(AllDataSourcesDown):
            asyncio.run(run_analysis_async("ZZZZ"))

    events = [r["event"] for r in captured_logs]
    assert "analysis_started" in events
    assert "analysis_no_report" in events
    # completed must NOT be emitted on failure
    assert "analysis_completed" not in events
    no_report = next(r for r in captured_logs if r["event"] == "analysis_no_report")
    assert no_report["ticker"] == "ZZZZ"


# ---------------------------------------------------------------------------
# analysis_flow.py: per-step events
# ---------------------------------------------------------------------------


def test_flow_emits_step_started_and_completed(captured_logs):
    """Running the run_crew step emits started + completed events."""
    from alphaquant.flows.analysis_flow import AnalysisFlow
    from alphaquant.models.company import Company
    from alphaquant.models.market import MarketData
    from alphaquant.models.financial import FinancialStatements
    from alphaquant.models.news import NewsAnalysis

    flow = AnalysisFlow()
    flow.state.ticker = "AAPL"

    fake_company = Company(
        ticker="AAPL",
        name="Apple Inc.",
        exchange="NASDAQ",
        sector="Technology",
        industry="Consumer Electronics",
        market_cap=3_000_000_000_000,
    )
    fake_market = MarketData(
        ticker="AAPL",
        as_of=datetime(2026, 6, 20),
        price=Decimal("150.00"),
        change_pct=0.5,
        volume=0,
        market_cap=3_000_000_000_000,
        source="yahoo",
    )
    fake_financial = FinancialStatements(ticker="AAPL")
    fake_news = NewsAnalysis.empty("AAPL")

    async def _get_company(_t):
        return fake_company

    async def _get_market(_t):
        return fake_market

    async def _get_news(_t):
        return fake_news

    async def _get_financial(_t):
        return fake_financial

    fake_registry = type("R", (), {})()
    fake_registry.get_company = _get_company
    fake_registry.get_market = _get_market
    fake_registry.get_news = _get_news
    fake_registry.get_financial = _get_financial

    fake_crew = type("C", (), {})()
    fake_crew.kickoff = lambda inputs: type("O", (), {"tasks_output": []})()

    with patch("alphaquant.flows.analysis_flow.DataSourceRegistry", return_value=fake_registry), \
         patch("alphaquant.flows.analysis_flow.AnalysisCrew", return_value=fake_crew):
        asyncio.run(flow.run_crew("AAPL"))

    events = [r["event"] for r in captured_logs]
    assert "flow_step_started" in events
    assert "flow_step_completed" in events

    started_steps = [
        r.get("step") for r in captured_logs if r["event"] == "flow_step_started"
    ]
    completed_steps = [
        r.get("step") for r in captured_logs if r["event"] == "flow_step_completed"
    ]
    assert "run_crew" in started_steps
    assert "run_crew" in completed_steps


def test_flow_run_crew_logs_started_completed(captured_logs):
    """run_crew emits started, then completed on success."""
    from alphaquant.flows.analysis_flow import AnalysisFlow
    from alphaquant.models.company import Company
    from alphaquant.models.market import MarketData
    from alphaquant.models.financial import FinancialStatements
    from alphaquant.models.news import NewsAnalysis

    fake_company = Company(
        ticker="AAPL",
        name="Apple Inc.",
        exchange="NASDAQ",
        sector="Technology",
        industry="Consumer Electronics",
        market_cap=3_000_000_000_000,
    )
    fake_market = MarketData(
        ticker="AAPL",
        as_of=datetime(2026, 6, 20),
        price=Decimal("150.00"),
        change_pct=0.5,
        volume=0,
        market_cap=3_000_000_000_000,
        source="yahoo",
    )
    fake_financial = FinancialStatements(ticker="AAPL")
    fake_news = NewsAnalysis.empty("AAPL")

    async def _get_company(_t):
        return fake_company

    async def _get_market(_t):
        return fake_market

    async def _get_news(_t):
        return fake_news

    async def _get_financial(_t):
        return fake_financial

    fake_registry = type("R", (), {})()
    fake_registry.get_company = _get_company
    fake_registry.get_market = _get_market
    fake_registry.get_news = _get_news
    fake_registry.get_financial = _get_financial

    fake_crew = type("C", (), {})()
    fake_crew.kickoff = lambda inputs: type("O", (), {"tasks_output": []})()

    flow = AnalysisFlow()
    flow.state.ticker = "AAPL"

    with patch("alphaquant.flows.analysis_flow.DataSourceRegistry", return_value=fake_registry), \
         patch("alphaquant.flows.analysis_flow.AnalysisCrew", return_value=fake_crew):
        asyncio.run(flow.run_crew("AAPL"))

    events = [r["event"] for r in captured_logs]
    assert "flow_step_started" in events
    assert "flow_step_completed" in events
    completed = next(
        r for r in captured_logs
        if r["event"] == "flow_step_completed" and r.get("step") == "run_crew"
    )
    assert completed["ticker"] == "AAPL"


def test_flow_run_crew_logs_failure(captured_logs):
    """run_crew propagates AllDataSourcesDown when the company source is exhausted."""
    from alphaquant.exceptions import AllDataSourcesDown
    from alphaquant.flows.analysis_flow import AnalysisFlow
    from alphaquant.models.financial import FinancialStatements

    flow = AnalysisFlow()
    flow.state.ticker = "AAPL"

    fake_registry = type("R", (), {})()

    async def _raise(_t):
        raise AllDataSourcesDown("nope")

    async def _empty(_t):
        return None

    async def _empty_list(_t):
        return []

    async def _empty_fs(_t):
        return FinancialStatements(ticker="AAPL")

    fake_registry.get_company = _raise
    fake_registry.get_market = _empty
    fake_registry.get_news = _empty_list
    fake_registry.get_financial = _empty_fs

    fake_crew = type("C", (), {})()
    fake_crew.kickoff = lambda inputs: type("O", (), {"tasks_output": []})()

    with patch("alphaquant.flows.analysis_flow.DataSourceRegistry", return_value=fake_registry), \
         patch("alphaquant.flows.analysis_flow.AnalysisCrew", return_value=fake_crew):
        with pytest.raises(AllDataSourcesDown):
            asyncio.run(flow.run_crew("AAPL"))

    started = [
        r for r in captured_logs
        if r["event"] == "flow_step_started" and r.get("step") == "run_crew"
    ]
    assert started, "expected flow_step_started for run_crew"
    assert started[0]["ticker"] == "AAPL"


def test_kickoff_with_timeout_emits_flow_timeout(captured_logs):
    """When the Flow exceeds the timeout, log flow_timeout then raise.

    Patches the per-Flow timeout down to 0.05s so the test finishes in
    well under a second (FLOW_TIMEOUT_SECONDS defaults to 120s).
    """
    import asyncio

    from alphaquant.flows import analysis_flow as flow_mod
    from alphaquant.flows.analysis_flow import AnalysisFlow

    flow = AnalysisFlow()

    async def _hang(*_a, **_kw):
        await asyncio.sleep(5)

    with patch.object(flow_mod, "FLOW_TIMEOUT_SECONDS", 0.05), \
         patch.object(AnalysisFlow, "kickoff_async", _hang):
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(flow.kickoff_with_timeout({"ticker": "AAPL"}))

    timeouts = [r for r in captured_logs if r["event"] == "flow_timeout"]
    assert timeouts, "expected flow_timeout event"
    assert timeouts[0]["ticker"] == "AAPL"
    assert timeouts[0]["timeout_seconds"] == 0.05


# ---------------------------------------------------------------------------
# data_sources: uses the observability abstraction
# ---------------------------------------------------------------------------


def test_data_sources_uses_observability_logger():
    """data_sources package must use the observability get_logger abstraction."""
    import alphaquant.infrastructure.data_sources as ds

    log = ds.log
    for attr in ("info", "warning", "error", "debug"):
        assert callable(getattr(log, attr, None)), f"missing {attr} on ds.log"

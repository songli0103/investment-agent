"""AnalysisFlow: thin shell wrapping the AnalysisCrew.

Two-step Flow:

1. ``run_crew`` (``@start``) — invokes :class:`AnalysisCrew.kickoff` inside
   :func:`asyncio.to_thread` with a per-step timeout, then dispatches to
   :func:`parse_crew_output` to fill the downstream state fields. The 4 data
   agents fetch their own data inside the Crew via tools (sub-project 2);
   the Flow no longer pre-fetches.
2. ``synthesize_report`` (``@listen(run_crew)``) — assembles the
   :class:`InvestmentReport` from the populated state.

Sub-project 1 kept the crew as a structural shell. Sub-project 2 makes the
4 data agents fetch data via their own tools, removes Flow pre-fetch, and
extends ``parse_crew_output`` to populate company/market/news/financial from
agent task outputs. Competitor/risk/valuation still use the deterministic
fallback; sub-project 3 will let those agents produce real structured
outputs and remove the fallback paths.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from crewai.flow import Flow, listen, start
from pydantic import BaseModel, Field

from alphaquant.crews import AnalysisCrew
from alphaquant.exceptions import (
    AllDataSourcesDown,
    InvalidTickerFormat,
    ReportGenerationError,
)
from alphaquant.models.company import Company
from alphaquant.models.competitor import CompetitorAnalysis
from alphaquant.models.financial import FinancialStatements
from alphaquant.models.market import MarketData
from alphaquant.models.news import NewsAnalysis, NewsItem
from alphaquant.models.report import InvestmentReport
from alphaquant.models.risk import RiskAssessment
from alphaquant.models.valuation import ValuationResult
from alphaquant.observability import get_logger

log = get_logger("alphaquant.flows.analysis_flow")

# Disclaimer text for all generated reports. Kept in Chinese per spec.
DISCLAIMER_TEXT = (
    "本报告由 AI 自动生成，仅供参考，不构成任何投资建议。"
    "投资有风险，决策需谨慎。"
)

# §3.4: whole-Flow timeout. Sub-project 2 widens 120→180s to absorb
# 4 parallel data fetches (~30s each) + manager LLM decisions (~2s each).
FLOW_TIMEOUT_SECONDS = 180.0


# Maps crew task descriptions to AnalysisState field keys. The order MUST
# match the order in crews/analysis_crew.py::_TASK_TEMPLATES.
_TASK_KEYWORDS: list[str] = [
    "company_resolver",
    "market_analyst",
    "news_analyst",
    "financial_analyst",
    "competitor_analyst",
    "risk_analyst",
    "valuation_analyst",
    "report_writer",
]


class AnalysisState(BaseModel):
    """State passed through Flow steps."""

    ticker: str = ""
    company: Any | None = None
    market: MarketData | None = None
    news: NewsAnalysis | None = None
    financial: FinancialStatements | None = None
    competitor: CompetitorAnalysis | None = None
    risk: RiskAssessment | None = None
    valuation: ValuationResult | None = None
    report: InvestmentReport | None = None
    errors: list[str] = Field(default_factory=list)


def _normalize_ticker(raw: str) -> str:
    t = raw.strip().upper()
    if not t or len(t) > 6:
        raise InvalidTickerFormat(raw)
    return t


def _news_items_to_analysis(items: list[NewsItem], ticker: str) -> NewsAnalysis:
    """Transform list[NewsItem] (registry contract) → NewsAnalysis (Flow contract).

    Aggregates sentiment counts and surfaces the top 3 by relevance as key events.
    Empty input → NewsAnalysis.empty() per §3.2 degradation.
    """
    if not items:
        return NewsAnalysis.empty(ticker)

    sentiment_to_score = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    pos = neg = neu = 0
    weighted = 0.0
    for it in items:
        if it.sentiment == "positive":
            pos += 1
        elif it.sentiment == "negative":
            neg += 1
        else:
            neu += 1
        weighted += sentiment_to_score[it.sentiment] * float(it.relevance_score)

    total = pos + neg + neu
    avg = weighted / total if total else 0.0
    avg = max(-1.0, min(1.0, avg))

    key_events = sorted(items, key=lambda x: x.relevance_score, reverse=True)[:3]
    source = key_events[0].source if key_events else "unavailable"

    return NewsAnalysis(
        ticker=ticker,
        as_of=datetime.utcnow(),
        total_count=total,
        positive_pct=pos / total if total else 0.0,
        negative_pct=neg / total if total else 0.0,
        neutral_pct=neu / total if total else 1.0,
        sentiment_score=avg,
        key_events=key_events,
        source=source,
    )


def _collect_sources(
    market: MarketData | None,
    news: NewsAnalysis | None,
    financial: FinancialStatements | None,
    competitor: CompetitorAnalysis | None,
) -> list[str]:
    """Compose ``InvestmentReport.sources`` from non-trivial upstreams.

    Excludes the literal ``"degraded"`` (a status, not a source) and dedupes
    while preserving first-seen order. Competitor sources are reported as
    ``"gics_peers"`` when the method is ``"gics"``; otherwise the underlying
    method is recorded.
    """
    raw: list[str] = []
    if market is not None and market.source and market.source != "degraded":
        raw.append(market.source)
    if news is not None and news.source:
        raw.append(news.source)
    if financial is not None and financial.source:
        raw.append(financial.source)
    if competitor is not None:
        if competitor.method == "gics":
            raw.append("gics_peers")
        elif competitor.method:
            raw.append(competitor.method)
    # dict.fromkeys preserves insertion order while deduping.
    return list(dict.fromkeys(raw))


def parse_crew_output(
    result: Any, state: "AnalysisState" | None = None
) -> dict[str, Any]:
    """Extract agent outputs from ``CrewOutput`` and (optionally) fill state.

    Sub-project 2: each task output's ``raw`` text is either JSON (success)
    or an error string matching the ``"Error..."`` / ``"No ..."`` /
    ``"...data available..."`` convention from the 4 data tools. We parse
    the 4 data fields (company, market, news, financial) and populate
    ``state`` accordingly. ``AllDataSourcesDown`` is raised for company
    fetch failure (preserves the FastAPI error code path).

    Returns a ``{role_key: parsed_data}`` mapping so callers (and tests)
    can inspect what was extracted. When ``state`` is provided, this
    function ALSO mutates ``state`` in place.
    """
    tasks_output = getattr(result, "tasks_output", []) or []
    extracted: dict[str, Any] = {}

    # Build a {key: raw_text} lookup from the actual tasks, indexed by role_key.
    raw_by_key: dict[str, str] = {}
    for idx, task_out in enumerate(tasks_output):
        if idx >= len(_TASK_KEYWORDS):
            break
        key = _TASK_KEYWORDS[idx]
        raw_by_key[key] = getattr(task_out, "raw", "") or ""
        extracted[key] = raw_by_key[key]

    # If no state was provided, we only collect the raw text.
    if state is None:
        return extracted

    # --- Sub-project 2: parse 4 data fields from agent task outputs ---

    # 1. Company (critical path — failure raises AllDataSourcesDown)
    company, company_err = _extract_data_field(
        raw_by_key.get("company_resolver", ""),
        Company,
        "company_data_unavailable",
    )
    if company is None:
        raise AllDataSourcesDown(
            f"Cannot resolve {state.ticker}: company data unavailable"
        )
    state.company = company

    # 2. Market (degraded: None + error)
    state.market, market_err = _extract_data_field(
        raw_by_key.get("market_analyst", ""),
        MarketData,
        "market_data_unavailable",
    )
    if market_err:
        state.errors.append(market_err)

    # 3. News (degraded: empty NewsAnalysis + error). Tool returns JSON list.
    news_raw = raw_by_key.get("news_analyst", "").strip()
    if not news_raw or news_raw.startswith("Error") or news_raw.startswith("No ") or "data available" in news_raw.lower():
        state.news = NewsAnalysis.empty(state.ticker)
        state.errors.append("news_data_unavailable")
    else:
        try:
            items_raw = json.loads(news_raw)
            news_items = [NewsItem(**i) for i in items_raw]
            state.news = _news_items_to_analysis(news_items, state.ticker)
        except Exception:
            state.news = NewsAnalysis.empty(state.ticker)
            state.errors.append("news_data_unavailable")

    # 4. Financial (degraded: empty FinancialStatements shell + error)
    state.financial, fin_err = _extract_data_field(
        raw_by_key.get("financial_analyst", ""),
        FinancialStatements,
        "financial_data_unavailable",
    )
    if state.financial is None:
        state.financial = FinancialStatements(ticker=state.ticker)
        state.errors.append(fin_err or "financial_data_unavailable")

    # --- Sub-project 3: 3 analysis fields + 1 report from Pydantic output_pydantic ---
    state.competitor = _extract_pydantic_field(
        tasks_output, 4, "competitor_analyst", CompetitorAnalysis, state
    )
    state.risk = _extract_pydantic_field(
        tasks_output, 5, "risk_analyst", RiskAssessment, state
    )
    state.valuation = _extract_pydantic_field(
        tasks_output, 6, "valuation_analyst", ValuationResult, state
    )
    state.report = _extract_pydantic_field(
        tasks_output, 7, "report_writer", InvestmentReport, state
    )

    return extracted


def _extract_pydantic_field(
    tasks_output: list[Any],
    idx: int,
    key: str,
    model_cls: type[BaseModel],
    state: "AnalysisState",
) -> BaseModel | None:
    """Extract a Pydantic model from a CrewAI task output.

    CrewAI 0.203.2 sets ``task_out.pydantic`` to the validated model instance when
    the task is configured with ``output_pydantic=...``. Per sub-3 decision
    (strict no-fallback), we ONLY read that attribute. If it is missing or not
    the expected model type, append "<key>_unavailable" to state.errors and
    return None. We do NOT attempt to recover by parsing task_out.raw.

    Returns the model instance, or ``None`` on any failure.
    """
    if idx >= len(tasks_output):
        state.errors.append(f"{key}_unavailable")
        return None
    task_out = tasks_output[idx]

    pyd_obj = getattr(task_out, "pydantic", None)
    if isinstance(pyd_obj, model_cls):
        return pyd_obj

    state.errors.append(f"{key}_unavailable")
    return None


def _extract_data_field(
    raw: str, model_cls: type, error_msg: str
) -> tuple[Any | None, str | None]:
    """Parse a tool output string into a Pydantic model, or return None + error.

    Order of failure detection:
      1. Empty / whitespace-only → failure
      2. Starts with "Error" / "No " / contains "data available" → failure
         (matches the error-string convention used by all 5 data tools)
      3. Try ``model_cls.model_validate_json``; on ValidationError → failure
      4. Otherwise → success, return parsed model

    Returns ``(model, None)`` on success or ``(None, error_msg)`` on failure.
    """
    raw = raw.strip() if raw else ""
    if not raw:
        return None, error_msg
    # Error-string convention: tools return "Error fetching X: ..." or "No X data..."
    lowered = raw.lower()
    if (
        raw.startswith("Error")
        or raw.startswith("No ")
        or "data available" in lowered
    ):
        return None, error_msg
    try:
        return model_cls.model_validate_json(raw), None
    except Exception:
        return None, error_msg


class AnalysisFlow(Flow[AnalysisState]):
    """Top-level Flow: 2-step thin shell wrapping AnalysisCrew."""

    @start()
    async def run_crew(
        self,
        ticker: str | None = None,
        crewai_trigger_payload: dict[str, Any] | None = None,
    ) -> None:
        """Step 1: Drive the 8-agent Crew to produce analysis results.

        Sub-project 2: the Flow no longer pre-fetches data. Each of the 4
        data agents (CompanyResolver, MarketAnalyst, NewsAnalyst,
        FinancialAnalyst) calls its own tool inside the Crew to fetch
        fresh data. We only pass the ticker to ``crew.kickoff``; the
        resulting task outputs are parsed back into ``state`` by
        ``parse_crew_output``.
        """
        # Resolve ticker from any of the supported channels.
        raw_ticker = (
            ticker
            or (crewai_trigger_payload or {}).get("ticker")
            or self.state.ticker
            or ""
        )
        normalized = _normalize_ticker(raw_ticker)
        self.state.ticker = normalized
        log.info("flow_step_started", step="run_crew", ticker=normalized)

        # Drive the 8-agent crew. The 4 data tasks (company_resolver,
        # market_analyst, news_analyst, financial_analyst) run in parallel
        # with async_execution=True; each calls its own tool to fetch
        # fresh data. Crew.kickoff is sync → wrap in to_thread.
        crew = AnalysisCrew()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    crew.kickoff,
                    inputs={"ticker": normalized},
                ),
                timeout=FLOW_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            log.error("crew_timeout", ticker=normalized)
            raise

        # Parse crew output → fill self.state fields downstream tasks consume.
        # Raises AllDataSourcesDown if company fetch failed.
        parse_crew_output(result, self.state)

        log.info("flow_step_completed", step="run_crew", ticker=normalized)

    @listen(run_crew)
    async def synthesize_report(self) -> None:
        """Sub-project 3: state.report is already populated by ReportWriter agent.

        This step fills only runtime-only fields (sources, disclaimer, generated_at)
        and raises ReportGenerationError if the report writer failed.
        """
        if self.state.report is None:
            log.error("report_writer_failed", ticker=self.state.ticker)
            raise ReportGenerationError(
                f"Report writer agent failed to produce InvestmentReport for {self.state.ticker}"
            )

        log.info("flow_step_started", step="synthesize_report", ticker=self.state.ticker)

        # Re-derive sources from upstream data (so they reflect actual data presence, not LLM guess)
        self.state.report.sources = _collect_sources(
            self.state.market, self.state.news, self.state.financial, self.state.competitor
        )

        # Runtime fields
        self.state.report.disclaimer = DISCLAIMER_TEXT  # constant from sub-1, kept verbatim
        self.state.report.generated_at = datetime.utcnow()

        log.info(
            "flow_step_completed",
            step="synthesize_report",
            ticker=self.state.ticker,
            report_id=self.state.report.report_id,
            rating=self.state.report.rating,
            confidence=self.state.report.confidence,
        )

    async def kickoff_with_timeout(self, inputs: dict[str, Any] | None = None) -> Any:
        """§3.4 whole-Flow 120s timeout wrapper.

        CrewAI Flow's ``kickoff`` is sync; ``kickoff_async`` returns a coroutine
        that we can wrap in ``asyncio.wait_for``. On timeout the underlying
        coroutine is cancelled and ``asyncio.TimeoutError`` propagates to the
        caller, which (per spec §5.2) maps to HTTP 504 GATEWAY_TIMEOUT.
        """
        try:
            return await asyncio.wait_for(
                self.kickoff_async(inputs=inputs),
                timeout=FLOW_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            log.error(
                "flow_timeout",
                ticker=(inputs or {}).get("ticker", self.state.ticker),
                timeout_seconds=FLOW_TIMEOUT_SECONDS,
            )
            raise

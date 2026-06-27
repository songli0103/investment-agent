"""AnalysisFlow: thin shell wrapping the AnalysisCrew.

Two-step Flow:

1. ``run_crew`` (``@start``) — invokes :class:`AnalysisCrew.kickoff` inside
   :func:`asyncio.to_thread` with a per-step timeout, then dispatches to
   :func:`parse_crew_output` to fill the downstream state fields. The 4 data
   agents fetch their own data inside the Crew via tools (sub-project 2);
   the Flow no longer pre-fetches.
2. ``synthesize_report`` (``@listen(run_crew)``) — computes the 3 analysis
   fields (competitor/risk/valuation) deterministically from the populated
   data, then assembles the full :class:`InvestmentReport` from data +
   deterministic analyses + the LLM's :class:`ReportWriterOutput`.

Sub-project 3 had the 3 analysis agents (competitor/risk/valuation) produce
structured Pydantic output. The LLM was emitting structurally invalid
output (wrong field names, conversational text) that caused the CrewAI
converter to retry-loop until the 180s flow timeout, blocking the frontend.
We reverted those tasks to text-only and compute the 3 analyses
deterministically in the Flow. The report_writer LLM now produces a slim
:class:`ReportWriterOutput` (rating, confidence, horizon, catalysts,
markdown); the Flow assembles the full :class:`InvestmentReport`.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from crewai.flow import Flow, listen, start
from pydantic import BaseModel, Field

from alphaquant.crews import AnalysisCrew
from alphaquant.exceptions import (
    AllDataSourcesDown,
    InvalidTickerFormat,
    ReportGenerationError,
)
from alphaquant.infrastructure.data_sources import DataSourceRegistry
from alphaquant.models.company import Company
from alphaquant.models.competitor import Competitor, CompetitorAnalysis
from alphaquant.models.financial import FinancialStatements
from alphaquant.models.market import MarketData
from alphaquant.models.news import NewsAnalysis, NewsItem
from alphaquant.models.report import InvestmentReport, ReportWriterOutput
from alphaquant.models.risk import RiskAssessment, RiskScore
from alphaquant.models.valuation import ValuationResult
from alphaquant.observability import get_logger
from alphaquant.scoring.dcf import compute_dcf_value
from alphaquant.scoring.financial_health import compute as compute_financial_health

log = get_logger("alphaquant.flows.analysis_flow")

# Disclaimer text for all generated reports. Kept in Chinese per spec.
DISCLAIMER_TEXT = (
    "本报告由 AI 自动生成，仅供参考，不构成任何投资建议。"
    "投资有风险，决策需谨慎。"
)

# §3.4: whole-Flow timeout. Sub-3 revert validation (Task 5) showed
# MiniMax-M3 routinely needs >300s for 7 LLM tasks (AAPL hit the 300s
# limit with 18 successful LLM calls still in progress on the cleanest
# run). Widening 300→600s restores the original spec ceiling and gives
# the LLM enough headroom for real-world latency.
FLOW_TIMEOUT_SECONDS = 600.0


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
    # Slim output from the report_writer LLM (sub-project-3 revert).
    # The Flow assembles the full ``InvestmentReport`` from this plus data
    # fields and the deterministic competitor/risk/valuation analyses.
    writer_output: ReportWriterOutput | None = None
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


# --- Sub-project-3 revert: deterministic helpers for the 3 analysis fields.
# The LLM agents (competitor/risk/valuation) are reverted to text-only; the
# Flow computes the structured Pydantic models here from the populated data.

# Fallback peer set when the competitor tool returns nothing. Mirrors the
# pre-sub-3 GICS_PEERS map (deleted in commit b646b75).
GICS_PEERS: dict[str, list[str]] = {
    "Technology": ["MSFT", "GOOGL", "META"],
    "Financial Services": ["JPM", "BAC", "WFC"],
    "Healthcare": ["JNJ", "PFE", "UNH"],
    "Energy": ["XOM", "CVX", "COP"],
    "Consumer Cyclical": ["WMT", "AMZN", "HD"],
    "Consumer Defensive": ["PG", "KO", "COST"],
    "Communication Services": ["META", "NFLX", "DIS"],
    "Industrials": ["CAT", "BA", "GE"],
    "Automotive": ["TM", "F", "GM"],
    "Basic Materials": ["LIN", "APD", "FCX"],
    "Real Estate": ["AMT", "PLD", "CCI"],
    "Utilities": ["NEE", "DUK", "SO"],
}


def _gics_peers_for(ticker: str, sector: str | None) -> list[Competitor]:
    """Build 3 GICS-fallback Competitor entries for a given ticker/sector."""
    peers = GICS_PEERS.get(sector or "", ["SPY", "QQQ", "DIA"])[:3]
    return [
        Competitor(
            ticker=t,
            name=t,
            market_cap=0,
            revenue_ttm=Decimal("0"),
        )
        for t in peers
    ]


def _compute_competitor_analysis(state: "AnalysisState") -> CompetitorAnalysis:
    """Sub-project-3 revert: deterministic competitor analysis from data.

    Uses a static GICS peer map keyed on the company's sector. We do not
    call ``CompetitorTool`` from this code path: the tool's nested event
    loop conflicts with the Flow's async runtime (Python 3.12 raises
    "Cannot run the event loop while another loop is running"), and the
    peer map is the deterministic source of truth for the MVP revert.
    """
    sector = getattr(state.company, "sector", None) if state.company else None
    peers = _gics_peers_for(state.ticker, sector)
    method = "gics"

    # Simple competitive score: 50 baseline, +/- per peer P/E difference.
    # The GICS_PEERS stub Competitors have no P/E so peer_pes is empty and
    # the score stays at 50; this is fine for the MVP revert (no real peer
    # data flowing through).
    target_pe = state.market.pe_ratio if state.market and state.market.pe_ratio else None
    score = 50
    if target_pe is not None and peers:
        peer_pes = [p.pe_ratio for p in peers if p.pe_ratio is not None]
        if peer_pes:
            median_pe = sorted(peer_pes)[len(peer_pes) // 2]
            if median_pe > 0:
                ratio = target_pe / median_pe
                # Lower P/E than peers = better value → higher score
                score = max(0, min(100, int(50 + (1.0 - ratio) * 50)))

    return CompetitorAnalysis(
        target_ticker=state.ticker,
        competitors=peers,
        industry_rank=1,
        industry_size=max(10, len(peers) + 1),
        competitive_score=score,
        strengths=[],
        weaknesses=[],
        method=method,
    )


def _default_risk_subscores(state: "AnalysisState") -> list[RiskScore]:
    """6-category risk subscores derived from data (sub-project-3 revert)."""
    fin_score = 50
    if state.financial and state.financial.balance_sheets:
        bs = state.financial.balance_sheets[0]
        if bs.total_assets and bs.total_assets > 0:
            debt_ratio = float(bs.total_liabilities / bs.total_assets * 100)
            # Lower debt → lower financial risk
            fin_score = max(0, min(100, int(100 - debt_ratio)))
    mkt_score = 50
    if state.market and state.market.beta is not None:
        # Higher beta → higher market risk
        mkt_score = max(0, min(100, int(abs(state.market.beta) * 50)))
    sentiment_score = 50
    if state.news and state.news.sentiment_score is not None:
        # Negative sentiment → higher risk
        sentiment_score = max(0, min(100, int(50 - state.news.sentiment_score * 50)))
    return [
        RiskScore(
            category="financial",
            score=fin_score,
            rationale=f"Debt-to-asset ratio suggests {fin_score}/100 financial risk",
            evidence=[],
        ),
        RiskScore(
            category="market",
            score=mkt_score,
            rationale=f"Beta-implied market risk: {mkt_score}/100",
            evidence=[],
        ),
        RiskScore(
            category="operational",
            score=50,
            rationale="Default neutral (no operational data)",
            evidence=[],
        ),
        RiskScore(
            category="regulatory",
            score=50,
            rationale="Default neutral (no regulatory data)",
            evidence=[],
        ),
        RiskScore(
            category="governance",
            score=50,
            rationale="Default neutral (no governance data)",
            evidence=[],
        ),
        RiskScore(
            category="macro",
            score=sentiment_score,
            rationale=f"News-sentiment-implied macro risk: {sentiment_score}/100",
            evidence=[],
        ),
    ]


def _compute_risk_assessment(state: "AnalysisState") -> RiskAssessment:
    """Sub-project-3 revert: deterministic risk assessment from data."""
    sub_scores = _default_risk_subscores(state)
    total = int(sum(s.score for s in sub_scores) / len(sub_scores))
    # Level mapping: 0-25 low, 26-50 medium, 51-75 high, 76-100 extreme
    if total <= 25:
        level = "low"
    elif total <= 50:
        level = "medium"
    elif total <= 75:
        level = "high"
    else:
        level = "extreme"
    return RiskAssessment(
        ticker=state.ticker,
        total_score=total,
        level=level,  # validator normalizes case (already lowercase here)
        sub_scores=sub_scores,
        top_risks=[s.rationale for s in sorted(sub_scores, key=lambda x: -x.score)[:3]],
        method="weighted_sum_v1",
    )


def _compute_valuation(state: "AnalysisState") -> ValuationResult:
    """Sub-project-3 revert: deterministic DCF + relative valuation."""
    current = state.market.price if state.market else Decimal("0")
    pe = state.market.pe_ratio if state.market and state.market.pe_ratio else 20.0
    peer_pe_avg = 20.0
    relative_value = current * Decimal(str(peer_pe_avg / pe)) if pe > 0 else current

    fcf_data = (
        state.financial.cash_flows[0].free_cash_flow
        if state.financial and state.financial.cash_flows
        else None
    )
    growth_pct = state.market.revenue_growth_yoy if state.market else None
    growth_rate = (growth_pct / 100.0) if growth_pct is not None else 0.05
    shares_outstanding = (
        int(state.market.market_cap / state.market.price)
        if state.market and state.market.price and state.market.price > 0
        else 0
    )
    dcf_value = None
    if fcf_data is not None and fcf_data > 0 and shares_outstanding > 0:
        dcf_value = compute_dcf_value(
            fcf=fcf_data,
            growth_rate=growth_rate,
            shares_outstanding=shares_outstanding,
        )
    if dcf_value is not None and relative_value is not None:
        intrinsic = (dcf_value + relative_value) / 2
        method = "dcf_relative_peg"
    else:
        intrinsic = relative_value
        method = "relative_only"
    upside = float((intrinsic - current) / current) if current else 0.0
    return ValuationResult(
        ticker=state.ticker,
        intrinsic_value_per_share=intrinsic,
        current_price=current,
        upside_pct=round(upside, 4),
        dcf_value=dcf_value,
        relative_value=relative_value,
        peg_ratio=None,
        method=method,  # validator coerces unknown to "dcf_relative_peg"
        assumptions={"peer_pe_avg": peer_pe_avg, "growth_rate": growth_rate},
    )


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

    # --- Sub-project-3 revert: 3 analysis tasks produce text only. The Flow
    # computes competitor/risk/valuation deterministically in
    # ``synthesize_report`` (see ``_compute_competitor_analysis`` etc.). The
    # report_writer LLM produces a slim ``ReportWriterOutput`` (rating,
    # confidence, horizon, catalysts, markdown) which the Flow combines
    # with data + deterministic analyses to assemble the full
    # ``InvestmentReport``.
    state.writer_output = _extract_pydantic_field(
        tasks_output, 7, "report_writer", ReportWriterOutput, state
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
        # fresh data. Crew.kickoff is sync → wrap in _kickoff_sync +
        # to_thread so ``asyncio.wait_for`` can cancel mid-execution
        # (sub-2 deferred blocker #1: asyncio shutdown race).
        def _kickoff_sync() -> Any:
            return AnalysisCrew().kickoff(inputs={"ticker": normalized})

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_kickoff_sync),
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
        """Sub-project-3 revert: build the full ``InvestmentReport`` from data +
        deterministic competitor/risk/valuation + the LLM's
        ``ReportWriterOutput`` (rating, confidence, horizon, catalysts, markdown).

        On any synthesis failure, raise ``ReportGenerationError`` so the caller
        (FastAPI handler per spec §5.2) can return HTTP 500.
        """
        log.info("flow_step_started", step="synthesize_report", ticker=self.state.ticker)
        assert self.state.company is not None
        assert self.state.news is not None
        assert self.state.financial is not None

        # §3.2: market may be None (degraded) — substitute a minimal placeholder
        # so InvestmentReport can still be constructed.
        market = self.state.market
        if market is None:
            market = MarketData(
                ticker=self.state.ticker,
                as_of=datetime.utcnow(),
                price=Decimal("0"),
                change_pct=0.0,
                volume=0,
                market_cap=self.state.company.market_cap,
                source="degraded",
            )
            self.state.market = market

        # Deterministic 3 analyses. These replace the deleted LLM-driven paths.
        self.state.competitor = _compute_competitor_analysis(self.state)
        self.state.risk = _compute_risk_assessment(self.state)
        self.state.valuation = _compute_valuation(self.state)

        # LLM synthesis (rating, confidence, horizon, catalysts, markdown).
        # If the LLM failed to produce a ReportWriterOutput, fall back to
        # conservative defaults so the frontend can still render something.
        wo = self.state.writer_output
        if wo is None:
            log.warning("writer_output_missing", ticker=self.state.ticker)
            wo = ReportWriterOutput(
                rating="Hold",
                confidence=None,
                investment_horizon="medium",
                catalysts=[],
                markdown=(
                    f"## {self.state.ticker} 投资研究报告\n\n"
                    "报告合成器未能从数据中提取完整结论。"
                    f"请参考风险评级 ({self.state.risk.level}) 和估值结果 "
                    f"(${self.state.valuation.intrinsic_value_per_share}) 判断投资价值。"
                ),
            )
            self.state.errors.append("writer_output_unavailable")

        try:
            health_score = compute_financial_health(self.state.financial)

            self.state.report = InvestmentReport(
                report_id=str(uuid.uuid4()),
                ticker=self.state.ticker,
                generated_at=datetime.utcnow(),
                data_as_of={
                    "market": market.as_of,
                    "news": self.state.news.as_of,
                },
                company=self.state.company,
                market=market,
                financial=self.state.financial,
                financial_health_score=health_score,
                news=self.state.news,
                competitors=self.state.competitor,
                risk=self.state.risk,
                valuation=self.state.valuation,
                rating=wo.rating,
                confidence=wo.confidence,
                investment_horizon=wo.investment_horizon,
                catalysts=wo.catalysts,
                markdown=wo.markdown,
                sources=_collect_sources(
                    market,
                    self.state.news,
                    self.state.financial,
                    self.state.competitor,
                ),
                disclaimer=DISCLAIMER_TEXT,
            )
            log.info(
                "flow_step_completed",
                step="synthesize_report",
                ticker=self.state.ticker,
                report_id=self.state.report.report_id,
                rating=wo.rating,
                confidence=wo.confidence,
                health_score=health_score,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.error(
                "flow_step_failed",
                step="synthesize_report",
                ticker=self.state.ticker,
                error=str(exc),
            )
            raise ReportGenerationError(
                f"Failed to synthesize report for {self.state.ticker}: {exc}"
            ) from exc

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

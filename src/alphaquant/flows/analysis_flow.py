"""AnalysisFlow: thin shell wrapping the AnalysisCrew (sub-project 1).

The deterministic 6-step Flow has been collapsed into two steps:

1. ``run_crew`` (``@start``) — pre-fetch all 4 raw data sources via
   :class:`DataSourceRegistry`, invoke :class:`AnalysisCrew.kickoff` inside
   :func:`asyncio.to_thread` with a per-step timeout, then dispatch to
   :func:`parse_crew_output` to fill the downstream state fields.
2. ``synthesize_report`` (``@listen(run_crew)``) — assemble the
   :class:`InvestmentReport` from the populated state.

Sub-project 1 keeps the crew as a structural shell. Agents run end-to-end
but their JSON outputs are normalized by ``parse_crew_output`` so the
existing deterministic fallback logic still drives competitor / risk /
valuation sub-scores. The visible result is byte-for-byte identical to the
pre-refactor Flow. Sub-project 3+ will let agents produce real structured
outputs and gradually remove the fallback paths.
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
from alphaquant.models.competitor import Competitor, CompetitorAnalysis
from alphaquant.models.financial import FinancialStatements
from alphaquant.models.market import MarketData
from alphaquant.models.news import NewsAnalysis, NewsItem
from alphaquant.models.report import InvestmentReport
from alphaquant.models.risk import RiskAssessment, RiskScore
from alphaquant.models.valuation import ValuationResult
from alphaquant.observability import get_logger
from alphaquant.scoring import financial_health, risk_score
from alphaquant.scoring.rating import determine_rating

log = get_logger("alphaquant.flows.analysis_flow")

# §3.4: whole-Flow timeout (per spec).
FLOW_TIMEOUT_SECONDS = 120.0


# §3.2: GICS peer fallback map. Each sector maps to 3 well-known US large-cap
# tickers used when the competitor tool returns nothing. If the sector isn't
# in the map, callers fall back to SPY (market-only) peers.
GICS_PEERS: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "GOOGL"],
    "Financial": ["JPM", "BAC", "WFC"],
    "Healthcare": ["JNJ", "PFE", "UNH"],
    "Energy": ["XOM", "CVX", "COP"],
    "Consumer": ["WMT", "PG", "COST"],
    "Communication": ["META", "NFLX", "DIS"],
}


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


def _gics_peers_for(company: Any, ticker: str) -> list[Competitor]:
    """Build 3 GICS-fallback Competitor entries for a given company."""
    sector = getattr(company, "sector", None) if company else None
    tickers = GICS_PEERS.get(sector or "", [])
    if not tickers or ticker.upper() in [t.upper() for t in tickers]:
        # Sector unknown OR target ticker is itself in the peer list —
        # use a market-only fallback so we still emit 3 peers.
        tickers = ["SPY", "SPY", "SPY"]
    return [
        Competitor(
            ticker=t,
            name=t,
            market_cap=0,
            revenue_ttm=Decimal("0"),
        )
        for t in tickers
    ]


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


def _default_risk_subscores(state: "AnalysisState") -> list[RiskScore]:
    """Sub-project 1 fallback: same risk subscores the deterministic Flow uses."""
    fin_score = 5
    if state.financial and state.financial.balance_sheets:
        bs = state.financial.balance_sheets[0]
        debt_ratio = float(bs.total_liabilities / bs.total_assets * 100) if bs.total_assets else 50
        fin_score = min(10, max(0, int(debt_ratio / 10)))
    mkt_score = 5
    if state.market and state.market.beta is not None:
        mkt_score = min(10, max(0, int(abs(state.market.beta) * 5)))
    return [
        RiskScore(
            category="financial",
            score=fin_score,
            rationale=f"Debt ratio suggests {fin_score}/10 financial risk",
            evidence=[],
        ),
        RiskScore(
            category="market",
            score=mkt_score,
            rationale=f"Beta-implied market risk: {mkt_score}/10",
            evidence=[],
        ),
        RiskScore(category="operational", score=5, rationale="Default neutral", evidence=[]),
        RiskScore(category="regulatory", score=5, rationale="Default neutral", evidence=[]),
        RiskScore(category="governance", score=5, rationale="Default neutral", evidence=[]),
        RiskScore(category="macro", score=5, rationale="Default neutral", evidence=[]),
    ]


def parse_crew_output(
    result: Any, state: "AnalysisState" | None = None
) -> dict[str, Any]:
    """Extract agent outputs from ``CrewOutput`` and (optionally) fill state.

    Sub-project 1: each task output's ``raw`` text is assumed to be JSON. We
    parse it and dispatch by task order into the per-key fallback logic.
    Sub-project 3 will let agents produce structured Pydantic outputs instead.

    Returns a ``{role_key: parsed_data}`` mapping so callers (and tests) can
    inspect what was extracted. When ``state`` is provided, this function
    ALSO mutates ``state`` in place, populating the ``competitor``,
    ``risk``, and ``valuation`` fields via the same deterministic fallback
    the legacy Flow used.
    """
    tasks_output = getattr(result, "tasks_output", []) or []
    extracted: dict[str, Any] = {}

    # Build a {key: data} lookup from the actual tasks, indexed by role_key.
    by_key: dict[str, dict[str, Any]] = {}
    for idx, task_out in enumerate(tasks_output):
        if idx >= len(_TASK_KEYWORDS):
            break
        key = _TASK_KEYWORDS[idx]
        raw = getattr(task_out, "raw", "") or ""
        try:
            data = json.loads(raw) if raw.strip().startswith("{") else {}
        except (json.JSONDecodeError, ValueError):
            data = {}
        by_key[key] = data
        extracted[key] = data

    # If no state was provided, we only collect the parsed data.
    if state is None:
        return extracted

    # Always populate competitor / risk / valuation with the deterministic
    # fallback (matching pre-refactor Flow behavior). The agent's JSON output,
    # when present, overrides the peer list / sub-scores; the math stays the
    # same. Sub-project 1 keeps this; sub-project 3 will let agents produce
    # fully structured Pydantic outputs and remove the fallback paths.
    _populate_competitor(state, by_key.get("competitor_analyst") or {})
    _populate_risk(state, by_key.get("risk_analyst") or {})
    _populate_valuation(state, by_key.get("valuation_analyst") or {})

    return extracted


def _populate_competitor(state: "AnalysisState", data: dict[str, Any]) -> None:
    from alphaquant.scoring import competitive as scoring_competitive

    peers_raw = data.get("peers", [])
    peers: list[Competitor] = []
    for p in peers_raw[:5]:
        try:
            peers.append(Competitor(**p))
        except Exception:
            continue
    if not peers and state.company is not None:
        peers = _gics_peers_for(state.company, state.ticker)
    target_metrics = {
        "market_cap": float(state.market.market_cap if state.market else 0),
        "revenue_growth_yoy": float(
            state.market.revenue_growth_yoy
            if state.market and state.market.revenue_growth_yoy
            else 0
        ),
        "gross_margin": 0,
        "net_margin": 0,
    }
    score = scoring_competitive.compute(target_metrics, peers)
    state.competitor = CompetitorAnalysis(
        target_ticker=state.ticker,
        competitors=peers,
        industry_rank=1,
        industry_size=max(10, len(peers) + 1),
        competitive_score=score,
        strengths=[],
        weaknesses=[],
        method="computed" if peers_raw else "fallback",
    )


def _populate_risk(state: "AnalysisState", data: dict[str, Any]) -> None:
    from alphaquant.scoring import risk_score as scoring_risk

    sub_scores_data = data.get("sub_scores", [])
    sub_scores: list[RiskScore] = (
        [
            RiskScore(
                category=s["category"],
                score=s["score"],
                rationale=s.get("rationale", ""),
                evidence=[],
            )
            for s in sub_scores_data
        ]
        if sub_scores_data
        else _default_risk_subscores(state)
    )
    total = scoring_risk.compute(sub_scores)
    level = scoring_risk.determine_level(total)
    state.risk = RiskAssessment(
        ticker=state.ticker,
        total_score=total,
        level=level,
        sub_scores=sub_scores,
        top_risks=[s.rationale for s in sub_scores[:3]],
    )


def _populate_valuation(state: "AnalysisState", data: dict[str, Any]) -> None:
    # Sub-project 1: deterministic fallback (current Flow logic).
    from alphaquant.scoring.dcf import compute_dcf_value

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
        if state.market and state.market.price > 0
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
    state.valuation = ValuationResult(
        ticker=state.ticker,
        intrinsic_value_per_share=intrinsic,
        current_price=current,
        upside_pct=round(upside, 4),
        dcf_value=dcf_value,
        relative_value=relative_value,
        peg_ratio=None,
        method=method,
        assumptions={"peer_pe_avg": peer_pe_avg},
    )


class AnalysisFlow(Flow[AnalysisState]):
    """Top-level Flow: 2-step thin shell wrapping AnalysisCrew."""

    @start()
    async def run_crew(
        self,
        ticker: str | None = None,
        crewai_trigger_payload: dict[str, Any] | None = None,
    ) -> None:
        """Step 1: Pre-fetch raw data via DataSourceRegistry, then drive the
        8-agent AnalysisCrew to produce analysis results. Sub-project 1
        keeps the crew as a structural shell; sub-project 3+ will let
        agents do real reasoning.
        """
        from alphaquant.models.company import Company

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

        # Pre-fetch all 4 raw data sources. The crew gets these as inputs.
        registry = DataSourceRegistry()
        try:
            company, market, news, financial = await asyncio.wait_for(
                asyncio.gather(
                    registry.get_company(normalized),
                    registry.get_market(normalized),
                    registry.get_news(normalized),
                    registry.get_financial(normalized),
                    return_exceptions=True,
                ),
                timeout=45.0,
            )
        except asyncio.TimeoutError:
            log.error("data_fetch_timeout", ticker=normalized)
            raise

        # Map exceptions → None (degraded mode).
        self.state.company = company if isinstance(company, Company) else None
        self.state.market = market if isinstance(market, MarketData) else None
        if isinstance(news, NewsAnalysis):
            self.state.news = news
        elif isinstance(news, list):
            self.state.news = _news_items_to_analysis(news, normalized)
        else:
            self.state.news = NewsAnalysis.empty(normalized)
        self.state.financial = (
            financial
            if isinstance(financial, FinancialStatements)
            else FinancialStatements(ticker=normalized)
        )

        if self.state.company is None:
            self.state.errors.append("company_data_unavailable")
        if self.state.market is None:
            self.state.errors.append("market_data_unavailable")
        if self.state.news.total_count == 0:
            self.state.errors.append("news_data_unavailable")
        if not isinstance(financial, FinancialStatements):
            self.state.errors.append("financial_data_unavailable")

        if self.state.company is None:
            raise AllDataSourcesDown(
                f"Cannot resolve {normalized}: company data unavailable"
            )

        # Drive the 8-agent crew. Crew.kickoff is sync → wrap in to_thread.
        crew = AnalysisCrew()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    crew.kickoff,
                    inputs={
                        "ticker": normalized,
                        "company": self.state.company.model_dump(mode="json") if self.state.company else None,
                        "market": self.state.market.model_dump(mode="json") if self.state.market else None,
                        "news": self.state.news.model_dump(mode="json") if self.state.news else None,
                        "financial": self.state.financial.model_dump(mode="json") if self.state.financial else None,
                    },
                ),
                timeout=FLOW_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            log.error("crew_timeout", ticker=normalized)
            raise

        # Parse crew output → fill self.state fields downstream tasks consume.
        parse_crew_output(result, self.state)

        log.info("flow_step_completed", step="run_crew", ticker=normalized)

    @listen(run_crew)
    async def synthesize_report(self) -> None:
        """Step 2: Synthesize InvestmentReport from crew-driven state.

        On any synthesis failure, raise ``ReportGenerationError`` so the
        caller (FastAPI handler per spec §5.2) can return HTTP 500.
        """
        log.info("flow_step_started", step="synthesize_report", ticker=self.state.ticker)
        assert self.state.company is not None
        assert self.state.news is not None
        assert self.state.financial is not None
        assert self.state.competitor is not None
        assert self.state.risk is not None
        assert self.state.valuation is not None

        # §3.2: market may be None (degraded) — substitute a minimal placeholder
        # so InvestmentReport.markdown and downstream consumers can render.
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

        try:
            rating, confidence = determine_rating(
                self.state.valuation, self.state.risk, self.state.news
            )
            health_score = financial_health.compute(self.state.financial)

            markdown = _build_markdown(
                self.state.company,
                market,
                self.state.financial,
                self.state.news,
                self.state.competitor,
                self.state.risk,
                self.state.valuation,
                rating,
                confidence,
                health_score,
            )

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
                rating=rating,
                confidence=confidence,
                catalysts=[],
                markdown=markdown,
                sources=_collect_sources(
                    market,
                    self.state.news,
                    self.state.financial,
                    self.state.competitor,
                ),
            )
            log.info(
                "flow_step_completed",
                step="synthesize_report",
                ticker=self.state.ticker,
                report_id=self.state.report.report_id,
                rating=rating,
                confidence=confidence,
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


def _build_markdown(
    company, market, financial, news, competitors, risk, valuation, rating, confidence, health_score
) -> str:
    upside_pct = valuation.upside_pct * 100
    return f"""# {company.ticker} {company.name} 投资研究报告

> **生成时间**: {datetime.utcnow().isoformat()}
> **免责声明**: 本报告由 AI 自动生成，仅供参考，不构成任何投资建议。投资有风险，决策需谨慎。

## 执行摘要
- **投资评级**: {rating} (置信度 {confidence}%)
- **当前价**: ${market.price} → 内在价值 ${valuation.intrinsic_value_per_share or market.price}
- **潜在涨跌**: {upside_pct:+.1f}%

## 公司概览
- 行业: {company.sector} / {company.industry}
- 交易所: {company.exchange}
- 市值: ${company.market_cap:,}

## 市场分析
- 当前价: ${market.price}
- 市盈率 (P/E): {market.pe_ratio or 'N/A'}
- 52 周区间: ${market.low_52w or 'N/A'} - ${market.high_52w or 'N/A'}
- Beta: {market.beta or 'N/A'}

## 财务分析
- 财务健康评分: {health_score}/100
- 数据源: {financial.source}

## 新闻情绪
- 情绪评分: {news.sentiment_score:.2f} (-1 ~ +1)
- 正面 {news.positive_pct:.0%} | 负面 {news.negative_pct:.0%} | 中性 {news.neutral_pct:.0%}

## 竞争对手
- 行业排名: #{competitors.industry_rank}/{competitors.industry_size}
- 竞争力评分: {competitors.competitive_score}/100
- 对标公司数: {len(competitors.competitors)}

## 风险评估
- 风险等级: {risk.level}
- 总分: {risk.total_score}/100

## 估值与建议
- 方法: {valuation.method}
- 内在价值: ${valuation.intrinsic_value_per_share or 'N/A'}
- 假设: {valuation.assumptions}

---
*本报告由 AI 自动生成，仅供参考，不构成任何投资建议。*"""

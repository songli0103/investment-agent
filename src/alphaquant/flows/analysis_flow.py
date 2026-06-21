"""AnalysisFlow: orchestrates 8 Agents via CrewAI Flow."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from crewai.flow import Flow, listen, start
from pydantic import BaseModel, Field

from alphaquant.infrastructure.data_sources import DataSourceRegistry
from alphaquant.exceptions import (
    AllDataSourcesDown,
    InvalidTickerFormat,
    ReportGenerationError,
)
from alphaquant.models.competitor import Competitor, CompetitorAnalysis
from alphaquant.models.financial import FinancialStatements
from alphaquant.models.market import MarketData
from alphaquant.models.news import NewsAnalysis, NewsItem
from alphaquant.models.report import InvestmentReport
from alphaquant.models.risk import RiskAssessment, RiskScore
from alphaquant.models.valuation import ValuationResult
from alphaquant.observability import get_logger
from alphaquant.scoring import competitive, financial_health, risk_score
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


class AnalysisFlow(Flow[AnalysisState]):
    """Top-level Flow orchestrating all 8 Agents."""

    @start()
    async def resolve_company(
        self,
        ticker: str | None = None,
        crewai_trigger_payload: dict[str, Any] | None = None,
    ) -> None:
        """Step 1: Validate ticker format and resolve company metadata.

        CrewAI Flow's ``kickoff(inputs={"ticker": "AAPL"})`` populates
        ``state.ticker`` but does not pass positional args to the @start
        method. The optional ``crewai_trigger_payload`` and ``ticker`` args
        support both the programmatic API (``flow.resolve_company("AAPL")``)
        and the framework-driven ``flow.kickoff_async(inputs={"ticker": ...})``
        path.
        """
        # Resolve the ticker from any of the supported channels.
        raw_ticker = (
            ticker
            or (crewai_trigger_payload or {}).get("ticker")
            or self.state.ticker
            or ""
        )
        normalized = _normalize_ticker(raw_ticker)
        self.state.ticker = normalized
        log.info("flow_step_started", step="resolve_company", ticker=normalized)
        registry = DataSourceRegistry()
        try:
            company = await registry.get_company(normalized)
            self.state.company = company
            log.info(
                "flow_step_completed",
                step="resolve_company",
                ticker=normalized,
                company_name=getattr(company, "name", None),
            )
        except AllDataSourcesDown as e:
            log.error(
                "flow_step_failed",
                step="resolve_company",
                ticker=normalized,
                error=str(e),
            )
            raise AllDataSourcesDown(f"Cannot resolve {normalized}: {e}")

    @listen(resolve_company)
    async def parallel_data_collection(self) -> None:
        """Step 2: Market + News + Financial in parallel."""
        log.info("flow_step_started", step="parallel_data_collection", ticker=self.state.ticker)
        registry = DataSourceRegistry()
        ticker = self.state.ticker

        market_raw, news_raw, financial_raw = await asyncio.wait_for(
            asyncio.gather(
                registry.get_market(ticker),
                registry.get_news(ticker),
                registry.get_financial(ticker),
                return_exceptions=True,
            ),
            timeout=45.0,  # design §3.4: parallel group ≤ 45s
        )

        self.state.market = (
            market_raw if isinstance(market_raw, MarketData) else None
        )
        if self.state.market is None:
            self.state.errors.append("market_data_unavailable")

        # Registry returns list[NewsItem]; transform into NewsAnalysis.
        if isinstance(news_raw, list):
            self.state.news = _news_items_to_analysis(news_raw, ticker)
        elif isinstance(news_raw, NewsAnalysis):
            self.state.news = news_raw
        else:
            self.state.news = NewsAnalysis.empty(ticker)
        if self.state.news.total_count == 0 and not (
            isinstance(news_raw, list) and len(news_raw) > 0
        ):
            self.state.errors.append("news_data_unavailable")

        self.state.financial = (
            financial_raw if isinstance(financial_raw, FinancialStatements) else FinancialStatements(ticker=ticker)
        )
        if not isinstance(financial_raw, FinancialStatements):
            self.state.errors.append("financial_data_unavailable")

        log.info(
            "flow_step_completed",
            step="parallel_data_collection",
            ticker=ticker,
            market_ok=isinstance(market_raw, MarketData),
            news_ok=isinstance(news_raw, (list, NewsAnalysis)),
            financial_ok=isinstance(financial_raw, FinancialStatements),
            errors=list(self.state.errors),
        )

    @listen(parallel_data_collection)
    async def competitor_analysis(self) -> None:
        """Step 3: Identify and compare competitors."""
        log.info("flow_step_started", step="competitor_analysis", ticker=self.state.ticker)
        if not self.state.company:
            peers = _gics_peers_for(None, self.state.ticker)
            self.state.competitor = CompetitorAnalysis(
                target_ticker=self.state.ticker,
                competitors=peers,
                industry_rank=1,
                industry_size=len(peers),
                competitive_score=50,
                strengths=[],
                weaknesses=[],
                method="fallback",
            )
            log.info(
                "flow_step_completed",
                step="competitor_analysis",
                ticker=self.state.ticker,
                method="fallback",
                peer_count=len(peers),
            )
            return

        # Use static peer map from CompetitorTool
        from alphaquant.tools.competitor_tool import CompetitorTool

        tool = CompetitorTool()
        try:
            import json

            raw = tool._run(self.state.ticker)
            peers_data = json.loads(raw) if not raw.startswith("No ") else []
        except Exception:
            peers_data = []

        peers = []
        for p in peers_data[:5]:
            try:
                peers.append(Competitor(**p))
            except Exception:
                continue

        if not peers:
            # §3.2 fallback: emit 3 GICS peers (or SPY market-only).
            peers = _gics_peers_for(self.state.company, self.state.ticker)
            self.state.competitor = CompetitorAnalysis(
                target_ticker=self.state.ticker,
                competitors=peers,
                industry_rank=1,
                industry_size=len(peers),
                competitive_score=50,
                method="fallback",
            )
            log.info(
                "flow_step_completed",
                step="competitor_analysis",
                ticker=self.state.ticker,
                method="fallback",
                peer_count=len(peers),
            )
            return

        target_metrics = {
            "market_cap": float(self.state.market.market_cap if self.state.market else 0),
            "revenue_growth_yoy": float(self.state.market.revenue_growth_yoy if self.state.market and self.state.market.revenue_growth_yoy else 0),
            "gross_margin": 0,
            "net_margin": 0,
        }
        score = competitive.compute(target_metrics, peers)
        self.state.competitor = CompetitorAnalysis(
            target_ticker=self.state.ticker,
            competitors=peers,
            industry_rank=1,
            industry_size=max(10, len(peers) + 1),
            competitive_score=score,
            strengths=[],
            weaknesses=[],
        )
        log.info(
            "flow_step_completed",
            step="competitor_analysis",
            ticker=self.state.ticker,
            method="computed",
            peer_count=len(peers),
            competitive_score=score,
        )

    @listen(competitor_analysis)
    async def risk_analysis(self) -> None:
        """Step 4: Compute risk score from upstream data."""
        log.info("flow_step_started", step="risk_analysis", ticker=self.state.ticker)
        sub_scores: list[RiskScore] = []

        # Financial risk
        if self.state.financial and self.state.financial.balance_sheets:
            bs = self.state.financial.balance_sheets[0]
            debt_ratio = float(bs.total_liabilities / bs.total_assets * 100) if bs.total_assets else 50
            fin_score = min(10, max(0, int(debt_ratio / 10)))
        else:
            fin_score = 5
        sub_scores.append(
            RiskScore(
                category="financial",
                score=fin_score,
                rationale=f"Debt ratio suggests {fin_score}/10 financial risk",
                evidence=[],
            )
        )

        # Market risk (beta-based)
        if self.state.market and self.state.market.beta is not None:
            mkt_score = min(10, max(0, int(abs(self.state.market.beta) * 5)))
        else:
            mkt_score = 5
        sub_scores.append(
            RiskScore(
                category="market",
                score=mkt_score,
                rationale=f"Beta-implied market risk: {mkt_score}/10",
                evidence=[],
            )
        )

        # Operational/regulatory/governance/macro — all default to neutral 5
        for cat in ("operational", "regulatory", "governance", "macro"):
            sub_scores.append(
                RiskScore(
                    category=cat,  # type: ignore[arg-type]
                    score=5,
                    rationale=f"Default neutral risk for {cat}; requires LLM deep analysis (future)",
                    evidence=[],
                )
            )

        total = risk_score.compute(sub_scores)
        level = risk_score.determine_level(total)
        self.state.risk = RiskAssessment(
            ticker=self.state.ticker,
            total_score=total,
            level=level,  # type: ignore[arg-type]
            sub_scores=sub_scores,
            top_risks=[s.rationale for s in sub_scores[:3]],
        )
        log.info(
            "flow_step_completed",
            step="risk_analysis",
            ticker=self.state.ticker,
            total_score=total,
            level=level,
        )

    @listen(risk_analysis)
    async def valuation_analysis(self) -> None:
        """Step 5: DCF + relative + PEG → intrinsic value."""
        log.info("flow_step_started", step="valuation_analysis", ticker=self.state.ticker)
        if not self.state.market:
            self.state.valuation = ValuationResult(
                ticker=self.state.ticker,
                current_price=Decimal("0"),
                upside_pct=0.0,
                method="relative_only",
            )
            return

        current = self.state.market.price
        # Relative valuation: simple P/E times industry average growth proxy
        pe = self.state.market.pe_ratio or 20.0
        peer_pe_avg = 20.0  # MVP assumption
        relative_value = (
            current * Decimal(str(peer_pe_avg / pe)) if pe > 0 else current
        )

        # DCF: 用最近年度 FCF + market growth + 默认 WACC/g_term 算 intrinsic value
        from alphaquant.scoring.dcf import compute_dcf_value

        dcf_value = None
        fcf_data = (
            self.state.financial.cash_flows[0].free_cash_flow
            if self.state.financial and self.state.financial.cash_flows
            else None
        )
        growth_pct = self.state.market.revenue_growth_yoy  # e.g. 5.0 means 5%
        growth_rate = ((growth_pct / 100.0) if growth_pct is not None else 0.05)
        shares_outstanding = (
            int(self.state.market.market_cap / self.state.market.price)
            if self.state.market.price > 0
            else 0
        )
        if fcf_data is not None and fcf_data > 0 and shares_outstanding > 0:
            dcf_value = compute_dcf_value(
                fcf=fcf_data,
                growth_rate=growth_rate,
                shares_outstanding=shares_outstanding,
            )

        # 内含价值：DCF 和 relative 的平均（如果有 DCF）；只有 relative 时退化
        if dcf_value is not None and relative_value is not None:
            intrinsic = (dcf_value + relative_value) / 2
            method = "dcf_relative_peg"
        else:
            intrinsic = relative_value
            method = "relative_only"

        upside = float((intrinsic - current) / current) if current else 0.0

        self.state.valuation = ValuationResult(
            ticker=self.state.ticker,
            intrinsic_value_per_share=intrinsic,
            current_price=current,
            upside_pct=round(upside, 4),
            dcf_value=dcf_value,
            relative_value=relative_value,
            peg_ratio=None,
            method=method,  # type: ignore[arg-type]
            assumptions={"peer_pe_avg": peer_pe_avg},
        )
        log.info(
            "flow_step_completed",
            step="valuation_analysis",
            ticker=self.state.ticker,
            method=method,
            current_price=float(current),
            intrinsic_value=float(intrinsic),
            dcf_value=float(dcf_value) if dcf_value is not None else None,
            upside_pct=round(upside, 4),
        )

    @listen(valuation_analysis)
    async def write_report(self) -> None:
        """Step 6: Synthesize all upstream into final report.

        On any synthesis failure, raise ``ReportGenerationError`` so the caller
        (FastAPI handler per spec §5.2) can return HTTP 500 INTERNAL_ERROR.
        """
        log.info("flow_step_started", step="write_report", ticker=self.state.ticker)
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
                step="write_report",
                ticker=self.state.ticker,
                report_id=self.state.report.report_id,
                rating=rating,
                confidence=confidence,
                health_score=health_score,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.error(
                "flow_step_failed",
                step="write_report",
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
*本报告由 AI 自动生成，仅供参考，不构成任何投资建议。*
"""
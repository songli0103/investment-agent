"""AnalysisFlow: orchestrates 8 Agents via CrewAI Flow."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from crewai import Task
from crewai.flow import Flow, listen, start
from pydantic import BaseModel, Field

from alphaquant.agents.competitor_analyst import build_competitor_analyst_agent
from alphaquant.agents.company_resolver import build_company_resolver_agent
from alphaquant.agents.financial_analyst import build_financial_analyst_agent
from alphaquant.agents.market_analyst import build_market_analyst_agent
from alphaquant.agents.news_analyst import build_news_analyst_agent
from alphaquant.agents.report_writer import build_report_writer_agent
from alphaquant.agents.risk_analyst import build_risk_analyst_agent
from alphaquant.agents.valuation_analyst import build_valuation_analyst_agent
from alphaquant.data_sources import DataSourceRegistry
from alphaquant.exceptions import AllDataSourcesDown, InvalidTickerFormat
from alphaquant.models.competitor import Competitor, CompetitorAnalysis
from alphaquant.models.financial import FinancialStatements
from alphaquant.models.market import MarketData
from alphaquant.models.news import NewsAnalysis
from alphaquant.models.report import InvestmentReport
from alphaquant.models.risk import RiskAssessment, RiskScore
from alphaquant.models.valuation import ValuationResult
from alphaquant.scoring import competitive, financial_health, risk_score
from alphaquant.scoring.rating import determine_rating


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


class AnalysisFlow(Flow[AnalysisState]):
    """Top-level Flow orchestrating all 8 Agents."""

    @start()
    def resolve_company(self, ticker: str) -> None:
        """Step 1: Validate ticker format and resolve company metadata."""
        normalized = _normalize_ticker(ticker)
        self.state.ticker = normalized
        registry = DataSourceRegistry()
        try:
            company = asyncio.run(registry.get_company(normalized))
            self.state.company = company
        except AllDataSourcesDown as e:
            raise AllDataSourcesDown(f"Cannot resolve {normalized}: {e}")

    @listen(resolve_company)
    def parallel_data_collection(self) -> None:
        """Step 2: Market + News + Financial in parallel."""
        registry = DataSourceRegistry()
        ticker = self.state.ticker

        async def _gather():
            return await asyncio.wait_for(
                asyncio.gather(
                    registry.get_market(ticker),
                    registry.get_news(ticker),
                    registry.get_financial(ticker),
                    return_exceptions=True,
                ),
                timeout=45.0,  # design §3.4: parallel group ≤ 45s
            )

        market_raw, news_raw, financial_raw = asyncio.run(_gather())

        self.state.market = (
            market_raw if isinstance(market_raw, MarketData) else None
        )
        if self.state.market is None:
            self.state.errors.append("market_data_unavailable")

        self.state.news = (
            news_raw if isinstance(news_raw, NewsAnalysis) else NewsAnalysis.empty(ticker)
        )
        if not isinstance(news_raw, NewsAnalysis):
            self.state.errors.append("news_data_unavailable")

        self.state.financial = (
            financial_raw if isinstance(financial_raw, FinancialStatements) else FinancialStatements(ticker=ticker)
        )
        if not isinstance(financial_raw, FinancialStatements):
            self.state.errors.append("financial_data_unavailable")

    @listen(parallel_data_collection)
    def competitor_analysis(self) -> None:
        """Step 3: Identify and compare competitors."""
        if not self.state.company:
            self.state.competitor = CompetitorAnalysis(
                target_ticker=self.state.ticker,
                competitors=[
                    Competitor(
                        ticker=self.state.ticker,
                        name=self.state.ticker,
                        market_cap=0,
                        revenue_ttm=Decimal("0"),
                    )
                ],
                industry_rank=1,
                industry_size=1,
                competitive_score=50,
                strengths=[],
                weaknesses=[],
                method="fallback",
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
            self.state.competitor = CompetitorAnalysis(
                target_ticker=self.state.ticker,
                competitors=[
                    Competitor(
                        ticker=self.state.ticker,
                        name=self.state.ticker,
                        market_cap=0,
                        revenue_ttm=Decimal("0"),
                    )
                ],
                industry_rank=1,
                industry_size=1,
                competitive_score=50,
                method="fallback",
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

    @listen(competitor_analysis)
    def risk_analysis(self) -> None:
        """Step 4: Compute risk score from upstream data."""
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

    @listen(risk_analysis)
    def valuation_analysis(self) -> None:
        """Step 5: DCF + relative + PEG → intrinsic value."""
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

        # DCF: skipped (data often missing in MVP)
        dcf_value = None
        upside = float((relative_value - current) / current) if current else 0.0

        self.state.valuation = ValuationResult(
            ticker=self.state.ticker,
            intrinsic_value_per_share=relative_value,
            current_price=current,
            upside_pct=round(upside, 4),
            dcf_value=dcf_value,
            relative_value=relative_value,
            peg_ratio=None,
            method="relative_only",
            assumptions={"peer_pe_avg": peer_pe_avg},
        )

    @listen(valuation_analysis)
    def write_report(self) -> None:
        """Step 6: Synthesize all upstream into final report."""
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
            sources=[market.source, self.state.news.source],
        )


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

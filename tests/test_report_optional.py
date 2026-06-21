"""Tests for optional InvestmentReport.confidence field.

Sub-plan for confidence-rubric spec (b0308df). confidence becomes int | None
so the LLM can return null when it cannot justify a number.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from alphaquant.models.competitor import Competitor, CompetitorAnalysis
from alphaquant.models.company import Company
from alphaquant.models.financial import (
    BalanceSheet,
    FinancialStatements,
    IncomeStatement,
)
from alphaquant.models.market import MarketData
from alphaquant.models.news import NewsAnalysis
from alphaquant.models.report import InvestmentReport
from alphaquant.models.risk import RiskAssessment, RiskScore
from alphaquant.models.valuation import ValuationResult


def _stub_report(**overrides: Any) -> InvestmentReport:
    """Build a minimal valid InvestmentReport, allowing field overrides.

    Fields other than `confidence` use known-good values; the test only cares
    about confidence validation. Caller can override any field via kwargs.
    """
    company = Company(
        ticker="AAPL",
        name="Apple Inc.",
        exchange="NASDAQ",
        sector="Technology",
        industry="Consumer Electronics",
        market_cap=3_000_000_000_000,
    )
    market = MarketData(
        ticker="AAPL",
        as_of=datetime(2024, 1, 1, 0, 0, 0),
        price=Decimal("150.00"),
        change_pct=0.5,
        volume=50_000_000,
        market_cap=3_000_000_000_000,
    )
    financial = FinancialStatements(
        ticker="AAPL",
        income_statements=[
            IncomeStatement(
                period="TTM",
                fiscal_year=2024,
                revenue=Decimal("400000000000"),
                net_income=Decimal("100000000000"),
            ),
        ],
        balance_sheets=[
            BalanceSheet(
                period="Q4",
                fiscal_year=2024,
                total_assets=Decimal("350000000000"),
                total_liabilities=Decimal("280000000000"),
                total_equity=Decimal("70000000000"),
            ),
        ],
        source="yahoo",
    )
    news = NewsAnalysis(
        ticker="AAPL",
        as_of=datetime(2024, 1, 1, 0, 0, 0),
        total_count=0,
        positive_pct=0.0,
        negative_pct=0.0,
        neutral_pct=1.0,
        sentiment_score=0.0,
    )
    competitors = CompetitorAnalysis(
        target_ticker="AAPL",
        competitors=[
            Competitor(
                ticker="MSFT",
                name="Microsoft",
                market_cap=2_500_000_000_000,
                revenue_ttm=Decimal("200000000000"),
            ),
        ],
        industry_rank=1,
        industry_size=5,
        competitive_score=75,
        method="gics",
    )
    risk = RiskAssessment(
        ticker="AAPL",
        total_score=40,
        level="medium",
        sub_scores=[
            RiskScore(
                category="financial",
                score=4,
                rationale="placeholder for test",
                evidence=[],
            )
        ],
        top_risks=[],
    )
    valuation = ValuationResult(
        ticker="AAPL",
        intrinsic_value_per_share=Decimal("180.00"),
        current_price=Decimal("150.00"),
        upside_pct=0.2,
        method="relative_only",
    )

    base: dict[str, Any] = {
        "report_id": "00000000-0000-0000-0000-000000000000",
        "ticker": "AAPL",
        "generated_at": datetime(2024, 1, 1, 0, 0, 0),
        "data_as_of": {},
        "company": company,
        "market": market,
        "financial": financial,
        "financial_health_score": 70,
        "news": news,
        "competitors": competitors,
        "risk": risk,
        "valuation": valuation,
        "rating": "Hold",
        "investment_horizon": "medium",
        "catalysts": [],
        "markdown": "## Summary",
        "sources": [],
        "disclaimer": "test",
    }
    base.update(overrides)
    return InvestmentReport(**base)


class TestConfidenceOptional:
    def test_none_accepted(self):
        """Confidence can be None (LLM returns null when not justified)."""
        rep = _stub_report(confidence=None)
        assert rep.confidence is None

    def test_default_is_none(self):
        """When confidence is omitted, default is None (not ValidationError)."""
        rep = _stub_report()
        assert rep.confidence is None

    def test_zero_accepted(self):
        rep = _stub_report(confidence=0)
        assert rep.confidence == 0

    def test_hundred_accepted(self):
        rep = _stub_report(confidence=100)
        assert rep.confidence == 100

    def test_seventy_accepted(self):
        """Regression: numeric confidence values still work."""
        rep = _stub_report(confidence=70)
        assert rep.confidence == 70

    def test_negative_rejected(self):
        with pytest.raises(ValidationError):
            _stub_report(confidence=-1)

    def test_above_100_rejected(self):
        with pytest.raises(ValidationError):
            _stub_report(confidence=101)
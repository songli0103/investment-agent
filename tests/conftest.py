"""Shared test fixtures and helpers for the AlphaQuant test suite."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

import pytest
from crewai.llm import LLM as _CrewLLM

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


class _FakeLLM(_CrewLLM):
    """Stub LLM that records calls and returns deterministic text. Used in unit tests."""

    def __init__(self) -> None:
        super().__init__(model="fake/model", api_key="fake")
        self.calls: list[dict[str, Any]] = []

    def call(self, messages, *args, **kwargs):  # type: ignore[override]
        from pydantic import BaseModel
        self.calls.append({"messages": messages, "kwargs": kwargs})
        # If a response_format (Pydantic schema) is requested, return a fake JSON string
        # that satisfies the schema. This avoids hitting the network.
        response_format = kwargs.get("response_format")
        if response_format is not None and isinstance(response_format, type) and issubclass(response_format, BaseModel):
            try:
                instance = response_format.model_construct()
                return instance.model_dump_json()
            except Exception:
                return "{}"
        return "fake llm response"


@pytest.fixture()
def stub_report():
    """Return a factory that builds a fully-populated valid InvestmentReport.

    Usage::

        def test_x(stub_report):
            rep = stub_report(confidence=None, ticker="TEST")

    The returned callable accepts the same kwargs as `InvestmentReport` and
    overrides them on top of sensible defaults. Used by confidence-rubric tests
    (test_report_optional.py, test_db.py) so the nested-model boilerplate lives
    in one place. Other tests in test_db.py continue to use the local
    `_make_report` helper, which has different defaults.
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

    def _factory(**overrides: Any) -> InvestmentReport:
        merged = dict(base)
        merged.update(overrides)
        return InvestmentReport(**merged)

    return _factory
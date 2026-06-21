"""Tests for alphaquant.scoring functions."""
from __future__ import annotations

from decimal import Decimal

import pytest

from alphaquant.models.competitor import Competitor
from alphaquant.models.financial import (
    BalanceSheet,
    CashFlowStatement,
    FinancialStatements,
    IncomeStatement,
)
from alphaquant.models.risk import RiskScore
from alphaquant.scoring import competitive, dcf, financial_health, risk_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_income(
    revenue: Decimal,
    gross_profit: Decimal | None,
    net_income: Decimal,
    period: str = "TTM",
    fiscal_year: int = 2024,
) -> IncomeStatement:
    return IncomeStatement(
        period=period,
        fiscal_year=fiscal_year,
        revenue=revenue,
        gross_profit=gross_profit,
        net_income=net_income,
    )


def make_balance(
    total_assets: Decimal,
    total_liabilities: Decimal,
    total_equity: Decimal,
    fiscal_year: int = 2024,
) -> BalanceSheet:
    return BalanceSheet(
        period="FY",
        fiscal_year=fiscal_year,
        total_assets=total_assets,
        total_liabilities=total_liabilities,
        total_equity=total_equity,
    )


def make_cash(ocf: Decimal, fiscal_year: int = 2024) -> CashFlowStatement:
    return CashFlowStatement(
        period="TTM",
        fiscal_year=fiscal_year,
        operating_cash_flow=ocf,
    )


# ---------------------------------------------------------------------------
# Financial Health
# ---------------------------------------------------------------------------

class TestFinancialHealth:
    def test_empty_statements_returns_neutral(self):
        stmts = FinancialStatements(ticker="X")
        assert financial_health.compute(stmts) == 50

    def test_zero_revenue_returns_low(self):
        stmts = FinancialStatements(
            ticker="X",
            income_statements=[
                make_income(
                    revenue=Decimal("0"),
                    gross_profit=Decimal("0"),
                    net_income=Decimal("0"),
                )
            ],
        )
        # Only an income statement is present. With revenue=0:
        # profitability=0, growth=50 (only 1 stmt), solvency=50 (no BS),
        # cash_quality=50 (no cash_flows → early return default).
        # weighted: 0*0.35 + 50*0.25 + 50*0.20 + 50*0.20 = 32.5 → 32 (banker's)
        assert financial_health.compute(stmts) == 32

    def test_high_profitability_high_score(self):
        # Revenue 100, gross 50 (50% margin), net 20 (20% margin)
        # profitability: 50*1.5 + 20*2 = 75 + 40 = 115 → capped at 100
        # growth defaults to 50, solvency defaults to 50, cash_quality to 50
        stmts = FinancialStatements(
            ticker="X",
            income_statements=[
                make_income(
                    revenue=Decimal("100"),
                    gross_profit=Decimal("50"),
                    net_income=Decimal("20"),
                )
            ],
        )
        # weighted: 100*0.35 + 50*0.25 + 50*0.20 + 50*0.20 = 35 + 12.5 + 10 + 10 = 67.5 → 68
        assert financial_health.compute(stmts) == 68

    def test_growth_yoy(self):
        # Latest revenue 200, previous 100 → 100% growth → score 100
        stmts = FinancialStatements(
            ticker="X",
            income_statements=[
                make_income(
                    revenue=Decimal("200"),
                    gross_profit=Decimal("80"),
                    net_income=Decimal("20"),
                    fiscal_year=2024,
                ),
                make_income(
                    revenue=Decimal("100"),
                    gross_profit=Decimal("40"),
                    net_income=Decimal("10"),
                    fiscal_year=2023,
                ),
            ],
        )
        # growth score = min(100, 50 + 100) = 100
        # profitability: revenue=200, gp=80 → 40% margin; net 20 → 10% margin
        #   → 40*1.5 + 10*2 = 60+20 = 80
        # solvency default 50, cash default 50
        # weighted: 80*0.35 + 100*0.25 + 50*0.20 + 50*0.20 = 28 + 25 + 10 + 10 = 73
        assert financial_health.compute(stmts) == 73

    def test_no_growth_data_defaults_to_50(self):
        stmts = FinancialStatements(
            ticker="X",
            income_statements=[
                make_income(
                    revenue=Decimal("100"),
                    gross_profit=Decimal("30"),
                    net_income=Decimal("10"),
                )
            ],
        )
        # Only one income statement → growth score is 50
        # profitability: gp=30 → 30% margin, net=10 → 10% margin
        #   → 30*1.5 + 10*2 = 45 + 20 = 65
        # weighted: 65*0.35 + 50*0.25 + 50*0.20 + 50*0.20
        #   = 22.75 + 12.5 + 10 + 10 = 55.25 → 55
        assert financial_health.compute(stmts) == 55

    def test_negative_growth_caps_at_zero(self):
        # Latest revenue 50, previous 100 → -50% growth → 50-50 = 0
        stmts = FinancialStatements(
            ticker="X",
            income_statements=[
                make_income(
                    revenue=Decimal("50"),
                    gross_profit=Decimal("10"),
                    net_income=Decimal("0"),
                    fiscal_year=2024,
                ),
                make_income(
                    revenue=Decimal("100"),
                    gross_profit=Decimal("30"),
                    net_income=Decimal("10"),
                    fiscal_year=2023,
                ),
            ],
        )
        # growth = max(0, 50 + (-50)) = 0
        # profitability: gp=10, revenue=50 → 20% margin, net=0 → 0%
        #   → 20*1.5 + 0*2 = 30
        # weighted: 30*0.35 + 0*0.25 + 50*0.20 + 50*0.20 = 10.5 + 0 + 10 + 10 = 30.5 → 30 (banker's)
        assert financial_health.compute(stmts) == 30

    def test_solvency(self):
        # total_assets 100, total_liabilities 50 → debt ratio 50% → score 50
        stmts = FinancialStatements(
            ticker="X",
            income_statements=[
                make_income(
                    revenue=Decimal("100"),
                    gross_profit=Decimal("30"),
                    net_income=Decimal("10"),
                )
            ],
            balance_sheets=[
                make_balance(
                    total_assets=Decimal("100"),
                    total_liabilities=Decimal("50"),
                    total_equity=Decimal("50"),
                )
            ],
        )
        # profitability 65, growth 50, solvency 50, cash default 50
        # weighted: 65*0.35 + 50*0.25 + 50*0.20 + 50*0.20 = 55.25 → 55
        assert financial_health.compute(stmts) == 55

    def test_cash_quality(self):
        # OCF 150, NI 100 → ratio 150 → capped at 100
        stmts = FinancialStatements(
            ticker="X",
            income_statements=[
                make_income(
                    revenue=Decimal("100"),
                    gross_profit=Decimal("30"),
                    net_income=Decimal("100"),
                )
            ],
            cash_flows=[make_cash(ocf=Decimal("150"))],
        )
        # cash_quality = 100
        # profitability: gp=30 → 30% margin, net=100 → 100% margin → 30*1.5+100*2 = 245 → 100
        # growth default 50, solvency default 50
        # weighted: 100*0.35 + 50*0.25 + 50*0.20 + 100*0.20 = 35 + 12.5 + 10 + 20 = 77.5 → 78
        assert financial_health.compute(stmts) == 78

    def test_cash_quality_zero_ni(self):
        stmts = FinancialStatements(
            ticker="X",
            income_statements=[
                make_income(
                    revenue=Decimal("100"),
                    gross_profit=Decimal("30"),
                    net_income=Decimal("0"),
                )
            ],
            cash_flows=[make_cash(ocf=Decimal("50"))],
        )
        # cash_quality: ni=0 → 0
        # profitability: gp=30 → 30%, net=0 → 0% → 30*1.5+0*2 = 45
        # weighted: 45*0.35 + 50*0.25 + 50*0.20 + 0*0.20
        #   = 15.75 + 12.5 + 10 + 0 = 38.25 → 38
        assert financial_health.compute(stmts) == 38


# ---------------------------------------------------------------------------
# Risk Score
# ---------------------------------------------------------------------------

class TestRiskScore:
    def test_empty_subscores_returns_50(self):
        assert risk_score.compute([]) == 50

    def test_zero_subscores_returns_zero(self):
        subs = [
            RiskScore(category="financial", score=0, rationale="very low risk here"),
            RiskScore(category="operational", score=0, rationale="very low risk here"),
        ]
        assert risk_score.compute(subs) == 0

    def test_max_subscores(self):
        subs = [
            RiskScore(category="financial", score=10, rationale="max score category 1"),
            RiskScore(category="operational", score=10, rationale="max score category 2"),
        ]
        # 10*10*0.30 + 10*10*0.15 = 30 + 15 = 45
        assert risk_score.compute(subs) == 45

    def test_known_weights_sum(self):
        # Just verify known categories sum correctly.
        # financial only: 10*10*0.30 = 30
        subs = [
            RiskScore(category="financial", score=10, rationale="financial risk text"),
        ]
        assert risk_score.compute(subs) == 30

    def test_caps_at_100(self):
        # Construct a set that would exceed 100 if not capped
        subs = [
            RiskScore(category="financial", score=10, rationale="max text here"),
            RiskScore(category="operational", score=10, rationale="max text here"),
            RiskScore(category="market", score=10, rationale="max text here"),
            RiskScore(category="regulatory", score=10, rationale="max text here"),
            RiskScore(category="governance", score=10, rationale="max text here"),
            RiskScore(category="macro", score=10, rationale="max text here"),
        ]
        # 10*10*(0.30+0.15+0.15+0.15+0.10+0.15) = 100*1.0 = 100
        assert risk_score.compute(subs) == 100

    def test_determine_level_low(self):
        assert risk_score.determine_level(0) == "low"
        assert risk_score.determine_level(25) == "low"

    def test_determine_level_medium(self):
        assert risk_score.determine_level(26) == "medium"
        assert risk_score.determine_level(50) == "medium"

    def test_determine_level_high(self):
        assert risk_score.determine_level(51) == "high"
        assert risk_score.determine_level(75) == "high"

    def test_determine_level_extreme(self):
        assert risk_score.determine_level(76) == "extreme"
        assert risk_score.determine_level(100) == "extreme"


# ---------------------------------------------------------------------------
# Competitive
# ---------------------------------------------------------------------------

def make_competitor(
    ticker: str,
    market_cap: int,
    revenue_growth_yoy: float | None = None,
    gross_margin: float | None = None,
    net_margin: float | None = None,
) -> Competitor:
    return Competitor(
        ticker=ticker,
        name=ticker + " Inc.",
        market_cap=market_cap,
        revenue_ttm=Decimal("100"),
        revenue_growth_yoy=revenue_growth_yoy,
        gross_margin=gross_margin,
        net_margin=net_margin,
    )


class TestCompetitive:
    def test_no_peers_returns_50(self):
        assert competitive.compute(
            target_metrics={"market_cap": 1000.0},
            peers=[],
        ) == 50

    def test_all_dimensions_missing_returns_50(self):
        peers = [make_competitor("AAA", 1000)]
        assert competitive.compute(
            target_metrics={}, peers=peers,
        ) == 50

    def test_top_quartile_market_cap(self):
        # Peers: 100, 200, 300, 400. Target 500 → 100th percentile (4/4 below)
        peers = [
            make_competitor("A", 100),
            make_competitor("B", 200),
            make_competitor("C", 300),
            make_competitor("D", 400),
        ]
        score = competitive.compute(
            target_metrics={"market_cap": 500.0}, peers=peers,
        )
        assert score == 100

    def test_bottom_quartile_market_cap(self):
        # Peers: 100, 200, 300, 400. Target 50 → 0th percentile
        peers = [
            make_competitor("A", 100),
            make_competitor("B", 200),
            make_competitor("C", 300),
            make_competitor("D", 400),
        ]
        score = competitive.compute(
            target_metrics={"market_cap": 50.0}, peers=peers,
        )
        assert score == 0

    def test_middle_percentile(self):
        # Peers: 100, 300, 400. Target 200 → 1/3 below = 33.33%
        peers = [
            make_competitor("A", 100),
            make_competitor("B", 300),
            make_competitor("C", 400),
        ]
        score = competitive.compute(
            target_metrics={"market_cap": 200.0}, peers=peers,
        )
        assert score == 33

    def test_avg_of_multiple_dimensions(self):
        # Peers: market_cap values, gross_margin values
        peers = [
            make_competitor("A", 100, gross_margin=10.0),
            make_competitor("B", 200, gross_margin=20.0),
            make_competitor("C", 300, gross_margin=30.0),
        ]
        # market_cap 250: below = [100, 200] = 2/3 → 66.67
        # gross_margin 25: below = [10, 20] = 2/3 → 66.67
        # mean = 66.67 → 67
        score = competitive.compute(
            target_metrics={"market_cap": 250.0, "gross_margin": 25.0},
            peers=peers,
        )
        assert score == 67

    def test_skips_missing_dimensions(self):
        # Peers missing net_margin; target also missing it → skip that dimension
        peers = [
            make_competitor("A", 100, gross_margin=10.0),
            make_competitor("B", 200, gross_margin=20.0),
            make_competitor("C", 300, gross_margin=30.0),
        ]
        # Only market_cap and gross_margin populated. Target on market_cap=250
        # gives 66.67. gross_margin=25 → 66.67. net_margin not in target → skip.
        # mean = 66.67 → 67
        score = competitive.compute(
            target_metrics={"market_cap": 250.0, "gross_margin": 25.0, "net_margin": None},
            peers=peers,
        )
        assert score == 67


# ---------------------------------------------------------------------------
# DCF Valuation
# ---------------------------------------------------------------------------


class TestComputeDcfValue:
    def test_normal_inputs_returns_decimal(self):
        # FCF=$1B, growth=8%, shares=100M, WACC=9%, g_term=2.5%.
        # Sanity: terminal value dominates equity (~75%), so per-share ≈ $199.
        result = dcf.compute_dcf_value(
            fcf=Decimal("1000000000"),
            growth_rate=0.08,
            shares_outstanding=100_000_000,
        )
        assert isinstance(result, Decimal)
        # Hand-computed expected ≈ $199.18; allow ±5% for rounding.
        assert Decimal("189") < result < Decimal("210")

    def test_fcf_zero_returns_none(self):
        assert dcf.compute_dcf_value(
            fcf=Decimal("0"),
            growth_rate=0.05,
            shares_outstanding=1_000_000,
        ) is None

    def test_fcf_negative_returns_none(self):
        assert dcf.compute_dcf_value(
            fcf=Decimal("-1000000"),
            growth_rate=0.05,
            shares_outstanding=1_000_000,
        ) is None

    def test_shares_zero_returns_none(self):
        assert dcf.compute_dcf_value(
            fcf=Decimal("1000000"),
            growth_rate=0.05,
            shares_outstanding=0,
        ) is None

    def test_shares_negative_returns_none(self):
        assert dcf.compute_dcf_value(
            fcf=Decimal("1000000"),
            growth_rate=0.05,
            shares_outstanding=-100,
        ) is None

    def test_wacc_le_terminal_growth_returns_none(self):
        # WACC=2.5% == terminal → invalid (Gordon formula 0/0).
        assert dcf.compute_dcf_value(
            fcf=Decimal("1000000"),
            growth_rate=0.05,
            shares_outstanding=1_000_000,
            wacc=0.025,
            terminal_growth=0.025,
        ) is None

    def test_wacc_below_terminal_growth_returns_none(self):
        # WACC < g_term → negative terminal value would explode.
        assert dcf.compute_dcf_value(
            fcf=Decimal("1000000"),
            growth_rate=0.05,
            shares_outstanding=1_000_000,
            wacc=0.02,
            terminal_growth=0.025,
        ) is None

    def test_higher_growth_higher_intrinsic(self):
        low = dcf.compute_dcf_value(
            fcf=Decimal("1000000"),
            growth_rate=0.03,
            shares_outstanding=1_000_000,
        )
        high = dcf.compute_dcf_value(
            fcf=Decimal("1000000"),
            growth_rate=0.10,
            shares_outstanding=1_000_000,
        )
        assert low is not None and high is not None
        assert high > low

    def test_higher_wacc_lower_intrinsic(self):
        low_wacc = dcf.compute_dcf_value(
            fcf=Decimal("1000000"),
            growth_rate=0.05,
            shares_outstanding=1_000_000,
            wacc=0.07,
        )
        high_wacc = dcf.compute_dcf_value(
            fcf=Decimal("1000000"),
            growth_rate=0.05,
            shares_outstanding=1_000_000,
            wacc=0.12,
        )
        assert low_wacc is not None and high_wacc is not None
        assert low_wacc > high_wacc


# ---------------------------------------------------------------------------
# Package init
# ---------------------------------------------------------------------------

def test_package_exports():
    from alphaquant.scoring import competitive, dcf, financial_health, risk_score
    assert competitive is not None
    assert dcf is not None
    assert financial_health is not None
    assert risk_score is not None

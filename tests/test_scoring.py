"""Tests for alphaquant.scoring functions."""
from __future__ import annotations

from decimal import Decimal

from alphaquant.models.financial import (
    BalanceSheet,
    CashFlowStatement,
    FinancialStatements,
    IncomeStatement,
)
from alphaquant.scoring import dcf, financial_health


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
    from alphaquant.scoring import dcf, financial_health
    assert dcf is not None
    assert financial_health is not None

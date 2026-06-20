"""Metrics panel component (price / market cap / multiples / ratios)."""
from __future__ import annotations

from decimal import Decimal

import streamlit as st

from alphaquant.models.report import InvestmentReport


def _fmt_price(value: Decimal | None) -> str:
    if value is None:
        return "—"
    return f"${float(value):,.2f}"


def _fmt_market_cap(value: int) -> str:
    """Format market cap with B/M suffix for readability."""
    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,}"


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _compute_roe(report: InvestmentReport) -> float | None:
    """Net income / total equity, percent. None when data insufficient."""
    income = report.financial.income_statements
    balance = report.financial.balance_sheets
    if not income or not balance:
        return None
    latest_is = income[0]
    latest_bs = balance[0]
    if latest_bs.total_equity == 0:
        return None
    return float(latest_is.net_income / latest_bs.total_equity * 100)


def _compute_debt_to_equity(report: InvestmentReport) -> float | None:
    """Total liabilities / total equity."""
    balance = report.financial.balance_sheets
    if not balance:
        return None
    bs = balance[0]
    if bs.total_equity == 0:
        return None
    return float(bs.total_liabilities / bs.total_equity)


def _compute_revenue_ttm(report: InvestmentReport) -> Decimal | None:
    """Revenue from the latest TTM/FY income statement."""
    for stmt in report.financial.income_statements:
        if stmt.period in ("TTM", "FY"):
            return stmt.revenue
    if report.financial.income_statements:
        return report.financial.income_statements[0].revenue
    return None


def render_metrics_panel(report: InvestmentReport) -> None:
    """Render the metrics grid (price / multiples / ratios)."""
    market = report.market

    row1 = st.columns(4)
    row1[0].metric("Current Price", _fmt_price(market.price))
    row1[1].metric("Market Cap", _fmt_market_cap(market.market_cap))
    row1[2].metric("P/E (TTM)", _fmt_ratio(market.pe_ratio))
    row1[3].metric("P/B", _fmt_ratio(market.pb_ratio))

    roe = _compute_roe(report)
    de = _compute_debt_to_equity(report)
    rev_ttm = _compute_revenue_ttm(report)

    row2 = st.columns(3)
    row2[0].metric("ROE", _fmt_ratio(roe) + "%" if roe is not None else "—")
    row2[1].metric("Debt/Equity", _fmt_ratio(de))
    row2[2].metric("Revenue TTM", _fmt_market_cap(int(rev_ttm)) if rev_ttm is not None else "—")
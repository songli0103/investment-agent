"""指标面板组件(价格 / 市值 / 倍数 / 比率)。"""
from __future__ import annotations

from decimal import Decimal

import streamlit as st

from alphaquant.models.report import InvestmentReport


def _fmt_price(value: Decimal | None) -> str:
    if value is None:
        return "—"
    return f"${float(value):,.2f}"


def _fmt_market_cap(value: int) -> str:
    """用 B/M 后缀格式化市值,以提升可读性。"""
    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,}"


def _fmt_revenue(value: Decimal) -> str:
    """把 Decimal 营收格式化为 $X.XB/T(保留 1 位小数)。"""
    billions = float(value) / 1e9
    if billions >= 1000:
        return f"${billions / 1000:.1f}T"
    if billions >= 1:
        return f"${billions:.1f}B"
    millions = float(value) / 1e6
    return f"${millions:.1f}M"


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _compute_roe(report: InvestmentReport) -> float | None:
    """净利润 / 股东权益,百分比。数据不足时为 None。"""
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
    """总负债 / 股东权益。"""
    balance = report.financial.balance_sheets
    if not balance:
        return None
    bs = balance[0]
    if bs.total_equity == 0:
        return None
    return float(bs.total_liabilities / bs.total_equity)


def _compute_revenue_ttm(report: InvestmentReport) -> Decimal | None:
    """取最近 TTM/FY 利润表的营收。"""
    for stmt in report.financial.income_statements:
        if stmt.period in ("TTM", "FY"):
            return stmt.revenue
    if report.financial.income_statements:
        return report.financial.income_statements[0].revenue
    return None


def render_metrics_panel(report: InvestmentReport) -> None:
    """渲染指标网格(价格 / 倍数 / 比率)。"""
    market = report.market

    row1 = st.columns(4)
    row1[0].metric("当前价格", _fmt_price(market.price))
    row1[1].metric("市值", _fmt_market_cap(market.market_cap))
    row1[2].metric("市盈率(TTM)", _fmt_ratio(market.pe_ratio))
    row1[3].metric("市净率", _fmt_ratio(market.pb_ratio))

    roe = _compute_roe(report)
    de = _compute_debt_to_equity(report)
    rev_ttm = _compute_revenue_ttm(report)

    row2 = st.columns(3)
    roe_text = (_fmt_ratio(roe) + "%") if roe is not None else "—"
    row2[0].metric("净资产收益率", roe_text)
    row2[1].metric("负债权益比", _fmt_ratio(de))
    row2[2].metric("TTM 营收", _fmt_revenue(rev_ttm) if rev_ttm is not None else "—")
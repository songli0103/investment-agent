"""财务健康度评分(0–100)。"""
from __future__ import annotations

from alphaquant.models.financial import FinancialStatements


def _profitability(statements: FinancialStatements) -> float:
    """基于毛利率/净利率和 ROE 的 0-100 评分。"""
    if not statements.income_statements:
        return 50.0
    latest = statements.income_statements[0]
    if latest.revenue == 0:
        return 0.0
    gross_margin = (
        float(latest.gross_profit / latest.revenue * 100) if latest.gross_profit else 30.0
    )
    net_margin = float(latest.net_income / latest.revenue * 100)
    return min(100.0, max(0.0, gross_margin * 1.5 + net_margin * 2))


def _growth(statements: FinancialStatements) -> float:
    """同比收入增长评分。"""
    if len(statements.income_statements) < 2:
        return 50.0
    latest = statements.income_statements[0]
    prev = statements.income_statements[1]
    if prev.revenue == 0:
        return 50.0
    growth_pct = float((latest.revenue - prev.revenue) / prev.revenue * 100)
    return min(100.0, max(0.0, 50 + growth_pct))


def _solvency(statements: FinancialStatements) -> float:
    """资产负债率和流动比率。"""
    if not statements.balance_sheets:
        return 50.0
    bs = statements.balance_sheets[0]
    if bs.total_assets == 0:
        return 50.0
    debt_ratio = float(bs.total_liabilities / bs.total_assets * 100)
    return min(100.0, max(0.0, 100 - debt_ratio))


def _cash_quality(statements: FinancialStatements) -> float:
    """OCF / 净利润 比率。"""
    if not statements.cash_flows or not statements.income_statements:
        return 50.0
    ocf = statements.cash_flows[0].operating_cash_flow
    ni = statements.income_statements[0].net_income
    if ni == 0:
        return 0.0
    ratio = float(ocf / ni * 100)
    return min(100.0, max(0.0, ratio))


def compute(statements: FinancialStatements) -> int:
    """整体财务健康度 0-100。"""
    if not statements.income_statements:
        return 50  # 无数据 → 中性
    scores = {
        "profitability": _profitability(statements),
        "growth": _growth(statements),
        "solvency": _solvency(statements),
        "cash_quality": _cash_quality(statements),
    }
    weights = {"profitability": 0.35, "growth": 0.25, "solvency": 0.20, "cash_quality": 0.20}
    total = sum(scores[k] * weights[k] for k in scores)
    return round(total)

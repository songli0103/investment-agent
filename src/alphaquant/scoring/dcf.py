"""Discounted Cash Flow (DCF) valuation — intrinsic value per share.

MVP simplification: ignores net debt (equity ≈ enterprise value). Uses
2-stage DCF: 5-year explicit forecast + Gordon perpetuity. Assumptions
mirror ``DCFTool`` defaults (WACC=9%, g_term=2.5%); production should
derive WACC from industry/beta.
"""
from __future__ import annotations

from decimal import Decimal


def compute_dcf_value(
    fcf: Decimal,
    growth_rate: float,
    shares_outstanding: int,
    wacc: float = 0.09,
    terminal_growth: float = 0.025,
    forecast_years: int = 5,
) -> Decimal | None:
    """Returns intrinsic value per share, or None if inputs invalid.

    Formula:
      PV(FCF_t) = FCF_0 * (1+g)^t / (1+WACC)^t  for t in 1..5
      TV = FCF_5 * (1+g_term) / (WACC - g_term)
      PV(TV) = TV / (1+WACC)^5
      Equity = sum(PV(FCF)) + PV(TV)        # MVP: 简化忽略净债务
      Per share = Equity / shares_outstanding
    """
    # Guard rails (return None on invalid input).
    if fcf <= 0:
        return None  # DCF meaningless for non-profitable companies
    if shares_outstanding <= 0:
        return None
    if wacc <= terminal_growth:
        return None  # Gordon formula explodes
    if growth_rate < -0.5:
        growth_rate = -0.5  # clamp to prevent single-year outliers

    # 5-year explicit forecast: PV of each year's FCF.
    pv_fcf_total = 0.0
    fcf_t = float(fcf)
    for t in range(1, forecast_years + 1):
        fcf_t *= 1 + growth_rate
        pv_fcf_total += fcf_t / ((1 + wacc) ** t)

    # Terminal value via Gordon Growth Model, discounted back to t=0.
    fcf_terminal_year = fcf_t * (1 + terminal_growth)
    terminal_value = fcf_terminal_year / (wacc - terminal_growth)
    pv_terminal = terminal_value / ((1 + wacc) ** forecast_years)

    equity = pv_fcf_total + pv_terminal  # MVP: equity ≈ enterprise value
    per_share = equity / shares_outstanding
    return Decimal(str(per_share)).quantize(Decimal("0.01"))

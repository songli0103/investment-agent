"""现金流折现(DCF)估值 —— 每股内在价值。

MVP 简化:忽略净债务(股权 ≈ 企业价值)。采用两阶段 DCF:5 年明确预测 + Gordon 永续增长。
假设与 ``DCFTool`` 默认值一致(WACC=9%,g_term=2.5%);生产环境应根据行业/beta 推导 WACC。
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
    """返回每股内在价值,如果输入无效则返回 None。

    公式:
      PV(FCF_t) = FCF_0 * (1+g)^t / (1+WACC)^t  for t in 1..5
      TV = FCF_5 * (1+g_term) / (WACC - g_term)
      PV(TV) = TV / (1+WACC)^5
      Equity = sum(PV(FCF)) + PV(TV)        # MVP: 简化忽略净债务
      Per share = Equity / shares_outstanding
    """
    # 守卫(对无效输入返回 None)。
    if fcf <= 0:
        return None  # 对非盈利公司 DCF 无意义
    if shares_outstanding <= 0:
        return None
    if wacc <= terminal_growth:
        return None  # Gordon 公式会发散
    if growth_rate < -0.5:
        growth_rate = -0.5  # 钳制以防止单年度异常值

    # 5 年明确预测:每年 FCF 的现值。
    pv_fcf_total = 0.0
    fcf_t = float(fcf)
    for t in range(1, forecast_years + 1):
        fcf_t *= 1 + growth_rate
        pv_fcf_total += fcf_t / ((1 + wacc) ** t)

    # 通过 Gordon 增长模型计算终值,折现回 t=0。
    fcf_terminal_year = fcf_t * (1 + terminal_growth)
    terminal_value = fcf_terminal_year / (wacc - terminal_growth)
    pv_terminal = terminal_value / ((1 + wacc) ** forecast_years)

    equity = pv_fcf_total + pv_terminal  # MVP:股权 ≈ 企业价值
    per_share = equity / shares_outstanding
    return Decimal(str(per_share)).quantize(Decimal("0.01"))

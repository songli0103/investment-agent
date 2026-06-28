"""大号彩色评级卡片组件。"""
from __future__ import annotations

import streamlit as st

from alphaquant.models.report import InvestmentReport


# 各评级等级的集中调色板(此处以及需要时在其他位置使用)。
# 键必须保持英文,因为它们与 Pydantic Literal 字段约束相匹配。
RATING_COLORS: dict[str, str] = {
    "Strong Buy": "#1a8f3a",  # 绿色
    "Buy": "#7cc242",          # 黄绿
    "Hold": "#8a8a8a",         # 灰色
    "Sell": "#e89a3c",         # 橙色
    "Strong Sell": "#c0392b",  # 红色
}

_DEFAULT_COLOR = "#8a8a8a"


def _format_date(dt) -> str:
    """为显示格式化 datetime/date。"""
    if hasattr(dt, "strftime"):
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    return str(dt)


def render_rating_card(report: InvestmentReport) -> None:
    """渲染包含评级、置信度、ticker 和日期的大号彩色评级卡片。"""
    rating = report.rating
    color = RATING_COLORS.get(rating, _DEFAULT_COLOR)
    confidence = report.confidence if report.confidence is not None else "N/A"
    ticker = report.ticker
    generated_at = _format_date(report.generated_at)

    html = f"""
    <div style="
        background-color: {color};
        color: white;
        padding: 24px 28px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
        margin-bottom: 8px;
    ">
      <div style="display:flex; align-items:baseline; justify-content:space-between; flex-wrap:wrap;">
        <div>
          <div style="font-size: 0.85rem; opacity: 0.9; letter-spacing: 0.05em;">评级</div>
          <div style="font-size: 2.6rem; font-weight: 700; line-height: 1.1; margin-top: 4px;">
            {rating}
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size: 0.85rem; opacity: 0.9; letter-spacing: 0.05em;">置信度</div>
          <div style="font-size: 2.0rem; font-weight: 600; line-height: 1.1; margin-top: 4px;">
            {confidence}{"%" if isinstance(confidence, int) else ""}
          </div>
        </div>
      </div>
      <div style="margin-top: 14px; font-size: 0.95rem; opacity: 0.95;">
        <strong>{ticker}</strong> &middot; {generated_at}
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
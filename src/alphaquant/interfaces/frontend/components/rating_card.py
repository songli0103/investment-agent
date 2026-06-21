"""Large colored rating card component."""
from __future__ import annotations

import streamlit as st

from alphaquant.models.report import InvestmentReport


# Centralized color palette for rating levels (used here and elsewhere if needed).
RATING_COLORS: dict[str, str] = {
    "Strong Buy": "#1a8f3a",  # green
    "Buy": "#7cc242",          # lime
    "Hold": "#8a8a8a",         # gray
    "Sell": "#e89a3c",         # orange
    "Strong Sell": "#c0392b",  # red
}

_DEFAULT_COLOR = "#8a8a8a"


def _format_date(dt) -> str:
    """Format datetime/date for display."""
    if hasattr(dt, "strftime"):
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    return str(dt)


def render_rating_card(report: InvestmentReport) -> None:
    """Render a large colored rating card with rating, confidence, ticker, and date."""
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
          <div style="font-size: 0.85rem; opacity: 0.9; letter-spacing: 0.05em;">RATING</div>
          <div style="font-size: 2.6rem; font-weight: 700; line-height: 1.1; margin-top: 4px;">
            {rating}
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size: 0.85rem; opacity: 0.9; letter-spacing: 0.05em;">CONFIDENCE</div>
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
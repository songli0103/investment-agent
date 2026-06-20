"""Analyze page: enter a ticker, run the full analysis, render the report."""
from __future__ import annotations

import asyncio

import streamlit as st

from alphaquant.exceptions import (
    AllDataSourcesDown,
    InvalidTickerFormat,
    TickerNotFound,
)
from alphaquant.frontend.components.charts import (
    render_risk_radar,
    render_sentiment_bar,
)
from alphaquant.frontend.components.metrics_panel import render_metrics_panel
from alphaquant.frontend.components.rating_card import render_rating_card
from alphaquant.frontend.db import DB
from alphaquant.main import run_analysis_async


st.title("Analyze")
st.write(
    "Enter a stock ticker to generate a full investment research report. "
    "The analysis is persisted to the local history database."
)


def _normalize_ticker(raw: str) -> str:
    """Validate and normalize ticker input per the flow's contract.

    Mirrors ``alphaquant.flows.analysis_flow._normalize_ticker``:
    strip whitespace, uppercase, require non-empty and len <= 6.
    """
    if raw is None:
        raise InvalidTickerFormat("")
    t = raw.strip().upper()
    if not t or len(t) > 6:
        raise InvalidTickerFormat(raw)
    return t


db = DB()
db.init()

with st.form(key="analyze_form", clear_on_submit=False):
    ticker_input = st.text_input("Ticker", placeholder="e.g. AAPL")
    submitted = st.form_submit_button("Analyze", type="primary")

if submitted:
    try:
        ticker = _normalize_ticker(ticker_input)
    except InvalidTickerFormat:
        st.error(
            "Invalid ticker format. Expected like 'AAPL' or 'BRK.B' "
            "(letters only, up to 6 characters)."
        )
        st.stop()

    try:
        with st.spinner("Running analysis..."):
            report = asyncio.run(run_analysis_async(ticker))
    except InvalidTickerFormat as exc:
        st.error(f"Invalid ticker format: {exc.ticker!r}.")
    except TickerNotFound as exc:
        st.error(
            f"Ticker {exc.ticker!r} could not be resolved by any data source. "
            "Double-check the symbol and try again."
        )
    except AllDataSourcesDown:
        st.error(
            "All data sources are currently unavailable. "
            "Please try again in a few minutes."
        )
    else:
        db.insert_report(ticker, report)

        render_rating_card(report)
        render_metrics_panel(report)

        charts_left, charts_right = st.columns(2)
        with charts_left:
            st.plotly_chart(render_risk_radar(report.risk), use_container_width=True)
        with charts_right:
            st.plotly_chart(
                render_sentiment_bar(report.news), use_container_width=True
            )

        st.subheader("Competitors")
        competitor_rows = [
            {
                "Ticker": c.ticker,
                "Name": c.name,
                "Market Cap": c.market_cap,
                "Revenue TTM": float(c.revenue_ttm),
                "P/E": c.pe_ratio,
            }
            for c in report.competitors.competitors
        ]
        st.table(competitor_rows)

        st.subheader("Full Report")
        st.markdown(report.markdown)

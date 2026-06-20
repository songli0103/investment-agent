"""Compare page: side-by-side analysis of up to 5 tickers.

Runs ``run_analysis_async`` concurrently for each ticker, persists successful
reports to the local SQLite history DB, then renders:

- A row of rating cards (one per successful ticker).
- A side-by-side metrics table for key metrics (price, P/E, market cap, confidence).
- An overlaid radar chart using the same 5 axes as ``render_risk_radar``.
- A short verdict identifying the ticker with the best rating and the highest
  confidence.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alphaquant.exceptions import (
    AllDataSourcesDown,
    InvalidTickerFormat,
    TickerNotFound,
)
from alphaquant.frontend.components.charts import (
    RADAR_AXES,
    RATING_TO_NUMERIC,
    _risk_axis_value,
)
from alphaquant.frontend.components.rating_card import render_rating_card
from alphaquant.frontend.db import DB
from alphaquant.main import run_analysis_async
from alphaquant.models.report import InvestmentReport


st.title("Compare")
st.write(
    "Enter up to 5 tickers (comma-separated) to run analyses in parallel and "
    "compare the resulting ratings, metrics, and risk profiles side by side."
)


MAX_TICKERS = 5


db = DB()
db.init()


def _parse_tickers(raw: str) -> list[str]:
    """Split a comma-separated string into a normalized, de-duplicated list.

    Rules:
    - Strip whitespace, uppercase, drop empties.
    - Cap at ``MAX_TICKERS`` entries (extras are dropped silently; we validate
      separately via ``st.warning`` so the user knows).
    """
    parts = [p.strip().upper() for p in raw.split(",")]
    parts = [p for p in parts if p]
    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


async def _compare_all(tickers: list[str]) -> list[InvestmentReport | BaseException]:
    """Run ``run_analysis_async`` concurrently for every ticker.

    ``return_exceptions=True`` ensures one bad ticker does not abort the rest
    of the comparison. The caller is responsible for filtering out the
    exception results before persisting or rendering.
    """
    results = await asyncio.gather(
        *(run_analysis_async(t) for t in tickers), return_exceptions=True
    )
    return list(results)


def _build_metrics_row(report: InvestmentReport) -> dict[str, Any]:
    """Extract a small set of comparison-friendly metrics from a report."""
    market = report.market
    return {
        "Price": float(market.price) if market.price is not None else None,
        "Market Cap": market.market_cap,
        "P/E": market.pe_ratio,
        "Confidence": report.confidence,
    }


def _build_overlay_radar(reports: list[InvestmentReport]) -> go.Figure:
    """Return a single Figure with one Scatterpolar trace per report."""
    # Close the polygon by repeating the first axis/value.
    closed_axes = RADAR_AXES + [RADAR_AXES[0]]

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig = go.Figure()
    for idx, report in enumerate(reports):
        values = [_risk_axis_value(report.risk, axis) for axis in RADAR_AXES]
        closed_values = values + [values[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=closed_values,
                theta=closed_axes,
                fill="toself",
                name=report.ticker,
                line=dict(color=palette[idx % len(palette)]),
                opacity=0.55,
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        title="Risk Profile Comparison",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def _format_market_cap(value: int | None) -> str:
    if value is None:
        return "—"
    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,}"


with st.form(key="compare_form", clear_on_submit=False):
    tickers_input = st.text_input(
        "Tickers (comma-separated, max 5)",
        value="AAPL, MSFT, GOOGL",
    )
    submitted = st.form_submit_button("Compare", type="primary")

if not submitted:
    st.stop()

tickers = _parse_tickers(tickers_input)

if not tickers:
    st.error("Please enter at least one ticker.")
    st.stop()

if len(tickers) > MAX_TICKERS:
    st.warning(
        f"More than {MAX_TICKERS} tickers supplied; only the first "
        f"{MAX_TICKERS} will be analyzed."
    )
    tickers = tickers[:MAX_TICKERS]

with st.spinner(f"Running analysis for {', '.join(tickers)}..."):
    results = asyncio.run(_compare_all(tickers))

# Split into successes and per-ticker failures.
successful: list[InvestmentReport] = []
failures: list[tuple[str, BaseException]] = []
for ticker, result in zip(tickers, results):
    if isinstance(result, BaseException):
        failures.append((ticker, result))
        continue
    successful.append(result)
    try:
        db.insert_report(ticker, result)
    except Exception as exc:  # pragma: no cover - DB write best-effort
        failures.append((ticker, exc))

if not successful:
    st.error("All ticker analyses failed. See messages below.")
    for ticker, exc in failures:
        if isinstance(exc, TickerNotFound):
            st.error(f"{ticker}: ticker could not be resolved.")
        elif isinstance(exc, InvalidTickerFormat):
            st.error(f"{ticker}: invalid ticker format.")
        elif isinstance(exc, AllDataSourcesDown):
            st.error(f"{ticker}: all data sources unavailable.")
        else:
            st.error(f"{ticker}: {type(exc).__name__}: {exc}")
    st.stop()

# Show any per-ticker failures (partial success path).
for ticker, exc in failures:
    if isinstance(exc, TickerNotFound):
        st.error(f"{ticker}: ticker could not be resolved.")
    elif isinstance(exc, InvalidTickerFormat):
        st.error(f"{ticker}: invalid ticker format.")
    elif isinstance(exc, AllDataSourcesDown):
        st.error(f"{ticker}: all data sources unavailable.")
    else:
        st.warning(f"{ticker}: {type(exc).__name__}: {exc}")

# Rating cards in a row of equal-width columns.
st.subheader("Ratings")
columns = st.columns(len(successful))
for col, report in zip(columns, successful):
    with col:
        render_rating_card(report)

# Side-by-side metrics table, indexed by ticker.
st.subheader("Side-by-side Metrics")
metrics_df = pd.DataFrame(
    [_build_metrics_row(r) for r in successful],
    index=[r.ticker for r in successful],
)
metrics_display = metrics_df.copy()
metrics_display["Price"] = metrics_display["Price"].apply(
    lambda v: f"${v:,.2f}" if v is not None else "—"
)
metrics_display["Market Cap"] = metrics_display["Market Cap"].apply(_format_market_cap)
metrics_display["P/E"] = metrics_display["P/E"].apply(
    lambda v: f"{v:.2f}" if v is not None else "—"
)
metrics_display["Confidence"] = metrics_display["Confidence"].apply(
    lambda v: f"{int(v)}%"
)
st.dataframe(metrics_display, use_container_width=True)

# Overlaid risk radar.
st.subheader("Risk Profile Overlay")
st.plotly_chart(_build_overlay_radar(successful), use_container_width=True)

# Verdict: best rating (highest RATING_TO_NUMERIC) and highest confidence.
best_rating_ticker = max(
    successful,
    key=lambda r: (RATING_TO_NUMERIC.get(r.rating, 0), r.confidence),
).ticker
best_confidence_ticker = max(successful, key=lambda r: r.confidence).ticker

st.subheader("Verdict")
st.write(f"**Best rating:** {best_rating_ticker}")
st.write(f"**Highest confidence:** {best_confidence_ticker}")
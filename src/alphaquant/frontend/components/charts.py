"""Plotly chart components: risk radar, sentiment bar, history lines."""
from __future__ import annotations

from typing import Iterable

import plotly.graph_objects as go

from alphaquant.infrastructure.persistence import ReportRecord
from alphaquant.models.news import NewsAnalysis
from alphaquant.models.risk import RiskAssessment


# Rating string → numeric score (5 = best, 1 = worst). Unknown → None.
RATING_TO_NUMERIC: dict[str, int] = {
    "Strong Buy": 5,
    "Buy": 4,
    "Hold": 3,
    "Sell": 2,
    "Strong Sell": 1,
}

# Axes shown on the risk radar. The backend RiskScore categories are
# {financial, operational, market, regulatory, governance, macro}; we map them
# onto the 5 user-facing axes specified in the brief.
RADAR_AXES: list[str] = [
    "financial_health",
    "valuation",
    "competitive",
    "operational",
    "market",
]

_RISK_AXIS_TO_CATEGORY: dict[str, str] = {
    "financial_health": "financial",
    "valuation": "financial",
    "competitive": "market",
    "operational": "operational",
    "market": "market",
}


def _risk_axis_value(risk: RiskAssessment, axis: str) -> int:
    """Return 0-10 score for a UI axis, averaging matching backend sub-scores."""
    target_cat = _RISK_AXIS_TO_CATEGORY[axis]
    matching = [s for s in risk.sub_scores if s.category == target_cat]
    if not matching:
        return 0
    return round(sum(s.score for s in matching) / len(matching))


def render_risk_radar(risk: RiskAssessment) -> go.Figure:
    """Return a 5-axis radar chart figure for the risk assessment."""
    axes = RADAR_AXES
    values = [_risk_axis_value(risk, a) for a in axes]
    # Close the polygon by repeating the first point.
    closed_axes = axes + [axes[0]]
    closed_values = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=closed_values,
            theta=closed_axes,
            fill="toself",
            name=risk.ticker,
            line=dict(color="#1f77b4"),
            opacity=0.7,
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        title=f"Risk Profile — {risk.ticker} (level: {risk.level})",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def render_sentiment_bar(news: NewsAnalysis) -> go.Figure:
    """Return a horizontal bar chart of positive / neutral / negative counts."""
    total = news.total_count
    positive = int(round(news.positive_pct * total))
    neutral = int(round(news.neutral_pct * total))
    negative = int(round(news.negative_pct * total))

    labels = ["Positive", "Neutral", "Negative"]
    counts = [positive, neutral, negative]
    colors = ["#1a8f3a", "#8a8a8a", "#c0392b"]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=counts,
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"News Sentiment — {news.ticker} (n={total})",
        xaxis_title="Articles",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def _format_history_points(history: Iterable[ReportRecord]) -> tuple[list, list[float | None], list[int | None]]:
    """Convert ReportRecords to parallel x / price / rating_numeric lists, sorted by time."""
    records = sorted(history, key=lambda r: r.generated_at)
    xs = [r.generated_at for r in records]
    prices: list[float | None] = [r.market_price for r in records]
    ratings: list[int | None] = [RATING_TO_NUMERIC.get(r.rating) for r in records]
    return xs, prices, ratings


def render_history_lines(history: list[ReportRecord]) -> go.Figure:
    """Return a dual-axis line chart: market price (left) + rating (right)."""
    xs, prices, ratings = _format_history_points(history)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=prices,
            name="Market Price",
            mode="lines+markers",
            yaxis="y1",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=8),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ratings,
            name="Rating",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#e89a3c", width=2, dash="dot"),
            marker=dict(size=8, symbol="diamond"),
            connectgaps=False,
        )
    )

    fig.update_layout(
        title="Report History",
        xaxis_title="Generated at",
        yaxis=dict(title=dict(text="Market Price (USD)"), side="left"),
        yaxis2=dict(
            title=dict(text="Rating (1=Strong Sell ... 5=Strong Buy)"),
            side="right",
            overlaying="y",
            range=[0.5, 5.5],
            tickmode="array",
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return fig
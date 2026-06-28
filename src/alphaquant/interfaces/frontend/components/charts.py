"""Plotly 图表组件:风险雷达、情绪柱状图、历史折线图。"""
from __future__ import annotations

from typing import Iterable

import plotly.graph_objects as go

from alphaquant.infrastructure.persistence import ReportRecord
from alphaquant.models.news import NewsAnalysis
from alphaquant.models.risk import RiskAssessment


# 评级字符串 → 数值(5 = 最好,1 = 最差)。未知 → None。
# 键必须保持英文,因为它们与 Pydantic Literal 字段约束相匹配。
RATING_TO_NUMERIC: dict[str, int] = {
    "Strong Buy": 5,
    "Buy": 4,
    "Hold": 3,
    "Sell": 2,
    "Strong Sell": 1,
}

# 风险雷达上显示的轴。后端 RiskScore 类别为
# {financial, operational, market, regulatory, governance, macro};
# 我们按需求将其映射到 5 个面向用户的轴。
RADAR_AXES: list[str] = [
    "financial_health",
    "valuation",
    "competitive",
    "operational",
    "market",
]

# 雷达轴的显示标签(图表渲染时使用)。
_RADAR_AXIS_DISPLAY: dict[str, str] = {
    "financial_health": "财务健康",
    "valuation": "估值",
    "competitive": "竞争",
    "operational": "运营",
    "market": "市场",
}

_RISK_AXIS_TO_CATEGORY: dict[str, str] = {
    "financial_health": "financial",
    "valuation": "financial",
    "competitive": "market",
    "operational": "operational",
    "market": "market",
}


def _risk_axis_value(risk: RiskAssessment, axis: str) -> int:
    """返回 UI 轴的 0-10 分,对匹配的后端子分数取平均。"""
    target_cat = _RISK_AXIS_TO_CATEGORY[axis]
    matching = [s for s in risk.sub_scores if s.category == target_cat]
    if not matching:
        return 0
    return round(sum(s.score for s in matching) / len(matching))


def _radar_axis_labels(axes: list[str]) -> list[str]:
    """把内部轴 key 翻译成图表上显示的中文标签。"""
    return [_RADAR_AXIS_DISPLAY.get(a, a) for a in axes]


def render_risk_radar(risk: RiskAssessment) -> go.Figure:
    """返回风险评估的 5 轴雷达图。"""
    axes = RADAR_AXES
    values = [_risk_axis_value(risk, a) for a in axes]
    # 通过重复第一个点闭合多边形。
    closed_axes = _radar_axis_labels(axes) + [_radar_axis_labels(axes)[0]]
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
        title=f"风险概览 — {risk.ticker} (等级:{risk.level})",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def render_sentiment_bar(news: NewsAnalysis) -> go.Figure:
    """返回正面 / 中性 / 负面文章数的水平柱状图。"""
    total = news.total_count
    positive = int(round(news.positive_pct * total))
    neutral = int(round(news.neutral_pct * total))
    negative = int(round(news.negative_pct * total))

    labels = ["正面", "中性", "负面"]
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
        title=f"新闻情绪 — {news.ticker} (n={total})",
        xaxis_title="文章数",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def _format_history_points(history: Iterable[ReportRecord]) -> tuple[list, list[float | None], list[int | None]]:
    """把 ReportRecord 转换为按时间排序的并行 x / 价格 / 评级数值列表。"""
    records = sorted(history, key=lambda r: r.generated_at)
    xs = [r.generated_at for r in records]
    prices: list[float | None] = [r.market_price for r in records]
    ratings: list[int | None] = [RATING_TO_NUMERIC.get(r.rating) for r in records]
    return xs, prices, ratings


def render_history_lines(history: list[ReportRecord]) -> go.Figure:
    """返回双轴折线图:市场价格(左)+ 评级(右)。"""
    xs, prices, ratings = _format_history_points(history)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=prices,
            name="市场价格",
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
            name="评级",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#e89a3c", width=2, dash="dot"),
            marker=dict(size=8, symbol="diamond"),
            connectgaps=False,
        )
    )

    fig.update_layout(
        title="报告历史",
        xaxis_title="生成时间",
        yaxis=dict(title=dict(text="市场价格(美元)"), side="left"),
        yaxis2=dict(
            title=dict(text="评级(1=强烈卖出 ... 5=强烈买入)"),
            side="right",
            overlaying="y",
            range=[0.5, 5.5],
            tickmode="array",
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["强烈卖出", "卖出", "持有", "买入", "强烈买入"],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return fig
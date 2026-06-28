"""对比页面:最多 5 个 ticker 的并排分析。

为每个 ticker 并发运行 ``run_analysis_async``,将成功的报告
持久化到本地 SQLite 历史数据库,然后渲染:

- 一排评级卡片(每个成功的 ticker 一张)。
- 一张关键指标的并排指标表(价格、市盈率、市值、置信度)。
- 一张使用与 ``render_risk_radar`` 相同 5 个轴的叠加雷达图。
- 一段简短的结论,标识最佳评级和最高置信度的 ticker。
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
from alphaquant.interfaces.frontend.components.charts import (
    RADAR_AXES,
    RATING_TO_NUMERIC,
    _risk_axis_value,
)
from alphaquant.interfaces.frontend.components.rating_card import render_rating_card
from alphaquant.infrastructure.persistence import DB
from alphaquant.main import run_analysis_async
from alphaquant.models.report import InvestmentReport


st.title("对比")
st.write(
    "输入最多 5 个 ticker(逗号分隔)以并行运行分析,"
    "并并排对比生成的评级、指标和风险概览。"
)


MAX_TICKERS = 5


db = DB()
db.init()


def _parse_tickers(raw: str) -> list[str]:
    """将逗号分隔的字符串切分为标准化、去重的列表。

    规则:
    - 去除空白、大写、丢弃空项。
    - 上限 ``MAX_TICKERS`` 个条目(超出部分静默丢弃;
      另行通过 ``st.warning`` 校验,告知用户)。
    """
    parts = [p.strip().upper() for p in raw.split(",")]
    parts = [p for p in parts if p]
    # 去重同时保持顺序。
    seen: set[str] = set()
    unique: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


async def _compare_all(tickers: list[str]) -> list[InvestmentReport | BaseException]:
    """为每个 ticker 并发运行 ``run_analysis_async``。

    ``return_exceptions=True`` 保证单个 ticker 失败不会中断其他 ticker
    的对比。调用方负责在持久化或渲染前过滤掉异常结果。
    """
    results = await asyncio.gather(
        *(run_analysis_async(t) for t in tickers), return_exceptions=True
    )
    return list(results)


def _build_metrics_row(report: InvestmentReport) -> dict[str, Any]:
    """从报告中提取一小组便于对比的指标。"""
    market = report.market
    return {
        "价格": float(market.price) if market.price is not None else None,
        "市值": market.market_cap,
        "市盈率": market.pe_ratio,
        "置信度": report.confidence,
    }


def _build_overlay_radar(reports: list[InvestmentReport]) -> go.Figure:
    """返回一个 Figure,每个报告对应一条 Scatterpolar 轨迹。"""
    # 通过重复第一个轴/值闭合多边形。
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
        title="风险概览对比",
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
        "Ticker(逗号分隔,最多 5 个)",
        value="AAPL, MSFT, GOOGL",
    )
    submitted = st.form_submit_button("对比", type="primary")

if not submitted:
    st.stop()

tickers = _parse_tickers(tickers_input)

if not tickers:
    st.error("请至少输入一个 ticker。")
    st.stop()

if len(tickers) > MAX_TICKERS:
    st.warning(
        f"输入的 ticker 超过 {MAX_TICKERS} 个;只会分析前 {MAX_TICKERS} 个。"
    )
    tickers = tickers[:MAX_TICKERS]

with st.spinner(f"正在分析 {', '.join(tickers)}..."):
    results = asyncio.run(_compare_all(tickers))

# 拆分为成功结果和逐 ticker 的失败。
successful: list[InvestmentReport] = []
failures: list[tuple[str, BaseException]] = []
for ticker, result in zip(tickers, results):
    if isinstance(result, BaseException):
        failures.append((ticker, result))
        continue
    successful.append(result)
    try:
        db.insert_report(ticker, result)
    except Exception as exc:  # pragma: no cover - DB 写入尽力而为
        failures.append((ticker, exc))

if not successful:
    st.error("所有 ticker 分析均失败。详见下方消息。")
    for ticker, exc in failures:
        if isinstance(exc, TickerNotFound):
            st.error(f"{ticker}:无法解析该 ticker。")
        elif isinstance(exc, InvalidTickerFormat):
            st.error(f"{ticker}: ticker 格式无效。")
        elif isinstance(exc, AllDataSourcesDown):
            st.error(f"{ticker}: 所有数据源不可用。")
        else:
            st.error(f"{ticker}: 分析失败。请查看服务器日志了解详情。")
    st.stop()

# 展示逐 ticker 的失败(部分成功路径)。
for ticker, exc in failures:
    if isinstance(exc, TickerNotFound):
        st.error(f"{ticker}:无法解析该 ticker。")
    elif isinstance(exc, InvalidTickerFormat):
        st.error(f"{ticker}: ticker 格式无效。")
    elif isinstance(exc, AllDataSourcesDown):
        st.error(f"{ticker}: 所有数据源不可用。")
    else:
        st.warning(f"{ticker}: 分析失败。请查看服务器日志了解详情。")

# 评级卡片,等宽列一排。
st.subheader("评级")
columns = st.columns(len(successful))
for col, report in zip(columns, successful):
    with col:
        render_rating_card(report)

# 并排指标表,以 ticker 为索引。
st.subheader("并排指标")
metrics_df = pd.DataFrame(
    [_build_metrics_row(r) for r in successful],
    index=[r.ticker for r in successful],
)
metrics_display = metrics_df.copy()
metrics_display["价格"] = metrics_display["价格"].apply(
    lambda v: f"${v:,.2f}" if v is not None else "—"
)
metrics_display["市值"] = metrics_display["市值"].apply(_format_market_cap)
metrics_display["市盈率"] = metrics_display["市盈率"].apply(
    lambda v: f"{v:.2f}" if v is not None else "—"
)
metrics_display["置信度"] = metrics_display["置信度"].apply(
    lambda v: f"{int(v)}%" if v is not None else "—"
)
st.dataframe(metrics_display, width="stretch")

# 叠加的风险雷达。
st.subheader("风险概览叠加图")
st.plotly_chart(
    _build_overlay_radar(successful), key="overlay_radar", width="stretch"
)

# 结论:最佳评级(最高的 RATING_TO_NUMERIC)和最高置信度。
best_rating_ticker = max(
    successful,
    key=lambda r: (
        RATING_TO_NUMERIC.get(r.rating, 0),
        r.confidence if r.confidence is not None else -1,
    ),
).ticker
best_confidence_ticker = max(
    successful,
    key=lambda r: (r.confidence if r.confidence is not None else -1),
).ticker

st.subheader("结论")
st.write(f"**最佳评级:** {best_rating_ticker}")
st.write(f"**最高置信度:** {best_confidence_ticker}")
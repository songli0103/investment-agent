"""历史页面:过去报告的时序图和表格,支持 CSV 导出。"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pandas as pd
import streamlit as st

from alphaquant.interfaces.frontend.components.charts import render_history_lines
from alphaquant.infrastructure.persistence import DB


st.title("历史")
st.write(
    "浏览历史分析记录。在侧边栏选择 ticker 子集和起始日期以缩小视图,"
    "然后将结果下载为 CSV。"
)


db = DB()
db.init()


@st.cache_data(ttl=10)
def _load_history(tickers: tuple[str, ...], since_iso: str | None) -> list[dict]:
    """缓存数据库读取 10 秒,避免每次 Streamlit 重跑时反复打 SQLite。

    ``tickers`` 使用元组(可哈希),``since_iso`` 使用字符串,
    以便参数在 Streamlit 重跑之间保持稳定。这里再转换回 list / datetime,
    让底层的 ``DB.get_history`` 看到它期望的类型化值。
    """
    since_dt = datetime.fromisoformat(since_iso) if since_iso else None
    records = db.get_history(tickers=list(tickers) or None, since=since_dt)
    return [
        {
            "id": r.id,
            "ticker": r.ticker,
            "generated_at": r.generated_at,
            "rating": r.rating,
            "confidence": r.confidence,
            "market_price": r.market_price,
        }
        for r in records
    ]


with st.sidebar:
    st.header("筛选")
    all_tickers = db.list_tickers()
    selected = st.multiselect(
        "Ticker",
        options=all_tickers,
        default=all_tickers,
        help="至少有一条已存储报告的 ticker。",
    )
    today = date.today()
    default_since = today - timedelta(days=365)
    since_date = st.date_input("起始日期", value=default_since, max_value=today)

# 标准化筛选条件并加载历史。
since_dt = datetime.combine(since_date, time.min)
since_iso = since_dt.isoformat()
rows = _load_history(tuple(selected), since_iso)
df = pd.DataFrame(rows)

if df.empty:
    st.info("还没有报告。请先运行分析。")
    st.stop()

# 表格按最新时间倒序排列;绘图函数会在内部重新排序。
df_display = df.sort_values("generated_at", ascending=False).reset_index(drop=True)

# 绘图函数需要 ReportRecord 形状的对象。把缓存中的 dict 重新水合为
# 公共模型,以满足 ``render_history_lines`` 的类型契约。
from alphaquant.infrastructure.persistence import ReportRecord  # 局部导入:避免测试中循环依赖

records = [
    ReportRecord(
        id=int(row["id"]),
        ticker=row["ticker"],
        generated_at=row["generated_at"],
        rating=row["rating"],
        confidence=(int(row["confidence"]) if row["confidence"] is not None else None),
        market_price=(
            float(row["market_price"]) if row["market_price"] is not None else None
        ),
        report_json="",  # 绘图不需要
    )
    for row in rows
]

st.plotly_chart(
    render_history_lines(records), key="history_lines", width="stretch"
)

st.subheader("历史记录表")

st.dataframe(
    df_display[["generated_at", "ticker", "rating", "confidence", "market_price"]],
    hide_index=True,
    width="stretch",
    column_config={
        "generated_at": st.column_config.DatetimeColumn(
            "日期",
            format="YYYY-MM-DD HH:mm",
        ),
        "ticker": st.column_config.TextColumn("股票代码"),
        "rating": st.column_config.TextColumn("评级"),
        "confidence": st.column_config.ProgressColumn(
            "置信度",
            min_value=0,
            max_value=100,
            format="%d",
        ),
        "market_price": st.column_config.NumberColumn(
            "价格",
            format="$%.2f",
        ),
    },
)

csv_bytes = df_display[
    ["generated_at", "ticker", "rating", "confidence", "market_price"]
].to_csv(index=False).encode("utf-8")
st.download_button(
    label="导出 CSV",
    data=csv_bytes,
    file_name="alphaquant_history.csv",
    mime="text/csv",
)

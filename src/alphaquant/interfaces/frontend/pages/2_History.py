"""History page: time-series chart and table of past reports, with CSV export."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pandas as pd
import streamlit as st

from alphaquant.interfaces.frontend.components.charts import render_history_lines
from alphaquant.infrastructure.persistence import DB


st.title("History")
st.write(
    "Browse past analysis runs. Pick a subset of tickers and a start date in the "
    "sidebar to narrow the view, then download the result as CSV."
)


db = DB()
db.init()


@st.cache_data(ttl=10)
def _load_history(tickers: tuple[str, ...], since_iso: str | None) -> list[dict]:
    """Cache the DB read for 10s to avoid hammering SQLite on every rerun.

    ``tickers`` is a tuple (hashable) and ``since_iso`` is a string so the args
    are stable across Streamlit reruns. We convert back to a list/datetime here
    so the underlying ``DB.get_history`` sees the typed values it expects.
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
    st.header("Filters")
    all_tickers = db.list_tickers()
    selected = st.multiselect(
        "Tickers",
        options=all_tickers,
        default=all_tickers,
        help="Tickers with at least one stored report.",
    )
    today = date.today()
    default_since = today - timedelta(days=365)
    since_date = st.date_input("Since", value=default_since, max_value=today)

# Normalize filters and load history.
since_dt = datetime.combine(since_date, time.min)
since_iso = since_dt.isoformat()
rows = _load_history(tuple(selected), since_iso)
df = pd.DataFrame(rows)

if df.empty:
    st.info("No reports yet. Run Analyze first.")
    st.stop()

# Sort newest first for the table; the chart helper re-sorts internally.
df_display = df.sort_values("generated_at", ascending=False).reset_index(drop=True)

# The chart wants ReportRecord-shaped objects. Re-hydrate the cache's dicts
# into the public model so render_history_lines' type contract is satisfied.
from alphaquant.infrastructure.persistence import ReportRecord  # local import: avoid cycles in tests

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
        report_json="",  # not needed for the chart
    )
    for row in rows
]

st.plotly_chart(render_history_lines(records), use_container_width=True)

st.subheader("History Table")

st.dataframe(
    df_display[["generated_at", "ticker", "rating", "confidence", "market_price"]],
    hide_index=True,
    use_container_width=True,
    column_config={
        "generated_at": st.column_config.DatetimeColumn(
            "Date",
            format="YYYY-MM-DD HH:mm",
        ),
        "ticker": st.column_config.TextColumn("Ticker"),
        "rating": st.column_config.TextColumn("Rating"),
        "confidence": st.column_config.ProgressColumn(
            "Confidence",
            min_value=0,
            max_value=100,
            format="%d",
        ),
        "market_price": st.column_config.NumberColumn(
            "Price",
            format="$%.2f",
        ),
    },
)

csv_bytes = df_display[
    ["generated_at", "ticker", "rating", "confidence", "market_price"]
].to_csv(index=False).encode("utf-8")
st.download_button(
    label="Export CSV",
    data=csv_bytes,
    file_name="alphaquant_history.csv",
    mime="text/csv",
)

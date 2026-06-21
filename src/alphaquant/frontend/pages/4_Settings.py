"""Settings page: database stats, history management, and service info.

Provides a read-only view of the local SQLite history, destructive
"clear all" action gated behind a ``st.dialog`` confirmation, a JSONL
export, and a quick look at the LLM service configuration.
"""
from __future__ import annotations

import os

import streamlit as st

from alphaquant.infrastructure.persistence import DB

st.title("Settings")

db = DB()
db.init()

# --- Database -----------------------------------------------------------
st.subheader("Database")
metric_cols = st.columns(3)
with metric_cols[0]:
    st.metric("Total reports", db.count())
with metric_cols[1]:
    st.metric("Unique tickers", len(db.list_tickers()))
with metric_cols[2]:
    size_kb = os.path.getsize(db.path) / 1024
    st.metric("DB size", f"{size_kb:.1f} KB")

st.caption(f"Path: `{db.path}`")

# --- Manage -------------------------------------------------------------
st.subheader("Manage")


@st.dialog("Confirm")
def _confirm_clear() -> None:
    """Modal confirmation for destructive ``delete_all`` action."""
    st.write(
        "This will permanently delete every stored report. "
        "This action cannot be undone."
    )
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("Yes, clear all", type="primary", use_container_width=True):
            db.delete_all()
            st.cache_data.clear()
            st.rerun()
    with col_no:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


if st.button("Clear all history", type="secondary"):
    _confirm_clear()

# Export: collect the generator into a single JSONL string, then encode.
try:
    jsonl_text = "\n".join(db.export_jsonl())
    jsonl_bytes = jsonl_text.encode("utf-8")
    st.download_button(
        label="Export all (JSONL)",
        data=jsonl_bytes,
        file_name="alphaquant_reports.jsonl",
        mime="application/jsonl",
    )
except Exception as exc:  # pragma: no cover - export best-effort
    st.error("Failed to build JSONL export. See server logs.")

# --- Service ------------------------------------------------------------
st.subheader("Service")

try:
    from alphaquant.config import get_settings  # type: ignore[import-not-found]

    settings = get_settings()
    st.write(f"**LLM model:** `{settings.litellm_model}`")
    api_key = settings.minimax_api_key
    if api_key.startswith("sk-cp-REPLACE_ME"):
        st.warning(
            "MINIMAX_API_KEY looks like a placeholder. "
            "Set a real key in `.env` to enable live analyses."
        )
    else:
        st.write("**MINIMAX_API_KEY:** set")
except Exception as exc:
    st.error(
        "Could not load service configuration. Check that `.env` exists "
        "and contains MINIMAX_API_KEY."
    )

st.write("**FastAPI docs:** http://localhost:8000/docs")

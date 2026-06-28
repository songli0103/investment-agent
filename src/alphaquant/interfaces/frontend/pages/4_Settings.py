"""设置页面:数据库统计、历史管理和服务信息。

提供本地 SQLite 历史数据库的只读视图,被 ``st.dialog`` 二次确认
保护的危险"全部清空"操作,JSONL 导出,以及 LLM 服务配置的快速查看。
"""
from __future__ import annotations

import os

import streamlit as st

from alphaquant.infrastructure.persistence import DB

st.title("设置")

db = DB()
db.init()

# --- 数据库 ---------------------------------------------------------------
st.subheader("数据库")
metric_cols = st.columns(3)
with metric_cols[0]:
    st.metric("报告总数", db.count())
with metric_cols[1]:
    st.metric("唯一 ticker 数", len(db.list_tickers()))
with metric_cols[2]:
    size_kb = os.path.getsize(db.path) / 1024
    st.metric("数据库大小", f"{size_kb:.1f} KB")

st.caption(f"路径: `{db.path}`")

# --- 管理 -----------------------------------------------------------------
st.subheader("管理")


@st.dialog("确认")
def _confirm_clear() -> None:
    """对危险 ``delete_all`` 操作的模态确认。"""
    st.write(
        "这将永久删除所有已存储的报告。"
        "此操作无法撤销。"
    )
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("确认清空", type="primary", width="stretch"):
            db.delete_all()
            st.cache_data.clear()
            st.rerun()
    with col_no:
        if st.button("取消", width="stretch"):
            st.rerun()


if st.button("清空所有历史", type="secondary"):
    _confirm_clear()

# 导出:将生成器收集为单个 JSONL 字符串,然后编码。
try:
    jsonl_text = "\n".join(db.export_jsonl())
    jsonl_bytes = jsonl_text.encode("utf-8")
    st.download_button(
        label="导出全部(JSONL)",
        data=jsonl_bytes,
        file_name="alphaquant_reports.jsonl",
        mime="application/jsonl",
    )
except Exception as exc:  # pragma: no cover - 导出尽力而为
    st.error("构建 JSONL 导出失败。请查看服务器日志。")

# --- 服务 -----------------------------------------------------------------
st.subheader("服务")

try:
    from alphaquant.infrastructure.config import get_settings  # type: ignore[import-not-found]

    settings = get_settings()
    st.write(f"**LLM 模型:** `{settings.litellm_model}`")
    api_key = settings.minimax_api_key
    if api_key.startswith("sk-cp-REPLACE_ME"):
        st.warning(
            "MINIMAX_API_KEY 看起来是占位符。"
            "请在 .env 中设置真实密钥以启用实际分析。"
        )
    else:
        st.write("**MINIMAX_API_KEY:** 已设置")
except Exception as exc:
    st.error(
        "无法加载服务配置。请检查 .env 文件是否存在并包含 MINIMAX_API_KEY。"
    )

st.write("**FastAPI 文档:** http://localhost:8000/docs")

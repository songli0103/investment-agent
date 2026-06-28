import streamlit as st

st.set_page_config(page_title="AlphaQuant", page_icon=None, layout="wide")

pages = [
    st.Page("pages/1_Analyze.py", title="分析", default=True),
    st.Page("pages/2_History.py", title="历史"),
    st.Page("pages/3_Compare.py", title="对比"),
    st.Page("pages/4_Settings.py", title="设置"),
]
nav = st.navigation(pages)
nav.run()

import streamlit as st

st.set_page_config(page_title="AlphaQuant", page_icon=None, layout="wide")

pages = [
    st.Page("pages/1_Analyze.py", title="Analyze", default=True),
    st.Page("pages/2_History.py", title="History"),
    st.Page("pages/3_Compare.py", title="Compare"),
    st.Page("pages/4_Settings.py", title="Settings"),
]
nav = st.navigation(pages)
nav.run()

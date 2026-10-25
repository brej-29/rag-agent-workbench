"""Compliance Workbench UI: entry point and app shell (top navigation, API status)."""
import streamlit as st

from services.shell import status_badge

st.set_page_config(
    page_title="Compliance Workbench",
    page_icon=":material/fact_check:",
    layout="wide",
)

pages = st.navigation(
    [
        st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
        st.Page("views/ask.py", title="Ask", icon=":material/forum:"),
        st.Page("views/checklist.py", title="Checklist", icon=":material/checklist:"),
        st.Page("views/gapcheck.py", title="Gap check", icon=":material/rule:"),
        st.Page("views/evaluation.py", title="Evaluation", icon=":material/monitoring:"),
    ],
    position="top",
)

with st.container(horizontal=True, horizontal_alignment="right"):
    status_badge()

pages.run()

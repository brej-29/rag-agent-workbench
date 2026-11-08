"""Compliance Workbench UI: entry point and app shell (top navigation, API status)."""
import streamlit as st

from services.shell import inject_theme, status_badge, theme_switch

st.set_page_config(
    page_title="Compliance Workbench",
    page_icon=":material/fact_check:",
    layout="wide",
)
inject_theme()

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

with st.container(key="app-header", horizontal=True, horizontal_alignment="right", vertical_alignment="center"):
    st.html('<span class="cw-wordmark">Compliance Workbench</span>')
    status_badge()
    theme_switch()

pages.run()

"""Home: what the workbench does, what it covers, and how well it is measured."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

import httpx
import pandas as pd
import streamlit as st

from services.backend_client import post_checklist
from services.eval_data import coverage_rows, headline, load_results
from services.presenters import profile_words
from services.shell import PROSE_WIDTH, api_key, backend_state, backend_url, facts_or_none

s = st.session_state
s.setdefault("profile", {})
results = load_results()

JOBS = [
    ("views/ask.py", "Ask", ":material/forum:",
     "Ask about the AI Act, GDPR, NIS2, DORA, the Cyber Resilience Act or NIST. Every answer cites the provision "
     "and says whether it applies on the date you choose."),
    ("views/checklist.py", "Checklist", ":material/checklist:",
     "Answer a few questions about your product and organisation. See which articles bind you now, which "
     "come later, and what would decide the rest."),
    ("views/gapcheck.py", "Gap check", ":material/rule:",
     "Load your own policy and check it, duty by duty, against those articles. Each verdict quotes the law and "
     "your policy."),
]


@st.cache_data(ttl=300, show_spinner=False)
def _counts(url: str, key: Optional[str], profile: Dict[str, Any], as_of: str) -> Dict[str, int]:
    return post_checklist(url, key, profile, as_of)["counts"]


st.title("Compliance Workbench", anchor=False)
st.markdown("EU and NIST rules for AI, data and cyber, checked against your profile and your policies.",
            width=PROSE_WIDTH)

words = profile_words(s.profile, facts_or_none()) if s.profile else ""
with st.container(border=True, horizontal=True, vertical_alignment="center"):
    st.markdown(f":material/person: **Your profile:** "
                f"{words or 'Not set yet. Answer a few questions to see what applies to you.'}")
    st.page_link("views/checklist.py", label="Edit your profile" if words else "Set up your profile",
                 icon=":material/tune:")

counts = None
if s.profile and backend_state() == "ready":
    try:
        counts = _counts(backend_url(), api_key(), s.profile, date.today().isoformat())
    except httpx.HTTPError:
        counts = None

st.header("What you can do", anchor=False)
for col, (page, title, icon, text) in zip(st.columns(3), JOBS):
    with col.container(border=True, height="stretch"):
        st.subheader(title, anchor=False)
        st.markdown(text)
        if page.endswith("checklist.py") and counts:
            st.markdown(f"For you today: **{counts['now']}** apply now · **{counts['coming']}** coming · "
                        f"**{counts['depends']}** depend on your answers")
        st.page_link(page, label=f"Open {title}", icon=icon)

st.header("What's covered", anchor=False)
corpus = results.get("corpus")
if corpus:
    st.dataframe(pd.DataFrame(coverage_rows(corpus)), hide_index=True, width="stretch", column_config={
        "Instrument": st.column_config.TextColumn("Instrument", width="large"),
        "Type": st.column_config.TextColumn("Type", width="medium"),
        "Provisions": st.column_config.NumberColumn("Provisions", format="%d", width="small"),
        "Source": st.column_config.LinkColumn("Source", display_text="Official text", width="small"),
        "Retrieved": st.column_config.TextColumn("Retrieved", width="small")})
else:
    st.caption("Not published yet: run `python eval/publish_results.py`.")
st.markdown(":material/lock: **ISO/IEC 27001 and 42001** are licensed: load your own copy on the Gap check page. "
            "It stays in your browser session.  \n"
            ":material/block: **Anything else**, such as HIPAA or CCPA, is declined rather than guessed.",
            width=PROSE_WIDTH)

st.header("How it's measured", anchor=False)
numbers = headline(results)
if numbers:
    with st.container(horizontal=True):
        for n in numbers:
            st.metric(n["label"], n["value"], border=True, help=n["note"])
    st.page_link("views/evaluation.py", label="See how each number is measured", icon=":material/monitoring:")
else:
    st.caption("Not published yet: run `python eval/publish_results.py`.")

st.divider()
st.caption(f"Information from official EU and NIST text, not legal advice. The same jobs are available to Claude "
           f"and other MCP clients at `{backend_url()}/mcp`.", width=PROSE_WIDTH)

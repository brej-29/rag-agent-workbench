"""Home: a plain-language path from uncertainty to a measured compliance result."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

import httpx
import streamlit as st

from services.backend_client import post_checklist
from services.eval_data import load_results, trust_proof
from services.presenters import RULEBOOK_LINES, dot_grid_html, esc, profile_words, rulebook_stack_html
from services.shell import PROSE_WIDTH, api_key, backend_state, backend_url, facts_or_none

s = st.session_state
s.setdefault("profile", {})
results = load_results()


@st.cache_data(ttl=300, show_spinner=False)
def _counts(url: str, key: Optional[str], profile: Dict[str, Any], as_of: str) -> Dict[str, int]:
    return post_checklist(url, key, profile, as_of)["counts"]


hero, stack = st.columns([1.05, 0.95], gap="large", vertical_alignment="center")
with hero:
    st.html('<div class="cw-hero-copy"><h1>Which EU and US rules apply to your AI, data or cyber product?</h1>'
            '<p>Plain answers quoted from the official text, with the date each rule starts to apply.</p></div>')
    with st.container(key="hero-actions", horizontal=True, gap="small"):
        with st.container(key="cta-primary", width="content"):
            st.page_link("views/checklist.py", label="Try a ready-made example", icon=":material/auto_awesome:")
        with st.container(key="cta-secondary", width="content"):
            st.page_link("views/ask.py", label="Ask a question", icon=":material/forum:")
with stack:
    st.html(rulebook_stack_html())

words = profile_words(s.profile, facts_or_none()) if s.profile else ""
st.html(f'<div class="cw-profile"><strong>Your profile:</strong> '
        f'{esc(words or "Not set yet. Answer a few questions to see what applies to you.")}</div>')
st.page_link("views/checklist.py", label="Edit your profile" if words else "Set up your profile")

counts = None
if s.profile and backend_state() == "ready":
    try:
        counts = _counts(backend_url(), api_key(), s.profile, date.today().isoformat())
    except httpx.HTTPError:
        counts = None

count_line = ""
if counts:
    count_line = (f"<p>For your saved profile today: <strong>{counts['now']}</strong> apply now, "
                  f"<strong>{counts['coming']}</strong> are coming, and "
                  f"<strong>{counts['depends']}</strong> depend on your answers.</p>")
st.html(f"""
<section class="cw-section"><h2>How it works</h2><div class="cw-steps">
  <div class="cw-step"><b>1</b><strong>Tell it about you</strong><p>Choose an example or answer only what you know.</p></div>
  <div class="cw-step"><b>2</b><strong>See what applies</strong><p>Get the articles that apply now, later, or after one more answer.</p></div>
  <div class="cw-step"><b>3</b><strong>Check your policy</strong><p>Compare each duty with your own words and see the supporting quotes.</p></div>
</div></section>
""")
st.html('<section class="cw-section cw-tools-head"><h2>Three tools, one set of sources</h2></section>')
TOOLS = [("views/ask.py", "Ask", "Use this when you need one answer with the official passage and date.", "",
          "Ask a question"),
         ("views/checklist.py", "Checklist", "Use this when you need to know which rules apply to your product.",
          count_line, "Build a checklist"),
         ("views/gapcheck.py", "Gap check", "Use this when you need to compare a policy with the duties that apply.",
          "", "Check a policy")]
for col, (page, title, text, extra, cta) in zip(st.columns(3, gap="medium"), TOOLS):
    with col.container(key=f"tool-{title.lower().replace(' ', '-')}"):
        st.html(f'<div class="cw-tool"><h3>{title}</h3><p>{text}</p>{extra}</div>')
        st.page_link(page, label=cta)


proof = trust_proof(results)
if proof:
    sentences = "".join(f"<p>{esc(item['sentence'])}</p>" for item in proof)
    st.html(f'<section class="cw-section cw-proof"><div class="cw-proof-copy"><h2>Can you trust it?</h2>'
            f'{sentences}</div>'
            f'{dot_grid_html(proof[0]["value"], proof[0]["total"])}</section>')
    st.page_link("views/evaluation.py", label="See how each number is measured", icon=":material/monitoring:")
else:
    st.html('<section class="cw-section"><h2>Can you trust it?</h2>'
            '<p>The published evaluation is not available yet.</p></section>')

library = "".join(f'<div class="cw-library-row"><strong style="--tab:{esc(color)}">{esc(name)}</strong>'
                  f'<span>{esc(line)}</span></div>' for name, color, line in RULEBOOK_LINES)
st.html(f'<section class="cw-section"><h2>What is in the library</h2><div class="cw-library">{library}</div>'
        '<p>ISO/IEC 27001 and 42001 are licensed. Bring your own copy on Gap check. Questions about other rules '
        'are declined instead of guessed.</p></section>')

st.divider()
st.caption(f"Information from official EU and NIST text, not legal advice. You can use the same Ask, Checklist and "
           f"Gap check tools from an MCP client at `{backend_url()}/mcp`.", width=PROSE_WIDTH)

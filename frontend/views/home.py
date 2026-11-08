"""Home: what this is, who it is for, what it covers and when the rules start, all in plain words."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

import httpx
import streamlit as st

from services.backend_client import post_checklist
from services.eval_data import load_results
from services.home_html import (
    REVEAL_JS, demo_html, proof_html, proof_panels, stats, stats_html, ticker_html, timeline, timeline_html,
    timeline_summary,
)
from services.presenters import RULEBOOK_LINES, esc, profile_words, rulebook_stack_html
from services.shell import PROSE_WIDTH, api_key, backend_state, backend_url, facts_or_none

s = st.session_state
s.setdefault("profile", {})
results = load_results()
today = date.today()
milestones = timeline(results.get("timeline") or {}, today)


@st.cache_data(ttl=300, show_spinner=False)
def _counts(url: str, key: Optional[str], profile: Dict[str, Any], as_of: str) -> Dict[str, int]:
    return post_checklist(url, key, profile, as_of)["counts"]


def section(title: str, lead: str = "") -> None:
    st.html(f'<header class="cw-section-head cw-reveal"><h2>{esc(title)}</h2>'
            + (f"<p>{esc(lead)}</p>" if lead else "") + "</header>")


# Hero ------------------------------------------------------------------------------------------------
with st.container(key="hero"):
    hero, stack = st.columns([1.05, 0.95], gap="large", vertical_alignment="center")
    with hero:
        st.html('<div class="cw-hero-copy"><h1>Which EU and US rules apply to your AI, data or cyber product?</h1>'
                '<p>Plain answers quoted from the official text, with the date each rule starts to apply. '
                'For founders, product teams and engineers, not only lawyers.</p></div>' + ticker_html())
        with st.container(key="hero-actions", horizontal=True, gap="small"):
            with st.container(key="cta-primary", width="content"):
                st.page_link("views/checklist.py", label="Try a ready-made example", icon=":material/auto_awesome:")
            with st.container(key="cta-secondary", width="content"):
                st.page_link("views/ask.py", label="Ask a question", icon=":material/forum:")
    with stack:
        st.html(rulebook_stack_html())
        st.caption("Hover or tab to a rulebook to see what it covers.")

numbers = stats(results.get("corpus"), results.get("timeline") or {}, len(milestones))
if numbers:
    st.html(stats_html(numbers))

words = profile_words(s.profile, facts_or_none()) if s.profile else ""
counts = None
if s.profile and backend_state() == "ready":
    try:
        counts = _counts(backend_url(), api_key(), s.profile, today.isoformat())
    except httpx.HTTPError:
        counts = None
mine = (f" Today <mark>{counts['now']} articles apply</mark>, {counts['coming']} are coming and "
        f"{counts['depends']} depend on your answers." if counts else "")
with st.container(key="profile-row", horizontal=True, vertical_alignment="center"):
    st.html(f'<div class="cw-profile"><strong>Your profile:</strong> '
            f'{esc(words or "Not set yet. Pick an example or answer a few questions.")}{mine}</div>')
    st.page_link("views/checklist.py", label="Edit your profile" if words else "Set up your profile",
                 icon=":material/tune:")

# A real answer, replayed -----------------------------------------------------------------------------------
section("See an answer, start to finish",
        "Every answer shows how it was found, quotes the law, and says whether the rule applies today.")
left, right = st.columns([0.9, 1.1], gap="large", vertical_alignment="center")
with left:
    st.html('<ul class="cw-points cw-reveal">'
            '<li><b>It finds the passage first.</b> The answer is written only from the official text it retrieved.</li>'
            '<li><b>It says no when it should.</b> If nothing relevant is found, you get a plain decline, not a guess.</li>'
            '<li><b>It checks dates.</b> Each cited rule shows whether it is in force on the date you choose.</li>'
            '</ul>')
    st.caption("Replayed from a real answer on 8 November 2026. Live answers take a few seconds.")
with right:
    st.html(demo_html())

# How it works -------------------------------------------------------------------------------------------
st.html("""<section class="cw-section cw-reveal"><h2>How it works</h2><ol class="cw-steps">
  <li><b>1</b><strong>Tell it about you</strong><p>Pick an example such as “chatbot startup” or answer only what you know.</p></li>
  <li><b>2</b><strong>See what applies</strong><p>Articles that apply now, ones coming later, and the one question that would settle the rest.</p></li>
  <li><b>3</b><strong>Check your policy</strong><p>Paste your policy and see, duty by duty, what it covers, with quotes from both sides.</p></li>
</ol></section>""")

# Timeline -----------------------------------------------------------------------------------------------
if milestones:
    section("When the rules start", timeline_summary(milestones))
    st.html(f'<div class="cw-timeline-wrap cw-reveal">{timeline_html(milestones)}</div>')
    st.caption("Scroll sideways for later dates. Hover or tab to a date to read the sentence of the law that sets it.")

# Tools --------------------------------------------------------------------------------------------------
section("Three tools, one set of sources")
TOOLS = [("views/ask.py", "Ask", "forum", "One question, one cited answer with its date.", "Ask a question"),
         ("views/checklist.py", "Checklist", "checklist", "Which articles bind your product, now and later.",
          "Build a checklist"),
         ("views/gapcheck.py", "Gap check", "rule", "What your policy covers and what it misses.", "Check a policy")]
for col, (page, title, icon, text, cta) in zip(st.columns(3, gap="medium"), TOOLS):
    with col.container(key=f"tool-{title.lower().replace(' ', '-')}"):
        st.html(f'<div class="cw-tool"><h3>{title}</h3><p>{text}</p></div>')
        st.page_link(page, label=cta, icon=f":material/{icon}:")

# Trust --------------------------------------------------------------------------------------------------
panels = proof_panels(results)
if panels:
    section("Can you trust it?", "Measured on labelled test questions. In the first panel each dot is one percentage point; in the others each dot is one question.")
    st.html(proof_html(panels))
    st.page_link("views/evaluation.py", label="See how each number is measured", icon=":material/monitoring:")

# Library ------------------------------------------------------------------------------------------------
library = "".join(f'<div class="cw-library-row"><strong style="--tab:{esc(color)}">{esc(name)}</strong>'
                  f'<span>{esc(line)}</span></div>' for name, color, line in RULEBOOK_LINES)
st.html(f'<section class="cw-section cw-reveal"><h2>What is in the library</h2><div class="cw-library">{library}</div>'
        '<p>ISO/IEC 27001 and 42001 are licensed. Bring your own copy on Gap check. Questions about other rules '
        'are declined instead of guessed.</p></section>')

st.divider()
st.caption(f"Information from official EU and NIST text, not legal advice. You can use the same Ask, Checklist and "
           f"Gap check tools from an MCP client at `{backend_url()}/mcp`.", width=PROSE_WIDTH)
st.html(REVEAL_JS, unsafe_allow_javascript=True)

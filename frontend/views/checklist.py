"""Checklist: which EU-law articles bind your product on a date, from facts you choose."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

from services.backend_client import get_facts, post_checklist
from services.presenters import (
    CROSSWALK_CAVEAT, GROUP_TITLES, checklist_csv, checklist_markdown, crosswalk_lines, dates_note, esc, profile_words,
    status_label,
)
from services.shell import PROSE_WIDTH, api_key, backend_state, backend_url, wake_watcher

NOT_SURE, NONE = "Not sure", "none of these"
s = st.session_state
s.setdefault("profile", {})


@st.cache_data(ttl=600, show_spinner=False)
def _facts(url: str, key: Optional[str]) -> List[Dict[str, Any]]:
    return get_facts(url, key)


def _sync(f: Dict[str, Any], key: str) -> None:
    """Widget -> session profile, before the rerun computes the checklist. A deciding question shows twice (in the
    form and as a shortcut in "Answer these"), so the other copy takes the same value."""
    raw = s[key]
    s[("q-" if key.startswith("f-") else "f-") + f["id"]] = raw
    if f["type"] == "bool":
        value = None if raw == NOT_SURE else raw == "Yes"
    elif f["type"] == "choice":
        value = None if raw == NOT_SURE else raw
    else:
        value = None if not raw else [v for v in raw if v != NONE]
    if value is None:
        s.profile.pop(f["id"], None)
    else:
        s.profile[f["id"]] = value


def fact_input(f: Dict[str, Any], key: str) -> None:
    labels, current = f.get("labels") or {}, s.profile.get(f["id"])
    common = {"key": key, "on_change": _sync, "args": (f, key), "help": f.get("help")}
    # Once a key holds a value (set by the widget or by _sync), the value comes from session state alone.
    held = key in s
    if f["type"] == "bool":
        index = 0 if current is None else (1 if current else 2)
        st.radio(f["label"], [NOT_SURE, "Yes", "No"], horizontal=True, **({} if held else {"index": index}), **common)
    elif f["type"] == "choice":
        options = [NOT_SURE] + f["values"]
        index = options.index(current) if current in options else 0
        st.selectbox(f["label"], options, format_func=lambda v: labels.get(v, v),
                     **({} if held else {"index": index}), **common)
    else:
        default = [] if current is None else (current or [NONE])
        st.multiselect(f["label"], f["values"] + [NONE], placeholder=NOT_SURE, format_func=lambda v: labels.get(v, v),
                       **({} if held else {"default": default}), **common)


@st.dialog("Article", position="right", width="medium")
def article_drawer(item: Dict[str, Any], as_of: str) -> None:
    st.caption(item["framework"])
    st.subheader(f"{item['ref']} — {item['title']}", anchor=False)
    label = status_label(item)
    if label:
        st.badge(f"{label[1]} (on {as_of})", color=label[0])
    if item.get("status_note"):
        st.caption(f"Dates: {dates_note(item['status_note'])}")
    if item.get("url"):
        st.markdown(f"[Official text]({item['url']})")
    for p in item["paragraphs"]:
        with st.container(border=True):
            st.caption(p["ref"])
            st.html(f'<div style="white-space:pre-wrap;line-height:1.6">{esc(p.get("text") or "")}</div>')


def item_card(item: Dict[str, Any], facts: List[Dict[str, Any]], as_of: str) -> None:
    by_id = {f["id"]: f for f in facts}
    with st.container(border=True):
        st.markdown(f"**{item['ref']}** · {item['title']}")
        with st.container(horizontal=True, gap="small", vertical_alignment="center"):
            label = status_label(item)
            if label:
                st.badge(label[1], color=label[0])
            if item["kind"] == "prohibition":
                st.badge("prohibition", color="red")
            if item["binding"] == "directive":
                st.badge("via national law", color="gray")
            if st.button("Read", key=f"read-{item['article']}", type="tertiary", icon=":material/menu_book:"):
                article_drawer(item, as_of)
        if item["depends_on"]:
            st.caption("Depends on: " + "; ".join(by_id[f]["label"] for f in item["depends_on"] if f in by_id))
        elif item.get("why"):
            st.caption(f"Applies because: {profile_words(item['why'], facts)}")
        if item.get("note"):
            st.caption(item["note"])
        groups = crosswalk_lines(item.get("crosswalk"), s.get("workspace", {}).get("standards"))
        if groups:
            with st.expander(f"Matches in other frameworks · {sum(len(l) + h for _, l, h in groups)}"):
                for group, lines, hidden in groups:
                    st.markdown(f"**{group}**")
                    for label, color, note in lines:
                        st.markdown(f":{color}-badge[{esc(label)}] {esc(note)}")
                    if hidden:
                        st.caption(f"and {hidden} more")
                st.caption(CROSSWALK_CAVEAT)


st.title("Your checklist", anchor=False)
st.markdown("Which articles of the EU AI Act, GDPR, NIS2, DORA and the Cyber Resilience Act bind you, and since or "
            "from when. Answer what you know; the rest stays “not sure”. Built from a rules table, not a model. "
            "Not legal advice.", width=PROSE_WIDTH)
state = backend_state()
if state == "waking":
    wake_watcher()
    st.skeleton(height=240)
    st.stop()
if state != "ready":
    st.error(f"Cannot reach the API at `{backend_url()}`. Check BACKEND_BASE_URL.", icon=":material/cloud_off:")
    st.stop()
try:
    facts = _facts(backend_url(), api_key())
except httpx.HTTPError as exc:
    st.error(f"Could not load the questions: {exc}", icon=":material/error:")
    st.stop()

as_of = st.sidebar.date_input("Checklist as of", value=date.today(), min_value=date(2016, 1, 1),
                              max_value=date(2035, 12, 31)).isoformat()
try:
    cl = post_checklist(backend_url(), api_key(), s.profile, as_of)
except httpx.HTTPError as exc:
    st.error(f"The checklist could not be built: {exc}", icon=":material/error:")
    st.stop()

left, right = st.columns([2, 3], gap="large")
with left:
    st.subheader("Your situation", anchor=False)
    for group in dict.fromkeys(f["group"] for f in facts):
        with st.expander(group, expanded=group == facts[0]["group"]):
            for f in facts:
                if f["group"] == group:
                    fact_input(f, f"f-{f['id']}")
with right:
    c = cl["counts"]
    with st.container(horizontal=True):
        st.metric("Apply now", c["now"])
        st.metric("Coming", c["coming"])
        st.metric("Depend on answers", c["depends"])
        st.metric("Do not apply", c["not_applicable"])
    if cl["questions"]:
        with st.container(border=True):
            st.markdown("**Answer these to narrow it down**")
            by_id = {f["id"]: f for f in facts}
            for q in cl["questions"]:
                fact_input(by_id[q["fact"]], f"q-{q['fact']}")
                st.caption(f"Decides {q['decides']} article{'s' if q['decides'] != 1 else ''}")
    for key, title in GROUP_TITLES:
        if cl["groups"][key]:
            st.subheader(f"{title} · {len(cl['groups'][key])}", anchor=False)
            for item in cl["groups"][key]:
                item_card(item, facts, as_of)
    for pr in cl["practices"]:
        tail = " · linked to your items" if pr.get("narrowed") else ""
        with st.expander(f"Recommended practices (voluntary): {pr['framework']}{tail}"):
            st.caption(pr["reason"])
            for g in pr["groups"]:
                st.markdown(f"**{g['name']}**")
                for i in g["items"]:
                    st.markdown(f"- {i['ref']}: {i['title']}")
    if cl["not_applicable"]:
        with st.expander(f"{c['not_applicable']} articles do not apply to you"):
            for x in cl["not_applicable"]:
                st.markdown(f"- {x['ref']} · {x['title']} — because: {profile_words(x['because'], facts) or 'not stated'}")
    with st.container(horizontal=True):
        st.download_button("Download Markdown", checklist_markdown(cl, facts), file_name=f"checklist-{as_of}.md",
                           mime="text/markdown", icon=":material/download:")
        st.download_button("Download CSV", checklist_csv(cl), file_name=f"checklist-{as_of}.csv", mime="text/csv",
                           icon=":material/table:")

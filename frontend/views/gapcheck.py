"""Gap check: your policy against the articles that bind you, duty by duty, with quotes from both."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import httpx
import streamlit as st

from services.backend_client import gapcheck_plan, parse_document, post_checklist, stream_gapcheck
from services.file_convert import convert_uploaded_file_to_text
from services.presenters import (POLICY_CHARS, STANDARD_CHARS, UPLOAD_MB, gap_csv, gap_markdown, text_limit_error,
                                 verdict_label)
from services.shell import api_key, backend_state, backend_url, wake_watcher

s = st.session_state
s.setdefault("workspace", {"policies": {}, "standards": {}})
s.setdefault("profile", {})
s.setdefault("gap_results", None)
TYPES = ["pdf", "md", "txt", "docx", "html", "htm"]


def _load(uploaded, kind: str) -> None:
    try:
        uploaded.seek(0)
        text, _ = convert_uploaded_file_to_text(uploaded)
        err = text_limit_error(kind, len(text))
        if err:
            st.error(err, icon=":material/error:")
            return
        parsed = parse_document(backend_url(), api_key(), kind, Path(uploaded.name).stem, text)
    except httpx.HTTPStatusError as exc:
        st.error(exc.response.json().get("detail", "The API refused this file."), icon=":material/error:")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read this file: {exc}", icon=":material/error:")
        return
    if kind == "policy":
        s.workspace["policies"][parsed["title"]] = {"text": text, "passages": len(parsed["passages"])}
    else:
        s.workspace["standards"][parsed["standard"]] = {"name": parsed["name"], "clauses": parsed["clauses"],
                                                        "coverage": parsed["coverage"], "title": parsed["title"]}


def documents_panel() -> None:
    st.subheader("Your documents", anchor=False)
    st.caption("Kept in this browser session only. The server stores nothing; closing the tab forgets them.")
    with st.form("policy_form", clear_on_submit=True, border=False):
        pol = st.file_uploader("Add a policy", type=TYPES, max_upload_size=UPLOAD_MB["policy"])
        st.caption(f"PDF, Word, Markdown, HTML or text · up to {POLICY_CHARS:,} characters of text")
        if st.form_submit_button("Load policy") and pol:
            _load(pol, "policy")
    for title, p in list(s.workspace["policies"].items()):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f":material/description: **{title}** · {p['passages']} passages")
            if st.button("Remove", key=f"rm-p-{title}", type="tertiary"):
                del s.workspace["policies"][title]
                st.rerun()
    with st.form("standard_form", clear_on_submit=True, border=False):
        std = st.file_uploader("Add your licensed copy of ISO/IEC 27001 or 42001", type=TYPES,
                               max_upload_size=UPLOAD_MB["standard"])
        st.caption(f"Up to {STANDARD_CHARS:,} characters of text · read for this session only")
        if st.form_submit_button("Load standard") and std:
            _load(std, "standard")
    for key, c in list(s.workspace["standards"].items()):
        cov = c["coverage"]
        found = f"{cov['annex_a']} of {cov['annex_a_expected']} Annex A controls" if "annex_a_expected" in cov else f"{cov['clauses']} clauses"
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f":material/lock: **{c['name']}** (your copy) · {found} found")
            if st.button("Remove", key=f"rm-s-{key}", type="tertiary"):
                del s.workspace["standards"][key]
                st.rerun()


def _options(cl: Dict[str, Any]) -> Dict[str, str]:
    opts = {}
    for g in ("now", "coming"):
        for it in cl["groups"][g]:
            opts[it["article"]] = f"{it['ref']} · {it['title']}"
            sections = sorted({p["id"].split("pt-")[1].split(".")[0] for p in it["paragraphs"] if "/pt-" in p["id"]}, key=int)
            for sec in sections:
                opts[f"{it['paragraphs'][-1]['id'].split('/pt-')[0]}#{sec}"] = f"{it['ref'].split(' Art.')[0]} Annex, section {sec}"
    return opts


def show_result(r: Dict[str, Any]) -> None:
    c = r["counts"]
    with st.expander(f"{r['ref']} {r['title']} · {c['covered']} covered · {c['partial']} partial · "
                     f"{c['missing']} missing · {c['unverified']} to review", expanded=bool(c["missing"] or c["unverified"])):
        for d in r["duties"]:
            color, label = verdict_label(d["verdict"])
            st.markdown(f":{color}-badge[{label}] **{d['ref']}** {d.get('reason', '')}")
            if d.get("law_quote"):
                st.caption(f"Law: “{d['law_quote']}”")
            if d.get("policy_quote") and d["verdict"] != "unverified":
                st.caption(f"Your policy ({d['passage_id']}): “{d['policy_quote']}”")
            if d["verdict"] == "unverified":
                st.caption(f"Could not be verified: {d['problem']}" + (f" (model said {d['claimed']})" if d.get("claimed") else ""))


st.title("Gap check", anchor=False)
st.caption("Checks your own policy against the articles that bind you, duty by duty. Every verdict quotes the law "
           "and your policy word for word, or says it could not be verified. Advisory, not legal advice.")
state = backend_state()
if state == "waking":
    wake_watcher()
    st.skeleton(height=240)
    st.stop()
if state != "ready":
    st.error(f"Cannot reach the API at `{backend_url()}`. Check BACKEND_BASE_URL.", icon=":material/cloud_off:")
    st.stop()

left, right = st.columns([2, 3], gap="large")
with left:
    documents_panel()
with right:
    st.subheader("What to check", anchor=False)
    if not s.workspace["policies"]:
        st.info("Load a policy on the left to start.", icon=":material/upload_file:")
        st.stop()
    title = st.selectbox("Policy", list(s.workspace["policies"]))
    cl = post_checklist(backend_url(), api_key(), s.profile, date.today().isoformat())
    opts = _options(cl)
    if not opts:
        st.warning("Your checklist has no binding articles yet. Fill in your profile on the Checklist page.")
        st.page_link("views/checklist.py", label="Open the checklist", icon=":material/checklist:")
        st.stop()
    picked = st.multiselect("Articles and annex sections", list(opts), default=[k for k in opts if "#" not in k][:6],
                            format_func=opts.get, max_selections=12)
    body = {"policy_text": s.workspace["policies"][title]["text"], "articles": picked, "profile": s.profile or None}
    if picked:
        plan = gapcheck_plan(backend_url(), api_key(), body)
        st.caption(f"{sum(u['duties'] for u in plan['units'])} duties · about {plan['est_tokens']:,} tokens · "
                   f"roughly {max(plan['est_minutes'], 0.2)} min on the free model tier")
        if plan["too_big"]:
            st.warning("That is more than one run allows. Choose fewer articles or sections.", icon=":material/warning:")
        if st.button("Run gap check", type="primary", disabled=plan["too_big"], icon=":material/play_arrow:"):
            results, summary = [], None
            progress = st.status("Checking…", expanded=True)
            try:
                for kind, data in stream_gapcheck(backend_url(), api_key(), body):
                    if kind == "article":
                        results.append(data)
                        with progress:
                            show_result(data)
                    elif kind == "done":
                        summary = data["summary"]
            except (httpx.HTTPError, RuntimeError) as exc:
                progress.update(label="Stopped", state="error")
                st.error(f"The gap check stopped: {exc}", icon=":material/error:")
            if summary:
                progress.update(label="Done", state="complete", expanded=False)
                s.gap_results = {"title": title, "results": results, "summary": summary}
    if s.gap_results:
        res = s.gap_results
        with st.container(horizontal=True):
            for v in ("covered", "partial", "missing", "unverified"):
                st.metric(verdict_label(v)[1].capitalize(), res["summary"][v])
        for r in res["results"]:
            show_result(r)
        with st.container(horizontal=True):
            st.download_button("Download Markdown", gap_markdown(res["results"], res["summary"], res["title"]),
                               file_name="gap-check.md", mime="text/markdown", icon=":material/download:")
            st.download_button("Download CSV", gap_csv(res["results"]), file_name="gap-check.csv", mime="text/csv",
                               icon=":material/table:")

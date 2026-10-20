"""Ask: grounded chat with a live pipeline trace and per-source evidence."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

from services.backend_client import namespace_stats, post_upload_text, stream_chat
from services.file_convert import DocumentConverter, convert_uploaded_file_to_text
from services.presenters import (
    EXAMPLES, FRAMEWORKS, HIGH_RISK, REGS_EXAMPLES, REGS_NAMESPACE, chip_label, cite_markdown, citing_sentences, esc,
    floor_bars_html, profile_words, status_label, step_line, trace_summary, verdict,
)
from services.shell import api_key, backend_state, backend_url, is_dark, wake_watcher

DEFAULT_NAMESPACE = REGS_NAMESPACE
AVATARS = {"user": ":material/person:", "assistant": ":material/fact_check:"}
s = st.session_state
s.setdefault("messages", [])
s.setdefault("uploads", [])


# ---------------------------------------------------------------------------
# Evidence drawer and trace
# ---------------------------------------------------------------------------

@st.dialog("Evidence", position="right", width="medium")
def evidence_drawer(n: int, src: Dict[str, Any], answer: str, floor: Optional[float],
                    as_of: Optional[str] = None) -> None:
    web = src.get("source") == "web"
    st.caption(f"Source {n} · {'web result' if web else src.get('source') or 'knowledge base'}")
    st.subheader(src.get("title") or f"Source {n}", anchor=False)
    label = status_label(src)
    if label:
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.badge(label[1], color=label[0])
            if as_of:
                st.caption(f"on {as_of}")
        if src.get("status_note"):
            st.caption(f"Dates: {src['status_note']}")
    if src.get("url"):
        st.markdown(f"[Official text]({src['url']})" if label else f"[{src['url']}]({src['url']})")
    if floor is not None and not web:
        st.html(floor_bars_html([float(src.get("score") or 0.0)], floor, is_dark()))
    cited = citing_sentences(answer, n)
    st.markdown("**Cited for**" if cited else "**Not cited in the answer**")
    for sentence in cited:
        st.markdown(f"> {cite_markdown(sentence, n_sources=n)}")
    st.markdown("**Passage**")
    with st.container(border=True):
        st.html(f'<div style="white-space:pre-wrap;line-height:1.6">{esc(src.get("chunk_text") or "")}</div>')
    if src.get("doc_id"):
        st.caption(f"Document `{src['doc_id']}`")


def render_trace(steps: List[Dict[str, Any]]) -> None:
    for step in steps:
        line = step_line(step)
        if line:
            st.markdown(("" if step["status"] == "running" else ":material/check_small: ") + line)
        detail = step.get("detail") or {}
        if step["id"] == "gate" and detail.get("scores"):
            st.html(floor_bars_html(detail["scores"], detail["floor"], is_dark()))


def _floor(steps: List[Dict[str, Any]]) -> Optional[float]:
    gate = next((x for x in steps if x["id"] == "gate"), None)
    return (gate.get("detail") or {}).get("floor") if gate else None


def render_answer(msg: Dict[str, Any], idx: int) -> None:
    resp, steps = msg.get("response") or {}, msg.get("steps") or []
    sources = resp.get("sources") or []
    abstained = bool(resp.get("insufficient_context"))
    total_ms = float((resp.get("timings") or {}).get("total_ms") or 0.0)
    with st.status(trace_summary(abstained, total_ms, len(sources), resp.get("abstain_reason")),
                   state="complete", expanded=False):
        render_trace(steps)
    st.markdown(cite_markdown(msg["content"], len(sources)))
    for warning in resp.get("warnings") or []:
        st.caption(f":material/warning: {warning}")

    color, label = verdict(resp)
    usage = resp.get("usage") or {}
    meta = [f"{total_ms / 1000:.1f} s"]
    if usage.get("total_tokens"):
        meta.append(f"{usage['total_tokens']:,} tokens")
    if usage.get("estimated_cost_usd") is not None:
        meta.append(f"≈ ${usage['estimated_cost_usd']:.4f}")
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.badge(label, color=color)
        if resp.get("as_of"):
            meta.insert(0, f"as of {resp['as_of']}")
            meta.append("not legal advice")
        st.caption(" · ".join(meta))
    noted = profile_words(resp.get("profile_updates") or {})
    if noted:
        st.caption(f":material/person_check: Noted from your question: {noted}")

    if sources:
        floor = _floor(steps)
        with st.container(horizontal=True, gap="small", wrap=True):
            for n, src in enumerate(sources, 1):
                if st.button(chip_label(n, src), key=f"src-{idx}-{n}", type="secondary",
                             icon=":material/gavel:" if src.get("ref") else ":material/description:"):
                    evidence_drawer(n, src, msg["content"], floor, resp.get("as_of"))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def _namespaces(url: str, key: Optional[str]) -> Dict[str, int]:
    return namespace_stats(url, key)


@st.dialog("Add a document")
def upload_dialog(namespace: str) -> None:
    st.caption(f"The document is split into passages and indexed into `{namespace}`.")
    with st.form("upload_form", border=False):
        uploaded = st.file_uploader("File", type=["pdf", "md", "txt", "docx", "pptx", "xlsx", "html", "htm"])
        title = st.text_input("Title", placeholder="Defaults to the file name")
        high_fidelity = st.checkbox(
            "High-fidelity conversion (slower)", disabled=DocumentConverter is None,
            help=None if DocumentConverter else "Needs docling, which this deployment does not include.",
        )
        submitted = st.form_submit_button("Add document", type="primary")
    if not submitted:
        return
    if uploaded is None:
        st.error("Choose a file to add.")
        return
    title = title.strip() or Path(uploaded.name).stem
    with st.spinner("Converting and indexing…"):
        try:
            uploaded.seek(0)
            text, meta = convert_uploaded_file_to_text(uploaded, use_high_fidelity=high_fidelity)
            if len(text.strip()) < 200:
                st.error("Only a few characters of text came out of this file. Check it is not a scanned image.")
                return
            post_upload_text(backend_url(), api_key(), {
                "title": title, "source": "ui-upload", "text": text, "namespace": namespace, "metadata": meta,
            })
        except httpx.HTTPStatusError as exc:
            st.error(f"The API refused the document ({exc.response.status_code}): {exc.response.text[:200]}")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not convert this file: {exc}")
            return
    s.uploads = (s.uploads + [{"title": title, "namespace": namespace,
                               "at": datetime.now(timezone.utc).strftime("%H:%M UTC")}])[-5:]
    _namespaces.clear()
    st.rerun()


def sidebar(ready: bool) -> Dict[str, Any]:
    with st.sidebar:
        st.subheader("Knowledge base", anchor=False)
        counts = _namespaces(backend_url(), api_key()) if ready else {}
        options = sorted(counts) or [DEFAULT_NAMESPACE]
        namespace = st.selectbox(
            "Namespace", options, index=options.index(DEFAULT_NAMESPACE) if DEFAULT_NAMESPACE in options else 0,
            format_func=lambda ns: f"{ns} · {counts[ns]} passages" if ns in counts else ns,
        )
        if st.button("Add a document", icon=":material/upload_file:", disabled=not ready, width="stretch"):
            upload_dialog(namespace)
        for u in reversed(s.uploads):
            st.caption(f":material/check: {u['title']} → `{u['namespace']}` · {u['at']}")

        if namespace == REGS_NAMESPACE:
            settings = regs_controls()
        else:
            st.subheader("Retrieval", anchor=False)
            top_k = st.slider("Passages to retrieve", 1, 20, 5)
            web = st.toggle("Search the web when the knowledge base is weak", value=False)
            web_threshold = st.slider(
                "Web search threshold", 0.0, 1.0, 0.25, 0.05, disabled=not web,
                help="If the best passage scores below this, web results are added (Tavily).",
            )
            settings = {"top_k": top_k, "use_web_fallback": web, "min_score": web_threshold}
        if s.messages and st.button("Clear conversation", icon=":material/delete_sweep:", type="tertiary"):
            s.messages = []
            st.rerun()
    return {"namespace": namespace, **settings}


def regs_controls() -> Dict[str, Any]:
    st.subheader("Your situation", anchor=False)
    as_of = st.date_input("Answer as of", value=date.today(), min_value=date(2016, 1, 1),
                          max_value=date(2035, 12, 31), help="Each cited rule shows whether it applies on this date.")
    frameworks = st.multiselect("Frameworks", FRAMEWORKS, placeholder="All covered frameworks")
    high_risk = HIGH_RISK[st.selectbox("High-risk AI class", list(HIGH_RISK),
                                       help="Sets which AI Act dates apply to high-risk obligations.")]
    st.caption("Answers come from official EU and NIST text only. Not legal advice.")
    return {"as_of": as_of.isoformat(), "frameworks": frameworks or None,
            "profile": {"high_risk": high_risk} if high_risk else None, "use_web_fallback": False}


# ---------------------------------------------------------------------------
# A turn
# ---------------------------------------------------------------------------

def run_turn(question: str, settings: Dict[str, Any]) -> None:
    history = [{"role": m["role"], "content": m["content"]} for m in s.messages]
    s.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)

    payload = {"query": question, "max_web_results": 5, "chat_history": history, **settings}
    steps: List[Dict[str, Any]] = []
    text, done, error = "", None, None
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        status = st.status(":shimmer[Searching the knowledge base…]", expanded=True)
        trace = status.empty()
        answer_box = st.empty()
        try:
            for kind, data in stream_chat(backend_url(), api_key(), payload):
                if kind == "step":
                    steps = [x for x in steps if x["id"] != data["id"]] + [data]
                    with trace.container():
                        render_trace(steps)
                elif kind == "token":
                    text += data
                    answer_box.markdown(cite_markdown(text, 99) + " ▍")
                elif kind == "done":
                    done = data
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            error = ("The API rejected this app's key. Check API_KEY in the app settings."
                     if code in (401, 403) else f"The API returned an error ({code}). Try again in a moment.")
        except httpx.TimeoutException:
            error = "The API took too long to answer. It may still be waking up; try again in a minute."
        except (httpx.HTTPError, RuntimeError) as exc:
            error = f"The answer could not be completed: {exc}"
        if error or done is None:
            status.update(label="Stopped", state="error", expanded=True)
            answer_box.error(error or "The stream ended without an answer. Try again.", icon=":material/error:")
            s.messages.pop()
            return
    s.messages.append({"role": "assistant", "content": str(done.get("answer") or text),
                       "response": done, "steps": steps})
    st.rerun()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

state = backend_state()
ready = state == "ready"
settings = sidebar(ready)

regs = settings["namespace"] == REGS_NAMESPACE
if not s.messages:
    if regs:
        st.title("Ask the rules for AI, data and cyber", anchor=False)
        st.markdown(
            "Answers quote the EU AI Act, GDPR, NIS2, DORA, the Cyber Resilience Act and NIST frameworks, "
            "cite the exact provision, and say whether each one **applies on your date**. Questions about "
            "rules not covered here get a plain decline instead of a guess."
        )
    else:
        st.title("Ask the knowledge base", anchor=False)
        st.markdown(
            "Answers cite the passages they use. When no passage clears the **relevance floor**, "
            "you get a plain “not in the knowledge base” instead of a guess."
        )
    if state == "waking":
        wake_watcher()
        st.skeleton(height=120)
    elif ready:
        pick = st.pills("Try a question", REGS_EXAMPLES if regs else EXAMPLES, key="example",
                        label_visibility="visible")
        st.caption("The last one is outside the covered rules: watch it decline." if regs
                   else "The last one is not in the knowledge base: watch it decline.")
    else:
        st.error(f"Cannot reach the API at `{backend_url()}`. Check BACKEND_BASE_URL.",
                 icon=":material/cloud_off:")

for i, m in enumerate(s.messages):
    with st.chat_message(m["role"], avatar=AVATARS[m["role"]]):
        if m["role"] == "assistant":
            render_answer(m, i)
        else:
            st.markdown(m["content"])

question = st.chat_input(
    "Ask a question" if ready else "Waiting for the API…",
    disabled=not ready or not api_key(),
)
if not api_key():
    st.info("Set API_KEY in the app's secrets to start asking.", icon=":material/key:")
if s.get("example") and not s.messages:
    question = s.example
if question:
    run_turn(question, settings)

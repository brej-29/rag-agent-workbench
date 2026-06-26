import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import httpx
import streamlit as st

from services.backend_client import post_upload_text
from services.file_convert import convert_uploaded_file_to_text


def get_backend_base_url() -> str:
    """Prefer Streamlit secrets, then environment variable, then localhost."""
    if "BACKEND_BASE_URL" in st.secrets:
        base_url = st.secrets["BACKEND_BASE_URL"]
    else:
        base_url = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
    return str(base_url).rstrip("/")


def get_api_key() -> Optional[str]:
    """Read API key from Streamlit secrets or environment."""
    if "API_KEY" in st.secrets:
        return str(st.secrets["API_KEY"])
    return os.getenv("API_KEY")


def ping_health(base_url: str, api_key: Optional[str]) -> Dict[str, Any]:
    url = f"{base_url}/health"
    headers: Dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key
    resp = httpx.get(url, headers=headers, timeout=10.0)
    resp.raise_for_status()
    return resp.json()


def call_chat(
    base_url: str,
    api_key: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    url = f"{base_url}/chat"
    headers: Dict[str, str] = {"Content-Type": "application/json", "X-API-Key": api_key}
    resp = httpx.post(url, json=payload, headers=headers, timeout=60.0)
    resp.raise_for_status()
    return resp.json()


def iter_chat_stream(
    base_url: str,
    api_key: str,
    payload: Dict[str, Any],
) -> Generator[Tuple[str, Optional[Dict[str, Any]]], None, None]:
    """Stream tokens from /chat/stream and yield (partial_answer, final_payload).

    Consumes the T2.9 SSE protocol:
      event: token  data: {"text": "..."} — yields (accumulated_text, None)
      event: done   data: {full payload}  — yields (full_answer, payload)
      event: error  data: {"message": "..."} — raises RuntimeError

    Also handles the legacy bare-data format for backward compatibility.
    """
    url = f"{base_url}/chat/stream"
    headers: Dict[str, str] = {"Content-Type": "application/json", "X-API-Key": api_key}

    full_answer = ""
    final_payload: Optional[Dict[str, Any]] = None
    current_event: Optional[str] = None
    error_message: Optional[str] = None

    with httpx.Client(timeout=60.0) as client:
        with client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    current_event = None  # blank line = end of SSE frame, reset
                    continue

                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                    continue

                if line.startswith("data:"):
                    raw = line.split(":", 1)[1].lstrip()

                    if current_event == "token":
                        try:
                            data = json.loads(raw)
                            text = str(data.get("text") or "")
                        except json.JSONDecodeError:
                            text = raw
                        if text:
                            full_answer += text
                            yield full_answer, None

                    elif current_event in ("done", "end"):
                        try:
                            final_payload = json.loads(raw)
                        except json.JSONDecodeError:
                            final_payload = None

                    elif current_event == "error":
                        try:
                            data = json.loads(raw)
                            error_message = data.get("message", raw)
                        except json.JSONDecodeError:
                            error_message = raw

                    elif current_event is None:
                        # Legacy format: bare data line without an event type.
                        if raw:
                            full_answer += " " + raw if full_answer else raw
                            yield full_answer, None

    if error_message:
        raise RuntimeError(f"Stream error from backend: {error_message}")

    if final_payload is not None:
        answer_text = str(final_payload.get("answer") or full_answer)
        yield answer_text, final_payload
    elif full_answer:
        yield full_answer, None


# ---------------------------------------------------------------------------
# T2.8 rendering helpers
# ---------------------------------------------------------------------------


def _render_quality_indicator(response: Dict[str, Any]) -> None:
    """Render the grounding/quality state — three distinct states per T2.8 hard rule 3.

    States (NEVER collapsed or fabricated — absent fields show 'not evaluated'):
      1. insufficient_context=True → abstention; no LLM was called
      2. grounded=False → answered but not well-supported by context
      3. grounded=True + unverified_citations → grounded but has dangling citation markers
      4. grounded=True + no unverified → clean grounded answer
      5. grounded=None → faithfulness not evaluated (flag OFF)
    """
    insufficient = response.get("insufficient_context", False)
    grounded = response.get("grounded")  # True / False / None — never fabricated
    faithfulness_score = response.get("faithfulness_score")
    unverified = response.get("unverified_citations") or []

    if insufficient:
        st.warning(
            "**Insufficient context** — no relevant information found in the knowledge "
            "base. The LLM was not called; the answer is a deterministic abstention.",
            icon="⚠️",
        )
        return

    if grounded is None:
        return

    score_str = f"  Score: {faithfulness_score:.2f}" if faithfulness_score is not None else ""
    if grounded and not unverified:
        st.success(
            f"Grounded answer — all claims supported by the retrieved context.{score_str}",
            icon="✅",
        )
    elif grounded and unverified:
        st.warning(
            f"Grounded but has unverified citation markers: {unverified}.{score_str}",
            icon="⚠️",
        )
    else:
        st.error(
            f"Answer is **not well-supported** by the retrieved context.{score_str}",
            icon="❌",
        )


def _render_sources_panel(sources: List[Dict[str, Any]], web_fallback_used: bool) -> None:
    """Render retrieved-and-kept sources with cosine scores (T8.2)."""
    if not sources:
        return
    label = f"Sources ({len(sources)})"
    if web_fallback_used:
        label += " — includes web results"
    with st.expander(label, expanded=False):
        for idx, src in enumerate(sources, start=1):
            title = src.get("title") or f"Source {idx}"
            url = src.get("url") or ""
            score = float(src.get("score") or 0.0)
            source_tag = src.get("source") or "unknown"
            badge = " 🌐" if source_tag == "web" else ""
            st.markdown(f"**[{idx}] {title}**{badge}  (score={score:.3f})")
            if url:
                st.markdown(f"URL: {url}")
            chunk_text = src.get("chunk_text") or ""
            if chunk_text:
                preview = chunk_text[:800] + ("…" if len(chunk_text) > 800 else "")
                st.caption(preview)
            if idx < len(sources):
                st.divider()


def _render_retrieval_debug(response: Dict[str, Any]) -> None:
    """Render collapsible retrieval-debug panel (T8.3)."""
    timings = response.get("timings") or {}
    crag_iterations = int(response.get("crag_iterations") or 0)
    corrective_action = response.get("corrective_action")
    contextualized_query = response.get("contextualized_query")
    top_score = float(response.get("top_score") or 0.0)
    sources = response.get("sources") or []
    web_fallback_used = response.get("web_fallback_used", False)
    cached = response.get("cached", False)

    with st.expander("Retrieval debug", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Top cosine score", f"{top_score:.3f}")
            st.metric("Sources kept", len(sources))
            st.metric("CRAG iterations", crag_iterations)
        with col2:
            st.metric("Retrieve", f"{timings.get('retrieve_ms', 0.0):.0f} ms")
            st.metric("Generate", f"{timings.get('generate_ms', 0.0):.0f} ms")
            st.metric("Faithfulness", f"{timings.get('faithfulness_ms', 0.0):.0f} ms")

        if crag_iterations > 0:
            st.markdown(
                f"**CRAG**: {crag_iterations} correction iteration(s), "
                f"action=`{corrective_action or 'none'}`"
            )
        if contextualized_query:
            st.markdown(f"**Contextualized query** (T2.5): _{contextualized_query}_")
        if web_fallback_used:
            st.markdown("**Web fallback**: Tavily was used (retrieval score was below threshold).")
        if cached:
            st.markdown("**Cached**: this response was served from the in-memory cache.")


def _render_token_usage(usage: Optional[Dict[str, Any]]) -> None:
    """Render per-request token usage and estimated cost (T8.5).

    Never fabricates: if usage is None (e.g. cached response), renders nothing.
    The cost field is always labeled as an ESTIMATE, consistent with the backend.
    """
    if not usage:
        return
    with st.expander("Token usage & cost", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Prompt tokens", usage.get("prompt_tokens", 0))
        col2.metric("Completion tokens", usage.get("completion_tokens", 0))
        col3.metric("Total tokens", usage.get("total_tokens", 0))

        cost = usage.get("estimated_cost_usd")
        if cost is not None:
            st.caption(
                f"Estimated cost: **${cost:.6f}** USD  "
                "(ESTIMATE from an as-of-date pricing table — "
                "see `backend/app/core/cost_accounting.py`)"
            )
        else:
            st.caption("Estimated cost: not available (model not in pricing table).")

        by_call = usage.get("by_call_type") or {}
        if by_call:
            st.markdown("**By call type:**")
            for call_type, counts in by_call.items():
                if isinstance(counts, dict):
                    st.markdown(
                        f"- `{call_type}`: {counts.get('total_tokens', 0)} total  "
                        f"({counts.get('prompt_tokens', 0)} in / "
                        f"{counts.get('completion_tokens', 0)} out)"
                    )


def _render_assistant_extras(message: Dict[str, Any], show_sources: bool) -> None:
    """Render quality indicator, sources, debug panel, and token usage for one assistant turn."""
    _render_quality_indicator(message)

    sources = message.get("sources") or []
    web_fallback_used = message.get("web_fallback_used", False)
    if show_sources and sources:
        _render_sources_panel(sources, web_fallback_used)

    _render_retrieval_debug(message)
    _render_token_usage(message.get("usage"))


# ---------------------------------------------------------------------------
# Session + sidebar
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "supports_stream" not in st.session_state:
        st.session_state.supports_stream = True
    if "namespace" not in st.session_state:
        st.session_state.namespace = "dev"
    if "recent_uploads" not in st.session_state:
        st.session_state.recent_uploads: List[Dict[str, Any]] = []
    if "chat_prefill" not in st.session_state:
        st.session_state.chat_prefill = None


def render_sidebar(backend_base_url: str, api_key: Optional[str]) -> Dict[str, Any]:
    with st.sidebar:
        st.header("Backend")

        st.markdown(f"**Backend URL:** `{backend_base_url}`")
        if api_key:
            st.markdown("**API key:** configured in Streamlit secrets or environment.")
        else:
            st.warning(
                "API_KEY is not configured. The backend is expected to be protected; "
                "chat will be disabled until an API key is set."
            )

        if st.button("Ping /health"):
            try:
                health = ping_health(backend_base_url, api_key)
                st.success("Backend reachable.")
                st.json(health)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Health check failed: {exc}")

        st.markdown("---")
        st.subheader("Chat settings")

        top_k = st.slider("Top K", min_value=1, max_value=20, value=5, step=1)
        min_score = st.slider(
            "Minimum relevance score",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
        )
        use_web_fallback = st.checkbox(
            "Use web fallback (Tavily)",
            value=True,
            help="When enabled, /chat may call Tavily if retrieval is weak.",
        )

        st.session_state.show_sources = st.checkbox(
            "Show sources", value=st.session_state.show_sources
        )

        if st.button("Clear chat"):
            st.session_state.messages = []

        # Faithfulness mode indicator — derived from the last assistant response.
        # grounded=None means the faithfulness judge did not run this session.
        assistant_msgs = [
            m for m in st.session_state.get("messages", []) if m.get("role") == "assistant"
        ]
        if assistant_msgs:
            last_grounded = assistant_msgs[-1].get("grounded")
            if last_grounded is None:
                st.markdown("---")
                st.caption("Grounding verification: off — answers are not grounding-checked.")

        st.markdown("---")
        st.subheader("Recent uploads")
        recent = st.session_state.get("recent_uploads", [])
        if not recent:
            st.caption("No documents uploaded yet.")
        else:
            for idx, item in enumerate(recent):
                title = item.get("title") or "Untitled"
                ns = item.get("namespace") or st.session_state.get("namespace", "dev")
                ts = item.get("timestamp", "")
                st.markdown(f"- **{title}**  \n  Namespace: `{ns}`  \n  Uploaded: {ts}")
                if st.button("Search this document", key=f"search_upload_{idx}"):
                    st.session_state.chat_prefill = f"Summarize: {title}"

    return {
        "top_k": top_k,
        "min_score": float(min_score),
        "use_web_fallback": bool(use_web_fallback),
    }


def render_chat_history(show_sources: bool) -> None:
    for message in st.session_state.messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        with st.chat_message("assistant" if role == "assistant" else "user"):
            st.markdown(content)
            if role == "assistant":
                _render_assistant_extras(message, show_sources)


@st.dialog("Upload document")
def upload_dialog(backend_base_url: str, api_key: Optional[str]) -> None:
    """Modal dialog for uploading and ingesting a document via /documents/upload-text."""
    st.write("Upload a document to ingest it into the RAG backend.")

    with st.form("upload_form"):
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "md", "txt", "docx", "pptx", "xlsx", "html", "htm"],
            accept_multiple_files=False,
        )

        default_title = ""
        if uploaded_file is not None:
            default_title = Path(uploaded_file.name).stem

        title = st.text_input("Title", value=default_title)
        namespace = st.text_input(
            "Namespace",
            value=st.session_state.get("namespace", "dev"),
            help="Target Pinecone namespace.",
        )
        source = st.text_input("Source label", value="ui-upload")
        tags = st.text_input("Tags (comma separated)", value="")
        notes = st.text_area("Notes", value="", height=80)

        high_fidelity = st.checkbox(
            "High-fidelity Docling mode (slower)",
            value=False,
            help=(
                "When enabled, skip the fast text extractor and use Docling directly. "
                "Useful for complex layouts, but slower."
            ),
        )

        upload_anyway = st.checkbox(
            "Upload even if extracted text is very short",
            value=False,
            help="Enable to upload even when the extracted text is shorter than 200 characters.",
        )

        submit = st.form_submit_button("Upload")
    if not submit:
        return

    if uploaded_file is None:
        st.error("Please select a file to upload.")
        return

    if not title.strip():
        st.error("Please provide a title.")
        return

    if not api_key:
        st.error("API_KEY is not configured; cannot upload to a protected backend.")
        return

    with st.spinner("Converting and uploading document (fast text extraction first, "
                    "Docling fallback may take up to ~45s for complex PDFs)..."):
        try:
            uploaded_file.seek(0)
            text, conv_meta = convert_uploaded_file_to_text(
                uploaded_file,
                use_high_fidelity=high_fidelity,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error converting file: {exc}")
            return

        if len(text.strip()) < 200 and not upload_anyway:
            st.warning(
                "Extracted text is very short (< 200 characters). "
                "Check the file or enable the checkbox to upload anyway."
            )
            return

        meta: Dict[str, Any] = {
            **conv_meta,
            "tags": [t.strip() for t in tags.split(",") if t.strip()],
            "notes": notes,
        }

        payload = {
            "title": title.strip(),
            "source": source.strip() or "ui-upload",
            "text": text,
            "namespace": namespace.strip() or st.session_state.get("namespace", "dev"),
            "metadata": meta,
        }

        try:
            response = post_upload_text(backend_base_url, api_key, payload)
        except httpx.HTTPStatusError as exc:
            if exc.response is not None:
                detail = exc.response.text
                status_code = exc.response.status_code
            else:
                detail = str(exc)
                status_code = "error"
            st.error(f"Upload failed ({status_code}): {detail}")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Upload failed: {exc}")
            return

        rec = {
            "title": title.strip(),
            "namespace": payload["namespace"],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "response": response,
        }
        recent = st.session_state.get("recent_uploads", [])
        recent.append(rec)
        st.session_state.recent_uploads = recent[-5:]

        st.success(f"Uploaded and indexed: {title.strip()}")
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="RAG Agent Workbench", layout="wide")
    st.title("RAG Agent Workbench – Chatbot")

    init_session_state()

    backend_base_url = get_backend_base_url()
    api_key = get_api_key()

    if st.button("📄 Upload Document"):
        upload_dialog(backend_base_url, api_key)

    settings = render_sidebar(backend_base_url, api_key)
    render_chat_history(show_sources=st.session_state.show_sources)

    if not api_key:
        st.info(
            "Configure `API_KEY` in Streamlit secrets (and on the backend) to start chatting."
        )
        return

    prefill = st.session_state.get("chat_prefill")
    if prefill and "chat_input" not in st.session_state:
        st.session_state.chat_input = prefill

    user_message = st.chat_input(
        "Ask a question about your documents...", key="chat_input"
    )
    if not user_message:
        return

    st.session_state.chat_prefill = None

    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    chat_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
        if msg.get("role") in ("user", "assistant")
    ]
    payload: Dict[str, Any] = {
        "query": user_message,
        "namespace": st.session_state.namespace,
        "top_k": int(settings["top_k"]),
        "use_web_fallback": settings["use_web_fallback"],
        "min_score": float(settings["min_score"]),
        "max_web_results": 5,
        "chat_history": chat_history,
    }

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking..._")

        response: Optional[Dict[str, Any]] = None

        try:
            if st.session_state.get("supports_stream", True):
                try:
                    for partial_answer, final_payload in iter_chat_stream(
                        backend_base_url, api_key, payload,
                    ):
                        if partial_answer:
                            placeholder.markdown(partial_answer)
                        if final_payload is not None:
                            response = final_payload
                            break
                except httpx.HTTPStatusError as exc:
                    if exc.response is not None and exc.response.status_code == 404:
                        st.session_state.supports_stream = False
                    else:
                        raise

            if response is None:
                response = call_chat(backend_base_url, api_key, payload)
                answer_text = str(response.get("answer") or "")
                placeholder.markdown(answer_text if answer_text else "_No answer returned._")

        except Exception as exc:  # noqa: BLE001
            placeholder.markdown("")
            st.error(f"Error calling backend: {exc}")
            return

        if not response:
            return

        # T2.8: Render quality indicator, sources, debug panel, and token usage.
        _render_quality_indicator(response)

        sources = response.get("sources") or []
        web_fallback_used = response.get("web_fallback_used", False)
        if st.session_state.show_sources and sources:
            _render_sources_panel(sources, web_fallback_used)

        _render_retrieval_debug(response)
        _render_token_usage(response.get("usage"))

        # Persist all observability fields so render_chat_history can replay them.
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": str(response.get("answer") or ""),
                "sources": sources,
                "timings": response.get("timings") or {},
                "grounded": response.get("grounded"),
                "faithfulness_score": response.get("faithfulness_score"),
                "unverified_citations": response.get("unverified_citations") or [],
                "insufficient_context": response.get("insufficient_context", False),
                "crag_iterations": response.get("crag_iterations", 0),
                "corrective_action": response.get("corrective_action"),
                "contextualized_query": response.get("contextualized_query"),
                "usage": response.get("usage"),
                "web_fallback_used": web_fallback_used,
                "top_score": float(response.get("top_score") or 0.0),
                "cached": response.get("cached", False),
            }
        )


if __name__ == "__main__":
    main()

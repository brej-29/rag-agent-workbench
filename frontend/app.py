import json
import os
from typing import Any, Dict, Generator, List, Optional, Tuple

import httpx
import streamlit as st


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

    The final_payload is None for intermediate updates and populated once
    when the terminating SSE event is received.
    """
    url = f"{base_url}/chat/stream"
    headers: Dict[str, str] = {"Content-Type": "application/json", "X-API-Key": api_key}

    full_answer = ""
    final_payload: Optional[Dict[str, Any]] = None
    current_event: Optional[str] = None

    with httpx.Client(timeout=60.0) as client:
        with client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue

                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                    continue

                if line.startswith("data:"):
                    data = line.split(":", 1)[1].lstrip()
                    if current_event == "end":
                        # Final payload with full JSON response.
                        try:
                            final_payload = json.loads(data)
                        except json.JSONDecodeError:
                            final_payload = None
                    else:
                        if data:
                            if full_answer:
                                full_answer += " "
                            full_answer += data
                            # Yield intermediate answer text.
                            yield full_answer, None

    # After stream ends, make sure we yield at least once with final payload.
    if final_payload is not None:
        # If the backend included the final answer in the JSON payload, prefer it.
        answer_text = str(final_payload.get("answer") or full_answer)
        yield answer_text, final_payload
    elif full_answer:
        yield full_answer, None


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "supports_stream" not in st.session_state:
        st.session_state.supports_stream = True
    if "namespace" not in st.session_state:
        st.session_state.namespace = "dev"


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

        namespace = st.text_input(
            "Namespace",
            value=st.session_state.get("namespace", "dev"),
            help="Pinecone namespace to query.",
        )
        st.session_state.namespace = namespace.strip() or "dev"

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

    return {
        "namespace": st.session_state.namespace,
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
            if role == "assistant" and show_sources:
                sources = message.get("sources") or []
                if sources:
                    with st.expander("Sources", expanded=False):
                        for idx, src in enumerate(sources, start=1):
                            title = src.get("title") or f"Source {idx}"
                            url = src.get("url") or ""
                            score = src.get("score", 0.0)
                            st.markdown(f"**[{idx}] {title}** (score={score:.3f})")
                            if url:
                                st.markdown(f"- URL: {url}")
                            chunk_text = src.get("chunk_text") or ""
                            if chunk_text:
                                st.write(chunk_text[:1000] + ("..." if len(chunk_text) > 1000 else ""))


def main() -> None:
    st.set_page_config(page_title="RAG Agent Workbench", layout="wide")
    st.title("RAG Agent Workbench – Chatbot")

    init_session_state()

    backend_base_url = get_backend_base_url()
    api_key = get_api_key()

    settings = render_sidebar(backend_base_url, api_key)
    render_chat_history(show_sources=st.session_state.show_sources)

    if not api_key:
        st.info(
            "Configure `API_KEY` in Streamlit secrets (and on the backend) to start chatting."
        )
        return

    user_message = st.chat_input("Ask a question about your documents...")
    if not user_message:
        return

    # Record and display user message
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    # Prepare payload for backend
    chat_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
        if msg.get("role") in ("user", "assistant")
    ]
    payload: Dict[str, Any] = {
        "query": user_message,
        "namespace": settings["namespace"],
        "top_k": int(settings["top_k"]),
        "use_web_fallback": settings["use_web_fallback"],
        "min_score": float(settings["min_score"]),
        "max_web_results": 5,
        "chat_history": chat_history,
    }

    # Call backend and stream / display assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking..._")

        response: Optional[Dict[str, Any]] = None

        try:
            if st.session_state.get("supports_stream", True):
                try:
                    # Attempt to use streaming endpoint first.
                    for partial_answer, final_payload in iter_chat_stream(
                        backend_base_url,
                        api_key,
                        payload,
                    ):
                        if partial_answer:
                            placeholder.markdown(partial_answer)
                        if final_payload is not None:
                            response = final_payload
                            break
                except httpx.HTTPStatusError as exc:
                    # If /chat/stream is not available, fall back to /chat.
                    if exc.response is not None and exc.response.status_code == 404:
                        st.session_state.supports_stream = False
                    else:
                        raise

            if response is None:
                # Fallback to non-streaming /chat.
                response = call_chat(backend_base_url, api_key, payload)
                answer_text = str(response.get("answer") or "")
                if answer_text:
                    placeholder.markdown(answer_text)
                else:
                    placeholder.markdown("_No answer returned._")

        except Exception as exc:  # noqa: BLE001
            placeholder.markdown("")
            st.error(f"Error calling backend: {exc}")
            return

        if not response:
            return

        answer = str(response.get("answer") or "")
        sources = response.get("sources") or []
        timings = response.get("timings") or {}

        # Optionally render sources for this assistant turn.
        if st.session_state.show_sources and sources:
            with st.expander("Sources", expanded=False):
                for idx, src in enumerate(sources, start=1):
                    title = src.get("title") or f"Source {idx}"
                    url = src.get("url") or ""
                    score = src.get("score", 0.0)
                    st.markdown(f"**[{idx}] {title}** (score={score:.3f})")
                    if url:
                        st.markdown(f"- URL: {url}")
                    chunk_text = src.get("chunk_text") or ""
                    if chunk_text:
                        st.write(chunk_text[:1000] + ("..." if len(chunk_text) > 1000 else ""))

        # Persist assistant message with metadata.
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "timings": timings,
            }
        )


if __name__ == "__main__":
    main()
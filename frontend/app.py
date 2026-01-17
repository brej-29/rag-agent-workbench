import os
from typing import Any, Dict

import httpx
import streamlit as st


def get_backend_base_url() -> str:
    # Prefer Streamlit secrets, then environment variable, then localhost.
    secrets = getattr(st, "secrets", {})
    base_url = getattr(secrets, "get", lambda _k, _d=None: None)("BACKEND_BASE_URL", None)
    if not base_url:
        base_url = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
    return base_url.rstrip("/")


def get_api_key() -> str | None:
    secrets = getattr(st, "secrets", {})
    api_key = getattr(secrets, "get", lambda _k, _d=None: None)("API_KEY", None)
    if not api_key:
        api_key = os.getenv("API_KEY")
    return api_key


async def ping_health(base_url: str, api_key: str | None) -> Dict[str, Any]:
    url = f"{base_url}/health"
    headers: Dict[str, str] = {}
    if api_key:
        headers["X-API-Key"] = api_key
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, headers=headers)
    return resp.json()


async def call_chat(
    base_url: str,
    api_key: str | None,
    query: str,
    namespace: str,
) -> Dict[str, Any]:
    url = f"{base_url}/chat"
    payload: Dict[str, Any] = {
        "query": query,
        "namespace": namespace,
        "top_k": 5,
        "use_web_fallback": True,
    }
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
    return resp.json()


def main() -> None:
    st.set_page_config(page_title="RAG Agent Workbench", layout="wide")
    st.title("RAG Agent Workbench – Chat Demo")

    backend_base_url = get_backend_base_url()
    api_key = get_api_key()

    with st.sidebar:
        st.header("Connectivity")
        st.markdown(f"**Backend URL:** `{backend_base_url}`")
        if api_key:
            st.markdown("**API key:** configured in Streamlit secrets.")
        else:
            st.markdown("**API key:** not set (backend may be open).")

        if st.button("Ping /health"):
            try:
                import asyncio

                health = asyncio.run(ping_health(backend_base_url, api_key))
                st.success("Backend reachable.")
                st.json(health)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Health check failed: {exc}")

    st.subheader("Chat")

    namespace = st.text_input("Namespace", value="dev", help="Pinecone namespace to query.")
    query = st.text_area(
        "Your question",
        value="What is retrieval-augmented generation?",
        height=100,
    )

    if st.button("Send"):
        if not query.strip():
            st.warning("Please enter a question.")
            return
        with st.spinner("Calling backend /chat..."):
            try:
                import asyncio

                response = asyncio.run(
                    call_chat(backend_base_url, api_key, query.strip(), namespace.strip() or "dev")
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Error calling backend: {exc}")
                return

        answer = response.get("answer", "")
        timings = response.get("timings", {})
        sources = response.get("sources", [])

        st.markdown("### Answer")
        st.write(answer or "_No answer returned._")

        st.markdown("### Timings (ms)")
        st.json(timings)

        if sources:
            st.markdown("### Sources")
            for idx, src in enumerate(sources[:5], start=1):
                title = src.get("title") or f"Source {idx}"
                url = src.get("url") or ""
                score = src.get("score", 0.0)
                st.markdown(f"**[{idx}] {title}** (score={score:.3f})")
                if url:
                    st.markdown(f"- URL: {url}")
                chunk_text = src.get("chunk_text") or ""
                if chunk_text:
                    with st.expander("Snippet", expanded=False):
                        st.write(chunk_text)


if __name__ == "__main__":
    main()
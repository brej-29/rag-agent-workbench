"""HTTP client for the RAG backend (no Streamlit imports: unit-testable)."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import httpx


def _headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def probe_health(base_url: str, timeout: float = 3.0) -> Tuple[str, float]:
    """Return ("ready" | "waking" | "down", seconds).

    A sleeping free-tier Space holds or 5xx-es requests while it boots, so a
    timeout or 5xx means "waking", not "down".  Only connection refusals and
    unexpected 4xx count as down.
    """
    start = time.perf_counter()
    try:
        resp = httpx.get(f"{base_url}/health", timeout=timeout)
    except httpx.TimeoutException:
        return "waking", time.perf_counter() - start
    except httpx.HTTPError:
        return "down", time.perf_counter() - start
    elapsed = time.perf_counter() - start
    if resp.status_code == 200:
        return "ready", elapsed
    return ("waking" if resp.status_code >= 500 else "down"), elapsed


def namespace_stats(base_url: str, api_key: Optional[str]) -> Dict[str, int]:
    """Vector counts per namespace ({} when unavailable)."""
    try:
        resp = httpx.get(f"{base_url}/documents/stats", headers=_headers(api_key), timeout=10.0)
        resp.raise_for_status()
        return {ns: int(v.get("vector_count", 0)) for ns, v in (resp.json().get("namespaces") or {}).items()}
    except (httpx.HTTPError, ValueError, AttributeError):
        return {}


def parse_sse(lines: Iterable[str]) -> Iterator[Tuple[str, Any]]:
    """Turn /chat/stream lines into ("step", dict) | ("token", str) | ("done", dict).

    Raises RuntimeError on an `error` event.  Unknown event types are ignored, so
    the client keeps working when the backend adds new ones.
    """
    event: Optional[str] = None
    for line in lines:
        if not line:
            event = None
            continue
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
            continue
        if not line.startswith("data:"):
            continue
        raw = line.split(":", 1)[1].strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = raw
        if event == "token":
            text = data.get("text", "") if isinstance(data, dict) else str(data)
            if text:
                yield "token", text
        elif event in ("step", "done"):
            yield event, data
        elif event == "error":
            message = data.get("message", raw) if isinstance(data, dict) else raw
            raise RuntimeError(message)


def stream_chat(base_url: str, api_key: Optional[str], payload: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """POST /chat/stream and yield parsed events (see parse_sse)."""
    with httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        with client.stream("POST", f"{base_url}/chat/stream", json=payload, headers=_headers(api_key)) as resp:
            resp.raise_for_status()
            yield from parse_sse(resp.iter_lines())


def post_upload_text(
    base_url: str,
    api_key: Optional[str],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Call backend /documents/upload-text with the given payload.

    Sends X-API-Key when provided and raises for HTTP errors.
    """
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"{base_url.rstrip('/')}/documents/upload-text", json=payload, headers=_headers(api_key))
        resp.raise_for_status()
        return resp.json()

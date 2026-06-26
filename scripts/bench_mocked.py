"""
In-process load benchmark: 50 requests against the real FastAPI app via
httpx.ASGITransport (no real HTTP server, no real external services).

PURPOSE
  Measures framework overhead — FastAPI middleware, LangGraph graph.invoke(),
  Pydantic schema validation, response serialization — with zero I/O latency.
  This is NOT a throughput projection for production (which is dominated by
  Pinecone + Groq latency).  See docs/LOAD_TEST.md for the full interpretation.

WHAT RUNS FOR REAL
  FastAPI routing, auth dependency (require_api_key), slowapi rate-limit
  middleware (disabled via RATE_LIMIT_ENABLED=false), the LangGraph pipeline
  (all 7 nodes), prompt builders, ChatResponse schema.

WHAT IS MOCKED
  - pinecone_search → one realistic chunk hit (0.92 cosine score)
  - get_llm (graph + streaming) → MagicMock with instant .invoke()
  - is_tavily_configured → False
  - init_pinecone → no-op (startup event, not triggered by ASGITransport
    anyway, but patched for belt-and-suspenders)
  - cache_enabled (router) → False (avoids serving identical cached response)

RATE LIMITING
  RATE_LIMIT_ENABLED=false prevents slowapi from registering its middleware
  when the app is imported.  With 50 concurrent requests from a single IP
  the 30/minute limiter would otherwise fire.

TRANSPORT
  httpx.ASGITransport(app=_app) routes httpx requests directly through the
  ASGI interface.  The ASGITransport does NOT trigger lifespan events, so
  the @app.on_event("startup") hook (init_pinecone) never fires regardless
  of the patch — but we patch it for safety in case this changes.
"""
from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Environment must be set BEFORE any app import so:
#   - get_settings() reads RATE_LIMIT_ENABLED=false → rate-limit middleware
#     is not registered
#   - LRU-cached settings picks up the test values
# ---------------------------------------------------------------------------
os.environ.setdefault("PINECONE_API_KEY", "bench-dummy-key")
os.environ.setdefault("PINECONE_INDEX_NAME", "bench-dummy-index")
os.environ.setdefault("PINECONE_HOST", "https://bench-dummy.pinecone.io")
os.environ.setdefault("GROQ_API_KEY", "bench-dummy-groq")
os.environ["RATE_LIMIT_ENABLED"] = "false"
os.environ["CACHE_ENABLED"] = "false"
_BENCH_API_KEY = "bench-test-key"
os.environ["API_KEY"] = _BENCH_API_KEY

import httpx  # noqa: E402 (after env setup)

# Clear LRU caches populated by any earlier imports in this process
from app.core.config import get_settings as _gs  # noqa: E402

_gs.cache_clear()

from app.core.auth import _get_configured_api_key as _gak  # noqa: E402

_gak.cache_clear()

from app.services.llm.groq_llm import get_llm as _gllm  # noqa: E402

_gllm.cache_clear()

import app.services.chat.graph as _graph_mod  # noqa: E402

_graph_mod._graph = None

# ---------------------------------------------------------------------------
# Mock shapes
# ---------------------------------------------------------------------------

_FAKE_CHUNK = {
    "_score": 0.92,
    "fields": {
        "chunk_text": "RAG combines retrieval with generation to answer questions.",
        "title": "Retrieval-Augmented Generation",
        "source": "wiki",
        "url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    },
}


def _make_llm_response(answer: str = "RAG combines retrieval and generation.") -> MagicMock:
    resp = MagicMock()
    resp.content = answer
    resp.usage_metadata = {"input_tokens": 120, "output_tokens": 25, "total_tokens": 145}
    resp.response_metadata = {}
    return resp


_mock_llm = MagicMock()
_mock_llm.invoke.return_value = _make_llm_response()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

_CONCURRENCY = 10
_TOTAL_REQUESTS = 50
_NAMESPACE = "bench"


async def _one_request(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    sem: asyncio.Semaphore,
) -> tuple[float, bool]:
    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await client.post(url, json=payload, headers=headers)
            elapsed = (time.perf_counter() - t0) * 1000.0
            return elapsed, resp.status_code >= 400
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            print(f"  [error] {exc}", file=sys.stderr)
            return elapsed, True


async def _run(app) -> Dict[str, Any]:
    transport = httpx.ASGITransport(app=app)
    url = "http://testserver/chat"
    payload: Dict[str, Any] = {
        "query": "Briefly explain retrieval-augmented generation.",
        "namespace": _NAMESPACE,
        "top_k": 5,
        "use_web_fallback": False,
    }
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": _BENCH_API_KEY,
    }

    sem = asyncio.Semaphore(_CONCURRENCY)
    latencies: List[float] = []
    errors = 0

    async with httpx.AsyncClient(transport=transport, timeout=30.0) as client:
        tasks = [_one_request(client, url, payload, headers, sem) for _ in range(_TOTAL_REQUESTS)]
        wall_start = time.perf_counter()
        for coro in asyncio.as_completed(tasks):
            ms, is_err = await coro
            latencies.append(ms)
            if is_err:
                errors += 1
        wall_elapsed = (time.perf_counter() - wall_start) * 1000.0

    return {
        "latencies_ms": latencies,
        "errors": errors,
        "total": _TOTAL_REQUESTS,
        "wall_ms": wall_elapsed,
    }


def _print_report(result: Dict[str, Any]) -> None:
    lats = sorted(result["latencies_ms"])
    n = len(lats)
    errors = result["errors"]
    wall_ms = result["wall_ms"]

    avg = sum(lats) / n if n else 0.0
    p50 = statistics.median(lats) if lats else 0.0
    idx95 = max(0, int(round(0.95 * (n - 1))))
    p95 = lats[idx95] if lats else 0.0
    throughput = (_TOTAL_REQUESTS / (wall_ms / 1000.0)) if wall_ms > 0 else 0.0

    print("=== /chat in-process bench (mocked externals) ===")
    print(f"Requests:        {_TOTAL_REQUESTS}")
    print(f"Concurrency:     {_CONCURRENCY}")
    print(f"Errors:          {errors} ({errors / _TOTAL_REQUESTS * 100:.1f}%)")
    print(f"Wall time:       {wall_ms:.0f} ms")
    print(f"Throughput:      {throughput:.1f} req/s")
    print(f"Avg latency:     {avg:.2f} ms")
    print(f"p50 latency:     {p50:.2f} ms")
    print(f"p95 latency:     {p95:.2f} ms")


def main() -> None:
    with (
        patch("app.main.init_pinecone"),
        patch("app.services.chat.graph.pinecone_search", return_value=[_FAKE_CHUNK]),
        patch("app.services.chat.graph.get_llm", return_value=_mock_llm),
        patch("app.services.chat.streaming.get_llm", return_value=_mock_llm),
        patch("app.services.chat.graph.is_tavily_configured", return_value=False),
        patch("app.routers.chat.cache_enabled", return_value=False),
    ):
        from app.main import app as _app

        # The @limiter.limit("30/minute") decorator is baked into the route at
        # import time; RATE_LIMIT_ENABLED=false prevents SlowAPIMiddleware from
        # being added, but the limiter object still counts requests.  Disabling it
        # here ensures all 50 bench requests reach the handler.
        from app.core.rate_limit import limiter as _rate_limiter
        _rate_limiter.enabled = False

        result = asyncio.run(_run(_app))

    _print_report(result)


if __name__ == "__main__":
    main()

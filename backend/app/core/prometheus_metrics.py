"""
Prometheus instrumentation for RAG Agent Workbench (T2.6).

Uses prometheus-client only (no prometheus-fastapi-instrumentator).
Reason: pfi 8.0.2 requires starlette>=1.0.0, which is incompatible with the
starlette==0.50.0 pinned by the installed FastAPI version.  prometheus-client
has zero framework dependency and gives full label control.

Provides two things:
  1. HTTP request metrics (count + latency Histogram) via a thin middleware
     added by setup_prometheus().
  2. A Histogram for RAG pipeline phase durations — the AUTHORITATIVE source
     for p50/p95 percentiles, replacing the unreliable maxlen=20 deque in
     the legacy JSON endpoint.

Path layout
-----------
  /metrics/prometheus  — Prometheus text exposition (public, no API key)
  /metrics             — Legacy JSON snapshot (existing, unchanged, API-key-gated)

Low-cardinality label discipline
---------------------------------
HTTP middleware labels: method (GET/POST/…), path (fixed route string, e.g.
"/chat"), status_class (2xx/4xx/5xx).  All paths in this API are fixed strings
(no /resource/{id} routes) so raw URL path is bounded: ~15 paths × 6 methods ×
5 status classes = O(450) max series.

RAG pipeline Histogram: label is "phase" with a fixed enumeration of 6 values
(retrieve, web, generate, rerank, faithfulness, total).  Bounded by construction.

NEVER label by query text, user input, namespace, document ID, or any
unbounded-cardinality field — those cause Prometheus metric explosion.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Mapping

from fastapi import Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from app.core.logging import get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# HTTP request metrics
# ---------------------------------------------------------------------------
#
# Both metrics are excluded on the /metrics/prometheus path itself (by the
# middleware guard below) to avoid polluting scrape-traffic counts.
#
# label cardinality: method × path × status_class
#   ~6 methods × ~15 paths × 5 status_classes = ~450 max series — safe.
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP request count by method, path, and status class.",
    ["method", "path", "status_class"],
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency (seconds) by method and path.",
    ["method", "path"],
    # Buckets tuned for a JSON API: sub-10ms not meaningful at this level.
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)

# ---------------------------------------------------------------------------
# RAG pipeline phase Histogram
# ---------------------------------------------------------------------------
#
# Bucket rationale (all values in SECONDS, Prometheus convention):
#
#   0.05s ( 50ms) — fast Pinecone retrieval lower bound
#   0.10s (100ms) — typical Pinecone retrieval
#   0.25s (250ms) — retrieval upper / fast generation lower
#   0.50s (500ms) — observed Pinecone retrieval peak (~350ms + margin)
#   0.75s (750ms) — retrieval + CRAG grade, no rewrite
#   1.00s (  1s ) — typical Groq generation (small model, short answer)
#   1.50s (1.5s ) — Groq generation for longer answers
#   2.00s (  2s ) — Groq generation + Tavily web search single call
#   3.00s (  3s ) — one CRAG correction iteration (rewrite LLM + re-retrieve)
#   5.00s (  5s ) — two CRAG iterations or slow Tavily
#   10.0s ( 10s ) — timeout safety ceiling (Groq/Tavily slow paths)
#
# The +Inf bucket is always added by prometheus-client automatically.
_CHAT_DURATION_BUCKETS = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)

# Bounded phase label: {retrieve, web, generate, rerank, faithfulness, total}.
# 6 values × 11 explicit buckets × 3 series per bucket (_sum, _count, _bucket) =
# well within a sane cardinality budget.
RAG_PHASE_DURATION = Histogram(
    "rag_phase_duration_seconds",
    "RAG pipeline phase duration (seconds). "
    "Authoritative p50/p95 source — computed over ALL observations, not a fixed window.",
    ["phase"],
    buckets=_CHAT_DURATION_BUCKETS,
)

# Maps timing-dict keys (milliseconds) to Histogram phase label values.
_PHASE_MAP: dict[str, str] = {
    "retrieve_ms": "retrieve",
    "web_ms": "web",
    "generate_ms": "generate",
    "rerank_ms": "rerank",
    "faithfulness_ms": "faithfulness",
    "total_ms": "total",
}

# Path excluded from HTTP metrics tracking (scrape traffic).
_PROMETHEUS_PATH = "/metrics/prometheus"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record_chat_timings_prometheus(timings: Mapping[str, float]) -> None:
    """Observe RAG pipeline timings into the Prometheus Histogram.

    Reuses the timings dict already produced by the chat pipeline — no
    recomputation, no change to how timings are measured.  Converts ms -> s.
    Phases with a value of 0.0 are skipped (e.g. rerank_ms when rerank is OFF,
    faithfulness_ms when RAG_FAITHFULNESS_ENABLED is False).

    Called from routers/chat.py immediately after the pipeline completes.
    """
    for ms_key, phase_label in _PHASE_MAP.items():
        value_ms = float(timings.get(ms_key) or 0.0)
        if value_ms > 0.0:
            RAG_PHASE_DURATION.labels(phase=phase_label).observe(value_ms / 1000.0)


def setup_prometheus(app: "FastAPI") -> None:
    """Attach Prometheus HTTP instrumentation and expose /metrics/prometheus.

    Registers two pieces on the app:
      1. A lightweight HTTP middleware that increments HTTP_REQUESTS_TOTAL and
         HTTP_REQUEST_DURATION for every request except /metrics/prometheus itself.
      2. A GET route at /metrics/prometheus that returns the full prometheus-client
         exposition (text format) with no API-key dependency.

    The RAG pipeline Histogram (rag_phase_duration_seconds) is registered on the
    default REGISTRY at module-import time and appears in the same exposition.
    """

    async def _prometheus_middleware(request: Request, call_next):
        path = request.url.path
        if path == _PROMETHEUS_PATH:
            return await call_next(request)  # do not track scrape requests

        method = request.method
        start = perf_counter()
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            HTTP_REQUESTS_TOTAL.labels(
                method=method, path=path, status_class="5xx"
            ).inc()
            raise

        duration_s = perf_counter() - start
        status_class = f"{status // 100}xx"
        HTTP_REQUESTS_TOTAL.labels(
            method=method, path=path, status_class=status_class
        ).inc()
        HTTP_REQUEST_DURATION.labels(method=method, path=path).observe(duration_s)
        return response

    async def _prometheus_endpoint():
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    app.middleware("http")(_prometheus_middleware)
    app.add_api_route(
        _PROMETHEUS_PATH,
        _prometheus_endpoint,
        methods=["GET"],
        include_in_schema=False,
        tags=["metrics"],
    )
    logger.info(
        "Prometheus metrics endpoint mounted at %s (public, no API key).",
        _PROMETHEUS_PATH,
    )

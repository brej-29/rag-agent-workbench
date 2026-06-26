# Load Test Report — /chat endpoint

## Purpose

This report documents a benchmark run of the `/chat` pipeline under controlled
in-process conditions.  It establishes a baseline for **framework overhead**
(FastAPI routing, LangGraph traversal, Pydantic serialization) with no real
external I/O.  This is the T3-C Part 2 deliverable.

---

## Run conditions

| Parameter | Value |
|---|---|
| Date | 2026-06-26 |
| Tool | `scripts/bench_mocked.py` |
| Transport | `httpx.ASGITransport(app=app)` — in-process, no TCP |
| Server | No real server process; ASGI interface called directly |
| Requests | 50 |
| Concurrency | 10 |
| Python | 3.11.15 |
| Platform | Windows 11, Intel/AMD x86-64 (GIL-bound) |

### What was mocked

| Boundary | Mock behaviour |
|---|---|
| Pinecone vector search | Instant return of 1 chunk (cosine 0.92) |
| Groq LLM (`generate_answer`) | `MagicMock.invoke()` returning a fake AIMessage |
| Groq LLM (`streaming`) | No-op async generator |
| Tavily web search | Disabled (`is_tavily_configured=False`) |
| FastAPI startup (Pinecone init) | `init_pinecone` no-op |
| Response cache | Disabled (`cache_enabled=False`) |
| slowapi rate limiter | `limiter.enabled=False` (prevents 30/min limit from firing across 50 requests from one IP) |

### What ran for real

The full in-process request path: ASGI receive/send, FastAPI middleware
(CORS, metrics collection, auth header check), `require_api_key` dependency,
the `run_in_threadpool` dispatch into the LangGraph pipeline, all 7 graph
nodes (`normalize_input` → `contextualize_query` → `retrieve_context` →
`corrective_retrieve` → `decide_next` → `generate_answer` →
`format_response`), prompt building, `filter_chunks_by_score`, citation
verification, `ChatResponse` Pydantic serialization, and JSON response
encoding.

---

## Results

```
=== /chat in-process bench (mocked externals) ===
Requests:        50
Concurrency:     10
Errors:          0 (0.0%)
Wall time:       1321 ms
Throughput:      37.9 req/s
Avg latency:     252.02 ms
p50 latency:     272.73 ms
p95 latency:     448.47 ms
```

---

## Interpretation

### What these numbers measure

The p50 of **273 ms** is the cost of routing, middleware, auth, LangGraph
node traversal, schema validation, and JSON serialization — with zero I/O
latency.  It is a floor, not a ceiling: in production, Pinecone and Groq API
latency dominate (typically 100–800 ms combined), and the p50 would be
600–1500 ms end-to-end.

### Why p50 is ~270 ms with mocked externals

The primary bottleneck is Python's GIL combined with `run_in_threadpool`:

- The router dispatches `graph.invoke()` via `asyncio.run_in_threadpool`,
  which schedules the call on the default `ThreadPoolExecutor`.
- With 10 concurrent requests, 10 threads compete for the GIL to execute
  LangGraph's pure-Python node traversal.
- Each node call holds the GIL during its Python bytecode execution.
- Effective concurrency is constrained — threads execute interleaved, not
  truly parallel, under CPU-bound load.

The graph's self-reported `generate_ms ≈ 0.02 ms` (logged per request)
reflects only the mock's `.invoke()` call time, not the thread scheduling
overhead or GIL contention visible from the outside.

### Relationship to Prometheus latency (T2.6)

The T2.6 Prometheus histogram (`rag_request_duration_seconds`) records
**total time from request receipt to response dispatch**, matching what this
bench measures.  The p95 of 448 ms under 10-concurrency simulated load sets
an expectation: with real Groq and Pinecone I/O, the Prometheus p95 bucket
should track at 600–1500 ms in nominal operation (1–2 concurrent users).

A sharp rise in the Prometheus p95 above 2000 ms with mocked externals (if
reproduced) would point to GIL starvation at higher concurrency — a signal
to consider either reducing LangGraph node count or offloading to a
subprocess pool.

### Throughput ceiling

**37.9 req/s** with 10 concurrent threads and zero I/O represents an
upper bound on single-machine throughput with the current GIL-bound design.
Real throughput (with Groq + Pinecone) at 10 concurrency would be limited
by external I/O (Groq: ~200–800 ms) and would likely plateau at 5–15 req/s.

### What this run does NOT measure

| Gap | Reason |
|---|---|
| Real Pinecone latency | Mocked — would add 50–200 ms per request |
| Real Groq latency | Mocked — would add 200–800 ms per request |
| LangSmith tracing overhead | Disabled (no real `LANGSMITH_API_KEY`) |
| Cold start (graph compilation) | First request compiles the graph; amortized here |
| GZip compression middleware | Not added to this app |

---

## How to reproduce

```bash
cd backend
# Needs a Python env with dependencies installed
PYTHONPATH=backend python scripts/bench_mocked.py
```

The script self-configures dummy credentials and disables all real external
calls.  No Pinecone or Groq account is required.

---

## Next steps

If real-traffic profiling shows p95 > 2000 ms under ≥ 5 concurrent users:

1. Profile with `py-spy` to identify which LangGraph node holds the GIL
   longest.
2. Consider converting CPU-bound graph nodes to `async def` with direct
   `await` on I/O (removing the `run_in_threadpool` wrapper).
3. Evaluate LangGraph's async `astream` / `ainvoke` path for the `/chat`
   endpoint.

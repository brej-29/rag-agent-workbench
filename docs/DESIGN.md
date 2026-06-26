# RAG Agent Workbench — Design Document

> **Audience:** Engineers and recruiters reviewing this repo.  
> **Purpose:** Explain the *decisions* behind the system — not just what it does, but why each
> choice was made and what the real tradeoffs are.  
> Exhaustive detail lives in [`docs/CONTEXT.md`](CONTEXT.md); this document curates the decisions
> that matter most.

---

## What this is

A production-style RAG (Retrieval-Augmented Generation) backend built as a deliberate engineering
exercise in decision-driven design.  It ingests documents from Wikipedia, arXiv, and OpenAlex
into a Pinecone vector index, then answers questions over that corpus via a **7-node LangGraph
pipeline** backed by Groq (LLaMA) and optional Tavily web search.

The headline capability: agentic RAG with corrective retrieval, cosine-gated abstention,
two-layer faithfulness checking, honest token-level streaming, and per-request cost accounting —
all wired to a Streamlit chat UI and a Prometheus metrics endpoint.

Every major feature was preceded by a retrieval evaluation harness.  The rule: no parameter
change without a measurement that justifies it.

**Stack:** FastAPI · LangGraph/LangChain · Pinecone (`llama-text-embed-v2`, 1024-dim, cosine) ·
Groq (LLaMA 3.1 8B) · Tavily (optional) · Streamlit · Prometheus · Docker

---

## Architecture

See the [pipeline diagram in the README](../README.md#architecture) for the full node flow.

A request to `POST /chat` passes through:

1. **FastAPI middleware** — CORS, API key auth (`X-API-Key`), slowapi rate limit (30 req/min),
   Prometheus HTTP instrumentation, in-memory TTL cache check.
2. **`run_in_threadpool`** — dispatches the LangGraph graph into a thread.
3. **LangGraph pipeline** (7 nodes, synchronous) — see diagram.
4. **Response serialization** — Pydantic `ChatResponse` with grounding metadata, timings,
   token usage, and source citations.

`POST /chat/stream` runs phases 1 and 3 (pre-generation nodes + post-generation grounding)
in a thread pool, with phase 2 (token generation) streamed async via `llm.astream` for real
first-token latency improvement.

---

## Key Design Decisions

### 1. Eval-first, anti-circular-validation

The evaluation harness (`eval/`) was built before any parameter was tuned.  Golden-set
`relevant_doc_ids` are determined by reading document content — never by running the retriever
and labelling its own output.  Doing so would make recall@k tautological (the retriever would
appear to have perfect recall because labels were derived from its output).

**Tradeoff:** building the harness first added upfront cost with no immediate feature output.
The payoff is that every subsequent decision (reranking, top_k, cosine floor) is backed by a
number, not intuition.

---

### 2. Two-threshold retrieval gate

Two independently configurable cosine thresholds serve different purposes:

| Setting | Default | Purpose |
|---|---|---|
| `RAG_MIN_SCORE` | 0.25 | **Routing:** if `top_score < 0.25`, route to Tavily web fallback |
| `RAG_MIN_CHUNK_SCORE` | **0.20** | **Safety floor:** drop individual Pinecone chunks below this cosine score before they enter the LLM context |

The floor at 0.20 is a **data-derived safety bound**: the minimum cosine score of any
golden-relevant chunk across 30 evaluation queries was 0.2368.  Setting the floor at 0.20
places it below this bound so no known-relevant chunk is dropped.  It is not a tuned optimum —
sharp floor calibration requires chunk-level graded relevance labels.

**Tradeoff:** two thresholds with different semantics create configuration surface.  Keeping
them distinct (even at different defaults) avoids the silent failure mode of a single threshold
accidentally serving both routing and filtering purposes.

---

### 3. Reranking: evaluated and disabled

A Pinecone hosted reranker (`bge-reranker-v2-m3`) was implemented, A/B tested against the
baseline, and **disabled by default** after measurement showed it was flat-or-negative at every
metric:

| Metric | Baseline | Rerank | Δ |
|---|---|---|---|
| nDCG@3 | 0.875 | 0.818 | −0.057 |
| nDCG@5 | 0.900 | 0.869 | −0.031 |
| Precision@1 | 0.966 | 0.966 | 0.000 |
| Mean latency | 360 ms | 795 ms | +435 ms |

**Root cause:** the corpus (34 chunks / 23 docs) is too small and well-separated for the
dense retriever to miscalibrate top-of-list order.  The reranker cannot demonstrate headroom
it never had.  `RAG_RERANK_ENABLED=False` is the empirically-validated default — enable only
after the corpus grows to where dense retrieval misfires on precision.

---

### 4. top_k = 5: precision-first

The quality-vs-k curve (n=30 queries) shows:

| k | Recall@k | P@k |
|---|---|---|
| 5 | 0.914 | 0.360 |
| 8 | 0.969 | 0.242 |
| 10 | 0.981 | 0.197 |

The **recall-margin knee** is k=8 (both recall and nDCG within 0.02 of the k=10 ceiling).
Despite this, `RAG_DEFAULT_TOP_K` is kept at **5** — a precision-first choice: k=5 delivers
higher-signal context (P@5=0.36 vs P@8=0.24) at the accepted cost of 6.7 recall points.

**Tradeoff:** recall@k cannot settle this — it measures whether relevant docs appear in the
ranked list, not whether a larger-but-noisier context improves LLM answer quality.  The
tiebreaker is a head-to-head answer-quality evaluation, which does not yet exist.  Until it
does, context signal quality is preferred over recall coverage.

---

### 5. Bounded CRAG corrective loop

`corrective_retrieve` (between `retrieve_context` and `decide_next`) grades retrieval quality
by the cosine score already in state.  If weak, it rewrites the query with Groq and re-queries
Pinecone — up to `RAG_CRAG_MAX_ITERS=2` times (a hard, unconditional loop bound).

The bound is **non-negotiable**: without it, a query on a topic not in the knowledge base would
spin indefinitely on weak retrieval, exhausting rate limits and blocking the response.

**Disabled by default** (`RAG_CRAG_ENABLED=False`): the corpus is saturated at recall@10=0.97,
so the corrective loop fires rarely on in-corpus queries.  Enable it only after observing
out-of-corpus queries where initial retrieval fails and the rewrite demonstrably helps.

**Circular-validation avoidance:** the grader uses the cosine score already in state — it does
not re-embed with the retrieval model.  Re-embedding would assess the retriever's output with
the retriever's own semantic space.

---

### 6. Two-layer faithfulness check

| Layer | When | Model calls | What it checks |
|---|---|---|---|
| `verify_citations` | Always | Zero | `[n]` citation markers that reference out-of-range chunk indices |
| `judge_faithfulness` | When `RAG_FAITHFULNESS_ENABLED=True` + not abstaining | 1 (reuses Groq client) | Whether answer claims are supported by the retrieved context |

The judge uses the **existing Groq LLM** — not the retrieval embedder.  Re-embedding the answer
with the same model used for retrieval would encode the embedder's biases into the faithfulness
signal (circular validation).

**Flag default OFF:** every `/chat` request would otherwise pay for a second LLM call.  On
Groq's free tier the cost is latency, not money, but it is still undesirable for interactive
use.  When the flag is OFF, `grounded` and `faithfulness_score` in `ChatResponse` are `null` —
the UI renders this as "not evaluated", never fabricates a value.

---

### 7. Honest streaming

`/chat/stream` uses `llm.astream` for the generation phase only (the nodes where it matters for
TTFT).  Pre-generation nodes (retrieval, CRAG, web search) are run synchronously in a thread
pool — making them async-native would add complexity with no meaningful latency improvement.

**Non-streamable paths are honest:**
- Cache hit → one token event with the full cached answer, `done.cached=true`
- Abstention → one token event with the deterministic abstention text
- Neither path calls the LLM or simulates token-by-token output

The previous implementation yielded whitespace-split words from a completed string.  That
misrepresented itself as streaming.

---

### 8. Cost and token observability

Token counts come from the **actual API response** (`response.usage_metadata`), not a local
tokenizer estimate.  All four LLM call types (generation, faithfulness judge, CRAG rewrite,
history contextualization) are tracked by `call_type` in `ChatResponse.usage.by_call_type` and
emitted as a Prometheus counter (`llm_tokens_total{call_type=...}`).

Dollar cost is an **estimate** from an as-of-date pricing table (`2026-06-25`) and is labeled
as such.  Embedding token counts are not reported — the Pinecone SDK does not expose them.

---

### 9. Reproducible corpus + pinned dimension

A corpus manifest (`eval/corpus_manifest.py generate`) snapshots vector IDs from the live
Pinecone index to `eval/corpus_manifest.json`.  A validator (`corpus_manifest.py validate`)
compares the committed manifest against the live index and reports drift without auto-reconciling.
Both operations are read-only.

The embedding model (`llama-text-embed-v2`) and dimension (1024) are now explicit in `Settings`
(`PINECONE_EMBED_MODEL`, `PINECONE_EMBED_DIMENSION`) and logged at startup — removing the
implicit dependency on Pinecone's default dimension.

---

## Limitations & Tradeoffs

These are the real constraints.  A design doc that only lists strengths reads as incomplete.

**1. Saturated eval corpus.**
The evaluation golden set covers 34 chunks / 23 documents.  At this scale, baseline dense
retrieval is already at recall@10=0.97 — the metrics are ceiling-bound.  Any apparent
improvement (whether from reranking, CRAG, or parameter changes) may be noise rather than
signal.  No feature can be conclusively validated until the corpus is at least 10× larger.

**2. Prompt injection mitigation, not elimination.**
The RAG system prompt instructs the LLM to use only the supplied context and cite inline.
This reduces prompt injection risk but does not eliminate it: a sufficiently adversarial document
can still attempt to override instructions via embedded directives in chunk text.

**3. Same-model faithfulness judge.**
The faithfulness judge calls the same Groq LLM that generated the answer.  A model grading its
own output has a self-preference bias — it may rate its own claims as grounded even when they
are not.  A second independent model (e.g. a different provider) would give a less biased
verdict but at higher cost and latency.

**4. Cost is an estimate.**
`estimated_cost_usd` is computed from a static pricing table pinned to 2026-06-25.  It does
not account for free-tier credits, batch pricing, or promotional rates.  Treat it as an order-
of-magnitude indicator, not a billing source of truth.

**5. Reranking and hybrid search deferred — not for lack of trying.**
Reranking was implemented and A/B tested; it is disabled because the measurement showed no
improvement on this corpus size, not because the implementation is absent.  Hybrid search
(sparse + dense) is documented and designed but not implemented — the recall gap it would address
(proper-noun queries) does not exist at current corpus size, where baseline recall@10=0.97.

**6. Chunk size below recommended range.**
The `RecursiveCharacterTextSplitter` is configured to ~225 tokens per chunk (900 chars ÷ ~4
chars/token).  Pinecone's guidance for `llama-text-embed-v2` suggests 400–500 tokens for best
retrieval quality.  The current chunks are too short to exploit the model's full context window.
Changing `chunk_size` requires re-ingestion and re-evaluation against the golden set.

**7. CRAG threshold and faithfulness threshold are placeholders.**
`RAG_CRAG_GOOD_SCORE=0.45` (the cosine threshold that triggers query rewriting) and
`RAG_FAITHFULNESS_THRESHOLD=0.5` (the faithfulness score below which `grounded=False`) are
reasonable midpoints — not values calibrated against labeled data.  Both require a held-out
answer-quality evaluation to tune.

---

## Testing & Observability

**343 tests** (321 unit + 22 integration) run in CI with zero network calls, zero credentials.

| Layer | What it tests |
|---|---|
| Unit (321) | Pure functions: metrics, chunking, normalization, dedup, prompt builders, retrieval gating, faithfulness, CRAG, streaming, Prometheus, cost accounting |
| Integration (22) | Real FastAPI app via `TestClient` — HTTP routing, auth dependency, LangGraph pipeline, SSE protocol, abstention path, faithfulness wiring; externals mocked at boundaries |

CI runs from the fully-pinned `backend/requirements.txt` lock (compiled with `uv pip compile`,
constrained to tested versions) — every CI run is a clean-environment reproducibility check.

Observability:
- **`/metrics`** (JSON, auth-gated) — request counts, error counts, 20-sample timing ring buffer
- **`/metrics/prometheus`** (Prometheus text, public) — `http_requests_total` (Counter),
  `http_request_duration_seconds` (Histogram), `rag_phase_duration_seconds` (Histogram),
  `llm_tokens_total` (Counter by `call_type`)
- **LangSmith** — optional trace collection via `LANGCHAIN_TRACING_V2=true`

---

## How to Run

```bash
# Backend
cd backend
pip install -r requirements.txt
cp .env.example .env           # fill in PINECONE_*, GROQ_API_KEY, optional API_KEY
uvicorn app.main:app --port 8000

# Frontend
pip install -r requirements.txt   # root (Streamlit)
streamlit run frontend/app.py

# Run tests (zero credentials needed)
pytest tests/ -v

# Evaluate retrieval (requires live Pinecone — reads only)
make eval

# Load benchmark (in-process, mocked externals)
PYTHONPATH=backend python scripts/bench_mocked.py
```

Full configuration reference: [`backend/.env.example`](../backend/.env.example)  
Operational runbook (key rotation, rate-limit toggle, deployment): [`docs/CONTEXT.md`](CONTEXT.md)

<div align="center">
  <h1>🧠 rag-agent-workbench</h1>
  <p><i>Production-style RAG backend with an 8-node LangGraph pipeline, cosine-gated abstention,
  corrective retrieval, two-layer faithfulness checking, and honest token streaming.</i></p>
</div>

<br>

<div align="center">
  <img alt="Language" src="https://img.shields.io/badge/Language-Python-blue">
  <img alt="Backend" src="https://img.shields.io/badge/Backend-FastAPI-009688">
  <img alt="Vector Store" src="https://img.shields.io/badge/Vector%20Store-Pinecone-3776AB">
  <img alt="Frameworks" src="https://img.shields.io/badge/Frameworks-LangChain%20%7C%20LangGraph-ff9800">
  <img alt="Frontend" src="https://img.shields.io/badge/Frontend-Streamlit-ff4b4b">
  <img alt="Tests" src="https://img.shields.io/badge/Tests-351%20passing-brightgreen">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-black">
</div>

---

## What this is

An agentic RAG system built as a deliberate engineering exercise: every design decision is
measurement-driven, every limitation is documented, and every feature ships with a test.

**Stack:** FastAPI · LangGraph/LangChain · Pinecone (`llama-text-embed-v2`, 1024-dim) · Groq
(`openai/gpt-oss-20b`, fallback `gpt-oss-120b`) · Tavily · Streamlit · Prometheus · Docker

**Standout decisions** (details in [`docs/DESIGN.md`](docs/DESIGN.md)):
- **Eval-first:** retrieval evaluation harness built before any parameter was tuned; labels
  set by reading documents, not by running the retriever (anti-circular-validation).
- **Reranking measured and disabled:** `bge-reranker-v2-m3` A/B tested — nDCG@3 fell 0.057,
  +435 ms latency, no tradeoff found. `RAG_RERANK_ENABLED=False` is empirically validated.
- **top_k=5, precision-first:** the recall-margin knee is k=8 (recall@8=0.97), but k=5
  (P@5=0.36 vs P@8=0.24) delivers higher-signal context. Tiebreaker: an answer-quality eval
  that doesn't yet exist.
- **Cosine floor = 0.20, data-derived:** set below the 0.2368 minimum cosine score of any
  golden-relevant chunk — a safety bound, not a tuned optimum.
- **Bounded CRAG:** hard `max_iters=2` loop guard, disabled by default on a saturated corpus.
- **Honest streaming:** `llm.astream` for real TTFT; cache hits and abstentions are served
  honestly with explicit `done.cached` / `insufficient_context` flags — never fake word-splits.

---

## Architecture

```mermaid
flowchart TD
    Client(["Client\n(HTTP / Streamlit)"])

    subgraph FastAPI["FastAPI (backend/app)"]
        MW["Middleware\nCORS · Auth · Rate limit · Prometheus · Cache"]
        ROUTER["/chat · /chat/stream\n/search · /ingest · /metrics"]
    end

    subgraph LangGraph["LangGraph Pipeline"]
        N1["normalize_input\n(defaults, history)"]
        N2["contextualize_query\n(multi-turn rewrite — flag)"]
        N3["retrieve_context\n(Pinecone top-k)"]
        N4["corrective_retrieve\n(CRAG grade → rewrite — flag)"]
        N5{"decide_next\n(top_score < RAG_MIN_SCORE?)"}
        N6["web_search\n(Tavily — optional)"]
        N7["generate_answer\nCosine floor RAG_MIN_CHUNK_SCORE=0.20\nUsable context? No → ABSTAIN"]
        N8["format_response\nverify_citations (always)\njudge_faithfulness (flag)"]
    end

    subgraph External["External Services"]
        PC[("Pinecone\nllama-text-embed-v2\n1024-dim cosine")]
        GR[("Groq\ngpt-oss-20b, 120b fallback")]
        TV[("Tavily\nWeb Search")]
        LS[("LangSmith\nTracing — optional")]
    end

    ABSTAIN(["Abstention\ninsufficient_context=True\nno LLM call"])
    RESP(["ChatResponse\nanswer · sources · timings\ngrounded · faithfulness_score\nusage · crag_iterations"])

    Client --> MW --> ROUTER
    ROUTER --> N1 --> N2 --> N3 --> N4 --> N5
    N5 -- "yes (web fallback)" --> N6 --> N7
    N5 -- "no" --> N7
    N7 -- "no usable context" --> ABSTAIN
    N7 -- "context OK" --> GR
    GR --> N8 --> RESP
    RESP --> Client

    N3 <--> PC
    N6 <--> TV
    N8 -.->|faithfulness judge| GR
    N8 -.->|traces| LS
```

> **Node-to-code mapping:**
> `normalize_input` · `contextualize_query` · `retrieve_context` · `corrective_retrieve` ·
> `decide_next` · `generate_answer` · `format_response` — all in
> [`backend/app/services/chat/graph.py`](backend/app/services/chat/graph.py)

---

## Live, evaluation and limits

| | |
|---|---|
| API (HF Space, Docker, CPU Basic) | https://brejbala-rag-agent-workbench-api.hf.space · [`/docs`](https://brejbala-rag-agent-workbench-api.hf.space/docs) |
| UI (Streamlit Community Cloud) | https://rag-agent-workbench.streamlit.app |

**Retrieval eval** (`make eval`, 30 hand-labelled queries, `eval` namespace, 2026-10-19):

| Metric | Value | | k sweep | Recall@k | nDCG@k | P@k |
|---|---|---|---|---|---|---|
| Recall@10 | 0.9694 | | k=5 (production) | 0.9139 | 0.9030 | 0.36 |
| MRR | 0.9833 | | k=8 (knee) | 0.9694 | 0.9280 | 0.24 |
| nDCG@10 | 0.9280 | | k=10 | 0.9806 | 0.9326 | 0.20 |

**Environment variables (names only):** backend `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`,
`PINECONE_HOST`, `PINECONE_NAMESPACE`, `PINECONE_TEXT_FIELD`, `GROQ_API_KEY`, `GROQ_BASE_URL`,
`GROQ_MODEL`, `LLM_FALLBACK_MODELS`, `LLM_REASONING_EFFORT`, `TAVILY_API_KEY`, `API_KEY`, `ENV`,
`ALLOWED_ORIGINS`, `RAG_*` flags; UI `BACKEND_BASE_URL`, `API_KEY`. Full list:
[`backend/README.md`](backend/README.md).

**Known limits (free tiers):**
- The HF Space sleeps after 48 h idle. The first request after that waits for a restart
  (about 10 s measured on 2026-10-19).
- The Streamlit app sleeps after 12 h without traffic; a visitor clicks "wake up".
- Pinecone Starter: one region (`us-east-1`), 2 GB, 1M read units a month.
  Tavily free: 1,000 credits a month.
- Model retirement: `make check-models` (also run weekly in CI) fails loudly when a configured
  model is retired or appears on Groq's deprecations page. `LLM_FALLBACK_MODELS` keeps `/chat`
  answering while you switch.

---

## Screenshot

<img width="1919" height="967" alt="image" src="https://github.com/user-attachments/assets/fe979aa1-b125-415c-9a1d-64a96289af87" />

<img width="1918" height="896" alt="image" src="https://github.com/user-attachments/assets/1d40bb2b-f477-4741-8014-0b2c2a7356bf" />

---

## Getting Started

### Backend

```bash
cd backend
pip install --require-hashes -r requirements.txt   # hash-pinned lock, Python 3.11
cp .env.example .env      # set PINECONE_*, GROQ_API_KEY; API_KEY optional
uvicorn app.main:app --port 8000
```

The backend needs a reachable Pinecone index at startup (it validates the index and exits on
a bad key or host).

Browse: `http://localhost:8000/health` · `http://localhost:8000/docs`

### Frontend

```bash
pip install --require-hashes -r requirements.txt   # root (Streamlit), Python 3.13
streamlit run frontend/app.py
```

High-fidelity PDF/Office conversion in the upload dialog uses docling (torch, several GB).
It is not in the hosted UI lock; for local use install `requirements-docling.txt` instead.
Without it, PDFs go through pypdf and Office files show an install hint.

Set `BACKEND_BASE_URL` and (if the backend is protected) `API_KEY` in
`.streamlit/secrets.toml` or environment variables.

### Tests

```bash
pytest tests/ -v        # 351 tests, zero network, zero credentials (never reads .env)
make check-models       # live: fails if a configured model is retired or deprecated
```

---

## Project Structure

```text
rag-agent-workbench/
├─ backend/
│  ├─ app/
│  │  ├─ routers/        # chat, search, ingest, metrics, health
│  │  ├─ services/       # chat/graph.py (LangGraph), pinecone_store, llm, rerank, crag, …
│  │  ├─ core/           # config, auth, cache, rate_limit, prometheus_metrics, …
│  │  └─ schemas/        # Pydantic models (ChatResponse, SourceHit, …)
│  ├─ requirements.in    # loose intent
│  └─ requirements.txt   # pinned lock (uv pip compile)
├─ frontend/app.py       # Streamlit chatbot UI
├─ eval/                 # retrieval harness, golden.jsonl, corpus manifest
├─ tests/                # 351 tests — unit + integration (+ tests/ui/smoke_app.py)
├─ scripts/              # bench_local.py, bench_mocked.py, smoke tests
└─ docs/
   ├─ DESIGN.md          # ← design decisions + limitations (start here)
   ├─ CONTEXT.md         # full decision log, work package history, runbook
   └─ LOAD_TEST.md       # in-process benchmark report
```

---

## Documentation

| Document | What it covers |
|---|---|
| **[`docs/DESIGN.md`](docs/DESIGN.md)** | Design decisions, tradeoffs, limitations — the engineering story |
| [`docs/CONTEXT.md`](docs/CONTEXT.md) | Full decision log, eval results, dependency management, runbook |
| [`docs/LOAD_TEST.md`](docs/LOAD_TEST.md) | In-process benchmark report (framework overhead, GIL analysis) |
| [`backend/README.md`](backend/README.md) | API catalogue, env vars, HF Spaces deployment |

---

## License

MIT — see [`LICENSE`](LICENSE).

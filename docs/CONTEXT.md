# RAG Agent Workbench – Context and Design

## Project Purpose

RAG Agent Workbench is a lightweight experimentation backend for retrieval-augmented generation (RAG). It focuses on:
- Fast ingestion of documents into a Pinecone index with integrated embeddings.
- Simple, production-style APIs for search and chat-style question answering.
- Keeping the backend slim: no local embedding or LLM models, relying instead on managed services.

---

## Current Architecture

- **Client(s)**  
  - Any HTTP client (curl, scripts in `scripts/`, future UI) talks to the FastAPI backend.

- **Backend (FastAPI, `backend/app`)**
  - `routers/`
    - `health.py` – service status.
    - `ingest.py` – /ingest/wiki, /ingest/openalex, /ingest/arxiv.
    - `documents.py` – manual uploads and stats.
    - `search.py` – semantic search over Pinecone.
    - `chat.py` – agentic RAG chat using LangGraph + LangChain.
  - `services/`
    - `ingestors/` – fetch content from arXiv, OpenAlex, Wikipedia.
    - `chunking.py` – chunk documents into Pinecone-ready records.
    - `dedupe.py` – in-memory duplicate record removal.
    - `normalize.py` – text normalisation and doc id generation.
    - `pinecone_store.py` – Pinecone init, search, upsert, stats.
    - `llm/groq_llm.py` – Groq-backed chat model wrapper.
    - `tools/tavily_tool.py` – Tavily web search integration.
    - `prompts/rag_prompt.py` – RAG system + user prompts.
    - `chat/graph.py` – LangGraph state graph for /chat.
  - `core/`
    - `config.py` – env-driven configuration.
    - `errors.py` – app-specific exceptions + handlers.
    - `logging.py` – basic logging setup.
    - `tracing.py` – LangSmith / LangChain tracing helper.
  - `schemas/` – Pydantic models for all endpoints.

- **Vector Store**
  - Pinecone index with integrated embeddings.
  - Text field configurable via `PINECONE_TEXT_FIELD`.

- **LLM and Tools**
  - Groq OpenAI-compatible chat model via `langchain-openai`.
  - Tavily web search via `langchain-community` tool (optional).
  - LangGraph orchestrates retrieval → routing → web search → generation.

---

## Implemented Endpoints

| HTTP Method | Path                    | Description                                                      |
|------------|-------------------------|------------------------------------------------------------------|
| GET        | `/health`               | Health check with service name and version.                      |
| POST       | `/ingest/arxiv`         | Ingest recent arXiv entries matching a query.                    |
| POST       | `/ingest/openalex`      | Ingest OpenAlex works matching a query.                          |
| POST       | `/ingest/wiki`          | Ingest Wikipedia pages by title.                                 |
| POST       | `/documents/upload-text`| Upload raw/manual text or Docling-converted content.             |
| GET        | `/documents/stats`      | Get vector counts per namespace from Pinecone.                   |
| POST       | `/search`               | Semantic search over Pinecone using integrated embeddings.       |
| POST       | `/chat`                 | Production-style RAG chat using LangGraph + Groq + Pinecone.     |
| POST       | `/chat/stream`          | SSE streaming variant of `/chat`.                                |

---

## Key Design Decisions

- **Integrated embeddings only**
  - No local embedding models; Pinecone is configured with integrated embeddings.
  - Backend stays light and easy to deploy in constrained environments.

- **OpenAI-compatible LLM interface**
  - Groq is accessed via the OpenAI-compatible API (`langchain-openai`).
  - Avoids additional provider-specific SDKs and keeps integration simple.

- **Agentic RAG flow using LangGraph**
  - Chat pipeline is modelled as a state graph:
    1. `normalize_input` – set defaults, normalise chat history.
    2. `retrieve_context` – Pinecone retrieval.
    3. `corrective_retrieve` – CRAG loop (pass-through when `RAG_CRAG_ENABLED=False`).
    4. `decide_next` – route to web search or generation.
    5. `web_search` – Tavily search (optional).
    6. `generate_answer` – Groq LLM with RAG prompts.
    7. `format_response` – post-generation grounding checks.
  - This makes the flow explicit and easy to extend.

- **Web search as a conditional fallback**
  - Tavily web search is used only when:
    - Retrieval returns no hits, or
    - Top score is below a threshold (`min_score`), and
    - `use_web_fallback=true` and `TAVILY_API_KEY` is configured.
  - When Tavily is not configured, the system degrades gracefully to retrieval-only.

- **LangSmith tracing via environment flags**
  - Tracing is enabled purely via environment:
    - `LANGCHAIN_TRACING_V2=true`
    - `LANGCHAIN_API_KEY` set
    - Optional: `LANGCHAIN_PROJECT`
  - `core/tracing.py` exposes helper functions that:
    - Check if tracing is enabled.
    - Construct callback handlers (`LangChainTracer`) for LangGraph/LangChain.
    - Expose trace metadata in API responses.

- **Error handling boundary**
  - External dependencies (Pinecone, Groq, Tavily) are wrapped so that:
    - Configuration errors return 500s with clear messages.
    - Upstream service failures raise `UpstreamServiceError` and surface as HTTP 502.
  - This keeps failure modes explicit for clients.

---

## Work Package History

### Work Package A

- **Scope**
  - Initial backend setup with FastAPI, Pinecone integration, and ingestion/search endpoints.
- **Highlights**
  - `/ingest/wiki`, `/ingest/openalex`, `/ingest/arxiv` for sourcing content.
  - `/documents/upload-text` for manual/Docling-based uploads.
  - `/search` and `/documents/stats` endpoints to query and inspect the index.
- **How to test**
  - Use `scripts/seed_ingest.py` and `scripts/smoke_arxiv.py` to seed and smoke-test ingestion.

### Work Package B (this change)

- **Scope**
  - Add a production-style `/chat` RAG endpoint using LangGraph and LangChain.
  - Integrate Groq as the LLM and Tavily as an optional web search fallback.
  - Introduce LangSmith tracing hooks and update documentation and smoke tests.

- **Functional changes**
  - New router: `backend/app/routers/chat.py`
    - `POST /chat`
      - Runs a LangGraph state graph:
        1. Normalises inputs and defaults.
        2. Retrieves context from Pinecone.
        3. Decides whether to invoke web search.
        4. Runs Tavily web search when enabled and needed.
        5. Calls Groq LLM with a RAG prompt to generate the answer.
        6. Returns answer, sources, timings, and trace metadata.
    - `POST /chat/stream`
      - Same pipeline as `/chat` but returns Server-Sent Events (SSE).
      - Streams tokens from the final answer plus a terminating event with the full JSON payload.

  - New schemas: `backend/app/schemas/chat.py`
    - `ChatRequest` with:
      - `query`, `namespace`, `top_k`, `use_web_fallback`,
        `min_score`, `max_web_results`, and `chat_history`.
    - `SourceHit` representing document/web snippets.
    - `ChatTimings` and `ChatTraceMetadata` for timings and LangSmith info.
    - `ChatResponse` combining answer, sources, timings, and trace metadata.

  - New services:
    - `backend/app/services/llm/groq_llm.py`
      - `get_llm()` returns a Groq-backed `ChatOpenAI` with:
        - `base_url` = `GROQ_BASE_URL` (default `https://api.groq.com/openai/v1`).
        - `model` = `GROQ_MODEL` (default `llama-3.1-8b-instant`).
        - Timeouts and retries from HTTP settings.
      - Raises a configuration error if `GROQ_API_KEY` is missing.

    - `backend/app/services/tools/tavily_tool.py`
      - `is_tavily_configured()` checks `TAVILY_API_KEY`.
      - `get_tavily_tool(max_results)` wraps `TavilySearchResults` from
        `langchain-community`.
      - Logs a warning and returns `None` when Tavily is not configured, disabling web fallback gracefully.

    - `backend/app/services/prompts/rag_prompt.py`
      - Defines RAG system and user prompts.
      - `build_rag_messages(chat_history, question, sources)` builds
        LangChain messages that:
        - Use only supplied context.
        - Label context snippets as `[1]`, `[2]`, etc., and instruct the model
          to cite them inline.

    - `backend/app/services/chat/graph.py`
      - Implements the LangGraph `ChatState` and state graph with nodes:
        - `normalize_input`
        - `retrieve_context`
        - `decide_next`
        - `web_search`
        - `generate_answer`
        - `format_response`
      - Uses Pinecone search for retrieval and Tavily for optional web search.
      - Calls the Groq LLM via `get_llm()` with LangChain Runnable config
        (`callbacks`) so LangSmith traces are collected when enabled.
      - Records `retrieve_ms`, `web_ms`, and `generate_ms` in `timings`.

  - New core utility:
    - `backend/app/core/tracing.py`
      - `is_tracing_enabled()` checks `LANGCHAIN_TRACING_V2` and `LANGCHAIN_API_KEY`.
      - `get_tracing_callbacks()` returns a `LangChainTracer` callback list when enabled.
      - `get_tracing_response_metadata()` returns `{langsmith_project, trace_enabled}`.

  - Configuration changes:
    - `backend/app/core/config.py` adds:
      - `GROQ_API_KEY`, `GROQ_BASE_URL`, `GROQ_MODEL`.
      - `TAVILY_API_KEY`.
      - `RAG_DEFAULT_TOP_K`, `RAG_MIN_SCORE`, `RAG_MAX_WEB_RESULTS`.
    - `backend/.env.example` updated with the new env vars, including LangSmith options.

  - Error handling:
    - `backend/app/core/errors.py` introduces `UpstreamServiceError`.
    - Centralised handler converts `UpstreamServiceError` into HTTP 502 responses.

  - Documentation and scripts:
    - `backend/README.md` updated with `/chat` and `/chat/stream` usage,
      env vars, and a local test checklist.
    - New scripts:
      - `scripts/smoke_chat.py` – uses `/ingest/wiki` and `/chat` for a local smoke test.
      - `scripts/smoke_chat_web.py` – tests `/chat` with `use_web_fallback=true`
        and a query that should trigger web search.

- **How to test**
  1. Start the backend:
     ```bash
     cd backend
     uvicorn app.main:app --reload --port 8000
     ```
  2. Ingest some Wikipedia pages:
     ```bash
     python ../scripts/smoke_chat.py --backend-url http://localhost:8000 --namespace dev
     ```
  3. Test web fallback (requires `TAVILY_API_KEY`):
     ```bash
     python ../scripts/smoke_chat_web.py --backend-url http://localhost:8000 --namespace dev
     ```
  4. Verify LangSmith traces:
     - Set `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, and optionally `LANGCHAIN_PROJECT`.
     - Run `/chat` again and confirm traces appear in LangSmith.

---

## Known Issues / Limits

- **No local models**
  - The backend intentionally does not host local embedding or LLM models.
  - All intelligence is delegated to Pinecone (integrated embeddings), Groq, and Tavily.

- **Retrieval quality depends on ingestion**
  - The usefulness of `/chat` depends heavily on the quality and coverage of the ingested documents.
  - For some queries, even the best matching chunks may not be sufficient to answer without web fallback.

- **Best-effort web search**
  - Tavily integration is optional and depends on the external Tavily API.
  - When Tavily is unavailable or misconfigured, the backend falls back to retrieval-only answers.

- **Simple SSE streaming**
  - `/chat/stream` streams tokens derived from the final answer string rather than streaming directly from the LLM.
  - This keeps implementation simple while still providing a streaming interface.

---

## Work Package C

### Scope

- Make the backend deploy-ready on Hugging Face Spaces using Docker.
- Add a minimal Streamlit frontend suitable for Streamlit Community Cloud (no Docker).
- Add production polish: basic API protection, rate limiting, caching, metrics, and a small benchmarking script.
- Keep configuration sane by default, with environment variables as overrides rather than hard requirements.

### Backend changes (HF Spaces deploy + runtime)

- **Docker / port behaviour**
  - `backend/Dockerfile` now:
    - Exposes port **7860** (the default for many Hugging Face Spaces deployments).
    - Uses a shell-form `CMD` so `PORT` can be honoured when set:
      - `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}`
  - New helper: `backend/app/core/runtime.py`
    - `get_port()`:
      - Reads `PORT` from the environment.
      - Defaults to `7860` when unset or invalid.
      - Logs: `Starting on port=<port> hf_spaces_mode=<bool>` using a simple heuristic (`SPACE_ID` / `SPACE_REPO_ID` env vars).
    - Called from `app.main` at import time so the log line is visible in container logs during startup.

### API key protection and CORS

- **API key protection**
  - New module: `backend/app/core/auth.py`
    - Defines `require_api_key` FastAPI dependency using `APIKeyHeader` (`X-API-Key`).
    - `validate_api_key_configuration()` runs at startup and enforces:
      - In production-like environments (`ENV=production` or on Hugging Face Spaces via `SPACE_ID` / `HF_HOME`):
        - `API_KEY` **must** be set or the backend fails fast with a clear error.
      - In local development:
        - If `API_KEY` is missing, the backend runs open but logs a prominent warning.
    - `require_api_key` behaviour:
      - If `API_KEY` is not configured (dev mode), the dependency is a no-op.
      - If `API_KEY` is configured:
        - Missing or mismatched `X-API-Key` results in HTTP 403.
  - Wiring:
    - All routers except `/health` are registered with `dependencies=[Depends(require_api_key)]`.
    - Docs and OpenAPI endpoints are explicitly secured:
      - `GET /openapi.json` – returns `app.openapi()`, protected by `require_api_key`.
      - `GET /docs` – Swagger UI via `get_swagger_ui_html`, protected by `require_api_key`.
      - `GET /redoc` – ReDoc UI via `get_redoc_html`, protected by `require_api_key`.
    - Effect:
      - In HF Spaces / production:
        - `/docs`, `/redoc`, `/openapi.json`, `/chat`, `/search`, `/documents/*`, `/ingest/*`, `/metrics` all require `X-API-Key`.
        - `/health` remains public for simple uptime checks.
      - In local dev with no `API_KEY`:
        - All endpoints (including docs) are accessible without a key for convenience.

- **CORS configuration**
  - `backend/app/core/security.py` now focuses solely on CORS:
    - Reads `ALLOWED_ORIGINS` env var as a comma-separated list.
    - If unset or empty:
      - Defaults to `["*"]` (permissive, useful for local dev and quick demos).
    - Applies FastAPI `CORSMiddleware` with:
      - `allow_origins=origins`
      - `allow_methods=["*"]`
      - `allow_headers=["*"]`
  - API key enforcement is handled entirely via `core/auth.py` and router/dependency wiring.

### Rate limiting (SlowAPI)

- New module: `backend/app/core/rate_limit.py`
  - Uses `slowapi.Limiter` with `get_remote_address` as the key function.
  - `setup_rate_limiter(app)`:
    - Reads `RATE_LIMIT_ENABLED` from `Settings` (defaults to `True`).
    - If disabled:
      - Logs `"Rate limiting is disabled via settings."`
      - Does **not** attach middleware (decorators become no-ops at runtime).
    - If enabled:
      - Attaches SlowAPI middleware: `app.middleware("http")(limiter.middleware)`.
      - Registers a custom `RateLimitExceeded` handler returning JSON:
        - HTTP `429`
        - Body: `{"detail": "Rate limit exceeded. Please slow down your requests.", "retry_after": ...}` when available.
      - Logs violations with client IP and path.

- Endpoint-specific limits (per IP):
  - `/chat` and `/chat/stream`:
    - Decorated with `@limiter.limit("30/minute")`.
  - `/ingest` endpoints:
    - `/ingest/arxiv`, `/ingest/openalex`, `/ingest/wiki`:
      - `@limiter.limit("10/minute")`.
  - `/search`:
    - `@limiter.limit("60/minute")`.

- Operational toggle:
  - New config flag in `Settings`:
    - `RATE_LIMIT_ENABLED: bool = True`
  - `.env.example`:
    - `RATE_LIMIT_ENABLED=true` (set to `false` to disable entirely).

### Caching (cachetools, in-memory)

- New module: `backend/app/core/cache.py`
  - Uses `cachetools.TTLCache` with short in-memory TTLs (no external store):
    - **Search cache**:
      - `TTL = 60s`, `maxsize = 1024`.
      - Keys: `(namespace, query, top_k, filters_json)` where `filters_json` is a JSON-serialised, sorted representation of the `filters` dict.
    - **Chat cache**:
      - `TTL = 60s`, `maxsize = 512`.
      - Keys: `(namespace, query, top_k, min_score, use_web_fallback)`.
      - Only used when **no chat history** is provided.

  - API:
    - `cache_enabled() -> bool` (reads `CACHE_ENABLED` from settings, default `True`).
    - `get_search_cached(...)` / `set_search_cached(...)`.
    - `get_chat_cached(...)` / `set_chat_cached(...)`.
    - `get_cache_stats()` returns hit/miss counters:
      - `search_hits`, `search_misses`, `chat_hits`, `chat_misses`.

  - Hit/miss logging:
    - Each cache lookup logs a hit or miss with namespace and query for observability.

- Integration into endpoints:
  - `/search` (`backend/app/routers/search.py`):
    - On each request:
      1. Check `get_search_cached(...)`.
      2. If hit: use cached `hits_raw` list.
      3. If miss: call Pinecone search and then `set_search_cached(...)`.
    - Response construction (mapping text field to `chunk_text`) remains unchanged.

  - `/chat` (`backend/app/routers/chat.py`):
    - Caching is **only considered** when `chat_history` is empty and caching is enabled.
    - Flow:
      1. Test `cache_enabled()` and `not payload.chat_history`.
      2. Attempt `get_chat_cached(...)`.
      3. On hit:
         - Log and return the cached `ChatResponse`.
         - Still call `record_chat_timings(...)` so `/metrics` reflects cached responses.
      4. On miss:
         - Run the LangGraph pipeline as before.
         - Record timings via `record_chat_timings(...)`.
         - Store the `ChatResponse` in the chat cache via `set_chat_cached(...)`.

- Operational toggle:
  - New config flag in `Settings`:
    - `CACHE_ENABLED: bool = True`
  - `.env.example`:
    - `CACHE_ENABLED=true` (set to `false` to fully disable caching).

### Metrics and observability

- New module: `backend/app/core/metrics.py`
  - In-memory metrics only, with a small footprint and no external dependencies beyond stdlib.
  - Tracks:
    - **Request counts by path**:
      - `_request_counts[path]` incremented for every request, via `metrics_middleware`.
    - **Error counts by path**:
      - `_error_counts[path]` incremented for any response with `status_code >= 400` or for unhandled exceptions.
    - **Chat timing metrics**:
      - Focused on `/chat` and `/chat/stream`.
      - Expected fields:
        - `retrieve_ms`, `web_ms`, `generate_ms`, `total_ms`.
      - Stored in:
        - `_timing_samples`: `deque(maxlen=20)` for the last 20 samples.
        - `_timing_sums` and `_timing_count` for averages.

  - Middleware:
    - `metrics_middleware(request, call_next)`:
      - Records per-path request and error counts.
      - Logs debug-level timing for each request.

  - API functions:
    - `record_chat_timings(timings: Mapping[str, float])`:
      - Updates sums, counts, and the ring buffer.
      - Called from both `/chat` and `/chat/stream` after timings are known.
    - `get_metrics_snapshot()`:
      - Builds a snapshot dictionary containing:
        - `requests_by_path`
        - `errors_by_path`
        - `timings`:
          - `average_ms` for each timing field.
          - `p50_ms` and `p95_ms` based on the last 20 samples.
        - `cache`:
          - `search_hits`, `search_misses`, `chat_hits`, `chat_misses` from `core.cache`.
        - `sample_count` and `samples` (the last 20 timing entries).

- `/metrics` endpoint
  - New router: `backend/app/routers/metrics.py`
    - `GET /metrics` returns `get_metrics_snapshot()` as JSON.
  - Registered in `app.main` with tag `["metrics"]`.
  - Left **public** (not behind API key) to simplify monitoring and demos.

- App wiring (`backend/app/main.py`)
  - After creating the FastAPI app:
    - `configure_security(app)` – CORS + optional API key.
    - `setup_rate_limiter(app)` – SlowAPI middleware when enabled.
    - `setup_metrics(app)` – metrics middleware.
  - Routers:
    - `health`, `ingest`, `search`, `documents`, `chat`, `metrics` all included.
  - Exception handlers:
    - Still configured via `setup_exception_handlers(app)`.

### Benchmarking script

- New script: `scripts/bench_local.py`
  - Purpose:
    - Provide a simple, cross-platform (including Windows) asyncio load tester for the backend.
    - Focused on `/chat`, with optional `/search` benchmarking.
  - Implementation:
    - Uses `httpx.AsyncClient` and `asyncio`.
    - Command-line arguments:
      - `--backend-url` (default: `http://localhost:8000`)
      - `--namespace` (default: `dev`)
      - `--concurrency` (default: `10`)
      - `--requests` (default: `50`)
      - `--include-search` (optional flag to also benchmark `/search`)
      - `--api-key` (optional `X-API-Key` value)
    - For each benchmark:
      - Issues the specified number of requests with the provided concurrency.
      - Records per-request latency (ms) and whether an error occurred.
    - Outputs:
      - Total requests, successes, errors, and error rate.
      - Average latency.
      - p50 and p95 latencies.
  - Entrypoint:
    - `python scripts/bench_local.py --backend-url http://localhost:8000 --namespace dev --concurrency 10 --requests 50`

### Streamlit frontend (Streamlit Community Cloud)

- New directory: `frontend/`
  - Main app: `frontend/app.py`
    - Dependencies:
      - `streamlit`
      - `httpx`
    - Backend configuration:
      - Reads `BACKEND_BASE_URL` from `st.secrets["BACKEND_BASE_URL"]` or the `BACKEND_BASE_URL` environment variable.
      - Reads `API_KEY` from `st.secrets["API_KEY"]` or the `API_KEY` environment variable.
    - Sidebar ("Backend" + settings):
      - Shows backend URL and API key status.
      - "Ping /health" button that calls the backend and shows the JSON response.
      - `top_k` slider, `min_score` slider, `use_web_fallback` checkbox.
      - "Show sources" toggle and "Clear chat" button.
      - "Recent uploads" section with quick actions:
        - For each recent upload, displays title, namespace, timestamp.
        - A "Search this document" button pre-fills the chat input with a prompt such as `Summarize: <title>`.
    - Chatbot UI:
      - Uses `st.chat_message` and `st.chat_input` with conversation stored in `st.session_state.messages`.
      - When the user sends a message:
        - Appends it to history and displays it.
        - Calls `/chat/stream` with `X-API-Key` (if available) and streams tokens into the UI.
        - If `/chat/stream` is unavailable (e.g. 404), falls back to `/chat`.
      - Assistant messages:
        - Display the answer text.
        - Optionally show sources in an expandable "Sources" section with titles, URLs, scores, and truncated snippets.
      - If `API_KEY` is not configured in secrets or environment:
        - The app warns and disables sending messages to the protected backend.
    - UI document upload:
      - A top-level “📄 Upload Document” button opens a `@st.dialog` modal.
      - Inside the dialog:
        - `st.file_uploader` for `.pdf`, `.md`, `.txt`, `.docx`, `.pptx`, `.xlsx`, `.html`, `.htm`.
        - Inputs for title (defaulting to filename), namespace, source label, tags, and notes.
        - A checkbox to allow uploading even when extracted text is very short.
        - On submit:
          - The frontend converts the file to text/markdown (using Docling when installed, or raw text for `.md`/`.txt`).
          - Calls backend `POST /documents/upload-text` with `X-API-Key`.
          - On success, records the upload in `st.session_state.recent_uploads` and triggers a rerun to close the dialog.

- Root-level `requirements.txt`
  - Added to support Streamlit Community Cloud, where the root requirements file is used:
    - `streamlit`
    - `httpx`
  - Backend Docker image continues to use `backend/requirements.txt`, keeping the backend container small and independent.

---

## Operational Runbook

### Rotating keys and secrets

- **Backend (Hugging Face Spaces or other container hosts)**
  - Update environment variables / secrets:
    - `PINECONE_API_KEY`, `PINECONE_HOST`, `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE`, `PINECONE_TEXT_FIELD`
    - `GROQ_API_KEY`, `GROQ_BASE_URL`, `GROQ_MODEL`
    - `TAVILY_API_KEY`
    - `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT`
    - `API_KEY` for HTTP clients
  - Redeploy or restart the Space to apply changes.
  - Verify:
    - `GET /health` returns `status: ok`.
    - `/chat` and `/search` work as expected.
    - `/metrics` shows traffic and cache counters updating.

- **Frontend (Streamlit Community Cloud)**
  - Use Streamlit Secrets manager (no secrets in repo):
    - `BACKEND_BASE_URL` – full URL of the backend (e.g. HF Spaces URL).
    - `API_KEY` – must match backend `API_KEY` if API protection is enabled.
  - After rotating backend keys:
    - If `API_KEY` changed, update it in Streamlit secrets.
    - No code changes required.

### Disabling rate limiting and caching

- **Rate limiting**
  - Set `RATE_LIMIT_ENABLED=false` in the backend environment (or `.env` for local).
  - Restart the backend.
  - SlowAPI middleware will not be attached; `@limiter.limit(...)` decorators become effectively no-op for enforcement.
  - `/metrics` will still track request counts and errors.

- **Caching**
  - Set `CACHE_ENABLED=false` in the backend environment.
  - Restart the backend.
  - Search and chat endpoints will bypass in-memory TTL caches entirely.
  - `get_cache_stats()` will still report counters, which will stop increasing.

### Diagnosing common deployment issues

- **Symptom: 404 / connection errors on Hugging Face Spaces**
  - Check:
    - The Space is configured as **Docker** and points to the `backend/` subdirectory (or uses the provided `backend/Dockerfile`).
    - Logs show the startup message:
      - `"Starting on port=... hf_spaces_mode=..."`.
    - HF Spaces sets `PORT` automatically; the Docker `CMD` will honour it.
  - Verify:
    - Open `/docs` and `/health` in the browser using the Space URL.
    - If 404/500 persists:
      - Ensure `PINECONE_*` and `GROQ_API_KEY` are set.
      - Check logs for `PineconeIndexConfigError` or missing LLM configuration.

- **Symptom: 401 Unauthorized from frontend**
  - Ensure:
    - Backend `API_KEY` is set and matches the `API_KEY` in Streamlit secrets.
    - Requests include `X-API-Key` header (Streamlit app does this automatically when `API_KEY` is present).
  - Confirm `/health` is still reachable without a key (by design).

- **Symptom: 429 Too Many Requests**
  - Indicates SlowAPI rate limiting is active.
  - Options:
    - Reduce load (e.g. from `bench_local.py`).
    - Temporarily set `RATE_LIMIT_ENABLED=false` for heavy local testing.
  - Inspect `/metrics`:
    - Check request counts and error counts for affected paths.

- **Symptom: Stale results after ingestion**
  - By default, caches are short-lived (60 seconds) but may briefly serve stale results:
    - When ingesting new documents, `/search` or `/chat` responses may not immediately reflect new content.
  - Workarounds:
    - Wait a minute for TTL expiry.
    - For strict freshness, disable caching with `CACHE_ENABLED=false`.

- **Symptom: Streamlit frontend cannot reach backend**
  - Verify:
    - `BACKEND_BASE_URL` in Streamlit secrets is correct and publicly reachable.
    - CORS config on the backend:
      - For debugging, keep `ALLOWED_ORIGINS` unset (defaults to `"*"`).
      - For locked-down deployment, ensure the Streamlit app origin is included.
  - Use the Connectivity panel:
    - Click "Ping /health" and inspect the response or error message.

---

## Embedding configuration (resolved)

This section records the embedding model and vector dimension used by the Pinecone integrated
embedding index, the attribute paths used to read them at runtime, whether the dimension is
pinned in code, and a reconciliation of the chunk-size settings against the model's token limit.

### Model and dimension

- **Embedding model:** `llama-text-embed-v2`  
  Read at startup from `embed_config.model`, where `embed_config = getattr(index_model, "embed", None)`
  and `index_model = pc.describe_index(settings.PINECONE_INDEX_NAME)`.  
  SDK type: `pinecone.core.openapi.db_control.model.model_index_embed.ModelIndexEmbed`
  (Pinecone SDK v7.3.0, attribute map key `"model"` → `str`).

- **Vector dimension:** `1024` (Pinecone SDK attribute path: `embed_config.dimension`,
  also mirrored at `index_model.dimension`)  
  **Not pinned anywhere in application code.** A search for the keyword `dimension` across all
  backend Python source files returns zero hits in application code — only in audit documentation.
  The value is whatever the Pinecone index was configured with at creation time.
  `llama-text-embed-v2` supports configurable output dimensions from 384 to 2048; 1024 is the
  default when unspecified. This project relies on that default silently.

- **Startup log added** (`backend/app/services/pinecone_store.py`, `init_pinecone`):
  ```
  INFO  Pinecone embedding config model='llama-text-embed-v2' dimension=1024 top_k_default=5
  ```
  This makes the full stack self-documenting from the first startup log line without any
  additional module or dependency.

### Token-limit reconciliation

Assumption: ~4 characters per token (standard BPE tokenizer estimate for English prose).

| Setting | Characters | ≈ Tokens | Model limit (tokens) | Truncation risk? |
|---|---|---|---|---|
| `chunk_size=900` (RecursiveCharacterTextSplitter) | 900 | ~225 | 2048 | None — well under limit |
| `MAX_CHARS_PER_CHUNK=6000` (safety cap, `chunking.py:21`) | 6000 | ~1500 | 2048 | None — also under limit |

Both settings are safely within the 2048-token input limit of `llama-text-embed-v2`.
Because `chunk_size=900 < MAX_CHARS_PER_CHUNK=6000`, the safety cap at 6000 chars never
activates under normal splitter operation; its role is purely defensive.

**Silent truncation cannot occur with current settings** as long as the embedding model is
`llama-text-embed-v2` with a 2048-token limit.

### Chunk-size vs. recommended range (observation for a later tuning step)

`llama-text-embed-v2` achieves best retrieval quality with inputs in the 400–500 token range
(per Pinecone's published guidance). The current primary chunk size of ~225 tokens
(900 chars ÷ 4 chars/token) is roughly **half** the recommended minimum.

This is flagged here as an observation only. Changing `chunk_size` affects retrieval quality,
dedup behavior, and the number of chunks per document; it should be tuned with a retrieval
evaluation harness (recall@k against a golden set) rather than adjusted blindly.

---

## Evaluation

### Retrieval evaluation harness

All files are **additive** (no runtime code modified). The harness exercises only the retrieval
path — no LLM calls, no graph execution.

#### Directory layout

```
eval/
├── __init__.py
├── corpus.py          — pinned mini-corpus definition (Wikipedia titles + arXiv/OpenAlex queries)
├── golden.jsonl       — golden query set (human-labeled relevant_doc_ids)
├── metrics.py         — pure metric functions: recall@k, MRR, nDCG@k
├── run.py             — retrieval runner (read-only Pinecone queries, writes eval/reports/)
├── setup_corpus.py    — one-time ingestion script (Pinecone upserts; separate from run.py)
└── fixtures/
    └── retrieval_fixture.json  — committed synthetic data for CI tests (zero network calls)
tests/
└── test_eval_metrics.py        — unit tests + fixture-based tests (zero network calls)
Makefile                        — targets: eval-corpus, eval, test
```

#### Metric function signatures

```python
# eval/metrics.py — pure functions, no model/embedder/LLM imports, stdlib math only

def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """recall@k = |retrieved_top_k ∩ relevant| / |relevant|"""

def mrr(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
) -> float:
    """RR = 1 / rank_of_first_relevant  (0 if none found)"""

def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """nDCG@k = DCG@k / IDCG@k  (binary relevance, log2 discount)"""
```

#### Make targets

| Target | Effect |
|---|---|
| `make eval-corpus` | Ingests the fixed mini-corpus into the `eval` Pinecone namespace (upsert — run once). Prints doc_ids for golden-set labeling. |
| `make eval` | Read-only Pinecone QUERY calls per golden entry. Writes JSON + markdown report to `eval/reports/`. |
| `make test` | Runs `tests/test_eval_metrics.py` — zero network calls, CI-safe. |

#### Anti-circular-validation rule

`relevant_doc_ids` in `golden.jsonl` **must** be determined by reading document content, not by
running the retriever and copying its output. Encoding the embedder's own intuition into the
relevant set makes recall@k tautological — the retriever will appear to have perfect recall
because the labels were derived from its own output.

The four example entries in `golden.jsonl` use `PLACEHOLDER_*` strings. Replace them with real
SHA256 doc_ids (printed by `make eval-corpus`) after reading the corresponding documents.

#### Document ID formula (reproducibility anchor)

Doc IDs are deterministic SHA256 hashes computed by `backend/app/services/normalize.py`:

```python
sha256(f"{source}|{title}|{url or ''}".encode("utf-8")).hexdigest()
```

Wikipedia titles produce stable doc_ids because the URL is predictable
(`https://en.wikipedia.org/wiki/{title.replace(" ", "_")}`). arXiv/OpenAlex results vary
by date — use Wikipedia titles as the anchor for `relevant_doc_ids`.

#### Chunk-level vs. document-level IDs

Pinecone record IDs are `{doc_id}:{chunk_idx}` (chunk level). `eval/run.py` deduplicates at
the document level before computing metrics: `hit["fields"]["doc_id"]` (preferred) or
`_id.rsplit(":", 1)[0]` (fallback). Relevance labels in `golden.jsonl` are therefore
document-level SHA256 IDs, not chunk IDs.

---

## Retrieval gating

### Summary

This step closes the **input-side (front) half** of the audit's primary compounding failure
chain: *weak retrieval → unfiltered context → unverified generation → hallucinated citations*.
Before this change, all retrieved Pinecone chunks — including near-zero-score hits — were
forwarded unconditionally to the LLM.  After this change a per-chunk score floor filters them,
and if no usable context survives the pipeline short-circuits to a deterministic abstention
**without calling the LLM**.

The **output-side (back) half** — post-generation grounding / citation verification — is NOT
addressed here and remains open for a later step (T2.2).

---

### Per-chunk score floor

| Setting | File:line | Purpose | Default | Status |
|---|---|---|---|---|
| `RAG_MIN_SCORE` | `config.py:77` | Web-fallback routing threshold: if `top_score < RAG_MIN_SCORE`, invoke Tavily | `0.25` | Routing only |
| `RAG_MIN_CHUNK_SCORE` | `config.py:83` | Per-chunk cosine floor: Pinecone chunks below this value are excluded from context | `0.25` | **PLACEHOLDER — not tuned** |

These two settings are **distinct** and independently configurable despite sharing the same
default value.  `RAG_MIN_SCORE` controls *whether* to invoke Tavily web search; `RAG_MIN_CHUNK_SCORE`
controls *which individual Pinecone chunks* are usable context for generation.

**The `0.25` default for `RAG_MIN_CHUNK_SCORE` is a PLACEHOLDER** chosen to match the routing
threshold for initial consistency.  It is NOT evidence-backed.  Calibrate it against the T1.2
retrieval eval set (recall@k / nDCG@k in `eval/`) before treating it as a justified value.
Setting it too high silences the system unnecessarily; too low lets weak chunks poison context.

#### Where the filter runs

`filter_chunks_by_score()` ([rag_prompt.py](../backend/app/services/prompts/rag_prompt.py)) is a
pure function — no side effects, independently unit-testable:

```python
def filter_chunks_by_score(
    chunks: List[Dict[str, Any]],
    min_chunk_score: float,
) -> List[Dict[str, Any]]:
    return [c for c in chunks if float(c.get("score") or 0.0) >= min_chunk_score]
```

It is called at the **top of `generate_answer`** in `graph.py`, AFTER `decide_next` has already
read `top_score` from the full (unfiltered) hit list for routing purposes.  This preserves the
existing Tavily routing logic exactly.

#### Tavily web results bypass the filter

The filter applies **only to Pinecone vector chunks**.  Tavily web results (source `"web"`) are
never passed to `filter_chunks_by_score` — they carry no cosine score and are not comparable
to Pinecone's normalised cosine values.

---

### Empty-context guard

After filtering Pinecone chunks and checking web results, if `usable_sources` is empty
(filtered Pinecone chunks = 0 AND web results = 0), `generate_answer` returns a fixed
**deterministic abstention** WITHOUT calling the Groq LLM:

```python
ABSTENTION_ANSWER = (
    "I was unable to find sufficient information in the knowledge base to answer "
    "your question. No retrieved chunks met the minimum relevance score threshold. "
    "Try enabling the web search fallback, broadening your query, or ingesting "
    "additional documents."
)
```

`generate_ms` is set to `0.0` on this path (no LLM call = no generation time).
The caller can detect an abstention programmatically via `ChatResponse.insufficient_context`
(a new `bool` field, default `False` for backward compatibility).

**Calling the LLM with empty context was the failure mode being removed.**  The old advisory
"say you don't know" instruction in the system prompt was unenforced and relied on the model's
judgment; the guard replaces it with a hard, unconditional short-circuit.

---

### How new logic composes with existing Tavily routing

```
retrieve_context          ← unchanged; sets top_score from full hit list
       │
decide_next               ← unchanged; routes to web_search if top_score < RAG_MIN_SCORE
       │
  ┌────┴───────────────────┐
  │ (web_fallback_used)     │ (no fallback)
web_search              [skip]
  │                        │
  └────────────┬───────────┘
               │
      generate_answer  ← NEW: filter Pinecone chunks by RAG_MIN_CHUNK_SCORE
                         NEW: if usable_sources empty → ABSTENTION_ANSWER, no LLM
                         else → build_rag_messages → Groq LLM → answer
```

The Tavily routing branch fires first (as before).  The empty-context guard fires **after**
any web fallback, so if Tavily returns results they flow through to the LLM even when all
Pinecone chunks were filtered out.  The guard only triggers when BOTH Pinecone (filtered) and
web results are empty simultaneously.

---

### Files changed (additive runtime changes only)

| File | Change |
|---|---|
| `backend/app/core/config.py` | Added `RAG_MIN_CHUNK_SCORE` (new setting) |
| `backend/app/services/prompts/rag_prompt.py` | Added `filter_chunks_by_score()` pure function |
| `backend/app/services/chat/graph.py` | Added `ABSTENTION_ANSWER` constant; added `insufficient_context` to `ChatState`; rewired `generate_answer` |
| `backend/app/schemas/chat.py` | Added `insufficient_context: bool` to `ChatResponse` |
| `backend/app/routers/chat.py` | Propagated `insufficient_context` in `_build_chat_response` |
| `tests/test_retrieval_gating.py` | 17 new CI-safe unit tests (zero network) |

---

## Dependency management

### Problem

Both requirements files were fully unpinned.  Every `docker build` and `pip install` resolved
fresh from PyPI, meaning builds were non-reproducible and silently exposed to LangChain / LangGraph
/ Pinecone SDK breaking changes that ship on minor-version bumps.  This is the primary reason the
audit flagged the dependency situation as high-priority.

---

### Layout: `.in` (intent) vs `.txt` (lock)

| File | Role | Who edits it |
|---|---|---|
| `backend/requirements.in` | Loose top-level packages — human-readable intent | Developer |
| `backend/requirements.txt` | Fully-pinned lock including ALL transitive deps | Generated (do not edit) |
| `requirements.in` | Loose top-level packages for the Streamlit frontend | Developer |
| `requirements.txt` | Fully-pinned lock for the Streamlit frontend | Generated (do not edit) |

The `.in` files are the source of truth for developer intent.  The `.txt` files are machine-generated
and committed so CI, Docker, and teammates all get bit-for-bit identical installs.

---

### Why pin to tested versions, not floated-to-latest?

The LangChain / LangGraph ecosystem ships breaking API changes on minor version bumps
(e.g., `langchain-core` 0.x → 1.x removed and renamed public classes; Pinecone SDK 3.x → 7.x
changed the integrated-embedding API).  Unpinned installs silently pull these changes into
production.  Deterministic Docker builds also require a stable dependency set — two `docker build`
calls hours apart should produce identical images.

All direct-dependency versions in the pinned `.txt` files equal the versions **currently installed
and verified working** in the active development environment (confirmed via `pip show` against all
22 direct deps; all matched).

---

### Compile tool

**`uv pip compile`** (uv 0.11.21) — available without extra install, faster than pip-tools.

```bash
# Backend (targets Python 3.11 to match Dockerfile: FROM python:3.11-slim)
uv pip compile --python-version 3.11 backend/requirements.in \
    -o backend/requirements.txt

# Frontend / root
uv pip compile requirements.in -o requirements.txt
```

The compilation was constrained to the installed direct-dep versions by passing a temporary
constraint file (`--constraint constraints-current.txt`) during the initial generation.  This file
is **not committed** — it is a one-time generation aid.  Future regenerations (e.g., after adding
a new package to `.in`) should use the constraint argument again to preserve currently-working
versions, or omit it to allow uv to pick the newest compatible set.

#### Transitive dep note (conda env caveat)

The development environment is a **shared conda environment**, not a project-isolated pip venv.
Conda and pip use different dependency solvers: conda installs packages without strictly enforcing
pip's `Requires-Dist` metadata.  This means the conda env's installed transitive deps (e.g.,
`langsmith==0.4.38`) may co-exist with direct deps that PyPI says require a newer version
(`langchain-core==1.1.0` → `langsmith>=0.9.0`).  The pinned `.txt` files reflect what pip/uv
would install in a clean pip environment — they are internally consistent and pip-installable,
which is the correct target for Docker.  **All 22 direct-dep versions match the installed
versions exactly** (verified); transitive dep versions resolve at the smallest compatible value
per PyPI metadata and may differ slightly from what conda shows in `pip freeze`.

---

### Hash pinning — deferred next step

Hash pinning (`--generate-hashes`) would verify the integrity of every downloaded wheel, guarding
against supply-chain attacks (compromised PyPI mirrors, typosquatting).  It was **not included**
in this step because it significantly lengthens the `.txt` files and requires re-hashing on every
machine.  Adding it is the recognized next hardening step:

```bash
uv pip compile --generate-hashes --python-version 3.11 backend/requirements.in \
    -o backend/requirements.txt
```

---

### Dockerfile install line (unchanged)

The Dockerfile already installs `backend/requirements.txt`.  No Dockerfile change was required:

```dockerfile
# line 9-10 of Dockerfile — unchanged
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
```

The root `requirements.txt` (frontend) is installed separately in the Streamlit environment and
is not referenced by the Dockerfile.

---

### How to add or upgrade a dependency

```bash
# 1. Edit the relevant .in file (add/change the package name)
#    e.g.: echo "httpx>=0.29" >> backend/requirements.in

# 2. Recompile
uv pip compile --python-version 3.11 backend/requirements.in \
    -o backend/requirements.txt          # backend

uv pip compile requirements.in -o requirements.txt  # frontend

# 3. Commit BOTH the updated .in and .txt files

# 4. If using Dependabot: it will open PRs updating .in entries;
#    merge the PR, then recompile .txt and push.
```

---

### Dependabot

`.github/dependabot.yml` is configured to open weekly PRs for both `backend/requirements.in` and
`requirements.in`.  After merging a Dependabot PR, recompile the corresponding `.txt` file and
push the update.

---

## Testing & CI

### What the unit tests cover

| File | Test file | What is tested |
|---|---|---|
| `eval/metrics.py` | `tests/test_eval_metrics.py` | recall@k, MRR, nDCG@k pure functions; 4 fixture cases with hand-computed expected values |
| `backend/app/services/prompts/rag_prompt.py` | `tests/test_retrieval_gating.py` | `filter_chunks_by_score` pure function (9 cases) |
| `backend/app/services/chat/graph.py` | `tests/test_retrieval_gating.py` | empty-context guard (Groq NOT called), normal path with context (8 cases) |
| `backend/app/services/normalize.py` | `tests/test_normalize.py` | `normalize_text` whitespace collapsing (10 cases); `is_valid_document` length gate (7 cases); `make_doc_id` SHA-256 determinism + collision-distinctness (10 cases) |
| `backend/app/services/dedupe.py` | `tests/test_dedupe.py` | `dedupe_records` duplicate removal, order preservation, no-id records, immutability (10 cases) |
| `backend/app/services/chunking.py` | `tests/test_chunking.py` | `chunk_document` chunk_size / overlap, multi-chunk split, Document output (7 cases); `documents_to_records` record schema, _id format, chunk sequence, text_field, skip-on-missing-metadata, safety truncation (15 cases) |
| `backend/app/services/prompts/rag_prompt.py` | `tests/test_rag_prompt.py` | `build_context_string` numbering, labels, url/chunk_text inclusion/omission, separator (12 cases); `build_rag_messages` system message, HumanMessage / AIMessage ordering, citation instruction, history handling (13 cases) |

**Total: 134 tests** as of T1.5.  All tests make zero network calls; no Pinecone, Groq, Tavily, or
LangSmith credentials are required.

---

### CI pipeline — `.github/workflows/ci.yml`

Triggers on every push and pull request.  Runs on `ubuntu-latest` with Python 3.11.

**Key design decisions:**

1. **Installs from `backend/requirements.txt` (the pinned lock), not the `.in` file.**  This is
   deliberate — it is the clean-environment reproducibility check for the T1.4 dependency pins.
   Every CI run validates that the pinned set installs cleanly into a fresh pip environment and
   that the test suite passes under those exact versions.  The dev environment was a shared conda
   env during T1.4, so this CI job is the authoritative clean-install validation.

2. **Runs `pytest tests/ -v` only.**  `make eval` and `make eval-corpus` are NOT invoked.  Both
   targets issue live Pinecone queries (reads / upserts); they require secrets (`PINECONE_API_KEY`
   etc.) and incur index cost.  CI is offline-only by design.

3. **No secrets required.**  The CI job contains no `env:` secrets block and will pass on forks
   and first-party PRs alike.

---

### Defect discovery policy

Tests assert **intended behavior** as documented in each function's docstring / comments.
If a new test reveals an actual defect in existing application code:

- The test is **not silently patched** to match the broken behavior.
- The test is marked `@pytest.mark.xfail(reason="<defect description>")` or
  `@pytest.mark.skip(reason="...")` with a clear reason string.
- The defect is **reported in the session chat** for a dedicated fix step.
- The fix step modifies only the relevant application code; the test is then un-marked.

**T1.5 result: no defects found.**  All 84 new tests passed on first run against the existing
application code.

---

### Runtime vs dev dependency separation

Test tooling (`pytest` and its transitives) is **not** installed in the production image.  The
Dockerfile installs only `backend/requirements.txt`.  Test deps live in a separate lock:

| File | Role | Installed in |
|---|---|---|
| `backend/requirements.txt` | Pinned runtime lock | Production Docker image + CI |
| `requirements-dev.txt` | Pinned test-tooling lock | CI only (and local dev) |
| `requirements-dev.in` | Loose dev intent (just `pytest`) | Source of truth for dev lock |

**Detected test-only deps beyond stdlib and the runtime lock:** `pytest==8.3.2` (and its
transitives: colorama 0.4.6, iniconfig 2.3.0, packaging 24.2, pluggy 1.6.0).  The suite uses
`unittest.mock` (stdlib) for all mocking — `pytest-mock` and `pytest-asyncio` are not needed.

`requirements-dev.txt` was compiled with `backend/requirements.txt` as a `--constraint` so dev
transitives cannot drift from the runtime pins.

---

### Test import path — committed config

Tests import `app.*` (e.g., `from app.services.normalize import ...`) and `eval.*` (e.g.,
`from eval.metrics import ...`).  Both require `backend/` and the repo root on `sys.path`.

**Mechanism:** `pyproject.toml` `[tool.pytest.ini_options]` with `pythonpath = [".", "backend"]`.
pytest adds both directories to `sys.path` before any test module is collected.  This is
committed to the repo and requires no local `PYTHONPATH` env-var or conda-env manipulation.
Individual test files also carry their own `sys.path.insert` guards; these are now redundant but
harmless.

---

### Clean-venv validation of the T1.4 lock (deferred, now complete)

A temporary clean venv was created using Python 3.11.15 (via uv's managed interpreter, matching
the Dockerfile's `FROM python:3.11-slim`).  Both locks were installed from scratch:

```
uv venv --python cpython-3.11.15 .venv-ci-check
uv pip install --python .venv-ci-check -r backend/requirements.txt
uv pip install --python .venv-ci-check -r requirements-dev.txt
.venv-ci-check/Scripts/python.exe -m pytest tests/ -v
```

**Result: 134 passed in 22.70s** on Python 3.11.15.  No import errors, no transitive conflicts,
no version mismatches.  The backend lock installs cleanly into a vanilla Python 3.11 interpreter.
This closes the deferred validation from T1.4 (where the conda env could not provide this
guarantee).

---

### CI pipeline — updated

`.github/workflows/ci.yml` now:
- **Triggers only on `main` branch** (push and PR targeting main) — prevents double-runs on
  Dependabot branches and CI noise on every feature branch.
- **Caches pip** with `cache-dependency-path` covering both `backend/requirements.txt` and
  `requirements-dev.txt` — cache invalidates when either lock changes.
- **Installs both locks separately**: runtime first (`backend/requirements.txt`), then dev
  (`requirements-dev.txt`).  This mirrors the production/test separation exactly.
- **No secrets** in the workflow file; will pass on forks without any configuration.
- Does **not** invoke `make eval` or `make eval-corpus`.

---

### Dependabot posture

`.github/dependabot.yml` is configured for three ecosystems: `github-actions`, `pip /backend`,
and `pip /` (frontend).  For both pip ecosystems:

- **Grouped PRs**: all minor and patch bumps are batched into a single PR per ecosystem
  (`backend-minor-patch` / `frontend-minor-patch`) instead of one PR per package.
- **Majors gated**: `version-update:semver-major` is ignored — major bumps require manual
  review, lock regeneration, and validation.
- **Limit 5** open PRs per ecosystem.

**Important: accepting a Dependabot bump requires recompiling the lock.**  Dependabot edits
the `.txt` file directly (it does not run `uv pip compile`), which breaks the uv-generated
comment header and may silently skip transitive adjustments.  After merging a Dependabot PR,
always regenerate the affected lock:

```bash
# Backend bump
uv pip compile --python-version 3.11 backend/requirements.in -o backend/requirements.txt

# Frontend bump
uv pip compile requirements.in -o requirements.txt

# Dev tooling bump
uv pip compile --python-version 3.11 \
    --constraint backend/requirements.txt \
    requirements-dev.in -o requirements-dev.txt
```

---

## Faithfulness & grounding (T2.2)

### Overview

This step closes the **output-side (back) half** of the audit's primary compounding failure
chain: *weak retrieval → unfiltered context → unverified generation → hallucinated citations*.
Before this change, the LLM could generate citations to non-existent chunks or claims that were
not supported by the retrieved context, with no signal surfaced to the caller.  After this change,
every response carries grounding metadata: a deterministic citation check always runs, and an
optional LLM-judge faithfulness check is available behind a flag.

The **input-side (front) half** of the chain — per-chunk score filtering and the empty-context
abstention guard — was completed in the Retrieval gating section above (T1.3).

---

### Two-layer design

| Layer | When it runs | Model calls | What it checks |
|---|---|---|---|
| `verify_citations` | Always (even when `RAG_FAITHFULNESS_ENABLED=False`) | Zero | `[n]` markers in the answer that reference out-of-range chunk indices |
| `judge_faithfulness` | When `RAG_FAITHFULNESS_ENABLED=True` AND not an abstention | 1 (reuses Groq client) | Whether answer claims are supported by the retrieved context |

Both functions live in `backend/app/services/faithfulness.py` and are importable by
`eval/faithfulness.py` for offline scoring — no duplicate implementation.

---

### Circular-validation avoidance

The faithfulness check **must not** re-embed the answer with the same retrieval embedder and
threshold cosine similarity against the retrieved chunks.  That approach reuses the retriever's
own semantic space to validate the retriever's own output — a circular-validation anti-pattern
that encodes the embedder's biases into the faithfulness signal.

**Solution:** the judge uses the existing Groq LLM (an independent language model) as a semantic
reasoner over the already-retrieved text.  The deterministic layer is purely lexical (regex, no
model).  Neither layer touches the retrieval embedding model.

---

### Flag-default-OFF cost rationale

`RAG_FAITHFULNESS_ENABLED=False` (the default).  Every `/chat` request would otherwise pay for
a second LLM call (judge) on top of the generation call.  On Groq's free tier the cost is
latency rather than money, but it is still undesirable for interactive requests.

When the flag is OFF:
- `verify_citations` runs (free, deterministic).
- The Groq LLM is **not** called a second time.
- `grounded` and `faithfulness_score` in `ChatResponse` are `null`.

When the flag is ON:
- `verify_citations` runs.
- `judge_faithfulness` calls the Groq LLM with a strict faithfulness prompt.
- `grounded`, `faithfulness_score`, and `faithfulness_ms` are populated.

---

### Flag-and-report behavior

When a faithfulness problem is detected, the system **flags it in the response and reports it to
the caller** — it does NOT alter the answer text and does NOT hard-fail the request by default.

Rationale: the judge is an LLM and can itself be wrong.  A false "ungrounded" verdict would
silently corrupt a correct answer or block a valid response.  Callers who want to suppress
ungrounded answers can inspect `ChatResponse.grounded` and act accordingly.

`grounded` resolution from the judge verdict:
- If `faithfulness_score` is not None: `grounded = (faithfulness_score >= RAG_FAITHFULNESS_THRESHOLD)`
- Else if `grounded` (raw bool from model) is not None: use it as fallback
- Else (parse failure): `grounded = None` ("unknown") — never raise, never block

---

### Composition with T1.3 (distinct states)

`insufficient_context` (T1.3) and `grounded` (T2.2) are **distinct states** with different
meanings and different execution paths:

| Field | When True/False | LLM called? |
|---|---|---|
| `insufficient_context=True` | No usable context survived retrieval + chunk filtering | No — deterministic abstention |
| `grounded=False` | LLM answered but answer is not well-supported by context | Yes — generation happened |
| `grounded=None` | Judge not called (flag OFF, abstention path, or parse failure) | N/A |

**The abstention path (`insufficient_context=True`) bypasses the judge entirely.**  There is no
model-generated answer to evaluate, and calling the LLM a second time on the abstention path
would be both wasteful and meaningless.  `format_response` checks `insufficient_context` first
and skips the judge block unconditionally when it is True.

---

### Grounding threshold — PLACEHOLDER

`RAG_FAITHFULNESS_THRESHOLD=0.5` is **not evidence-backed**.  It is a reasonable midpoint chosen
as a neutral starting value.  To calibrate it:

1. Collect answer/context pairs from real queries.
2. Human-label each pair as grounded (1) or ungrounded (0).
3. Use the faithfulness judge to score each pair.
4. Choose the threshold that maximises F1 (or precision, depending on operational preference) on
   the labeled set.

The offline evaluation wrapper (`eval/faithfulness.py`) exposes the same `judge_faithfulness`
function so this calibration can be done without modifying the runtime.

---

### Files added / changed

| File | Change |
|---|---|
| `backend/app/services/faithfulness.py` | **New** — `verify_citations()` (deterministic) + `judge_faithfulness()` (LLM-as-judge) + `FaithfulnessVerdict` dataclass |
| `backend/app/services/prompts/faithfulness_prompt.py` | **New** — judge prompt builder (separate from `rag_prompt.py`) |
| `backend/app/core/config.py` | Added `RAG_FAITHFULNESS_ENABLED` (default False) + `RAG_FAITHFULNESS_THRESHOLD` (PLACEHOLDER 0.5) |
| `backend/app/schemas/chat.py` | Added `faithfulness_ms` to `ChatTimings`; added `grounded`, `faithfulness_score`, `unverified_citations` to `ChatResponse` |
| `backend/app/services/chat/graph.py` | Added grounding fields to `ChatState`; filled `format_response` node |
| `backend/app/routers/chat.py` | Propagated grounding fields in `_build_chat_response` |
| `eval/faithfulness.py` | **New** — thin offline wrapper importing runtime functions for eval use |
| `tests/test_faithfulness.py` | **New** — 29 CI-safe unit tests (zero network, judge mocked) |

---

## CORS & origins

### Problem fixed

`security.py` previously set `allow_credentials=True` alongside a default `allow_origins=["*"]`.
The **WHATWG Fetch Standard** (not RFC 7234, which covers HTTP caching) forbids this combination:
a wildcard origin paired with `credentials: include` is rejected by every major browser.  The
bug was latent because the API is consumed by a Streamlit frontend that does not send cookies,
but it would have silently broken any browser client that enabled credentials mode.

### Bearer-token auth — why credentials mode is never needed

The API authenticates with a bearer API key in the `X-API-Key` header.  `allow_credentials=True`
in CORS enables the `Access-Control-Allow-Credentials` response header, which signals to browsers
that the request may carry cookies, HTTP authentication, or TLS client certificates.  None of
those are used here.  `allow_credentials` is now permanently `False`.

### Origins resolution

`_get_allowed_origins()` in `security.py`:

| `ALLOWED_ORIGINS` env var | Result |
|---|---|
| Unset or empty string | `["*"]` — permissive dev default (safe: credentials=False) |
| `"https://a.com,https://b.com"` | `["https://a.com", "https://b.com"]` |
| Entries that are all whitespace | Falls back to `["*"]` |

Set `ALLOWED_ORIGINS` to a comma-separated list before deploying:

```
ALLOWED_ORIGINS=https://my-app.hf.space,https://my-frontend.com
```

### Prod-wildcard warning

On startup, `configure_security` checks whether origins resolved to `["*"]` **and** whether the
environment is production-like.  The prod-detection reuses `_is_production_like()` from
`app.core.auth` (same `ENV=production` / `SPACE_ID` / `HF_HOME` heuristic used by the API-key
startup check), so the two startup guards are consistent.

**Behaviour:** a `WARNING` is logged — not a `RuntimeError`.  The rationale for not hard-failing:
the API is still protected by bearer-token auth even with wildcard CORS, so the operational risk
of a misconfigured `ALLOWED_ORIGINS` is lower than a missing `API_KEY`; a hard-fail would block
deployments unnecessarily.  Operators are expected to respond to the warning before going live.

### Files changed

| File | Change |
|---|---|
| `backend/app/core/security.py` | Removed `allow_credentials=True`; added prod-wildcard warning; imported `_is_production_like` from auth |
| `tests/test_cors.py` | 12 new CI-safe unit tests (zero network) |

---

## Two-stage retrieval — Pinecone hosted reranker (T1.6)

### Overview

Two-stage retrieval adds an optional second step to the dense-retrieval path: after a first-stage
Pinecone cosine search, the surviving chunks are re-ordered by a **Pinecone hosted reranker**
(`pc.inference.rerank`) before the top-k subset is forwarded to the LLM.  The reranker captures
semantic nuance that dense retrieval sometimes misses (e.g. keyword-dense but query-irrelevant
documents).

The feature is **disabled by default** (`RAG_RERANK_ENABLED=false`).  When disabled, every code
path is byte-for-byte identical to the baseline; the rerank helper is never called.  Enable it only
after an A/B run (`make eval-ab`) shows a statistically meaningful nDCG@k improvement that justifies
the added latency.

### Critical scale constraint

Reranker scores and cosine similarity scores are **completely different numerical distributions**.
The existing thresholds (`RAG_MIN_SCORE`, `RAG_MIN_CHUNK_SCORE`) are calibrated for cosine
similarity and must **never** be applied to rerank scores.  The cosine floor runs **before**
reranking (on cosine scores); no threshold is applied to the output of the reranker.

### Configuration knobs

| Setting | File | Default | Purpose |
|---|---|---|---|
| `RAG_RERANK_ENABLED` | `config.py` | `False` | Master switch — OFF means baseline behavior |
| `RAG_RERANK_MODEL` | `config.py` | `bge-reranker-v2-m3` | Pinecone inference reranker model |
| `RAG_RERANK_CANDIDATES` | `config.py` | `20` | First-stage pool size; clamped to ≥ `RAG_DEFAULT_TOP_K` |
| `RAG_MIN_CHUNK_SCORE` | `config.py` | `0.25` | Cosine floor applied BEFORE reranking (unchanged) |

`RAG_RERANK_MODEL` must be available on the operator's Pinecone plan.  `pinecone-rerank-v0` is a
lower-throughput development-tier alternative.  See https://docs.pinecone.io/models/overview.

### Execution path (when enabled)

```
retrieve_context
  ├── Dense search: top_k=max(RAG_RERANK_CANDIDATES, top_k) candidates
  ├── top_score = max cosine score (UNCHANGED — routing reads this)
  └── state["retrieved"] = all candidates (with cosine scores)

decide_next
  └── Reads top_score (cosine) for web-fallback threshold (UNCHANGED)

generate_answer
  ├── Stage 1 — cosine floor: filter_chunks_by_score(retrieved, RAG_MIN_CHUNK_SCORE)
  │             Cosine-calibrated; floor runs on cosine scores only
  ├── Stage 2 — hosted rerank: pc.inference.rerank(model, query, survivors, top_n=top_k)
  │             rerank_ms recorded; rerank scores NOT threshold-compared
  └── state["retrieved"] = top_k reranked chunks → LLM context
```

When `RAG_RERANK_ENABLED=False`, Stage 2 is skipped entirely and `rerank_ms=0.0`.

### Graceful degradation

Any Pinecone Inference API error in `rerank_chunks()` is caught, logged, and the function
returns the pre-rerank cosine order truncated to `top_n`.  The user receives a valid response
from the baseline retrieval path without any exception propagating.

### Files added / changed

| File | Change |
|---|---|
| `backend/app/core/config.py` | Added `RAG_RERANK_ENABLED`, `RAG_RERANK_MODEL`, `RAG_RERANK_CANDIDATES` |
| `backend/app/services/pinecone_store.py` | Added `get_pinecone_client()` getter; fixed `_pc` type annotation |
| `backend/app/services/rerank.py` | **New** — `rerank_chunks()` helper using `pc.inference.rerank` |
| `backend/app/services/chat/graph.py` | `retrieve_context`: wider pool when ON; `generate_answer`: cosine floor → rerank → top_k; added `rerank_ms` timing |
| `backend/app/schemas/chat.py` | Added `rerank_ms: float` to `ChatTimings` (0.0 when OFF) |
| `backend/app/routers/chat.py` | Propagated `rerank_ms` in `_build_chat_response` |
| `eval/run_ab.py` | **New** — on-demand A/B harness (live Pinecone calls, NOT for CI) |
| `Makefile` | Added `eval-ab` target; added `CANDIDATES` and `RERANK_MODEL` variables |
| `tests/test_reranking.py` | **New** — 14 CI-safe unit tests; Pinecone Inference API mocked; zero network |

### A/B evaluation — `make eval-ab`

```bash
# Default: top_k=10, candidates=20, model=bge-reranker-v2-m3
make eval-ab

# Custom pool and model
make eval-ab CANDIDATES=30 RERANK_MODEL=pinecone-rerank-v0 TOP_K=5
```

Runs each golden-set query through both arms (baseline dense + rerank), computes
recall@k, MRR, nDCG@k, and mean latency per arm, and writes a delta table to
`eval/reports/ab_{timestamp}.{json,md}`.

**This target issues live Pinecone calls and incurs inference cost.  It is NOT invoked by
`make eval` and must NOT be added to CI.**  Use it on-demand to empirically justify flipping
`RAG_RERANK_ENABLED=True`.

A second target `make eval-ab-topk` runs the same arm comparison with `--multi-k`, adding
precision@1, recall@3/5, and nDCG@3/5 to address the ceiling-bound objection at recall@10.

---

### Top-heavy validation (T2.1 follow-up) — `eval/reports/ab_20260625T083719Z.*`

**Motivation:** the initial A/B (T2.1) showed recall@10 falling from 0.9684 to 0.9167 with
reranking.  Recall@10 on a 34-chunk, 23-document corpus is ceiling-bound (~30 % of the corpus
retrieved per query), meaning the retriever saturates the metric and any reranker can only
lose recall without the upside of a larger pool.  Reranking is designed to improve top-of-list
precision, not recall — so the T2.1 metric had all downside and no upside for the reranker.
This run (`make eval-ab-topk`, same 29 queries, same golden set, top_k=10, candidates=20)
computes the metrics reranking is supposed to optimise (precision@1, nDCG@3/5) to check for
a hidden precision/recall tradeoff.

**Results (n=29, top_k=10, candidates=20, model=bge-reranker-v2-m3):**

| Metric | Baseline (A) | Rerank (B) | D (B-A) | Verdict |
|--------|-------------|-----------|---------|---------|
| Mean Precision@1 | 0.9655 | 0.9655 | 0.0000 | Tie |
| Mean Recall@3 | 0.8534 | 0.7730 | -0.0805 | Baseline wins |
| Mean nDCG@3 | 0.8750 | 0.8176 | -0.0574 | Baseline wins |
| Mean Recall@5 | 0.9109 | 0.8822 | -0.0287 | Baseline wins |
| Mean nDCG@5 | 0.8997 | 0.8687 | -0.0310 | Baseline wins |
| Mean Recall@10 | 0.9684 | 0.9167 | -0.0517 | Baseline wins |
| Mean MRR | 0.9828 | 0.9828 | 0.0000 | Tie |
| Mean nDCG@10 | 0.9255 | 0.8853 | -0.0402 | Baseline wins |
| Mean latency (ms) | 359.6 | 795.2 | +435.5 | Baseline wins |

**Finding:** there is NO precision/recall tradeoff.  The reranker fails to improve even the
top-heavy metrics it is designed to target: nDCG@3 falls by 0.057, nDCG@5 by 0.031, and
Precision@1 ties at 0.9655 (no improvement).  The reranker is flat-or-negative at every
measured k value.

**Root cause:** `candidates=20` with `RAG_MIN_CHUNK_SCORE=0.25` causes the cosine floor to
silently drop borderline-relevant chunks from the reranker's input pool before it can
surface them.  The reranker never sees those documents, so it cannot fix the ordering deficit
it creates by pushing other documents down.  Two levers exist to revisit this if warranted:
raise `RAG_RERANK_CANDIDATES` to 40-50, or lower `RAG_MIN_CHUNK_SCORE` specifically for the
rerank path to give the model more to work with.

**Conclusion: `RAG_RERANK_ENABLED=False` is the correct default.  The decision is now
empirically validated at both recall-oriented and precision-oriented metrics.**  Report in
`eval/reports/ab_20260625T083719Z.json` and `.md`.

---

## Retrieval calibration

### Why "maximize retrieval quality" is the wrong framing here

Baseline dense retrieval on this corpus (34 chunks / 23 docs, `eval` Pinecone namespace)
is already saturated at the production setting: recall@10=0.97, MRR=0.98, nDCG@10=0.93.
Running a parameter sweep to "maximize" these metrics is uninformative when they barely
move — any apparent optimum is noise, not signal.

The framing is therefore recast from "maximize quality" to two distinct objectives:

1. **Context-cost floor (top_k):** find the SMALLEST `top_k` at which retrieval quality
   is near the ceiling.  Fewer chunks in the LLM prompt = lower token cost + less
   mid-context dilution (the "lost-in-the-middle" effect).  The deliverable is a
   quality-vs-k curve with a readable knee.

2. **Cosine floor (RAG_MIN_CHUNK_SCORE) — honest documentation only:** the floor's job
   is precision/noise-reduction, which recall@k does NOT measure.  A saturated recall
   eval cannot validate a precision floor sharply, so this calibration step documents
   purpose, limitations, and a data-derived safety bound — it does NOT claim an optimum.

---

### Top-k sweep — `eval/reports/sweep_20260625T100233Z.*`

**Setup:** `make eval-sweep` runs `eval/run_sweep.py`.  Retrieves `chunk_fetch_k=20`
chunks ONCE per query (dense-only, rerank OFF, matching shipped production defaults),
deduplicates to a doc-ranked list (~13 unique docs), then slices to compute metrics at
doc-level k = 1, 2, 3, 5, 8, 10 — no extra Pinecone calls per k value.

Note on chunk vs doc level: `RAG_DEFAULT_TOP_K` is a chunk-level budget in the production
pipeline.  For this corpus (avg 1.48 chunks/doc) the chunk:doc ratio is close to 1:1 at the
k values swept, so the curve directly informs the `RAG_DEFAULT_TOP_K` setting.

**Quality-vs-k table (n=30 queries):**

| k  | Recall@k | nDCG@k  | P@k    | D-Recall | D-nDCG  |
|----|----------|---------|--------|----------|---------|
|  1 | 0.5806   | 0.9667  | 0.9667 | -0.4000  | +0.0341 |
|  2 | 0.7389   | 0.8635  | 0.6833 | -0.2417  | -0.0690 |
|  3 | 0.8583   | 0.8791  | 0.5556 | -0.1222  | -0.0534 |
|  5 | 0.9139   | 0.9030  | 0.3600 | -0.0667  | -0.0295 |
|  8 | 0.9694   | 0.9280  | 0.2417 | -0.0111  | -0.0045 | **<- knee** |
| 10 | 0.9806   | 0.9326  | 0.1967 | +0.0000  | +0.0000 | <- ceiling |

Margin used: +/-0.02 on both recall AND nDCG (must satisfy both simultaneously).

**Knee: k=8.**  Recall@8=0.9694 (delta 0.011 below ceiling) and nDCG@8=0.9280
(delta 0.005 below ceiling) — both inside the 0.02 margin.  k=5 does NOT qualify:
recall@5=0.9139 (delta 0.067) and nDCG@5=0.9030 (delta 0.030), nDCG delta exceeds the
margin.

**Cost rationale:** moving from k=8 to k=10 retrieves two more chunks per query.  On this
corpus that costs ~15% more LLM-context tokens for less than 1% quality gain (delta-recall
0.011, delta-nDCG 0.005).  Moving from k=5 to k=8 recovers 6.7 recall points and 3
nDCG points for 60% more chunks.

**Precision@1 curiosity:** P@1=0.9667 at k=1 (96.7% of queries have the top-ranked doc
relevant) but recall@1=0.5806 — because many golden entries have 2-3 relevant docs, so
one retrieved doc is often only partial coverage.  This confirms the dense retriever is
excellent at ranking precision but needs k >= 8 to achieve full recall for multi-relevant
queries.

---

### RAG_DEFAULT_TOP_K recommendation

**Current default: `RAG_DEFAULT_TOP_K = 5`.**

The curve shows k=5 is meaningfully below the quality knee (recall delta 0.067, nDCG
delta 0.030 from ceiling).  The minimum k that holds quality within 0.02 of the ceiling
is **k=8**.

**k=8 is the recall-margin knee** — the smallest k where both recall AND nDCG are within
0.02 of the k=10 ceiling (recall@8=0.9694, D=0.011; nDCG@8=0.9280, D=0.005).

**However, precision@8=0.2417** — approximately 76% of the retrieved context at k=8 is
irrelevant.  Moving from k=5 to k=8 costs 60% more LLM-context chunks per query while
picking up 6.7 recall points and 3 nDCG points.

**recall@k cannot adjudicate this tradeoff.**  Recall measures whether relevant doc_ids
appear in the top-k list; it does NOT measure whether the LLM produces better answers
from a larger-but-noisier context.  The precision/recall tradeoff at this corpus size
is not resolvable from retrieval metrics alone.

**Decision: `RAG_DEFAULT_TOP_K` kept at 5** (T2.2 lock).  This is a precision-first
choice: k=5 delivers higher-signal context (P@5=0.36 vs P@8=0.24) at the accepted cost
of lower recall coverage (recall@5=0.91 vs recall@8=0.97).  The tiebreaker for
revisiting this decision is an **answer-quality eval** — a head-to-head comparison of
answers generated at k=5 vs k=8 against human relevance judgments.  Until that eval
exists, context signal quality is preferred over recall coverage.

---

### Cosine floor (RAG_MIN_CHUNK_SCORE) — honest documentation

**Current default: `RAG_MIN_CHUNK_SCORE = 0.25`.**

**Purpose:** drop low-cosine-similarity chunks before they enter the LLM context window.
The floor reduces noise (irrelevant chunks that passed the top-k retrieval budget) and
keeps the prompt tightly relevant.  It runs BEFORE any reranking step.

**Why this eval cannot tune the floor sharply:** recall@k measures whether relevant
doc_ids appear in the top-k ranked list — it does NOT measure whether those docs'
individual chunks are present in the filtered context.  On a corpus saturated at
recall@10=0.97, every floor value from 0.0 to 0.96 produces the same recall score
(because the relevant docs still rank within top-k even when their chunks are
hypothetically dropped).  A precision-oriented eval (where false positives reduce the
score) or a graded-relevance eval (where chunk quality matters, not just doc presence)
would be needed to tune the floor sharply.

**Data-derived safety bound:** the sweep at `chunk_fetch_k=20` recorded the minimum
cosine score of any retrieved chunk whose `doc_id` was in the relevant set, across all 30
golden queries.  That minimum was **0.2368**.

> A `RAG_MIN_CHUNK_SCORE` at or below **0.2368** is safe for this eval set: no
> golden-relevant chunk scored below that threshold in the top-20 results.

**Current floor vs safety bound:** `RAG_MIN_CHUNK_SCORE=0.25 > 0.2368`.  The one chunk
that scored 0.2368 would be dropped by the current floor.  Whether this matters in
practice depends on whether that chunk carries information not duplicated by the same
doc's other chunks (if any) or by other relevant docs retrieved for the same query.
Since the production recall is 0.97, the dropped chunk is evidently not the only path
to a correct answer — but this cannot be confirmed without a faithfulness or answer-
quality eval.

**What would tune it sharply:** (a) a per-answer quality eval comparing answers produced
with and without the low-scoring chunk; (b) graded relevance labels at the chunk level
(not just doc level) so a precision@k metric is meaningful; (c) a larger or noisier
corpus with more retrievable low-quality noise where the floor discriminates signal
from noise.

**Change (T2.2 lock): `RAG_MIN_CHUNK_SCORE` lowered from 0.25 to 0.20.**  The prior floor
of 0.25 was above the 0.2368 safety bound — meaning the one chunk that scored 0.2368
could be silently dropped.  Setting the floor at 0.20 places it explicitly below the
0.2368 bound so that no known-relevant chunk from the eval set is excluded.

0.20 is **not a tuned optimum** — it is the safety-bound floor.  It encodes only the
constraint "never drop a known-relevant chunk from the eval set" and nothing more.  Sharp
floor tuning (trading precision against recall at the chunk level) requires a
precision-oriented eval, graded relevance labels at chunk level, or a larger corpus where
low-quality noise chunks are routinely retrievable.

---

## Corrective retrieval (CRAG) (T2.4)

### Overview

CRAG (Corrective RAG) adds a self-correcting loop between initial Pinecone retrieval and
the `decide_next` routing node.  After the first retrieval attempt, the retrieved chunks
are **graded** using the cosine scores already in state (no re-embedding).  If retrieval
is graded weak, the query is **rewritten** by the Groq LLM and Pinecone is **re-queried**
with the new query.  This repeats up to `RAG_CRAG_MAX_ITERS` times.

The feature is **disabled by default** (`RAG_CRAG_ENABLED=False`).  When disabled, the
`corrective_retrieve` node is a byte-for-byte pass-through — the existing pipeline is
unchanged.

---

### Mandatory max-iterations guard

`RAG_CRAG_MAX_ITERS=2` (default) is a **hard, non-negotiable loop bound**.  The
corrective loop ALWAYS terminates after this many iterations regardless of whether the
grade improves.  This is not a soft "try up to N times" — the bound is enforced
unconditionally by a `for iteration in range(max_iters)` loop without any early-exit
other than a `break` on a "good" grade.

**Why this guard is mandatory:** an unbounded corrective loop on a query that always
produces weak retrieval (e.g. a topic not in the knowledge base) would spin indefinitely,
exhausting API rate limits and blocking the response.  The guard closes this audit
finding.  `test_max_iters_guard_always_terminates` in `tests/test_crag.py` uses a
threshold of 0.99 (unreachably high) to ensure the loop terminates after exactly
`max_iters` iterations even when retrieval is perpetually graded weak.

---

### Circular-validation avoidance

The CRAG grader uses the **top cosine score already returned by `retrieve_context`**
(stored in `state["top_score"]`).  It does NOT re-embed the query or the retrieved
chunks with the retrieval model.  Re-embedding would be circular validation: using the
retriever's own semantic space to assess the retriever's own output encodes the
embedder's biases into the grading signal.

This mirrors the same avoidance principle as T2.3 (faithfulness judge uses the Groq LLM,
not the retrieval embedder).

---

### Composition with existing components

CRAG is composed with the existing pipeline without duplicating any component:

| Existing component | How CRAG composes |
|---|---|
| `retrieve_context` | CRAG grades its output; re-retrieval reuses `pinecone_search` |
| Groq LLM (`get_llm`) | Query rewriter reuses the same client; no new LLM |
| Tavily web fallback | Unchanged — `decide_next` still routes to web_search if top_score < RAG_MIN_SCORE after CRAG |
| Cosine floor (`RAG_MIN_CHUNK_SCORE`) | Unchanged — runs in `generate_answer` after CRAG |
| Empty-context abstention | Unchanged — if CRAG exhausts its iterations with empty results, `generate_answer` still detects the empty context and returns the deterministic abstention |
| Faithfulness check | Unchanged — `format_response` runs after the full pipeline |

When `RAG_CRAG_ENABLED=False`, the `corrective_retrieve` node returns state unchanged
at the top of the function — no branches, no side effects, no calls to any dependency.
The OFF path is verified by `test_flag_off_is_exact_passthrough`.

---

### Flag-OFF posture

`RAG_CRAG_ENABLED=False` is the permanent default for new deployments.  The corrective
loop adds latency (one extra LLM call + one extra Pinecone query per iteration) and the
grading threshold (`RAG_CRAG_GOOD_SCORE=0.45`) is a PLACEHOLDER not backed by an
answer-quality eval.  Enable it only after:

1. Observing real out-of-corpus queries where initial retrieval fails.
2. Confirming that the rewritten query + re-retrieval actually improves answer quality
   (not just recall@k — recall@k is saturated at 0.97 on the golden set, meaning CRAG
   will rarely fire on in-corpus queries and metric lift is not the right signal here).

---

### Validation methodology

On this saturated corpus (MRR=0.98, recall@10=0.97), CRAG will rarely fire — initial
retrieval almost always returns a good result.  **Metric lift on the golden set is not
the right signal**: if CRAG almost never triggers, its contribution to eval metrics is
near zero by construction, not because it is broken.

The correct validation methodology is:
1. **Triggering mechanism**: design out-of-corpus or low-signal test queries that
   reliably produce top_score < `RAG_CRAG_GOOD_SCORE`.  Confirm the loop fires.
2. **Answer quality**: compare answers before/after CRAG correction on those queries.
   The tiebreaker is human judgment on answer quality, not recall@k.
3. **Guard test**: `test_max_iters_guard_always_terminates` is the CI-safe proxy for
   the termination invariant.

---

### Configuration

| Setting | Default | Purpose |
|---|---|---|
| `RAG_CRAG_ENABLED` | `False` | Master switch; OFF = byte-for-byte baseline behavior |
| `RAG_CRAG_MAX_ITERS` | `2` | Hard loop bound; mandatory, non-negotiable |
| `RAG_CRAG_GOOD_SCORE` | `0.45` | Cosine threshold for "good" grade; PLACEHOLDER, not tuned |

---

### Files added / changed

| File | Change |
|---|---|
| `backend/app/core/config.py` | Added `RAG_CRAG_ENABLED`, `RAG_CRAG_MAX_ITERS`, `RAG_CRAG_GOOD_SCORE` |
| `backend/app/services/crag.py` | **New** — `grade_retrieval()` + `rewrite_query()` (no circular validation, reuses existing LLM) |
| `backend/app/services/prompts/query_rewrite_prompt.py` | **New** — query rewrite prompt builder (separate from rag_prompt.py and faithfulness_prompt.py) |
| `backend/app/services/chat/graph.py` | Added `crag_iterations`/`corrective_action` to `ChatState`; added `corrective_retrieve` node; wired between `retrieve_context` and `decide_next`; added CRAG import |
| `backend/app/schemas/chat.py` | Added `crag_iterations: int` + `corrective_action: Optional[str]` to `ChatResponse` |
| `backend/app/routers/chat.py` | Propagated CRAG fields in `_build_chat_response` |
| `tests/test_crag.py` | **New** — 17 CI-safe unit tests (zero network, LLM/search mocked); includes mandatory max-iters guard test |

---

## Metrics & observability (T2.6)

### Overview

T2.6 replaces the unreliable 20-sample deque p95 in the legacy JSON `/metrics` endpoint with a
Prometheus Histogram as the **authoritative** latency percentile source, and adds a public
Prometheus text-exposition endpoint at `/metrics/prometheus`.

---

### Path layout and access control

| Path | Format | Auth | Owner |
|---|---|---|---|
| `/metrics` | JSON (existing, unchanged) | API-key gated (`X-API-Key`) | `backend/app/routers/metrics.py` |
| `/metrics/prometheus` | Prometheus text exposition (new) | Public — no API key | `backend/app/core/prometheus_metrics.py` |

The `/metrics/prometheus` endpoint is intentionally public to match the standard Prometheus
pull model: a Prometheus server polls the endpoint from inside a trusted network, not through
an authenticated HTTP client.  The endpoint does NOT call Pinecone, Groq, or Tavily — it reads
only the in-process `prometheus_client` REGISTRY, so no secrets are required.

**Public-endpoint decision (deliberate).** Prometheus endpoints are conventionally
network-restricted rather than token-gated; on Hugging Face Spaces there is no private network
in front of the container, so the endpoint is public by necessity.  The data it exposes is
deliberately low-sensitivity: route-level request counts and latencies only — no query text,
no document content, no user data.  In a production deployment with network control (e.g. a
private VPC or Kubernetes cluster), this endpoint would sit behind a network policy or a
scrape-authentication proxy (e.g. Prometheus `bearer_token` + nginx auth) rather than being
left publicly reachable.

---

### Why prometheus-client directly (not prometheus-fastapi-instrumentator)

`prometheus-fastapi-instrumentator==8.0.2` requires `starlette>=1.0.0`, which is incompatible
with `starlette==0.50.0` pinned by `fastapi==0.128.0` in the existing lock.  Upgrading starlette
to 1.3.1 broke FastAPI internals at test time.

`prometheus-client` has zero framework dependency and provides all needed primitives.
HTTP instrumentation (count + duration) is implemented as a thin middleware directly in
`backend/app/core/prometheus_metrics.py`, keeping all Prometheus code in one module.

---

### p95 fix — why the deque is unreliable

The legacy `_timing_samples: deque(maxlen=20)` computes p95 over the last 20 samples only.
With 20 observations, the 95th-percentile bucket contains exactly one sample (floor(0.95 × 20) = 19th),
and the result has a very wide confidence interval — approximately ±30 % CI on p95.

The Prometheus Histogram accumulates ALL observations (unbounded sample count, no ring-buffer
truncation) and computes percentiles from the cumulative bucket distribution.  Percentiles from
the Histogram converge as more requests are processed and reflect the true long-run distribution.

The deque p95 remains in `/metrics` for backward compatibility but is **documented as legacy /
indicative only**.  The Prometheus Histogram is the authoritative source.

---

### Metrics registered

**HTTP request metrics** (one series per method × path × status_class):
- `http_requests_total` (Counter) — total request count
- `http_request_duration_seconds` (Histogram) — request latency per method + path

**RAG pipeline metrics** (one series per phase):
- `rag_phase_duration_seconds` (Histogram) — per-phase latency in seconds
  - Phases: `retrieve`, `web`, `generate`, `rerank`, `faithfulness`, `total`
  - Only phases with `> 0 ms` measured are observed; zero-ms phases (e.g. `rerank_ms=0` when
    `RAG_RERANK_ENABLED=False`) are intentionally skipped

---

### Bucket rationale

RAG pipeline Histogram buckets (seconds):

| Bucket | Meaning |
|---|---|
| 0.05 | Fast Pinecone retrieval lower bound |
| 0.10 | Typical Pinecone retrieval |
| 0.25 | Retrieval ceiling / fast generation floor |
| 0.50 | Observed Pinecone retrieval peak (~350ms + margin) |
| 0.75 | Retrieval + CRAG grade, no rewrite |
| 1.00 | Typical Groq generation (small model, short answer) |
| 1.50 | Groq generation for longer answers |
| 2.00 | Groq generation + Tavily single call |
| 3.00 | One CRAG correction iteration (rewrite + re-retrieve) |
| 5.00 | Two CRAG iterations or slow Tavily |
| 10.0 | Timeout safety ceiling |

---

### Label cardinality discipline

Labels MUST be bounded — unbounded labels (raw query text, user input, per-request IDs) cause
"metric cardinality explosion" where Prometheus TSDB memory grows without bound.

| Metric | Labels | Max series |
|---|---|---|
| `http_requests_total` | method, path, status_class | ~6 × 15 × 5 = 450 |
| `http_request_duration_seconds` | method, path | ~6 × 15 = 90 |
| `rag_phase_duration_seconds` | phase (6 fixed values) | 6 |

The `path` label uses `request.url.path` (no query string), which is safe for this API because
all routes are fixed strings with no variable path segments (no `/resource/{id}` patterns).

NEVER add labels for: query text, user input, namespace, document ID, or any other per-request
unbounded-cardinality field.

---

### Prometheus scrape configuration example

```yaml
# prometheus.yml snippet
scrape_configs:
  - job_name: rag_agent_workbench
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics/prometheus
    scrape_interval: 30s
```

No `Authorization` header required.  The endpoint does not need the `X-API-Key` header.

---

### Dependency change (T2.6)

| Package | Before | After | Reason |
|---|---|---|---|
| `prometheus-client` | Not pinned (0.20.0 installed transitively) | `0.25.0` (now a direct dep, pinned in lock) | Direct dependency added to `backend/requirements.in` |
| `prometheus-fastapi-instrumentator` | — | Not used | starlette compatibility — see above |
| `fastapi` | `0.128.0` (lock but not pinned in `.in`) | `0.128.0` (now pinned in `.in`) | Prevents uv from resolving newer FastAPI + incompatible starlette |
| `starlette` | `0.50.0` | `0.50.0` (unchanged) | Held by fastapi==0.128.0 pin |

Recompile command:
```bash
uv pip compile --python-version 3.11 --no-strip-extras \
    --output-file backend/requirements.txt backend/requirements.in
```

---

### Files added / changed

| File | Change |
|---|---|
| `backend/app/core/prometheus_metrics.py` | **New** — all Prometheus code: Histogram definitions, `record_chat_timings_prometheus()`, HTTP middleware, `setup_prometheus()` |
| `backend/app/main.py` | Added `setup_prometheus(app)` call after `setup_metrics(app)` |
| `backend/app/routers/chat.py` | Added `record_chat_timings_prometheus(timings)` call after `record_chat_timings(...)` in both `/chat` and `/chat/stream` (NOT in cached-response path) |
| `backend/requirements.in` | Added `prometheus-client`; pinned `fastapi==0.128.0` (stability guard) |
| `backend/requirements.txt` | Recompiled — `prometheus-client==0.25.0` now pinned; `starlette==0.50.0` preserved |
| `tests/test_prometheus_metrics.py` | **New** — 10 CI-safe tests: endpoint 200 + content-type, histogram presence, HTTP metric after request, observation count + sum, zero-phase skip, ms→s conversion, legacy snapshot structure |
| `docs/CONTEXT.md` | This section |
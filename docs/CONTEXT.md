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
    3. `decide_next` – route to web search or generation.
    4. `web_search` – Tavily search (optional).
    5. `generate_answer` – Groq LLM with RAG prompts.
    6. `format_response` – reserved for post-processing.
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
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

## Next Steps (Work Package C – Proposed)

- Add lightweight evaluation endpoints (e.g. scoring relevance or answer quality).
- Introduce conversation/session management for multi-turn chat.
- Add optional caching of retrieval results and/or LLM responses.
- Provide a minimal web UI for interactive exploration of the RAG pipeline.
- Extend observability (metrics, tracing details) around LangGraph nodes and external calls.
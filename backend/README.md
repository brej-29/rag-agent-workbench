# RAG Agent Workbench – Backend

Lightweight FastAPI backend for ingesting documents into Pinecone (with integrated embeddings), searching over them, and serving a production-style RAG chat endpoint.

## Prerequisites

- Python 3.11+
- A Pinecone account and an index configured with **integrated embeddings**
- A Groq account and API key for chat
- (Optional) Tavily API key for web search fallback
- (Optional) LangSmith account + API key for tracing
- Environment variables set (see `.env.example`)

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # then edit with your Pinecone, Groq, and optional Tavily/LangSmith credentials
```

Required `.env` values:

- `PINECONE_API_KEY` – your Pinecone API key
- `PINECONE_INDEX_NAME` – the index name (used for configuration checks)
- `PINECONE_HOST` – the index host URL (use host targeting for production)
- `PINECONE_NAMESPACE` – default namespace (e.g. `dev`)
- `PINECONE_TEXT_FIELD` – text field name used by the integrated embedding index (e.g. `chunk_text` or `content`)
- `LOG_LEVEL` – e.g. `INFO`, `DEBUG`

Required for `/chat`:

- `GROQ_API_KEY` – your Groq API key
- `GROQ_BASE_URL` – Groq OpenAI-compatible endpoint (default `https://api.groq.com/openai/v1`)
- `GROQ_MODEL` – Groq chat model name (default `llama-3.1-8b-instant`)

Optional for web search fallback:

- `TAVILY_API_KEY` – Tavily API key (enables web search in `/chat` when retrieval is weak)

Optional for LangSmith tracing:

- `LANGCHAIN_TRACING_V2` – set to `true` to enable tracing
- `LANGCHAIN_API_KEY` – your LangSmith API key
- `LANGCHAIN_PROJECT` – project name for traces (e.g. `rag-agent-workbench`)

Optional for basic API protection:

- `API_KEY` – when set, `/ingest/*`, `/documents/*`, `/search`, and `/chat*` require `X-API-Key` header.

Optional for CORS:

- `ALLOWED_ORIGINS` – comma-separated list of allowed origins.
  - If unset, defaults to `"*"` (useful for local dev and quick demos).

Optional for rate limiting and caching:

- `RATE_LIMIT_ENABLED` – defaults to `true`. Set to `false` to disable SlowAPI limits.
- `CACHE_ENABLED` – defaults to `true`. Set to `false` to disable in-memory TTL caches.

Your Pinecone index **must** be configured for integrated embeddings (e.g. via `create_index_for_model` or `configure_index(embed=...)`), with a field mapping that includes the configured `PINECONE_TEXT_FIELD`.

## Run locally

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

## Sample endpoints

### Health

```bash
curl http://localhost:8000/health
```

### Ingest from arXiv

```bash
curl -X POST "http://localhost:8000/ingest/arxiv" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "retrieval augmented generation",
    "max_docs": 5,
    "namespace": "dev",
    "category": "papers"
  }'
```

### Ingest from OpenAlex

```bash
curl -X POST "http://localhost:8000/ingest/openalex" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "retrieval augmented generation",
    "max_docs": 5,
    "namespace": "dev",
    "mailto": "you@example.com"
  }'
```

### Ingest from Wikipedia

```bash
curl -X POST "http://localhost:8000/ingest/wiki" \
  -H "Content-Type: application/json" \
  -d '{
    "titles": ["Retrieval-augmented generation", "Vector database"],
    "namespace": "dev"
  }'
```

### Manual text upload

```bash
curl -X POST "http://localhost:8000/documents/upload-text" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "My manual note",
    "source": "manual",
    "text": "This is some example text describing RAG pipelines...",
    "namespace": "dev",
    "metadata": {
      "url": "https://example.com/my-note"
    }
  }'
```

### Search

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \  # only if API_KEY is enabled
  -d '{
    "query": "what is RAG",
    "top_k": 5,
    "namespace": "dev",
    "filters": {"source": "arxiv"}
  }'
```

### Document stats

```bash
curl "http://localhost:8000/documents/stats?namespace=dev"
```

### Chat (non-streaming)

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \  # only if API_KEY is enabled
  -d '{
    "query": "What is retrieval-augmented generation?",
    "namespace": "dev",
    "top_k": 5,
    "use_web_fallback": true,
    "min_score": 0.25,
    "max_web_results": 5,
    "chat_history": [
      {"role": "user", "content": "You are helping me understand RAG."}
    ]
  }'
```

Example JSON response:

```json
{
  "answer": "...",
  "sources": [
    {
      "source": "wiki",
      "title": "Retrieval-augmented generation",
      "url": "https://en.wikipedia.org/wiki/...",
      "score": 0.91,
      "chunk_text": "..."
    }
  ],
  "timings": {
    "retrieve_ms": 35.2,
    "web_ms": 0.0,
    "generate_ms": 420.7,
    "total_ms": 470.1
  },
  "trace": {
    "langsmith_project": "rag-agent-workbench",
    "trace_enabled": true
  }
}
```

### Chat (SSE streaming)

```bash
curl -N -X POST "http://localhost:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \  # only if API_KEY is enabled
  -d '{
    "query": "Summarise retrieval-augmented generation.",
    "namespace": "dev",
    "top_k": 5,
    "use_web_fallback": true
  }'
```

- The response will be `text/event-stream`.
- Individual SSE events stream tokens (space-delimited).
- The final event (`event: end`) includes the full JSON payload as in `/chat`.

### Metrics

```bash
curl "http://localhost:8000/metrics"
```

Returns JSON with:

- `requests_by_path` and `errors_by_path`
- `timings` (average and p50/p95 for `retrieve_ms`, `web_ms`, `generate_ms`, `total_ms`)
- `cache` stats
- Last 20 timing samples for chat.

## Seeding data

A helper script is provided to seed the index with multiple arXiv and OpenAlex queries:

```bash
python ../scripts/seed_ingest.py --base-url http://localhost:8000 --namespace dev --mailto you@example.com
```

## Docling integration (external script)

Docling is used via a separate script so the backend container stays small. To convert a local PDF and upload it as text:

```bash
cd scripts
pip install docling
python docling_convert_and_upload.py \
  --pdf-path /path/to/file.pdf \
  --backend-url http://localhost:8000 \
  --namespace dev \
  --title "My PDF via Docling" \
  --source docling
```

## Deploy Backend on Hugging Face Spaces (Docker)

1. **Create a new Space**
   - Go to Hugging Face → *New Space*.
   - Choose:
     - **SDK**: Docker
     - **Space name**: e.g. `your-name/rag-agent-workbench-backend`.
   - Point the Space to this repository and configure it to use the `backend/` subdirectory (or copy `backend/Dockerfile` to the root if you prefer).

2. **Environment variables / secrets**

   In the Space settings, configure the following (as “Secrets” where appropriate):

   Required:

   - `PINECONE_API_KEY`
   - `PINECONE_HOST`
   - `PINECONE_INDEX_NAME`
   - `PINECONE_NAMESPACE`
   - `PINECONE_TEXT_FIELD=content` (or your actual text field)
   - `GROQ_API_KEY`
   - `GROQ_BASE_URL` (optional, defaults to `https://api.groq.com/openai/v1`)
   - `GROQ_MODEL` (optional, defaults to `llama-3.1-8b-instant`)

   Optional:

   - `TAVILY_API_KEY` (web search fallback for `/chat`)
   - `LANGCHAIN_TRACING_V2`
   - `LANGCHAIN_API_KEY`
   - `LANGCHAIN_PROJECT`
   - `API_KEY` (to protect `/ingest/*`, `/documents/*`, `/search`, `/chat*`)
   - `ALLOWED_ORIGINS` (e.g. your Streamlit frontend origin)
   - `RATE_LIMIT_ENABLED` and `CACHE_ENABLED` (rarely need to change from defaults)

3. **Ports and startup**

   - The Docker image exposes port **7860** by default.
   - Hugging Face Spaces sets the `PORT` environment variable; the `CMD` honours it:
     - `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}`
   - On successful startup, logs include:
     - `Starting on port=<port> hf_spaces_mode=<bool>`

4. **Verify**

   - Open your Space URL:
     - `https://<your-space>.hf.space/docs` – interactive API docs.
     - `https://<your-space>.hf.space/health` – health check.
   - If `API_KEY` is set, test protected endpoints using `X-API-Key`.

## Deploy Frontend on Streamlit Community Cloud

1. **Prepare the repo**

   - The minimal Streamlit frontend lives under `frontend/app.py`.
   - Root `requirements.txt` includes:
     - `streamlit`
     - `httpx`

2. **Create Streamlit app**

   - Go to Streamlit Community Cloud and create a new app.
   - Point it at this repository.
   - Set the main file to `frontend/app.py`.

3. **Configure Streamlit secrets**

   - In the Streamlit app settings, configure *Secrets* (YAML):

     ```yaml
     BACKEND_BASE_URL: "https://<your-backend-space>.hf.space"
     API_KEY: "your-backend-api-key"  # only if backend API_KEY is set
     ```

   - **Do not** commit secrets into the repo.

4. **Verify connectivity**

   - Open the Streamlit app.
   - In the sidebar “Connectivity” panel:
     - Confirm the backend URL is correct.
     - Click “Ping /health” to verify backend connectivity.
   - Use the chat panel to send a question:
     - The app will call `/chat` on the backend and display answer, timings, and sources.

## Local Test Checklist – Work Package C

1. **Configure environment**

   - Set `PINECONE_*` variables for an integrated embeddings index.
   - Set `GROQ_API_KEY` (and optionally override `GROQ_BASE_URL`, `GROQ_MODEL`).
   - Optionally set `TAVILY_API_KEY` for web fallback.
   - Optionally enable LangSmith:
     - `LANGCHAIN_TRACING_V2=true`
     - `LANGCHAIN_API_KEY=...`
     - `LANGCHAIN_PROJECT=rag-agent-workbench`
   - Optionally set:
     - `API_KEY` for basic protection.
     - `ALLOWED_ORIGINS` if you are calling from a browser origin.
     - `RATE_LIMIT_ENABLED` / `CACHE_ENABLED` for tuning.

2. **Start the backend**

   ```bash
   cd backend
   uvicorn app.main:app --reload --port 8000
   ```

3. **Ingest data**

   - Quick Wikipedia smoke test (also see `scripts/smoke_chat.py`):

     ```bash
     python ../scripts/smoke_chat.py --backend-url http://localhost:8000 --namespace dev
     ```

4. **Test `/search`**

   ```bash
   curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: $API_KEY" \  # only if API_KEY is enabled
     -d '{"query": "what is RAG", "namespace": "dev", "top_k": 5}'
   ```

5. **Test `/chat`**

   - Use the curl example above or run:

     ```bash
     curl -X POST "http://localhost:8000/chat" \
       -H "Content-Type: application/json" \
       -H "X-API-Key: $API_KEY" \  # only if API_KEY is enabled
       -d '{"query": "What is retrieval-augmented generation?", "namespace": "dev"}'
     ```

6. **Test `/chat` with web fallback**

   - Requires `TAVILY_API_KEY`:

     ```bash
     python ../scripts/smoke_chat_web.py --backend-url http://localhost:8000 --namespace dev
     ```

7. **Inspect `/metrics`**

   ```bash
   curl "http://localhost:8000/metrics"
   ```

   - Confirm:
     - Request counts are increasing.
     - Timing stats (`average_ms`, `p50_ms`, `p95_ms`) are populated after several `/chat` calls.
     - Cache hit/miss counters change when repeating identical `/search` or `/chat` requests.

8. **Run the benchmark script**

   - From the repo root:

     ```bash
     python scripts/bench_local.py \
       --backend-url http://localhost:8000 \
       --namespace dev \
       --concurrency 10 \
       --requests 50 \
       --api-key "$API_KEY"
     ```

   - Review reported:
     - Average latency.
     - p50 / p95 latency.
     - Error rate.

9. **Optional: Test Streamlit frontend locally**

   - Install root requirements:

     ```bash
     pip install -r requirements.txt
     ```

   - Run:

     ```bash
     streamlit run frontend/app.py
     ```

   - Configure `BACKEND_BASE_URL` and `API_KEY` via environment or `.streamlit/secrets.toml`, and verify chat works end-to-end.
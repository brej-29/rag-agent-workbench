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

## Local test checklist – Work Package B

1. **Configure environment**
   - Set `PINECONE_*` variables for an integrated embeddings index.
   - Set `GROQ_API_KEY` (and optionally override `GROQ_BASE_URL`, `GROQ_MODEL`).
   - Optionally set `TAVILY_API_KEY` for web fallback.
   - Optionally enable LangSmith:
     - `LANGCHAIN_TRACING_V2=true`
     - `LANGCHAIN_API_KEY=...`
     - `LANGCHAIN_PROJECT=rag-agent-workbench`

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

4. **Test `/chat`**

   - Use the curl example above or run:

     ```bash
     curl -X POST "http://localhost:8000/chat" \
       -H "Content-Type: application/json" \
       -d '{"query": "What is retrieval-augmented generation?", "namespace": "dev"}'
     ```

5. **Test `/chat` with web fallback**

   - Requires `TAVILY_API_KEY`:

     ```bash
     python ../scripts/smoke_chat_web.py --backend-url http://localhost:8000 --namespace dev
     ```

6. **Verify tracing (optional)**

   - With LangSmith env vars set, hit `/chat` and confirm the run appears in your LangSmith project.
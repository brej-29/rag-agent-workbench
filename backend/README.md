# RAG Agent Workbench – Backend

Lightweight FastAPI backend for ingesting documents into Pinecone (with integrated embeddings) and searching over them.

## Prerequisites

- Python 3.11+
- A Pinecone account and an index configured with **integrated embeddings**
- Environment variables set (see `.env.example`)

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # then edit with your Pinecone credentials
```

Required `.env` values:

- `PINECONE_API_KEY` – your Pinecone API key
- `PINECONE_INDEX_NAME` – the index name (used for configuration checks)
- `PINECONE_HOST` – the index host URL (use host targeting for production)
- `PINECONE_NAMESPACE` – default namespace (e.g. `dev`)
- `LOG_LEVEL` – e.g. `INFO`, `DEBUG`

Your Pinecone index **must** be configured for integrated embeddings (e.g. via `create_index_for_model` or `configure_index(embed=...)`), with a field mapping that includes `chunk_text`.

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
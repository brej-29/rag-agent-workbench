# rag-agent-workbench

This repository contains a lightweight RAG backend built with FastAPI, Pinecone (integrated embeddings), and LangGraph/LangChain for agentic RAG flows, plus a Streamlit chatbot frontend.

- Backend API: see [`backend/README.md`](backend/README.md) for setup, environment variables, API key protection, endpoint examples, and deployment instructions (including `/chat`, `/chat/stream`, `/metrics`, and Hugging Face Spaces notes).
- Architecture and design context: see [`docs/CONTEXT.md`](docs/CONTEXT.md) for work package history, security hardening notes, and operational runbook.
- Frontend: Streamlit chat app under [`frontend/app.py`](frontend/app.py) intended for Streamlit Community Cloud or local runs.
- Utility scripts: see the `scripts/` directory for ingestion, smoke-test helpers, Docling-based local ingestion, and benchmarking (including `scripts/bench_local.py`). rag-agent-workbench

This repository contains a lightweight RAG backend built with FastAPI, Pinecone (integrated embeddings), and LangGraph/LangChain for agentic RAG flows, plus a minimal Streamlit frontend.

- Backend API: see [`backend/README.md`](backend/README.md) for setup, environment variables, endpoint examples, and deployment instructions (including `/chat`, `/chat/stream`, `/metrics`, and Hugging Face Spaces notes).
- Architecture and design context: see [`docs/CONTEXT.md`](docs/CONTEXT.md) for work package history and operational runbook.
- Frontend: minimal Streamlit app under [`frontend/app.py`](frontend/app.py) intended for Streamlit Community Cloud or local runs.
- Utility scripts: see the `scripts/` directory for ingestion, smoke tests, and benchmarking helpers (including `scripts/bench_local.py`).
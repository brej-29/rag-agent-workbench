# rag-agent-workbench

This repository contains a lightweight RAG backend built with FastAPI, Pinecone (integrated embeddings), and LangGraph/LangChain for agentic RAG flows.

- Backend API: see [`backend/README.md`](backend/README.md) for setup, environment variables, and endpoint examples (including `/chat` and `/chat/stream`).
- Architecture and design context: see [`docs/CONTEXT.md`](docs/CONTEXT.md).
- Utility scripts: see the `scripts/` directory for ingestion and smoke-test helpers.
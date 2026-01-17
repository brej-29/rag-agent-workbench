<div align="center">
  <h1>🧠 rag-agent-workbench</h1>
  <p><i>Lightweight, production-style RAG backend with a Streamlit chatbot frontend — built for Pinecone, Groq, and LangGraph/LangChain.</i></p>
</div>

<br>

<div align="center">
  <img alt="Language" src="https://img.shields.io/badge/Language-Python-blue">
  <img alt="Backend" src="https://img.shields.io/badge/Backend-FastAPI-009688">
  <img alt="Vector Store" src="https://img.shields.io/badge/Vector%20Store-Pinecone-3776AB">
  <img alt="Frameworks" src="https://img.shields.io/badge/Frameworks-LangChain%20%7C%20LangGraph-ff9800">
  <img alt="Frontend" src="https://img.shields.io/badge/Frontend-Streamlit-ff4b4b">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-black">
</div>

<div align="center">
  <br>
  <b>Built with the tools and technologies:</b>
  <br><br>
  <code>Python</code> |
  <code>FastAPI</code> |
  <code>Pinecone</code> |
  <code>LangChain</code> |
  <code>LangGraph</code> |
  <code>Groq</code> |
  <code>Streamlit</code> |
  <code>Docling</code> |
  <code>httpx</code>
</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Backend](#backend)
  - [Frontend](#frontend)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [License](#license)
- [Contact](#contact)

---

## Overview

This repository contains a lightweight RAG backend built with FastAPI, Pinecone (integrated embeddings), and LangGraph/LangChain for agentic RAG flows, plus a Streamlit chatbot frontend.

At a high level:

- The **backend** exposes ingestion, semantic search, and production-style RAG chat endpoints (with optional web-search fallback, rate limiting, caching, metrics, and API key protection).
- The **frontend** is a Streamlit chatbot UI that talks to the backend `/chat` endpoint, supports streaming responses, and offers a modal-based document upload workflow that ingests local files via `/documents/upload-text`.

---

## Features

- **Backend API**
  - FastAPI-based RAG backend with Pinecone integrated embeddings.
  - Agentic RAG chat powered by LangGraph and LangChain.
  - Groq LLM integration via OpenAI-compatible API.
  - Optional Tavily web-search fallback.
  - Ingestion endpoints for arXiv, OpenAlex, Wikipedia, and manual text uploads.
  - Caching, rate limiting, metrics endpoint, and API key protection for secured deployments.
  - Dockerized backend suitable for Hugging Face Spaces.

- **Frontend (Streamlit)**
  - Chatbot UI using `st.chat_message` and `st.chat_input`.
  - Streaming support via `/chat/stream` when available, with automatic fallback to `/chat`.
  - Sidebar controls for query behaviour (top_k, min_score, web fallback, show sources).
  - Modal **Upload Document** dialog to convert and upload local PDFs/MD/TXT/Office/HTML files to the backend.
  - Recent uploads panel with quick “Search this document” actions.

- **Developer Experience**
  - Simple configuration via `.env` and Streamlit secrets.
  - Utility scripts for seeding, smoke tests, benchmarking, and Docling-based local ingestion.
  - Clear work package history and operational runbook under `docs/`.

---

## Getting Started

### Backend

- Backend API: see [`backend/README.md`](backend/README.md) for setup, environment variables, API key protection, endpoint examples, and deployment instructions (including `/chat`, `/chat/stream`, `/metrics`, and Hugging Face Spaces notes).

Typical flow:

1. Create a Python 3.11+ virtual environment.
2. Install backend dependencies:

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. Copy `.env.example` → `.env` and configure:
   - Pinecone (integrated embeddings).
   - Groq LLM parameters.
   - Optional Tavily, LangSmith, rate limiting, caching, and API key (`API_KEY`) for protected deployments.
4. Run the backend locally:

   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

5. Browse:
   - `http://localhost:8000/health`
   - `http://localhost:8000/docs`

### Frontend

- Frontend: Streamlit chat app under [`frontend/app.py`](frontend/app.py) intended for Streamlit Community Cloud or local runs.

For local usage:

```bash
pip install -r requirements.txt  # root requirements (Streamlit + frontend deps)
streamlit run frontend/app.py
```

Configure:

- `BACKEND_BASE_URL` (e.g. `http://localhost:8000` or your HF Space URL).
- `API_KEY` (if the backend is protected) via:
  - `st.secrets` in `.streamlit/secrets.toml`, or
  - environment variables.

---

## Project Structure

- Backend API: see [`backend/README.md`](backend/README.md) for setup, environment variables, API key protection, endpoint examples, and deployment instructions (including `/chat`, `/chat/stream`, `/metrics`, and Hugging Face Spaces notes).
- Architecture and design context: see [`docs/CONTEXT.md`](docs/CONTEXT.md) for work package history, security hardening notes, and operational runbook.
- Frontend: Streamlit chat app under [`frontend/app.py`](frontend/app.py) intended for Streamlit Community Cloud or local runs.
- Utility scripts: see the `scripts/` directory for ingestion, smoke-test helpers, Docling-based local ingestion, and benchmarking (including `scripts/bench_local.py`).

A high-level layout:

```text
rag-agent-workbench/
├─ backend/            # FastAPI app, core logic, routers, services, config
├─ frontend/           # Streamlit chatbot UI
├─ docs/               # Context, worklog, and design documentation
├─ scripts/            # Ingestion, smoke tests, benchmark, and docling helpers
├─ requirements.txt    # Frontend / root-level dependencies
├─ backend/requirements.txt  # Backend dependencies
├─ LICENSE
└─ README.md
```

---

## Documentation

- **Backend API & operations**  
  See [`backend/README.md`](backend/README.md) for:
  - Environment variables and configuration.
  - Endpoint catalogue (ingest, search, chat, metrics).
  - Hugging Face Spaces deployment notes.
  - LangSmith, Tavily, Groq, and Pinecone configuration.

- **Architecture & work packages**  
  See [`docs/CONTEXT.md`](docs/CONTEXT.md) for:
  - Overall architecture and design decisions.
  - Work package history (A/B/C, security + UI + ingestion).
  - Operational runbook (key rotation, toggling rate limiting/caching, diagnosing issues).

- **Worklog**  
  See [`docs/WORKLOG.md`](docs/WORKLOG.md) for a chronological summary of changes and key files per work package.

---

## License

This project is licensed under the **MIT License**.  
See the [`LICENSE`](LICENSE) file for details.

---

## Contact

For questions, suggestions, or collaboration:

- Open an issue or discussion in this repository.
- Refer to the maintainers listed in project documentation or commit history. rag-agent-workbench

This repository contains a lightweight RAG backend built with FastAPI, Pinecone (integrated embeddings), and LangGraph/LangChain for agentic RAG flows, plus a minimal Streamlit frontend.

- Backend API: see [`backend/README.md`](backend/README.md) for setup, environment variables, endpoint examples, and deployment instructions (including `/chat`, `/chat/stream`, `/metrics`, and Hugging Face Spaces notes).
- Architecture and design context: see [`docs/CONTEXT.md`](docs/CONTEXT.md) for work package history and operational runbook.
- Frontend: minimal Streamlit app under [`frontend/app.py`](frontend/app.py) intended for Streamlit Community Cloud or local runs.
- Utility scripts: see the `scripts/` directory for ingestion, smoke tests, and benchmarking helpers (including `scripts/bench_local.py`).
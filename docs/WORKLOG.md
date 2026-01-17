# Worklog

## 2026-01-17 – Work Package C

- **Summary**
  - Productionised the backend for deployment on Hugging Face Spaces (Docker) and added a minimal Streamlit frontend suitable for Streamlit Community Cloud.
  - Introduced optional API key protection, rate limiting, and in-memory caching.
  - Added in-memory metrics with a `/metrics` endpoint and an asyncio-based benchmark script.

- **Key Files Changed**
  - Backend runtime and deployment:
    - `backend/Dockerfile`
    - `backend/app/core/runtime.py`
  - Security, CORS, rate limiting, and caching:
    - `backend/app/core/security.py`
    - `backend/app/core/rate_limit.py`
    - `backend/app/core/cache.py`
  - Metrics:
    - `backend/app/core/metrics.py`
    - `backend/app/routers/metrics.py`
  - Routers and configuration:
    - `backend/app/main.py`
    - `backend/app/routers/chat.py`
    - `backend/app/routers/search.py`
    - `backend/app/routers/ingest.py`
    - `backend/app/core/config.py`
  - Dependencies and environment:
    - `backend/requirements.txt`
    - `backend/.env.example`
  - Tooling and frontend:
    - `scripts/bench_local.py`
    - `frontend/app.py`
    - `requirements.txt` (root)
  - Documentation:
    - `backend/README.md`
    - `docs/CONTEXT.md`

- **Major Decisions**
  - Use port `7860` by default in the Docker image, while respecting the `PORT` environment variable for platforms like Hugging Face Spaces.
  - Keep API key protection opt-in via `API_KEY` with clear logging when disabled.
  - Enable rate limiting and caching by default, with simple boolean toggles (`RATE_LIMIT_ENABLED`, `CACHE_ENABLED`) for easy operational control.
  - Implement metrics as in-memory only (no external storage) and expose them via a JSON `/metrics` endpoint tailored for demos and lightweight monitoring.

## 2026-01-17 – Security + UI + Ingestion Hardening

- **Summary**
  - Hardened the backend for public deployment by enforcing API key protection for all non-health endpoints and for the OpenAPI/Swagger documentation.
  - Upgraded the Streamlit frontend to a conversational chat UI using Streamlit's chat primitives.
  - Improved local document ingestion workflows with Docling-aware scripts for single files and batch folder ingestion.

- **Key Files Changed**
  - Backend authentication and wiring:
    - `backend/app/core/auth.py`
    - `backend/app/core/security.py`
    - `backend/app/main.py`
  - Frontend chatbot UI:
    - `frontend/app.py`
  - Local ingestion scripts:
    - `scripts/docling_convert_and_upload.py`
    - `scripts/batch_ingest_local_folder.py`
  - Documentation:
    - `backend/README.md`
    - `docs/CONTEXT.md`
    - `docs/WORKLOG.md` (this file)

- **Major Decisions**
  - In production-like environments (`ENV=production` or on Hugging Face Spaces), require `API_KEY` and fail fast at startup when it is missing.
  - Use a single `require_api_key` dependency (based on `APIKeyHeader`) to protect all routers except `/health`, and to guard `/openapi.json`, `/docs`, and `/redoc`.
  - Treat Streamlit as a first-class chat client, using `st.chat_message`/`st.chat_input` with session-based history and optional streaming from `/chat/stream`.
  - Keep Docling as an optional local-only dependency and reuse its conversion logic via scripts that upload text to `/documents/upload-text` rather than extending the backend container.
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
  - Hardened the backend for public deployment by enforcing API key protection for all non-health endpoints and (initially) for the OpenAPI/Swagger documentation, then relaxed docs to be publicly viewable while keeping all functional endpoints protected.
  - Upgraded the Streamlit frontend to a conversational chat UI using Streamlit's chat primitives.
  - Improved local document ingestion workflows with Docling-aware scripts for single files and batch folder ingestion.
  - Added a UI-based document upload dialog in the Streamlit app that ingests files via `/documents/upload-text`.

- **Key Files Changed**
  - Backend authentication and wiring:
    - `backend/app/core/auth.py`
    - `backend/app/core/security.py`
    - `backend/app/main.py`
  - Frontend chatbot UI and upload:
    - `frontend/app.py`
    - `frontend/services/file_convert.py`
    - `frontend/services/backend_client.py`
  - Local ingestion scripts:
    - `scripts/docling_convert_and_upload.py`
    - `scripts/batch_ingest_local_folder.py`
  - Documentation:
    - `backend/README.md`
    - `docs/CONTEXT.md`
    - `docs/WORKLOG.md` (this file)

- **Major Decisions**
  - In production-like environments (`ENV=production` or on Hugging Face Spaces), require `API_KEY` and fail fast at startup when it is missing; Swagger/OpenAPI remain publicly accessible but all non-health API endpoints still enforce `X-API-Key`.
  - Use a single `require_api_key` dependency (based on `APIKeyHeader`) to protect all routers except `/health`.
  - Treat Streamlit as a first-class chat client, using `st.chat_message`/`st.chat_input` with session-based history and optional streaming from `/chat/stream`.
  - Keep Docling as an optional dependency used in:
    - Local ingestion scripts that upload text to `/documents/upload-text`.
    - The frontend upload dialog for converting PDFs/Office/HTML when available, while falling back to raw `.md`/`.txt` and showing clear errors otherwise.
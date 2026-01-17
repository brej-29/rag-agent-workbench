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
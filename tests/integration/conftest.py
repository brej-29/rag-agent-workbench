"""
Session-scoped integration fixture: real FastAPI app through TestClient with
the three external service boundaries mocked at realistic shapes.

INTERNAL WIRING THAT RUNS FOR REAL
  HTTP routing, CORS/security middleware, slowapi rate-limit middleware,
  the require_api_key auth dependency, LangGraph pipeline
  (normalize_input → contextualize_query → retrieve_context →
   corrective_retrieve → decide_next → generate_answer → format_response),
  all prompt builders, the ChatResponse schema, and the SSE frame encoder.

MOCKED (external network only)
  - init_pinecone: patched so the FastAPI startup event does not hit the
    Pinecone control plane.
  - pinecone_search / get_llm / is_tavily_configured: NOT patched here —
    patched function-scoped inside each test so each test controls the
    realistic response shapes independently.
  - cache_enabled: patched to return False so tests never receive stale
    cached responses from each other.

CI-SAFE: zero network calls; requires only the CI env vars already exported
  (PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_HOST).
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# API key for auth integration tests.  Set BEFORE any app import so
# _get_configured_api_key() (lru_cached in app.core.auth) picks it up.
# The fixture clears the cache after setting the value.
# ---------------------------------------------------------------------------
INTEGRATION_API_KEY = "integration-test-key"


@pytest.fixture(scope="session")
def integration_client():
    """
    TestClient wired to the real FastAPI app.  Session-scoped so the app is
    started once and reused across all integration tests.

    Setup order
    -----------
    1. Set API_KEY env var to a known value; clear auth LRU cache so the
       endpoint enforcer sees the new key.
    2. Clear get_settings() and get_llm() LRU caches to ensure a clean
       settings read (avoids contamination from unit tests earlier in the run).
    3. Reset the compiled LangGraph singleton so it is recompiled in this
       environment.
    4. Import app.main — module-level app construction runs here (routes,
       middleware, etc.).
    5. Patch init_pinecone in app.main's namespace before the TestClient
       context manager fires the FastAPI startup event.
    6. Patch cache_enabled at the router level so all tests get a fresh
       pipeline (no cross-test cache pollution).
    """
    # Step 1 — auth key
    prev_api_key = os.environ.get("API_KEY")
    os.environ["API_KEY"] = INTEGRATION_API_KEY

    # Step 2 — clear LRU caches that were populated before this fixture ran
    from app.core.config import get_settings
    from app.core.auth import _get_configured_api_key
    from app.services.llm.groq_llm import get_llm

    get_settings.cache_clear()
    _get_configured_api_key.cache_clear()
    get_llm.cache_clear()

    # Step 3 — reset compiled graph singleton
    import app.services.chat.graph as _graph_mod
    _graph_mod._graph = None

    # Step 4 — import the app (module-level construction)
    from app.main import app as _app

    # Steps 5+6 — patches active for the entire session
    with (
        patch("app.main.init_pinecone"),                       # startup: no Pinecone SDK call
        patch("app.routers.chat.cache_enabled", return_value=False),  # no cross-test cache
    ):
        with TestClient(_app, raise_server_exceptions=True) as client:
            yield client

    # Restore API_KEY to its pre-fixture state
    if prev_api_key is None:
        os.environ.pop("API_KEY", None)
    else:
        os.environ["API_KEY"] = prev_api_key
    _get_configured_api_key.cache_clear()

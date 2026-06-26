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

CI-SAFE: zero network calls.  The three required Pinecone env vars
  (PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_HOST) are set to
  obvious dummy values inside the fixture if they are not already present
  in the environment.  They satisfy Settings() validation but never reach
  the network — init_pinecone is patched before the startup event fires,
  and pinecone_search is mocked per-test.  No real Pinecone credentials
  are required to run these tests.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Required-without-default Settings fields (Field(...) in config.py).
# These are the ONLY three fields that have no default; every other field
# is either Optional[str] = Field(default=None) or has an explicit default.
# Setting obvious dummies here satisfies Settings() validation in CI where
# real credentials are absent.  The values NEVER reach the network:
#   - init_pinecone is patched (no Pinecone SDK call at startup)
#   - pinecone_search is patched per-test (no Pinecone query call)
# ---------------------------------------------------------------------------
_DUMMY_PINECONE: dict[str, str] = {
    "PINECONE_API_KEY": "test-key-not-real",
    "PINECONE_INDEX_NAME": "test-index",
    "PINECONE_HOST": "https://test.invalid",
}

# API key used by auth integration tests.
INTEGRATION_API_KEY = "integration-test-key"


@pytest.fixture(scope="session")
def integration_client():
    """
    TestClient wired to the real FastAPI app.  Session-scoped so the app is
    started once and reused across all integration tests.

    Setup order
    -----------
    1. Save and set API_KEY so the auth dependency sees a known value.
    2. Save and set required Pinecone env vars to obvious dummy values for
       any that are absent (CI has no .env; local dev keeps real values).
    3. Clear LRU caches that may have been populated by earlier unit tests.
    4. Reset the compiled LangGraph singleton.
    5. Import app.main (module-level construction: routes, middleware, etc.).
    6. Patch init_pinecone before the TestClient startup event fires.
    7. Patch cache_enabled to prevent cross-test cache pollution.
    8. Yield the TestClient.
    9. Teardown: restore all env vars to their original state and clear the
       auth LRU cache so later tests are not affected.
    """
    # ------------------------------------------------------------------
    # Step 1 — auth key
    # ------------------------------------------------------------------
    prev_api_key = os.environ.get("API_KEY")
    os.environ["API_KEY"] = INTEGRATION_API_KEY

    # ------------------------------------------------------------------
    # Step 2 — required Pinecone vars (CI has none; local keeps real values)
    # ------------------------------------------------------------------
    prev_pinecone = {key: os.environ.get(key) for key in _DUMMY_PINECONE}
    for key, dummy_val in _DUMMY_PINECONE.items():
        os.environ.setdefault(key, dummy_val)

    # ------------------------------------------------------------------
    # Step 3 — clear LRU caches populated before this fixture ran
    # ------------------------------------------------------------------
    from app.core.config import get_settings
    from app.core.auth import _get_configured_api_key
    from app.services.llm.groq_llm import get_llm

    get_settings.cache_clear()
    _get_configured_api_key.cache_clear()
    get_llm.cache_clear()

    # ------------------------------------------------------------------
    # Step 4 — reset compiled graph singleton
    # ------------------------------------------------------------------
    import app.services.chat.graph as _graph_mod
    _graph_mod._graph = None

    # ------------------------------------------------------------------
    # Step 5 — import the app (module-level construction)
    # ------------------------------------------------------------------
    from app.main import app as _app

    # ------------------------------------------------------------------
    # Steps 6+7 — patches active for the entire session
    # ------------------------------------------------------------------
    with (
        patch("app.main.init_pinecone"),                              # startup: no Pinecone SDK call
        patch("app.routers.chat.cache_enabled", return_value=False),  # no cross-test cache
    ):
        with TestClient(_app, raise_server_exceptions=True) as client:
            yield client

    # ------------------------------------------------------------------
    # Step 9 — teardown: restore env to its pre-fixture state
    # ------------------------------------------------------------------
    # Restore API_KEY
    if prev_api_key is None:
        os.environ.pop("API_KEY", None)
    else:
        os.environ["API_KEY"] = prev_api_key
    _get_configured_api_key.cache_clear()

    # Restore Pinecone vars (remove dummies we introduced; keep real values)
    for key, prev_val in prev_pinecone.items():
        if prev_val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev_val

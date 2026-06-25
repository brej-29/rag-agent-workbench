import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Reuse the same prod-detection heuristic used by validate_api_key_configuration
# (ENV=production or HF Spaces SPACE_ID/HF_HOME) so the two startup checks are
# consistent.
from app.core.auth import _is_production_like
from app.core.logging import get_logger

logger = get_logger(__name__)


def _get_allowed_origins() -> List[str]:
    """Return the CORS origin allowlist.

    - ALLOWED_ORIGINS env var (comma-separated): use that list.
    - Unset / empty: fall back to ["*"] for permissive local-dev behaviour.
      This default is only safe because allow_credentials is always False —
      the WHATWG Fetch Standard forbids wildcard origins + credentials together,
      and browsers reject that combination.
    """
    raw = os.getenv("ALLOWED_ORIGINS")
    if not raw:
        return ["*"]
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins if origins else ["*"]


def configure_security(app: FastAPI) -> None:
    """Configure CORS on the FastAPI app.

    API key enforcement is handled via dependencies in app.core.auth.

    allow_credentials is always False: the API authenticates with a bearer
    API key (X-API-Key header), not cookies.  Combining wildcard origins with
    allow_credentials=True is invalid per the WHATWG Fetch Standard and is
    rejected by browsers; bearer-token auth has no need for credentials mode.
    """
    origins = _get_allowed_origins()

    if origins == ["*"] and _is_production_like():
        logger.warning(
            "CORS origins resolved to wildcard ('*') in a production-like "
            "environment. Set ALLOWED_ORIGINS to a comma-separated list of "
            "trusted origins (e.g. 'https://my-app.hf.space,https://my-ui.com'). "
            "Wildcard origins are acceptable for local development only."
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # always False — bearer-token (X-API-Key) auth, not cookies.
        # Wildcard origins + credentials is invalid per the WHATWG Fetch Standard.
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS configured allow_origins=%s allow_credentials=False", origins)

import os
from typing import Iterable, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger

logger = get_logger(__name__)


def _get_allowed_origins() -> List[str]:
    raw = os.getenv("ALLOWED_ORIGINS")
    if not raw:
        # Default: permissive for local development and simple frontends.
        origins = ["*"]
    else:
        origins = [item.strip() for item in raw.split(",") if item.strip()]
        if not origins:
            origins = ["*"]
    return origins


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Optional API key protection for selected endpoints.

    When the API_KEY environment variable is set, this middleware enforces the
    presence of an `X-API-Key` header with a matching value for:

      - /ingest/*
      - /documents/*
      - /chat*
      - /search

    The following paths remain public regardless of API_KEY:

      - /health
      - /docs
      - /openapi.json
      - /redoc
      - /metrics

    When API_KEY is not set, the middleware is not installed and the API is open.
    """

    def __init__(self, app: FastAPI, api_key: str) -> None:  # type: ignore[override]
        super().__init__(app)
        self.api_key = api_key

        self._protected_prefixes: List[str] = [
            "/ingest",
            "/documents",
            "/chat",
            "/search",
        ]
        self._public_prefixes: List[str] = [
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/metrics",
        ]

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        path = request.url.path or "/"

        # Public endpoints stay open.
        if any(path.startswith(prefix) for prefix in self._public_prefixes):
            return await call_next(request)

        # Only enforce for protected prefixes.
        if not any(path.startswith(prefix) for prefix in self._protected_prefixes):
            return await call_next(request)

        header_key: Optional[str] = request.headers.get("X-API-Key")
        if not header_key or header_key != self.api_key:
            logger.warning("Rejected request with missing/invalid API key path=%s", path)
            return JSONResponse(
                status_code=401,
                content={
                    "detail": (
                        "Missing or invalid API key. Provide X-API-Key header with "
                        "a valid key to access this endpoint."
                    )
                },
            )

        return await call_next(request)


def configure_security(app: FastAPI) -> None:
    """Configure CORS and optional API key protection on the FastAPI app."""
    # CORS
    origins = _get_allowed_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS configured allow_origins=%s", origins)

    # Optional API key middleware
    api_key = os.getenv("API_KEY")
    if not api_key:
        logger.warning(
            "API key disabled; protected endpoints are open. "
            "Set API_KEY environment variable to enable X-API-Key protection."
        )
        return

    logger.info("API key protection enabled for ingest, documents, search, and chat.")
    app.add_middleware(APIKeyMiddleware, api_key=api_key)
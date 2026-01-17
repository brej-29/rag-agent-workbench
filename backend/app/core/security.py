import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


def configure_security(app: FastAPI) -> None:
    """Configure CORS on the FastAPI app.

    API key enforcement is handled via dependencies in app.core.auth.
    """
    origins = _get_allowed_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS configured allow_origins=%s", origins)
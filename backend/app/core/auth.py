import os
from functools import lru_cache
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.logging import get_logger

logger = get_logger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@lru_cache(maxsize=1)
def _get_configured_api_key() -> Optional[str]:
    """Return the configured API key, or None if not set.

    The key is read from the API_KEY environment variable.
    """
    raw = os.getenv("API_KEY")
    if raw is None or not raw.strip():
        return None
    return raw.strip()


def _is_production_like() -> bool:
    """Heuristic to detect production / hosted environments.

    - ENV=production
    - or running on Hugging Face Spaces (SPACE_ID or HF_HOME set)
    """
    env = os.getenv("ENV", "").strip().lower()
    if env == "production":
        return True
    if os.getenv("SPACE_ID") or os.getenv("HF_HOME"):
        return True
    return False


def validate_api_key_configuration() -> None:
    """Validate API key configuration at startup.

    Behaviour:
    - In production-like environments (HF Spaces or ENV=production):
        - API_KEY MUST be set, otherwise raise RuntimeError (fail fast).
    - In other environments:
        - If API_KEY is missing, allow running open but log a clear warning.
    """
    configured = _get_configured_api_key()
    if _is_production_like():
        if not configured:
            raise RuntimeError(
                "API_KEY environment variable must be set when running in "
                "production or on Hugging Face Spaces. Configure API_KEY in "
                "your environment or Space secrets."
            )
        logger.info("API key configured for production / hosted environment.")
    else:
        if not configured:
            logger.warning(
                "API_KEY is not set; backend is running without authentication. "
                "This is intended for local development only."
            )
        else:
            logger.info("API key configured for development mode.")


async def require_api_key(api_key: Optional[str] = Security(api_key_header)) -> None:
    """FastAPI dependency that enforces X-API-Key when configured.

    - If API_KEY is not configured (local/dev), this is a no-op.
    - If API_KEY is configured:
        - Missing or mismatched X-API-Key results in HTTP 403.
    """
    configured = _get_configured_api_key()
    if not configured:
        # No API key configured: dev mode, do not enforce.
        return

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing API key. Provide X-API-Key header.",
        )

    if api_key != configured:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
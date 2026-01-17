from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Global limiter instance used for decorators.
limiter = Limiter(key_func=get_remote_address)


def setup_rate_limiter(app: FastAPI) -> None:
    """Configure SlowAPI rate limiting middleware and handlers.

    Limits are enabled/disabled via Settings.RATE_LIMIT_ENABLED.
    """
    settings = get_settings()
    if not getattr(settings, "RATE_LIMIT_ENABLED", True):
        logger.info("Rate limiting is disabled via settings.")
        return

    logger.info("Rate limiting enabled with SlowAPI.")

    app.state.limiter = limiter  # type: ignore[attr-defined]

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_exceeded_handler(  # type: ignore[no-redef]
        request: Request,
        exc: RateLimitExceeded,
    ) -> JSONResponse:
        retry_after: str | None = None
        try:
            retry_after = exc.headers.get("Retry-After")  # type: ignore[assignment]
        except Exception:  # noqa: BLE001
            retry_after = None

        logger.warning(
            "Rate limit exceeded path=%s client=%s limit=%s",
            request.url.path,
            get_remote_address(request),
            exc.detail,
        )
        content: dict[str, Any] = {
            "detail": "Rate limit exceeded. Please slow down your requests.",
        }
        if retry_after is not None:
            content["retry_after"] = retry_after
        return JSONResponse(status_code=429, content=content)

    # Attach SlowAPI middleware
    app.middleware("http")(limiter.middleware)
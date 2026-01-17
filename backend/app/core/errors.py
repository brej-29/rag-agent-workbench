import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status

logger = logging.getLogger(__name__)


class PineconeIndexConfigError(RuntimeError):
    """Raised when the Pinecone index is not configured for integrated embeddings."""


class UpstreamServiceError(RuntimeError):
    """Raised when an upstream dependency (LLM, web search, etc.) fails."""

    def __init__(self, service: str, message: str) -> None:
        self.service = service
        super().__init__(message)


def setup_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI app."""

    @app.exception_handler(PineconeIndexConfigError)
    async def pinecone_index_config_error_handler(
        request: Request, exc: PineconeIndexConfigError
    ) -> JSONResponse:
        logger.error("Pinecone index configuration error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(exc)},
        )

    @app.exception_handler(UpstreamServiceError)
    async def upstream_service_error_handler(
        request: Request, exc: UpstreamServiceError
    ) -> JSONResponse:
        logger.error("Upstream service error from %s: %s", exc.service, exc)
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": str(exc)},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning("Request validation error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors()},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        # Let FastAPI-style HTTPException pass through with its status and detail.
        logger.warning(
            "HTTPException raised: status=%s detail=%s",
            exc.status_code,
            exc.detail,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled error", exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )


__all__ = ["PineconeIndexConfigError", "setup_exception_handlers"]
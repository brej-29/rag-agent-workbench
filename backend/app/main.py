from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from app.core.config import get_settings
from app.core.errors import PineconeIndexConfigError, setup_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.routers.documents import router as documents_router
from app.routers.health import router as health_router
from app.routers.ingest import router as ingest_router
from app.routers.search import router as search_router
from app.services.pinecone_store import init_pinecone

settings = get_settings()
configure_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    default_response_class=ORJSONResponse,
)

# Register routers
app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(search_router)
app.include_router(documents_router)

# Register exception handlers
setup_exception_handlers(app)


@app.on_event("startup")
async def startup_event() -> None:
    """Application startup hook."""
    try:
        init_pinecone(settings)
        logger.info("Pinecone initialisation completed")
    except PineconeIndexConfigError:
        # Let the exception handler and FastAPI/uvicorn deal with the error.
        # Re-raise to fail fast on misconfiguration.
        raise
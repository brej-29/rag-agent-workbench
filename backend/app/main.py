from fastapi import Depends, FastAPI
from fastapi.responses import ORJSONResponse

from app.core.auth import require_api_key, validate_api_key_configuration
from app.core.config import get_settings
from app.core.errors import PineconeIndexConfigError, setup_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.metrics import setup_metrics
from app.core.rate_limit import setup_rate_limiter
from app.core.runtime import get_port
from app.core.security import configure_security
from app.routers.documents import router as documents_router
from app.routers.health import router as health_router
from app.routers.ingest import router as ingest_router
from app.routers.search import router as search_router
from app.routers.chat import router as chat_router
from app.routers.metrics import router as metrics_router
from app.services.pinecone_store import init_pinecone

settings = get_settings()
configure_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)

# Validate API key configuration early so hosted deployments fail fast when misconfigured.
validate_api_key_configuration()

# Log runtime port / environment context at import time for easier diagnostics.
get_port()

app = FastAPI(
    title="RAG Agent Workbench API",
    version=settings.APP_VERSION,
    default_response_class=ORJSONResponse,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Core app configuration
configure_security(app)
setup_rate_limiter(app)
setup_metrics(app)

# Register routers with tags and ensure they are included in the schema.
# Health and docs remain public; all other routers are protected by API key dependency when configured.
app.include_router(health_router, tags=["health"])
app.include_router(ingest_router, tags=["ingest"], dependencies=[Depends(require_api_key)])
app.include_router(search_router, tags=["search"], dependencies=[Depends(require_api_key)])
app.include_router(documents_router, tags=["documents"], dependencies=[Depends(require_api_key)])
app.include_router(chat_router, tags=["chat"], dependencies=[Depends(require_api_key)])
app.include_router(metrics_router, tags=["metrics"], dependencies=[Depends(require_api_key)])

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
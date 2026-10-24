from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from mcp.server.transport_security import TransportSecuritySettings
from starlette.routing import Route

from app.core.auth import (
    _get_configured_api_key,
    require_api_key,
    ApiKeyGate,
    validate_api_key_configuration,
)
from app.core.config import get_settings
from app.core.errors import setup_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.metrics import setup_metrics
from app.core.prometheus_metrics import setup_prometheus
from app.core.rate_limit import setup_rate_limiter
from app.core.runtime import get_port
from app.core.security import configure_security
from app.routers.documents import router as documents_router
from app.routers.health import router as health_router
from app.routers.ingest import router as ingest_router
from app.routers.search import router as search_router
from app.routers.chat import router as chat_router
from app.routers.checklist import router as checklist_router
from app.routers.workspace import router as workspace_router
from app.routers.metrics import router as metrics_router
from app.mcp_server import mcp
from app.services.pinecone_store import init_pinecone

settings = get_settings()
configure_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)

# Validate API key configuration early so hosted deployments fail fast when misconfigured.
validate_api_key_configuration()

# Log runtime port / environment context at import time for easier diagnostics.
get_port()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Startup: initialise Pinecone.  PineconeIndexConfigError propagates so a
    misconfigured deployment fails fast instead of serving broken retrieval."""
    init_pinecone(settings)
    logger.info("Pinecone initialisation completed")
    # A mounted sub-app's lifespan never runs, so the MCP session manager is entered here.
    async with mcp.session_manager.run():
        yield


app = FastAPI(
    lifespan=lifespan,
    title="RAG Agent Workbench API",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Core app configuration
configure_security(app)
setup_rate_limiter(app)
setup_metrics(app)
setup_prometheus(app)

# Register routers with tags and ensure they are included in the schema.
# Health and docs remain public; all other routers are protected by API key dependency when configured.
app.include_router(health_router, tags=["health"])
app.include_router(ingest_router, tags=["ingest"], dependencies=[Depends(require_api_key)])
app.include_router(search_router, tags=["search"], dependencies=[Depends(require_api_key)])
app.include_router(documents_router, tags=["documents"], dependencies=[Depends(require_api_key)])
app.include_router(chat_router, tags=["chat"], dependencies=[Depends(require_api_key)])
app.include_router(checklist_router, tags=["checklist"], dependencies=[Depends(require_api_key)])
app.include_router(workspace_router, tags=["workspace"], dependencies=[Depends(require_api_key)])
app.include_router(metrics_router, tags=["metrics"], dependencies=[Depends(require_api_key)])

# MCP (Streamable HTTP) at /mcp, behind the same X-API-Key as the REST API.
# DNS-rebinding protection (localhost allowlist) applies when no API key is set;
# with a key, authentication is the control and the hosting proxy owns Host.
_mcp_security = (
    TransportSecuritySettings(enable_dns_rebinding_protection=False)
    if _get_configured_api_key()
    else TransportSecuritySettings(
        allowed_hosts=["localhost", "localhost:*", "127.0.0.1", "127.0.0.1:*"],
        allowed_origins=["http://localhost:*", "http://127.0.0.1:*"],
    )
)
# Exact Route (not Mount): a Mount answers POST /mcp with a 307 to /mcp/, which MCP
# clients do not follow.  The SDK app routes /mcp itself, so the path is passed through.
app.router.routes.append(Route("/mcp", endpoint=ApiKeyGate(
    mcp.streamable_http_app(streamable_http_path="/mcp", json_response=True, transport_security=_mcp_security)
), name="mcp"))

# Register exception handlers
setup_exception_handlers(app)


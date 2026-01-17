from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="Health check",
    description="Returns service status, name, and version.",
)
async def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }
from fastapi import APIRouter

from app.core.metrics import get_metrics_snapshot

router = APIRouter(tags=["metrics"])


@router.get(
    "/metrics",
    summary="In-memory metrics snapshot",
    description=(
        "Returns request and error counts by path, timing statistics for chat "
        "requests (average and p50/p95), cache hit/miss counters, and the last "
        "20 timing samples."
    ),
)
async def metrics() -> dict:
    return get_metrics_snapshot()
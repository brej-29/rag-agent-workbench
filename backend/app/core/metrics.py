from collections import defaultdict, deque
from threading import Lock
from time import perf_counter
from typing import Deque, Dict, List, Mapping, MutableMapping

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.cache import get_cache_stats
from app.core.logging import get_logger

logger = get_logger(__name__)


# Request and error counters by path.
_request_counts: MutableMapping[str, int] = defaultdict(int)
_error_counts: MutableMapping[str, int] = defaultdict(int)

# Timing samples for chat requests: last N samples.
_TIMING_FIELDS = ["retrieve_ms", "web_ms", "generate_ms", "total_ms"]
_TIMING_BUFFER_SIZE = 20
_timing_samples: Deque[Dict[str, float]] = deque(maxlen=_TIMING_BUFFER_SIZE)

# Aggregated sums and counts for averages.
_timing_sums: Dict[str, float] = {f: 0.0 for f in _TIMING_FIELDS}
_timing_count: int = 0

_lock = Lock()


async def metrics_middleware(request: Request, call_next):
    """Middleware capturing request counts and error counts by path."""
    path = request.url.path or "/"
    start = perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001
        elapsed = (perf_counter() - start) * 1000.0
        with _lock:
            _request_counts[path] += 1
            _error_counts[path] += 1
        logger.exception("Unhandled error for path=%s elapsed_ms=%.2f", path, elapsed)
        raise

    elapsed = (perf_counter() - start) * 1000.0
    status = response.status_code
    with _lock:
        _request_counts[path] += 1
        if status >= 400:
            _error_counts[path] += 1

    logger.debug(
        "Request path=%s status=%s elapsed_ms=%.2f", path, status, elapsed
    )
    return response


def record_chat_timings(timings: Mapping[str, float]) -> None:
    """Record timing metrics from a chat request.

    Expects a mapping with keys retrieve_ms, web_ms, generate_ms, total_ms.
    """
    global _timing_count
    sample = {field: float(timings.get(field, 0.0)) for field in _TIMING_FIELDS}
    with _lock:
        _timing_samples.append(sample)
        for field, value in sample.items():
            _timing_sums[field] += value
        _timing_count += 1


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = max(0, min(len(values_sorted) - 1, int(round((p / 100.0) * (len(values_sorted) - 1)))))
    return values_sorted[k]


def get_metrics_snapshot() -> Dict[str, object]:
    """Return a stable snapshot of metrics suitable for /metrics responses."""
    with _lock:
        requests_by_path = dict(_request_counts)
        errors_by_path = dict(_error_counts)
        samples = list(_timing_samples)
        sums = dict(_timing_sums)
        count = int(_timing_count)

    averages: Dict[str, float] = {}
    if count > 0:
        for field in _TIMING_FIELDS:
            averages[field] = sums.get(field, 0.0) / count
    else:
        for field in _TIMING_FIELDS:
            averages[field] = 0.0

    # Compute percentiles over the last N samples.
    p50: Dict[str, float] = {}
    p95: Dict[str, float] = {}
    if samples:
        for field in _TIMING_FIELDS:
            values = [s.get(field, 0.0) for s in samples]
            p50[field] = _percentile(values, 50.0)
            p95[field] = _percentile(values, 95.0)
    else:
        for field in _TIMING_FIELDS:
            p50[field] = 0.0
            p95[field] = 0.0

    cache_stats = get_cache_stats()

    return {
        "requests_by_path": requests_by_path,
        "errors_by_path": errors_by_path,
        "timings": {
            "average_ms": averages,
            "p50_ms": p50,
            "p95_ms": p95,
        },
        "cache": cache_stats,
        "sample_count": count,
        "samples": samples,
    }


def setup_metrics(app: FastAPI) -> None:
    """Attach metrics middleware to the app."""
    logger.info("Metrics middleware enabled.")
    app.middleware("http")(metrics_middleware)
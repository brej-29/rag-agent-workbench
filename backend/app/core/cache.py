import json
from threading import Lock
from typing import Any, Dict, Hashable, Optional, Tuple

from cachetools import TTLCache

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_settings = get_settings()
_CACHE_ENABLED: bool = getattr(_settings, "CACHE_ENABLED", True)

# TTLs are intentionally short and in-code defaults; no env required.
_SEARCH_TTL_SECONDS = 60
_CHAT_TTL_SECONDS = 60

_search_cache: TTLCache = TTLCache(maxsize=1024, ttl=_SEARCH_TTL_SECONDS)
_chat_cache: TTLCache = TTLCache(maxsize=512, ttl=_CHAT_TTL_SECONDS)

_lock = Lock()

_search_hits: int = 0
_search_misses: int = 0
_chat_hits: int = 0
_chat_misses: int = 0


def cache_enabled() -> bool:
    return _CACHE_ENABLED


def _make_search_key(
    namespace: str,
    query: str,
    top_k: int,
    filters: Optional[Dict[str, Any]],
) -> Hashable:
    filters_json = (
        json.dumps(filters, sort_keys=True, separators=(",", ":"))
        if filters is not None
        else ""
    )
    return (namespace, query, int(top_k), filters_json)


def _make_chat_key(
    namespace: str,
    query: str,
    top_k: int,
    min_score: float,
    use_web_fallback: bool,
) -> Hashable:
    return (namespace, query, int(top_k), float(min_score), bool(use_web_fallback))


def get_search_cached(
    namespace: str,
    query: str,
    top_k: int,
    filters: Optional[Dict[str, Any]],
) -> Optional[Any]:
    """Return cached search results or None."""
    global _search_hits, _search_misses
    if not _CACHE_ENABLED:
        return None

    key = _make_search_key(namespace, query, top_k, filters)
    with _lock:
        if key in _search_cache:
            _search_hits += 1
            logger.info(
                "Search cache hit namespace='%s' query='%s' top_k=%d",
                namespace,
                query,
                top_k,
            )
            return _search_cache[key]
        _search_misses += 1
    logger.info(
        "Search cache miss namespace='%s' query='%s' top_k=%d",
        namespace,
        query,
        top_k,
    )
    return None


def set_search_cached(
    namespace: str,
    query: str,
    top_k: int,
    filters: Optional[Dict[str, Any]],
    value: Any,
) -> None:
    if not _CACHE_ENABLED:
        return
    key = _make_search_key(namespace, query, top_k, filters)
    with _lock:
        _search_cache[key] = value


def get_chat_cached(
    namespace: str,
    query: str,
    top_k: int,
    min_score: float,
    use_web_fallback: bool,
) -> Optional[Any]:
    """Return cached chat response or None.

    Only used when chat_history is empty.
    """
    global _chat_hits, _chat_misses
    if not _CACHE_ENABLED:
        return None

    key = _make_chat_key(namespace, query, top_k, min_score, use_web_fallback)
    with _lock:
        if key in _chat_cache:
            _chat_hits += 1
            logger.info(
                "Chat cache hit namespace='%s' query='%s' top_k=%d",
                namespace,
                query,
                top_k,
            )
            return _chat_cache[key]
        _chat_misses += 1
    logger.info(
        "Chat cache miss namespace='%s' query='%s' top_k=%d",
        namespace,
        query,
        top_k,
    )
    return None


def set_chat_cached(
    namespace: str,
    query: str,
    top_k: int,
    min_score: float,
    use_web_fallback: bool,
    value: Any,
) -> None:
    if not _CACHE_ENABLED:
        return
    key = _make_chat_key(namespace, query, top_k, min_score, use_web_fallback)
    with _lock:
        _chat_cache[key] = value


def get_cache_stats() -> Dict[str, int]:
    """Return a snapshot of cache hit/miss counters."""
    return {
        "search_hits": _search_hits,
        "search_misses": _search_misses,
        "chat_hits": _chat_hits,
        "chat_misses": _chat_misses,
    }
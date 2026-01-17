import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from app.core.config import get_env_bool
from app.core.logging import get_logger

logger = get_logger(__name__)


def is_tracing_enabled() -> bool:
    """Return True if LangSmith / LangChain tracing is enabled via environment."""
    tracing_flag = get_env_bool("LANGCHAIN_TRACING_V2", False)
    api_key = os.getenv("LANGCHAIN_API_KEY")
    return tracing_flag and bool(api_key)


def get_langsmith_project() -> Optional[str]:
    """Return the LangSmith project name, if configured."""
    return os.getenv("LANGCHAIN_PROJECT")


@lru_cache(maxsize=1)
def get_tracing_callbacks() -> List[Any]:
    """Return LangChain callback handlers for tracing, if available.

    When LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY is set, this will
    attempt to create a LangChainTracer instance. If tracing is not enabled
    or the tracer is unavailable, an empty list is returned.
    """
    if not is_tracing_enabled():
        logger.info(
            "LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true and "
            "LANGCHAIN_API_KEY to enable)."
        )
        return []

    try:
        from langchain_core.tracers import LangChainTracer  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "LangSmith tracing requested but LangChainTracer is unavailable: %s", exc
        )
        return []

    project = get_langsmith_project()
    tracer = LangChainTracer(project_name=project)
    logger.info(
        "LangSmith tracing enabled for project='%s'",
        project or "(default)",
    )
    return [tracer]


def get_tracing_response_metadata() -> Dict[str, Any]:
    """Return trace metadata suitable for API responses."""
    return {
        "langsmith_project": get_langsmith_project(),
        "trace_enabled": is_tracing_enabled(),
    }
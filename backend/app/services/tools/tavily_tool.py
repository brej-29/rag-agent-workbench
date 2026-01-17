from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def is_tavily_configured() -> bool:
    """Return True if Tavily web search is configured."""
    settings = get_settings()
    return bool(settings.TAVILY_API_KEY)


def get_tavily_tool(max_results: int):
    """Return a TavilySearchResults tool instance or None if not configured.

    The underlying LangChain community tool reads the TAVILY_API_KEY environment
    variable, so we only need to ensure configuration is present and log if not.
    """
    if not is_tavily_configured():
        logger.warning(
            "Tavily web search requested but TAVILY_API_KEY is not set. "
            "Web fallback will be disabled for this request."
        )
        return None

    try:
        from langchain_community.tools.tavily_search import (  # type: ignore[import]
            TavilySearchResults,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to import TavilySearchResults tool: %s", exc)
        return None

    settings = get_settings()
    logger.info(
        "Initialising TavilySearchResults tool max_results=%d", max_results
    )

    # The TavilySearchResults tool picks up the API key from the TAVILY_API_KEY
    # environment variable; we just pass the max_results configuration.
    return TavilySearchResults(max_results=max_results)
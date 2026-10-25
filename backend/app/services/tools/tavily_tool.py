from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def is_tavily_configured() -> bool:
    """Return True if Tavily web search is configured."""
    settings = get_settings()
    return bool(settings.TAVILY_API_KEY)


def get_tavily_tool(max_results: int):
    """Return a Runnable Tavily search (list of result dicts) or None if not configured.

    Wraps tavily-python directly (langchain-community's TavilySearchResults is
    legacy in LangChain 1.x); as a Runnable it still receives config for tracing.
    """
    if not is_tavily_configured():
        logger.warning(
            "Tavily web search requested but TAVILY_API_KEY is not set. "
            "Web fallback will be disabled for this request."
        )
        return None

    from langchain_core.runnables import RunnableLambda  # noqa: PLC0415
    from tavily import TavilyClient  # noqa: PLC0415

    client = TavilyClient(api_key=get_settings().TAVILY_API_KEY)
    logger.info("Initialising Tavily search max_results=%d", max_results)
    return RunnableLambda(
        lambda inp: client.search(inp["query"], max_results=max_results).get("results", []),
        name="tavily_search",
    )

from typing import BaseException, List, Optional

import feedparser
import httpx
from langchain_core.documents import Document
from tenacity import RetryError, retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.normalize import make_doc_id, normalize_text, is_valid_document

logger = get_logger(__name__)

ARXIV_QUERY_URL_DEFAULT = "https://export.arxiv.org/api/query"
ARXIV_USER_AGENT = "rag-agent-workbench/0.1 (local-dev; contact: example@example.com)"


def _get_arxiv_query_url() -> str:
    """Return the arXiv API query URL, allowing optional env override."""
    import os

    return os.getenv("ARXIV_QUERY_URL") or ARXIV_QUERY_URL_DEFAULT


def _is_retryable_arxiv_error(exc: BaseException) -> bool:
    """Return True if an exception should trigger a retry."""
    if isinstance(exc, httpx.RequestError):
        # Network issues / timeouts
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        # Retry on rate limiting and server errors
        return status == 429 or 500 <= status < 600
    return False


@retry(
    reraise=True,
    stop=stop_after_attempt(get_settings().HTTP_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(_is_retryable_arxiv_error),
)
async def _fetch_arxiv_feed(query: str, max_results: int) -> str:
    """Fetch the arXiv Atom feed, following redirects and handling transient errors."""
    settings = get_settings()
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
    }

    url = _get_arxiv_query_url()
    headers = {"User-Agent": ARXIV_USER_AGENT}

    async with httpx.AsyncClient(
        timeout=settings.HTTP_TIMEOUT_SECONDS,
        follow_redirects=True,
        headers=headers,
    ) as client:
        response = await client.get(url, params=params)

        # Defensive manual redirect handling (should rarely be needed with follow_redirects=True)
        redirect_attempts = 0
        while (
            response.status_code in {301, 302, 307, 308}
            and "location" in response.headers
            and redirect_attempts < 3
        ):
            next_url = response.headers["location"]
            logger.info("Following manual arXiv redirect to %s", next_url)
            response = await client.get(next_url, headers=headers)
            redirect_attempts += 1

        if response.status_code != 200:
            logger.error(
                "arXiv API returned %s %s for url=%s",
                response.status_code,
                response.reason_phrase,
                str(response.url),
            )
            response.raise_for_status()

        return response.text


async def fetch_arxiv_documents(
    query: str,
    max_results: int,
    category: Optional[str] = None,
) -> List[Document]:
    """Fetch documents from the arXiv API and convert them into LangChain Documents."""
    feed_text = await _fetch_arxiv_feed(query=query, max_results=max_results)

    parsed = feedparser.parse(feed_text)
    entries = getattr(parsed, "entries", [])

    documents: List[Document] = []
    for entry in entries:
        title = (entry.get("title") or "").strip()
        summary = (entry.get("summary") or "").strip()
        if not summary:
            continue

        authors_list = entry.get("authors", [])
        authors = ", ".join(a.get("name") for a in authors_list if a.get("name"))

        published = entry.get("published") or ""
        link = entry.get("link") or entry.get("id") or ""

        raw_text = f"{title}\n\n{summary}"
        normalized = normalize_text(raw_text)

        if not is_valid_document(normalized):
            logger.info(
                "Skipping short arXiv document title='%s' (len=%d)",
                title,
                len(normalized),
            )
            continue

        doc_id = make_doc_id(source="arxiv", title=title, url=link)
        metadata = {
            "title": title,
            "source": "arxiv",
            "url": link,
            "published": published,
            "authors": authors,
            "doc_id": doc_id,
        }
        if category:
            metadata["category"] = category

        documents.append(Document(page_content=normalized, metadata=metadata))

    logger.info("Fetched %d arXiv documents for query='%s'", len(documents), query)
    return documents
from typing import List, Optional

import feedparser
import httpx
from langchain_core.documents import Document
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.normalize import make_doc_id, normalize_text, is_valid_document

logger = get_logger(__name__)

_ARXIV_API_URL = "http://export.arxiv.org/api/query"


@retry(
    reraise=True,
    stop=stop_after_attempt(get_settings().HTTP_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(httpx.HTTPError),
)
async def _fetch_arxiv_feed(query: str, max_results: int) -> str:
    settings = get_settings()
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
    }

    async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(_ARXIV_API_URL, params=params)
        response.raise_for_status()
        return response.text


async def fetch_arxiv_documents(
    query: str,
    max_results: int,
    category: Optional[str] = None,
) -> List[Document]:
    """Fetch documents from the arXiv API and convert them into LangChain Documents."""
    try:
        feed_text = await _fetch_arxiv_feed(query=query, max_results=max_results)
    except RetryError as exc:
        # Last attempt's exception is stored in .last_attempt
        logger.error("Failed to fetch arXiv feed after retries: %s", exc)
        raise

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
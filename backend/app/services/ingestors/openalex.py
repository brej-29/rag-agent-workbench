from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from langchain_core.documents import Document
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.normalize import make_doc_id, normalize_text, is_valid_document

logger = get_logger(__name__)

_OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def _reconstruct_abstract_from_inverted_index(inverted: Dict[str, List[int]]) -> str:
    """Reconstruct abstract text from OpenAlex's abstract_inverted_index structure."""
    if not inverted:
        return ""

    position_to_word: Dict[int, str] = {}
    for word, positions in inverted.items():
        for pos in positions:
            # If collisions happen, first word wins; this is acceptable for our use.
            position_to_word.setdefault(pos, word)

    if not position_to_word:
        return ""

    max_pos = max(position_to_word.keys())
    words: List[str] = []
    for idx in range(max_pos + 1):
        word = position_to_word.get(idx, "")
        words.append(word)
    return " ".join(words).strip()


@retry(
    reraise=True,
    stop=stop_after_attempt(get_settings().HTTP_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(httpx.HTTPError),
)
async def _fetch_openalex_works(
    query: str,
    per_page: int,
    mailto: str,
) -> Dict[str, Any]:
    settings = get_settings()
    params = {
        "search": query,
        "per-page": per_page,
        "mailto": mailto,
    }

    async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(_OPENALEX_WORKS_URL, params=params)
        response.raise_for_status()
        return response.json()


async def fetch_openalex_documents(
    query: str,
    max_results: int,
    mailto: str,
) -> List[Document]:
    """Fetch works from OpenAlex and convert them into LangChain Documents."""
    try:
        payload = await _fetch_openalex_works(
            query=query,
            per_page=max_results,
            mailto=mailto,
        )
    except RetryError as exc:
        logger.error("Failed to fetch OpenAlex works after retries: %s", exc)
        raise

    results: List[Dict[str, Any]] = payload.get("results") or []

    documents: List[Document] = []
    for work in results:
        title = (work.get("display_name") or "").strip()
        if not title:
            continue

        # Authorship
        authorships = work.get("authorships") or []
        authors: List[str] = []
        for auth in authorships:
            author_info = auth.get("author") or {}
            name = author_info.get("display_name")
            if name:
                authors.append(name)
        authors_str = ", ".join(authors)

        publication_date = work.get("publication_date") or ""
        if not publication_date:
            # fall back to year if needed
            year = work.get("publication_year")
            if year is not None:
                publication_date = str(year)

        primary_location = work.get("primary_location") or {}
        url = primary_location.get("landing_page_url") or work.get("id") or ""

        abstract: Optional[str] = work.get("abstract")
        if not abstract:
            inverted = work.get("abstract_inverted_index") or {}
            if isinstance(inverted, dict):
                abstract = _reconstruct_abstract_from_inverted_index(inverted)

        if not abstract:
            logger.info("Skipping OpenAlex work with no abstract title='%s'", title)
            continue

        raw_text = f"{title}\n\n{abstract}"
        normalized = normalize_text(raw_text)

        if not is_valid_document(normalized):
            logger.info(
                "Skipping short OpenAlex document title='%s' (len=%d)",
                title,
                len(normalized),
            )
            continue

        doc_id = make_doc_id(source="openalex", title=title, url=url)
        metadata: Dict[str, Any] = {
            "title": title,
            "source": "openalex",
            "url": url,
            "published": publication_date,
            "authors": authors_str,
            "doc_id": doc_id,
        }

        documents.append(Document(page_content=normalized, metadata=metadata))

    logger.info("Fetched %d OpenAlex documents for query='%s'", len(documents), query)
    return documents
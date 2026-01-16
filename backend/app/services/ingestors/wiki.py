from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.normalize import make_doc_id, normalize_text, is_valid_document

logger = get_logger(__name__)

_WIKI_REST_BASE = "https://en.wikipedia.org/api/rest_v1"


@retry(
    reraise=True,
    stop=stop_after_attempt(get_settings().HTTP_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(httpx.HTTPError),
)
async def _fetch_summary(title: str) -> Dict[str, Any]:
    settings = get_settings()
    url = f"{_WIKI_REST_BASE}/page/summary/{title}"
    async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(url)
        if response.status_code == 404:
            # Let caller handle fallback
            return {"status": 404}
        response.raise_for_status()
        return response.json()


@retry(
    reraise=True,
    stop=stop_after_attempt(get_settings().HTTP_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(httpx.HTTPError),
)
async def _fetch_html(title: str) -> str:
    settings = get_settings()
    url = f"{_WIKI_REST_BASE}/page/html/{title}"
    async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


def _strip_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return normalize_text(text)


async def fetch_wiki_documents(titles: List[str]) -> List[Document]:
    """Fetch documents from Wikimedia REST API based on page titles."""
    documents: List[Document] = []

    for title in titles:
        safe_title = title.replace(" ", "_")
        try:
            summary_payload = await _fetch_summary(safe_title)
        except RetryError as exc:
            logger.error("Failed to fetch Wikipedia summary for '%s': %s", title, exc)
            continue

        status_code = summary_payload.get("status")
        summary_text: str = ""
        page_url: str = ""
        full_text: Optional[str] = None

        if status_code == 404:
            # Fallback to HTML endpoint
            try:
                html = await _fetch_html(safe_title)
            except RetryError as exc:
                logger.error(
                    "Failed to fetch Wikipedia HTML for '%s' after retries: %s",
                    title,
                    exc,
                )
                continue
            full_text = _strip_html_to_text(html)
            page_url = f"https://en.wikipedia.org/wiki/{safe_title}"
        else:
            summary_text = (summary_payload.get("extract") or "").strip()
            content_urls = summary_payload.get("content_urls") or {}
            desktop = content_urls.get("desktop") or {}
            page_url = desktop.get("page") or f"https://en.wikipedia.org/wiki/{safe_title}"

            # In some cases summary may be empty; attempt HTML fetch for richer text
            if not summary_text:
                try:
                    html = await _fetch_html(safe_title)
                    full_text = _strip_html_to_text(html)
                except RetryError as exc:
                    logger.error(
                        "Failed to fetch Wikipedia HTML for '%s' after retries: %s",
                        title,
                        exc,
                    )

        combined_text_parts = [title]
        if summary_text:
            combined_text_parts.append(summary_text)
        if full_text:
            combined_text_parts.append(full_text)

        combined_text = "\n\n".join(part for part in combined_text_parts if part)
        normalized = normalize_text(combined_text)

        if not is_valid_document(normalized):
            logger.info(
                "Skipping short Wikipedia document title='%s' (len=%d)",
                title,
                len(normalized),
            )
            continue

        doc_id = make_doc_id(source="wiki", title=title, url=page_url)
        metadata: Dict[str, Any] = {
            "title": title,
            "source": "wiki",
            "url": page_url,
            "extract": summary_text,
            "doc_id": doc_id,
        }

        documents.append(Document(page_content=normalized, metadata=metadata))

    logger.info("Fetched %d Wikipedia documents", len(documents))
    return documents
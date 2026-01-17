from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.normalize import make_doc_id, normalize_text, is_valid_document

logger = get_logger(__name__)

_WIKI_REST_BASE = "https://en.wikipedia.org/api/rest_v1"
_ACTION_API_URL = "https://en.wikipedia.org/w/api.php"

DEFAULT_UA = (
    "rag-agent-workbench/0.1 "
    "(+https://github.com/<YOUR_GITHUB_USERNAME>/rag-agent-workbench; contact: <email>)"
)


class WikimediaRestTransientError(Exception):
    """Raised for transient Wikimedia REST API errors that should be retried."""
    pass


def _wikimedia_headers() -> Dict[str, str]:
    """Build Wikimedia-friendly headers with a descriptive User-Agent.

    Uses WIKIMEDIA_USER_AGENT from the environment if present, otherwise DEFAULT_UA.
    """
    ua = os.getenv("WIKIMEDIA_USER_AGENT") or DEFAULT_UA
    return {
        "User-Agent": ua,
        "Api-User-Agent": ua,
        "Accept": "application/json; charset=utf-8",
        "Accept-Language": "en",
    }


@retry(
    reraise=True,
    stop=stop_after_attempt(get_settings().HTTP_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(WikimediaRestTransientError)
    | retry_if_exception_type(httpx.RequestError),
)
async def _fetch_summary(title: str) -> Dict[str, Any]:
    """Fetch REST summary for a page title.

    - Retries on 429 and 5xx (transient) and on network errors.
    - Returns {"status": 404} for 404.
    - Returns {"status": 403} for 403 so caller can fallback to Action API without retries.
    """
    settings = get_settings()
    url = f"{_WIKI_REST_BASE}/page/summary/{title}"
    headers = _wikimedia_headers()

    async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS, headers=headers) as client:
        response = await client.get(url)

    if response.status_code == 404:
        return {"status": 404}

    if response.status_code == 403:
        # Do not retry; caller will fallback to Action API.
        return {"status": 403}

    if response.status_code == 429 or 500 <= response.status_code < 600:
        # Treat as transient and let tenacity retry.
        raise WikimediaRestTransientError(
            f"Transient Wikimedia REST status {response.status_code} for '{title}'"
        )

    response.raise_for_status()
    return response.json()


@retry(
    reraise=True,
    stop=stop_after_attempt(get_settings().HTTP_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(httpx.HTTPError),
)
async def _fetch_html(title: str) -> str:
    """Fetch HTML for a page as a secondary fallback."""
    settings = get_settings()
    url = f"{_WIKI_REST_BASE}/page/html/{title}"
    headers = _wikimedia_headers()

    async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


async def _fetch_action_extract(title: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetch plain-text extract via the MediaWiki Action API.

    Returns (extract, page_title) where either may be None if not available.
    """
    settings = get_settings()
    headers = _wikimedia_headers()
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "redirects": 1,
        "titles": title,
        "format": "json",
    }

    async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS, headers=headers) as client:
        response = await client.get(_ACTION_API_URL, params=params)
        response.raise_for_status()
        data = response.json()

    query = data.get("query", {})
    pages = query.get("pages", {}) or {}
    for page in pages.values():
        extract = page.get("extract")
        page_title = page.get("title")
        if extract:
            return extract, page_title

    return None, None


def _strip_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return normalize_text(text)


async def fetch_wiki_documents(titles: List[str]) -> List[Document]:
    """Fetch documents from Wikipedia using REST with Action API fallback.

    For each title:
      1. Try REST summary endpoint with retries for transient errors.
      2. On 403, 429, or repeated 5xx/network failures, fallback to Action API extracts.
      3. Optionally enrich content with HTML when summary is empty.
    """
    documents: List[Document] = []

    for title in titles:
        safe_title = title.replace(" ", "_")
        summary_payload: Dict[str, Any] = {}
        summary_text: str = ""
        page_url: str = ""
        full_text: Optional[str] = None
        backend_used: str = "none"

        # Step 1: Try REST summary with retries for transient errors.
        try:
            summary_payload = await _fetch_summary(safe_title)
        except RetryError as exc:
            logger.warning(
                "Wikimedia REST summary failed after retries for '%s': %s. "
                "Falling back to Action API.",
                title,
                exc,
            )
            summary_payload = {"status": "rest_unavailable"}

        status_code = summary_payload.get("status")

        # Step 2: Fallback to Action API on 403, 429/5xx (rest_unavailable), etc.
        if status_code in {"rest_unavailable", 403}:
            try:
                extract, page_title = await _fetch_action_extract(title)
            except httpx.HTTPError as exc:
                logger.error(
                    "Wikimedia Action API failed for '%s': %s",
                    title,
                    exc,
                )
                # Propagate so the router can translate to 502.
                raise

            if not extract:
                logger.info(
                    "No extract returned by Wikimedia Action API for title='%s'",
                    title,
                )
                continue

            summary_text = normalize_text(extract)
            resolved_title = page_title or title
            safe_resolved = resolved_title.replace(" ", "_")
            page_url = f"https://en.wikipedia.org/wiki/{safe_resolved}"
            backend_used = "action"
        else:
            # Handle 404 or successful REST summary.
            if status_code == 404:
                # Fallback to HTML endpoint when REST summary reports 404.
                try:
                    html = await _fetch_html(safe_title)
                    full_text = _strip_html_to_text(html)
                    page_url = f"https://en.wikipedia.org/wiki/{safe_title}"
                    backend_used = "rest_html"
                except RetryError as exc:
                    logger.error(
                        "Failed to fetch Wikipedia HTML for '%s' after retries: %s",
                        title,
                        exc,
                    )
                    continue
            else:
                summary_text = (summary_payload.get("extract") or "").strip()
                content_urls = summary_payload.get("content_urls") or {}
                desktop = content_urls.get("desktop") or {}
                page_url = desktop.get("page") or f"https://en.wikipedia.org/wiki/{safe_title}"
                backend_used = "rest"

                # In some cases summary may be empty; attempt HTML fetch for richer text.
                if not summary_text:
                    try:
                        html = await _fetch_html(safe_title)
                        full_text = _strip_html_to_text(html)
                        backend_used = "rest_html"
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
            "wikimedia_backend": backend_used,
        }

        documents.append(Document(page_content=normalized, metadata=metadata))

    logger.info("Fetched %d Wikipedia documents", len(documents))
    return documents
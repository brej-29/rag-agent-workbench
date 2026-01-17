from typing import Any, Dict, List
from collections import Counter

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.ingest import (
    ArxivIngestRequest,
    IngestResponse,
    OpenAlexIngestRequest,
    WikiIngestRequest,
)
from app.services import dedupe as dedupe_service
from app.services import chunking as chunking_service
from app.services.ingestors.arxiv import fetch_arxiv_documents
from app.services.ingestors.openalex import fetch_openalex_documents
from app.services.ingestors.wiki import fetch_wiki_documents
from app.services.pinecone_store import upsert_records

logger = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])


async def _process_and_upsert(
    documents: List[Document],
    namespace: str,
    source: str,
    details: dict | None = None,
) -> IngestResponse:
    """Shared helper to chunk, dedupe and upsert documents."""
    if not documents:
        return IngestResponse(
            namespace=namespace,
            source=source,
            ingested_documents=0,
            ingested_chunks=0,
            skipped_documents=0,
            details=details or {"reason": "no_documents_after_filtering"},
        )

    records = chunking_service.documents_to_records(documents)
    records = dedupe_service.dedupe_records(records)

    total_upserted = await run_in_threadpool(upsert_records, namespace, records)

    return IngestResponse(
        namespace=namespace,
        source=source,
        ingested_documents=len(documents),
        ingested_chunks=total_upserted,
        skipped_documents=0,
        details=details,
    )


@router.post(
    "/arxiv",
    response_model=IngestResponse,
    summary="Ingest documents from arXiv",
    description="Fetches recent arXiv entries for a query and upserts them into Pinecone.",
)
async def ingest_arxiv(payload: ArxivIngestRequest) -> IngestResponse:
    settings = get_settings()
    namespace = payload.namespace or settings.PINECONE_NAMESPACE
    max_docs = min(payload.max_docs, 20)

    logger.info(
        "Starting arXiv ingestion query='%s' max_docs=%d namespace='%s'",
        payload.query,
        max_docs,
        namespace,
    )

    try:
        documents = await fetch_arxiv_documents(
            query=payload.query,
            max_results=max_docs,
            category=payload.category,
        )
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        logger.error("Upstream arXiv error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Upstream arXiv error: unable to retrieve content. "
            "Try again later.",
        ) from exc

    return await _process_and_upsert(documents, namespace=namespace, source="arxiv")


@router.post(
    "/openalex",
    response_model=IngestResponse,
    summary="Ingest documents from OpenAlex",
    description="Fetches works from OpenAlex for a query and upserts them into Pinecone.",
)
async def ingest_openalex(payload: OpenAlexIngestRequest) -> IngestResponse:
    settings = get_settings()
    namespace = payload.namespace or settings.PINECONE_NAMESPACE
    max_docs = min(payload.max_docs, 20)

    logger.info(
        "Starting OpenAlex ingestion query='%s' max_docs=%d namespace='%s'",
        payload.query,
        max_docs,
        namespace,
    )

    try:
        documents = await fetch_openalex_documents(
            query=payload.query,
            max_results=max_docs,
            mailto=payload.mailto,
        )
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        logger.error("Upstream OpenAlex error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Upstream OpenAlex error: unable to retrieve content. "
            "Try again later.",
        ) from exc

    return await _process_and_upsert(documents, namespace=namespace, source="openalex")


@router.post(
    "/wiki",
    response_model=IngestResponse,
    summary="Ingest documents from Wikipedia",
    description=(
        "Fetches articles from Wikipedia using the REST API with Action API fallback "
        "and upserts them into Pinecone."
    ),
)
async def ingest_wiki(payload: WikiIngestRequest) -> IngestResponse:
    settings = get_settings()
    namespace = payload.namespace or settings.PINECONE_NAMESPACE

    titles = payload.titles[:20]
    logger.info(
        "Starting Wikipedia ingestion titles=%d namespace='%s'",
        len(titles),
        namespace,
    )

    try:
        documents = await fetch_wiki_documents(titles=titles)
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        logger.error("Upstream Wikimedia error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=(
                "Upstream Wikimedia error: unable to retrieve content. "
                "Try again later or use Action API fallback."
            ),
        ) from exc

    # Summarise which backend was used (REST vs Action API) for debugging.
    backend_counts: Dict[str, int] = Counter(
        doc.metadata.get("wikimedia_backend", "unknown") for doc in documents
    )
    details: Dict[str, Any] = {"wikimedia_backend_counts": dict(backend_counts)}

    return await _process_and_upsert(
        documents, namespace=namespace, source="wiki", details=details
    )
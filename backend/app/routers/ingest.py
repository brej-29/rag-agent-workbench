from typing import List

from fastapi import APIRouter

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
) -> IngestResponse:
    """Shared helper to chunk, dedupe and upsert documents."""
    if not documents:
        return IngestResponse(
            namespace=namespace,
            source=source,
            ingested_documents=0,
            ingested_chunks=0,
            skipped_documents=0,
            details={"reason": "no_documents_after_filtering"},
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
    )


@router.post("/arxiv", response_model=IngestResponse)
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

    documents = await fetch_arxiv_documents(
        query=payload.query,
        max_results=max_docs,
        category=payload.category,
    )

    return await _process_and_upsert(documents, namespace=namespace, source="arxiv")


@router.post("/openalex", response_model=IngestResponse)
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

    documents = await fetch_openalex_documents(
        query=payload.query,
        max_results=max_docs,
        mailto=payload.mailto,
    )

    return await _process_and_upsert(documents, namespace=namespace, source="openalex")


@router.post("/wiki", response_model=IngestResponse)
async def ingest_wiki(payload: WikiIngestRequest) -> IngestResponse:
    settings = get_settings()
    namespace = payload.namespace or settings.PINECONE_NAMESPACE

    titles = payload.titles[:20]
    logger.info(
        "Starting Wikipedia ingestion titles=%d namespace='%s'",
        len(titles),
        namespace,
    )

    documents = await fetch_wiki_documents(titles=titles)

    return await _process_and_upsert(documents, namespace=namespace, source="wiki")
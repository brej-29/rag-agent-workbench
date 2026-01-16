from typing import Any, Dict, List

from fastapi import APIRouter, Query
from fastapi.concurrency import run_in_threadpool
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.documents import (
    DocumentsStatsResponse,
    NamespaceStat,
    UploadTextRequest,
    UploadTextResponse,
)
from app.services import chunking as chunking_service
from app.services import dedupe as dedupe_service
from app.services.normalize import make_doc_id, normalize_text, is_valid_document
from app.services.pinecone_store import describe_index_stats, upsert_records

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload-text", response_model=UploadTextResponse)
async def upload_text(payload: UploadTextRequest) -> UploadTextResponse:
    settings = get_settings()
    namespace = payload.namespace or settings.PINECONE_NAMESPACE

    normalized = normalize_text(payload.text)
    if not is_valid_document(normalized):
        logger.info(
            "Skipping manual upload for title='%s' due to insufficient length (len=%d)",
            payload.title,
            len(normalized),
        )
        return UploadTextResponse(
            namespace=namespace,
            source=payload.source,
            ingested_documents=0,
            ingested_chunks=0,
        )

    metadata: Dict[str, Any] = payload.metadata.copy() if payload.metadata else {}
    url = metadata.get("url", "")
    published = metadata.get("published", "")

    doc_id = make_doc_id(source=payload.source, title=payload.title, url=url)
    metadata.update(
        {
            "title": payload.title,
            "source": payload.source,
            "url": url,
            "published": published,
            "doc_id": doc_id,
        }
    )

    document = Document(page_content=normalized, metadata=metadata)
    records = chunking_service.documents_to_records([document])
    records = dedupe_service.dedupe_records(records)

    total_upserted = await run_in_threadpool(upsert_records, namespace, records)

    return UploadTextResponse(
        namespace=namespace,
        source=payload.source,
        ingested_documents=1,
        ingested_chunks=total_upserted,
    )


@router.get("/stats", response_model=DocumentsStatsResponse)
async def documents_stats(
    namespace: str | None = Query(
        default=None,
        description="Optional namespace filter; if omitted, stats for all namespaces are returned",
    ),
) -> DocumentsStatsResponse:
    raw_stats = await run_in_threadpool(describe_index_stats, namespace)

    stats: Dict[str, NamespaceStat] = {
        ns_name: NamespaceStat(vector_count=ns_info.get("vector_count", 0))
        for ns_name, ns_info in raw_stats.items()
    }

    return DocumentsStatsResponse(namespaces=stats)
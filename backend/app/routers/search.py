from typing import Any, Dict, List

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.search import SearchHit, SearchRequest, SearchResponse
from app.services.pinecone_store import search as pinecone_search

logger = get_logger(__name__)

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest) -> SearchResponse:
    settings = get_settings()
    namespace = payload.namespace or settings.PINECONE_NAMESPACE

    logger.info(
        "Received search request namespace='%s' top_k=%d",
        namespace,
        payload.top_k,
    )

    hits_raw: List[Dict[str, Any]] = await run_in_threadpool(
        pinecone_search,
        namespace,
        payload.query,
        payload.top_k,
        payload.filters,
        None,
    )

    hits: List[SearchHit] = []
    for hit in hits_raw:
        hit_id = hit.get("_id") or hit.get("id") or ""
        score = float(hit.get("_score") or hit.get("score") or 0.0)
        fields = hit.get("fields") or {}
        hits.append(
            SearchHit(
                id=hit_id,
                score=score,
                fields=fields,
            )
        )

    return SearchResponse(
        namespace=namespace,
        query=payload.query,
        top_k=payload.top_k,
        hits=hits,
    )
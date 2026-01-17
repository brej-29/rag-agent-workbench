from __future__ import annotations

from typing import Any, Dict, List, Optional

from pinecone import Pinecone

from app.core.config import Settings, get_settings
from app.core.errors import PineconeIndexConfigError
from app.core.logging import get_logger

logger = get_logger(__name__)

_index: Optional[Any] = None
_pc: Optional[Pinenecone] = None
_default_namespace: str = "dev"


def init_pinecone(settings: Optional[Settings] = None) -> None:
    """Initialise the Pinecone client and Index.

    This function should be called once on application startup. It validates
    that the configured index is an integrated embedding index so that
    `upsert_records` and `search` can be used without local embedding models.
    """
    global _index, _pc, _default_namespace

    if settings is None:
        settings = get_settings()

    text_field = settings.PINECONE_TEXT_FIELD.strip()
    if not text_field:
        raise ValueError("PINECONE_TEXT_FIELD must not be empty")

    logger.info(
        "Initialising Pinecone client (host targeting). host=%s text_field=%s",
        settings.PINECONE_HOST,
        text_field,
    )

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)

    # Validate index configuration via control plane using index name.
    index_model = pc.describe_index(settings.PINECONE_INDEX_NAME)
    embed_config = getattr(index_model, "embed", None)

    if not embed_config:
        raise PineconeIndexConfigError(
            "The configured Pinecone index is not an integrated embedding index.\n"
            "Create or reconfigure an index using Pinecone's integrated inference "
            "(e.g. via `create_index_for_model` or `configure_index(embed=...)`) so "
            "that embeddings are generated server-side. This keeps the backend "
            "lightweight without local embedding models."
        )

    if not getattr(index_model, "status", None) or not getattr(
        index_model.status, "ready", False
    ):
        raise PineconeIndexConfigError(
            f"Pinecone index '{settings.PINECONE_INDEX_NAME}' is not ready. "
            "Please wait for the index to become ready in the Pinecone console."
        )

    index_host = settings.PINECONE_HOST
    logger.info("Connecting to Pinecone index via host %s", index_host)

    index = pc.Index(host=index_host)

    _pc = pc
    _index = index
    _default_namespace = settings.PINECONE_NAMESPACE

    logger.info(
        "Pinecone initialised successfully with namespace=%s",
        _default_namespace,
    )


def get_index() -> Any:
    """Return the initialised Pinecone Index client."""
    if _index is None:
        raise RuntimeError("Pinecone index has not been initialised")
    return _index


def get_default_namespace() -> str:
    return _default_namespace


def upsert_records(
    namespace: str, records: List[Dict[str, Any]], batch_size: int = 64
) -> int:
    """Upsert records into Pinecone using the RECORDS API.

    Returns the total number of records reported as upserted.
    """
    if not records:
        return 0

    index = get_index()
    total_upserted = 0

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        logger.info(
            "Upserting %d records into namespace='%s' (batch %d/%d)",
            len(batch),
            namespace,
            i // batch_size + 1,
            (len(records) + batch_size - 1) // batch_size,
        )
        response = index.upsert_records(namespace=namespace, records=batch)

        # The response type may be a dict-like or model; try to read upserted count.
        upserted_count = getattr(response, "upserted_count", None)
        if upserted_count is None and isinstance(response, dict):
            upserted_count = response.get("upserted_count")

        if isinstance(upserted_count, int):
            total_upserted += upserted_count
        else:
            # Fallback: assume all batch records were upserted
            total_upserted += len(batch)

    logger.info(
        "Finished upserting %d records into namespace='%s'", total_upserted, namespace
    )
    return total_upserted


def search(
    namespace: str,
    query_text: str,
    top_k: int,
    filters: Optional[Dict[str, Any]] = None,
    fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Search Pinecone using integrated embedding search.

    Returns a list of hits, each containing `_id`, `_score`, and `fields`.
    """
    index = get_index()
    if fields is None:
        settings = get_settings()
        text_field = settings.PINECONE_TEXT_FIELD
        fields = [
            text_field,
            "title",
            "source",
            "url",
            "published",
            "doc_id",
            "chunk_id",
        ]

    query: Dict[str, Any] = {
        "inputs": {"text": query_text},
        "top_k": top_k,
    }
    if filters:
        query["filter"] = filters

    logger.info(
        "Searching Pinecone namespace='%s' top_k=%d filters=%s",
        namespace,
        top_k,
        filters,
    )

    response = index.search(namespace=namespace, query=query, fields=fields)

    # The response should match the SearchRecordsResponse shape.
    data: Dict[str, Any]
    if hasattr(response, "to_dict"):
        data = response.to_dict()  # type: ignore[assignment]
    elif hasattr(response, "model_dump"):
        data = response.model_dump()  # type: ignore[assignment]
    elif isinstance(response, dict):
        data = response
    else:
        # Fallback to __dict__
        data = getattr(response, "__dict__", {})

    result = data.get("result", data)
    hits = result.get("hits", []) or result.get("matches", [])

    if not isinstance(hits, list):
        return []

    return hits  # type: ignore[return-value]


def describe_index_stats(namespace_filter: Optional[str] = None) -> Dict[str, Any]:
    """Return index statistics, optionally filtered to a specific namespace."""
    index = get_index()
    stats = index.describe_index_stats()

    # stats.namespaces is a mapping of namespace -> object with vector_count
    namespaces: Dict[str, Any] = getattr(stats, "namespaces", {}) or {}
    result: Dict[str, Any] = {}
    for name, ns_info in namespaces.items():
        if namespace_filter and name != namespace_filter:
            continue

        vector_count = getattr(ns_info, "vector_count", None)
        if vector_count is None and isinstance(ns_info, dict):
            vector_count = ns_info.get("vector_count", 0)

        result[name] = {"vector_count": int(vector_count or 0)}

    return result
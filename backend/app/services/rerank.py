"""Pinecone hosted reranker integration (Inference API).

Two-stage retrieval design
--------------------------
Stage 1 — dense retrieval (pinecone_store.search):
    Retrieve RAG_RERANK_CANDIDATES candidates by cosine similarity.
    The existing cosine thresholds (RAG_MIN_SCORE, RAG_MIN_CHUNK_SCORE) are
    applied on these cosine scores exactly as before — they are NEVER applied
    to rerank scores, which live on a completely different scale/distribution.

Stage 2 — hosted rerank (this module):
    pc.inference.rerank() re-orders the cosine-floor survivors by semantic
    relevance; the caller takes the top_k result.

Model availability
------------------
Default model : bge-reranker-v2-m3
Dev-tier alt  : pinecone-rerank-v0 (lower throughput, check plan limits)

The operator MUST confirm the chosen model is available on their Pinecone plan
before enabling RAG_RERANK_ENABLED.  Plan availability varies by tier.
See https://docs.pinecone.io/models/overview for the current model catalogue.

Graceful degradation
--------------------
Any Inference API error is caught, logged, and the function returns the
pre-rerank cosine order (truncated to top_n).  Reranking is an enhancement —
not a hard dependency — so errors must not propagate to the user.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.core.logging import get_logger
from app.services.pinecone_store import get_pinecone_client

# Hard upper limit on candidates passed to the Pinecone hosted reranker.
# bge-reranker-v2-m3 (and pinecone-rerank-v0) cap at 100 documents per call;
# exceeding this returns an API error.  Operators setting RAG_RERANK_CANDIDATES
# above this value would otherwise waste the call (graceful degradation catches
# it, but the latency is already paid).
RERANK_CANDIDATES_MAX = 100

logger = get_logger(__name__)


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_n: int,
    model: str,
) -> List[Dict[str, Any]]:
    """Rerank chunks using the Pinecone hosted Inference rerank API.

    Exact SDK call
    --------------
    pc.inference.rerank(
        model=model,
        query=query,
        documents=[{"text": chunk["chunk_text"]} for chunk in chunks],
        top_n=min(top_n, len(chunks)),
        return_documents=True,
    )
    Result: RerankResult.data — list of RankedDocument with .index and .score.

    Parameters
    ----------
    chunks : cosine-floor survivors from filter_chunks_by_score().
             The floor MUST run before this function (cosine threshold ≠ rerank threshold).
    top_n  : final number of chunks to return (= state["top_k"]).
    model  : Pinecone inference model (from RAG_RERANK_MODEL setting).

    Returns
    -------
    Reordered sub-list (len ≤ top_n) with "rerank_score" key added to each chunk.
    Rerank scores are NOT comparable to cosine scores — do not threshold them.

    On any API error: logs the exception and returns chunks[:top_n] in cosine order.
    """
    if not chunks:
        return chunks

    pc = get_pinecone_client()
    documents = [{"text": chunk.get("chunk_text") or ""} for chunk in chunks]
    effective_top_n = min(top_n, len(documents))

    try:
        result = pc.inference.rerank(
            model=model,
            query=query,
            documents=documents,
            top_n=effective_top_n,
            return_documents=True,
        )

        reranked: List[Dict[str, Any]] = []
        for ranked_doc in result.data:
            orig_idx = int(ranked_doc.index)
            rerank_score = float(getattr(ranked_doc, "score", 0.0))
            chunk = chunks[orig_idx].copy()
            chunk["rerank_score"] = rerank_score
            reranked.append(chunk)

        logger.info(
            "Pinecone rerank completed model=%s candidates=%d top_n=%d returned=%d",
            model,
            len(chunks),
            top_n,
            len(reranked),
        )
        return reranked

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Pinecone rerank call failed (model=%s candidates=%d): %s "
            "— falling back to cosine order",
            model,
            len(chunks),
            exc,
        )
        return chunks[:top_n]

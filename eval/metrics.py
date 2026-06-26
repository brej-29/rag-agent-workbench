"""
Retrieval quality metrics — pure functions, no model/embedder/LLM imports.

All metrics operate on ranked lists of retrieved document IDs versus a set of
human-labeled relevant document IDs.  They are intentionally self-contained so
they can be unit-tested without any network access or backend dependencies.
"""

from __future__ import annotations

import math
from typing import Sequence, Set


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """Fraction of relevant documents found in the top-k retrieved results.

    Formula:
        recall@k = |retrieved_top_k ∩ relevant| / |relevant|

    Returns 0.0 when relevant_ids is empty (undefined, treated as no signal).

    Args:
        retrieved_ids: Ranked list of document IDs returned by the retriever,
            ordered from most to least relevant.  May contain more than k items;
            only the first k are considered.
        relevant_ids: Set of document IDs judged relevant by a human annotator.
        k: Cutoff rank.

    Returns:
        Float in [0.0, 1.0].
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def mrr(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
) -> float:
    """Reciprocal rank of the first relevant document in the ranked list.

    Formula:
        RR = 1 / rank_of_first_relevant   (rank is 1-indexed)
        RR = 0                             if no relevant document appears

    For a single query MRR == RR.  Average across queries to get Mean RR.

    Returns 0.0 when retrieved_ids is empty or no relevant document is found.

    Args:
        retrieved_ids: Ranked list of document IDs, most relevant first.
        relevant_ids: Set of human-labeled relevant document IDs.

    Returns:
        Float in [0.0, 1.0].
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """Normalised Discounted Cumulative Gain at rank k (binary relevance).

    Formula (binary gain):
        DCG@k  = Σ_{i=1}^{k}  rel_i / log2(i + 1)
        IDCG@k = Σ_{i=1}^{min(|relevant|, k)}  1 / log2(i + 1)
        nDCG@k = DCG@k / IDCG@k

    where rel_i = 1 if retrieved_ids[i-1] ∈ relevant_ids else 0.
    Returns 0.0 when IDCG@k == 0 (no relevant documents exist).

    Args:
        retrieved_ids: Ranked list of document IDs, most relevant first.
        relevant_ids: Set of human-labeled relevant document IDs.
        k: Cutoff rank.

    Returns:
        Float in [0.0, 1.0].
    """
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1)
        if doc_id in relevant_ids
    )

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """Fraction of the top-k retrieved results that are relevant.

    Formula:
        precision@k = |retrieved_top_k ∩ relevant| / k

    Unlike recall@k this normalises by k (the retrieval budget), not by the
    number of relevant documents, so it directly measures top-of-list density.
    When k > len(retrieved_ids) the missing positions count as non-relevant
    (i.e. precision is penalised for returning fewer than k items).

    Returns 0.0 when k <= 0.

    Args:
        retrieved_ids: Ranked list of document IDs returned by the retriever,
            ordered from most to least relevant.
        relevant_ids: Set of document IDs judged relevant by a human annotator.
        k: Cutoff rank (must be >= 1).

    Returns:
        Float in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    return len(set(top_k) & relevant_ids) / k

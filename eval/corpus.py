"""
Fixed mini-corpus definition for reproducible retrieval evaluation.

REPRODUCIBILITY GUARANTEE
--------------------------
Wikipedia articles are pinned by exact title.  Because the backend computes
document IDs as SHA256("wiki|{title}|{url}"), ingesting the same title from
the same Wikipedia URL always produces the same doc_id — making the golden set
stable across re-ingestions.

arXiv / OpenAlex queries are included for corpus breadth but their results vary
by date (different recent papers each run).  Use Wikipedia titles as the anchor
for golden-set entries; treat arXiv/OpenAlex results as supplemental context
that improves retrieval without being tied to specific doc_ids.

INGESTION NAMESPACE
--------------------
The corpus is ingested into the "eval" Pinecone namespace (EVAL_NAMESPACE below)
to keep it isolated from the development namespace ("dev").  Pass
--namespace eval to both setup_corpus.py and run.py (the default for both).
"""

from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Wikipedia titles — STABLE.  These produce deterministic doc_ids.
# Each title must match the exact Wikipedia article title (case-sensitive).
# ---------------------------------------------------------------------------
WIKI_TITLES: List[str] = [
    "Retrieval-augmented generation",
    "Vector database",
    "Large language model",
    "Transformer (deep learning architecture)",
    "Approximate nearest neighbor search",
    "Information retrieval",
    "Semantic search",
    "Question answering",
]

# ---------------------------------------------------------------------------
# arXiv search queries — NON-DETERMINISTIC across dates.
# Results vary; do not pin golden-set relevant_doc_ids to arXiv doc_ids unless
# you also commit the exact fetched titles to this file.
# ---------------------------------------------------------------------------
ARXIV_QUERIES: List[Dict[str, Any]] = [
    {"query": "retrieval augmented generation survey", "max_docs": 5},
    {"query": "dense passage retrieval open domain", "max_docs": 5},
    {"query": "approximate nearest neighbor search", "max_docs": 5},
]

# ---------------------------------------------------------------------------
# OpenAlex queries — NON-DETERMINISTIC across dates (same caveat as arXiv).
# Requires a mailto contact address passed at ingest time.
# ---------------------------------------------------------------------------
OPENALEX_QUERIES: List[Dict[str, Any]] = [
    {"query": "retrieval augmented generation", "max_docs": 5},
    {"query": "vector similarity search embedding", "max_docs": 5},
]

# Pinecone namespace used for all eval ingestion and querying.
EVAL_NAMESPACE: str = "eval"

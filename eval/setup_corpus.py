#!/usr/bin/env python3
"""
One-time corpus ingestion for reproducible retrieval evaluation.

Reads the fixed corpus definition from eval/corpus.py and ingests it into the
"eval" Pinecone namespace.  Prints the doc_id of every ingested document so
you can copy them into eval/golden.jsonl.

This script issues Pinecone UPSERT calls — it is deliberately separate from
eval/run.py (which is read-only).  Run it once, then label eval/golden.jsonl
using the printed doc_ids (read the document content to decide relevance;
do NOT copy from retriever output).

Usage:
    python eval/setup_corpus.py [--namespace eval] [--mailto your@email.com]

GROQ_API_KEY is NOT required.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Path setup — must come before any backend or eval imports.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_BACKEND_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(_BACKEND_DIR / ".env", override=False)
except ImportError:
    pass

from eval.corpus import ARXIV_QUERIES, OPENALEX_QUERIES, WIKI_TITLES  # noqa: E402


def _print_doc_ids(source: str, records: List[Dict[str, Any]]) -> None:
    seen_docs: dict[str, str] = {}
    for rec in records:
        doc_id = rec.get("doc_id", "")
        title = rec.get("title", "")
        if doc_id and doc_id not in seen_docs:
            seen_docs[doc_id] = title
    for doc_id, title in seen_docs.items():
        print(f"  [{source}] doc_id={doc_id}  title={title!r}")


async def _ingest_wiki(namespace: str) -> int:
    from app.services import chunking as chunking_service   # noqa: PLC0415
    from app.services import dedupe as dedupe_service       # noqa: PLC0415
    from app.services.ingestors.wiki import fetch_wiki_documents  # noqa: PLC0415
    from app.services.pinecone_store import upsert_records  # noqa: PLC0415

    print(f"\n=== Wikipedia ({len(WIKI_TITLES)} titles) ===")
    documents = await fetch_wiki_documents(titles=WIKI_TITLES)
    records = chunking_service.documents_to_records(documents)
    records = dedupe_service.dedupe_records(records)
    _print_doc_ids("wiki", records)
    total = upsert_records(namespace=namespace, records=records)
    print(f"  Upserted {total} chunks from {len(documents)} Wikipedia documents.")
    return total


async def _ingest_arxiv(namespace: str) -> int:
    from app.services import chunking as chunking_service       # noqa: PLC0415
    from app.services import dedupe as dedupe_service           # noqa: PLC0415
    from app.services.ingestors.arxiv import fetch_arxiv_documents  # noqa: PLC0415
    from app.services.pinecone_store import upsert_records      # noqa: PLC0415

    total = 0
    print(f"\n=== arXiv ({len(ARXIV_QUERIES)} queries) ===")
    for spec in ARXIV_QUERIES:
        docs = await fetch_arxiv_documents(
            query=spec["query"],
            max_results=spec["max_docs"],
        )
        records = chunking_service.documents_to_records(docs)
        records = dedupe_service.dedupe_records(records)
        _print_doc_ids("arxiv", records)
        n = upsert_records(namespace=namespace, records=records)
        total += n
        print(f"  [{spec['query']!r}] Upserted {n} chunks from {len(docs)} docs.")
    return total


async def _ingest_openalex(namespace: str, mailto: str) -> int:
    from app.services import chunking as chunking_service           # noqa: PLC0415
    from app.services import dedupe as dedupe_service               # noqa: PLC0415
    from app.services.ingestors.openalex import fetch_openalex_documents  # noqa: PLC0415
    from app.services.pinecone_store import upsert_records          # noqa: PLC0415

    total = 0
    print(f"\n=== OpenAlex ({len(OPENALEX_QUERIES)} queries) ===")
    for spec in OPENALEX_QUERIES:
        docs = await fetch_openalex_documents(
            query=spec["query"],
            max_results=spec["max_docs"],
            mailto=mailto,
        )
        records = chunking_service.documents_to_records(docs)
        records = dedupe_service.dedupe_records(records)
        _print_doc_ids("openalex", records)
        n = upsert_records(namespace=namespace, records=records)
        total += n
        print(f"  [{spec['query']!r}] Upserted {n} chunks from {len(docs)} docs.")
    return total


async def _main_async(namespace: str, mailto: str, skip_arxiv: bool, skip_openalex: bool) -> None:
    from app.core.config import get_settings              # noqa: PLC0415
    from app.services.pinecone_store import init_pinecone # noqa: PLC0415

    settings = get_settings()
    print(f"Initialising Pinecone (namespace='{namespace}')...")
    init_pinecone(settings)

    total = 0
    total += await _ingest_wiki(namespace)

    if not skip_arxiv:
        total += await _ingest_arxiv(namespace)
    else:
        print("\n[SKIP] arXiv ingestion (--skip-arxiv)")

    if not skip_openalex:
        if not mailto:
            print(
                "\n[SKIP] OpenAlex ingestion — pass --mailto your@email.com to enable.",
                file=sys.stderr,
            )
        else:
            total += await _ingest_openalex(namespace, mailto)
    else:
        print("\n[SKIP] OpenAlex ingestion (--skip-openalex)")

    print(f"\n=== Done: {total} total chunks upserted into namespace='{namespace}' ===")
    print(
        "\nNEXT STEPS:\n"
        "1. Review the doc_ids printed above.\n"
        "2. For each golden query in eval/golden.jsonl, read the corresponding\n"
        "   documents (by title/URL) and decide which are genuinely relevant.\n"
        "3. Replace the PLACEHOLDER strings in relevant_doc_ids with the real\n"
        "   doc_id values from step 1.\n"
        "4. Add >=20-30 total entries and run: make eval\n"
        "\nDO NOT copy doc_ids from retriever output — that encodes the embedder's\n"
        "own biases and makes recall@k meaningless (circular validation)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest eval corpus into Pinecone (one-time setup).")
    parser.add_argument("--namespace", default="eval", help="Target Pinecone namespace (default: eval)")
    parser.add_argument("--mailto", default="", help="Contact email for OpenAlex API")
    parser.add_argument("--skip-arxiv", action="store_true", help="Skip arXiv ingestion")
    parser.add_argument("--skip-openalex", action="store_true", help="Skip OpenAlex ingestion")
    args = parser.parse_args()

    asyncio.run(
        _main_async(
            namespace=args.namespace,
            mailto=args.mailto,
            skip_arxiv=args.skip_arxiv,
            skip_openalex=args.skip_openalex,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

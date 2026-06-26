#!/usr/bin/env python3
"""
Corpus manifest generator and validator for the eval Pinecone namespace.

BOTH operations are READ-ONLY — no upsert, no delete.

generate
--------
Snapshot the live eval-namespace index into eval/corpus_manifest.json.
Enumerates all vector IDs via index.list(), deduplicates to unique doc_ids
(vectors follow the "<doc_id>:<chunk_idx>" naming scheme used by
chunking.documents_to_records), fetches metadata for one representative
chunk per doc_id, and writes a committed JSON manifest.

validate
--------
Compare the committed eval/corpus_manifest.json against the live index.
Reports missing docs (in manifest but absent from index) and extra docs
(in index but absent from manifest).  Exit code 0 means no drift.

Usage
-----
  # Snapshot the current index state
  python eval/corpus_manifest.py generate [--namespace eval] [--out eval/corpus_manifest.json]

  # Detect drift (no mutations)
  python eval/corpus_manifest.py validate [--namespace eval] [--manifest eval/corpus_manifest.json]

Requirements
------------
  PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_HOST — set via env or backend/.env
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

# ---------------------------------------------------------------------------
# Path setup — allow importing from backend/app/
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(_BACKEND_DIR / ".env", override=False)
except ImportError:
    pass

_DEFAULT_MANIFEST = _REPO_ROOT / "eval" / "corpus_manifest.json"


# ---------------------------------------------------------------------------
# Pure-function helpers — tested in tests/test_corpus_manifest.py
# ---------------------------------------------------------------------------

def parse_doc_id(vector_id: str) -> str:
    """Extract the doc_id from a vector ID of the form '<doc_id>:<chunk_idx>'."""
    return vector_id.rsplit(":", 1)[0]


def compute_drift(
    manifest_doc_ids: Set[str],
    live_doc_ids: Set[str],
) -> Dict[str, List[str]]:
    """Return missing and extra sets given two collections of doc_ids.

    missing: in manifest, absent from live index.
    extra:   in live index, absent from manifest.
    """
    missing = sorted(manifest_doc_ids - live_doc_ids)
    extra = sorted(live_doc_ids - manifest_doc_ids)
    return {"missing": missing, "extra": extra}


# ---------------------------------------------------------------------------
# Pinecone helpers
# ---------------------------------------------------------------------------

def _collect_all_vector_ids(index: Any, namespace: str) -> List[str]:
    """Return every vector ID in the given namespace via paginated list()."""
    all_ids: List[str] = []
    for page in index.list(namespace=namespace):
        # Pinecone SDK: list() yields lists of ID strings
        if isinstance(page, list):
            all_ids.extend(page)
        else:
            # Older SDK returns an object with .vectors / .ids attribute
            ids = getattr(page, "ids", None) or getattr(page, "vectors", [])
            all_ids.extend(ids)
    return all_ids


def _dedupe_to_doc_ids(vector_ids: List[str]) -> Set[str]:
    """Derive the set of unique doc_ids from a list of vector IDs."""
    return {parse_doc_id(vid) for vid in vector_ids}


def _fetch_metadata_batch(
    index: Any,
    ids: List[str],
    namespace: str,
) -> Dict[str, Dict[str, Any]]:
    """Fetch metadata for a batch of vector IDs.  Returns {id: metadata_dict}."""
    resp = index.fetch(ids=ids, namespace=namespace)
    vectors = getattr(resp, "vectors", {}) or {}
    return {
        vid: dict(getattr(v, "metadata", {}) or {})
        for vid, v in vectors.items()
    }


def _build_doc_record(
    doc_id: str,
    chunk_id: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "first_chunk_id": chunk_id,
        "source_type": metadata.get("source") or "unknown",
        "source_id": metadata.get("url") or metadata.get("title") or "",
        "title": metadata.get("title") or "",
        "url": metadata.get("url") or "",
        "published": metadata.get("published") or "",
    }


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def _cmd_generate(args: argparse.Namespace) -> None:
    from app.core.config import get_settings  # noqa: PLC0415
    from pinecone import Pinecone  # noqa: PLC0415
    from app.services.pinecone_store import init_pinecone  # noqa: PLC0415

    settings = get_settings()
    init_pinecone(settings)

    from app.services.pinecone_store import _pc  # noqa: PLC0415, SLF001
    index = _pc.Index(host=settings.PINECONE_HOST)

    namespace: str = args.namespace
    out_path = Path(args.out)

    print(f"Listing all vector IDs in namespace='{namespace}' …")
    vector_ids = _collect_all_vector_ids(index, namespace)
    print(f"  Found {len(vector_ids)} total chunk vectors.")

    doc_id_to_chunk: Dict[str, str] = {}
    for vid in vector_ids:
        did = parse_doc_id(vid)
        if did not in doc_id_to_chunk:
            doc_id_to_chunk[did] = vid

    print(f"  {len(doc_id_to_chunk)} unique doc_ids.")

    # Fetch metadata for one representative chunk per doc_id (batched)
    representative_ids = list(doc_id_to_chunk.values())
    id_to_meta: Dict[str, Dict[str, Any]] = {}
    batch_size = 100  # Pinecone fetch limit
    for i in range(0, len(representative_ids), batch_size):
        batch = representative_ids[i : i + batch_size]
        id_to_meta.update(_fetch_metadata_batch(index, batch, namespace))
        print(f"  Fetched metadata for chunks {i + 1}–{min(i + batch_size, len(representative_ids))} …")

    documents: List[Dict[str, Any]] = []
    for doc_id, chunk_id in sorted(doc_id_to_chunk.items()):
        meta = id_to_meta.get(chunk_id, {})
        documents.append(_build_doc_record(doc_id, chunk_id, meta))

    # Sort by source_type then title for stable diffs
    documents.sort(key=lambda d: (d["source_type"], d["title"].lower()))

    manifest: Dict[str, Any] = {
        "_note": (
            "Generated by `make corpus-manifest`.  NOT in CI — regenerate manually "
            "when the eval corpus changes.  arXiv and OpenAlex entries are non-deterministic "
            "(live sources change); Wikipedia entries are stable (SHA256 of wiki|title|url)."
        ),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "namespace": namespace,
        "index_name": settings.PINECONE_INDEX_NAME,
        "method": "index.list() for IDs + index.fetch() for metadata (read-only)",
        "document_count": len(documents),
        "documents": documents,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"\nManifest written to {out_path}  ({len(documents)} documents)")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def _cmd_validate(args: argparse.Namespace) -> None:
    from app.core.config import get_settings  # noqa: PLC0415
    from pinecone import Pinecone  # noqa: PLC0415
    from app.services.pinecone_store import init_pinecone  # noqa: PLC0415

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(
            f"ERROR: manifest file not found at {manifest_path}\n"
            "Run `make corpus-manifest` to generate it.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as fh:
        manifest: Dict[str, Any] = json.load(fh)

    manifest_doc_ids: Set[str] = {doc["doc_id"] for doc in manifest.get("documents", [])}
    print(
        f"Manifest: {len(manifest_doc_ids)} doc_ids  "
        f"(generated_at={manifest.get('generated_at', 'unknown')})"
    )

    settings = get_settings()
    init_pinecone(settings)

    from app.services.pinecone_store import _pc  # noqa: PLC0415, SLF001
    index = _pc.Index(host=settings.PINECONE_HOST)

    namespace: str = args.namespace
    print(f"Querying live index namespace='{namespace}' …")
    vector_ids = _collect_all_vector_ids(index, namespace)
    live_doc_ids = _dedupe_to_doc_ids(vector_ids)
    print(f"  Live index: {len(live_doc_ids)} unique doc_ids across {len(vector_ids)} chunks.")

    drift = compute_drift(manifest_doc_ids, live_doc_ids)

    if not drift["missing"] and not drift["extra"]:
        print("\nOK — no drift detected.  Live index matches the committed manifest.")
        return

    if drift["missing"]:
        print(f"\nMISSING from live index ({len(drift['missing'])} doc_ids in manifest but absent from index):")
        for did in drift["missing"]:
            print(f"  {did}")

    if drift["extra"]:
        print(f"\nEXTRA in live index ({len(drift['extra'])} doc_ids in index but absent from manifest):")
        for did in drift["extra"]:
            print(f"  {did}")

    print(
        "\nDrift detected.  Run `make corpus-manifest` to regenerate the manifest "
        "from the live index, or re-ingest to restore expected documents."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Corpus manifest generator and validator (read-only Pinecone operations).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Snapshot the live index into a manifest JSON file.")
    gen.add_argument(
        "--namespace",
        default="eval",
        help="Pinecone namespace to snapshot (default: eval).",
    )
    gen.add_argument(
        "--out",
        default=str(_DEFAULT_MANIFEST),
        help=f"Output path for the manifest JSON (default: {_DEFAULT_MANIFEST}).",
    )

    val = sub.add_parser(
        "validate",
        help="Compare the committed manifest to the live index and report drift.",
    )
    val.add_argument(
        "--namespace",
        default="eval",
        help="Pinecone namespace to validate against (default: eval).",
    )
    val.add_argument(
        "--manifest",
        default=str(_DEFAULT_MANIFEST),
        help=f"Path to the committed manifest JSON (default: {_DEFAULT_MANIFEST}).",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        _cmd_generate(args)
    elif args.command == "validate":
        _cmd_validate(args)


if __name__ == "__main__":
    main()

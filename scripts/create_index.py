#!/usr/bin/env python3
"""
Bootstrap script: create the Pinecone integrated embedding index.

This script pins the embedding model and vector dimension EXPLICITLY in code so
the index is reproducible from configuration, not console clicks.  Model and
dimension come from Settings (app.core.config) — the single source of truth
shared with the running application.

Safety contract
---------------
- IDEMPOTENT by default: if the index already exists, the script exits with a
  clear message and does NOT modify or overwrite it.
- Destructive recreate is behind an explicit double opt-in (--recreate AND
  --confirm-recreate) and prints a loud warning before proceeding.

Usage
-----
  # Create index (safe — skips if already exists)
  python scripts/create_index.py

  # Specify a different index name or cloud region
  python scripts/create_index.py --index-name my-index --cloud aws --region us-east-1

  # Recreate (DESTRUCTIVE — deletes the existing index first)
  python scripts/create_index.py --recreate --confirm-recreate

Requirements
------------
  PINECONE_API_KEY, PINECONE_INDEX_NAME (or --index-name), PINECONE_HOST
  (PINECONE_HOST is only needed at runtime, not for index creation).
  Set via environment variables or backend/.env.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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

from app.core.config import get_settings  # noqa: E402


# ---------------------------------------------------------------------------
# Constants — sourced from Settings; documented here for readability
# ---------------------------------------------------------------------------
# Cloud/region defaults.  These match the serverless config assumed during
# initial setup; override via CLI flags if your index is in a different region.
_DEFAULT_CLOUD = "aws"
_DEFAULT_REGION = "us-east-1"


def _describe_existing(pc: object, index_name: str) -> bool:
    """Return True if the index already exists, False otherwise."""
    try:
        existing = [idx.name for idx in pc.list_indexes().indexes]  # type: ignore[attr-defined]
        return index_name in existing
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] Could not list indexes: {exc}", file=sys.stderr)
        return False


def _create(pc: object, index_name: str, settings: object, cloud: str, region: str) -> None:
    """Issue the create_index_for_model call with explicit model + dimension."""
    # Settings fields are the single source of truth — do not inline magic numbers here.
    model: str = settings.PINECONE_EMBED_MODEL          # type: ignore[attr-defined]
    dimension: int = settings.PINECONE_EMBED_DIMENSION  # type: ignore[attr-defined]
    text_field: str = settings.PINECONE_TEXT_FIELD      # type: ignore[attr-defined]

    print(f"  Creating index '{index_name}' …")
    print(f"    model     = {model}")
    print(f"    dimension = {dimension}")
    print(f"    metric    = cosine")
    print(f"    text_field= {text_field}  (field_map key)")
    print(f"    cloud     = {cloud}  region = {region}")

    pc.create_index_for_model(  # type: ignore[attr-defined]
        name=index_name,
        cloud=cloud,
        region=region,
        embed={
            "model": model,
            "field_map": {"text": text_field},
            "metric": "cosine",
            "dimension": dimension,
        },
    )
    print("  Waiting for index to become ready …")
    for attempt in range(30):
        try:
            desc = pc.describe_index(index_name)  # type: ignore[attr-defined]
            ready = getattr(getattr(desc, "status", None), "ready", False)
            if ready:
                print(f"  Index '{index_name}' is ready.")
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
        print(f"  … still waiting ({(attempt + 1) * 5}s elapsed)")
    print("  [WARN] Timed out waiting for index to become ready. Check the Pinecone console.")


def _delete(pc: object, index_name: str) -> None:
    print(f"  Deleting index '{index_name}' …", flush=True)
    pc.delete_index(index_name)  # type: ignore[attr-defined]
    # Brief pause to let the control-plane propagate the deletion.
    time.sleep(5)
    print(f"  Index '{index_name}' deleted.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the Pinecone integrated embedding index (idempotent by default).",
    )
    parser.add_argument(
        "--index-name",
        default=None,
        help="Pinecone index name.  Defaults to PINECONE_INDEX_NAME from settings.",
    )
    parser.add_argument(
        "--cloud",
        default=_DEFAULT_CLOUD,
        help=f"Serverless cloud provider (default: {_DEFAULT_CLOUD}).",
    )
    parser.add_argument(
        "--region",
        default=_DEFAULT_REGION,
        help=f"Serverless cloud region (default: {_DEFAULT_REGION}).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        default=False,
        help="DELETE the existing index and recreate it.  DESTRUCTIVE — also requires --confirm-recreate.",
    )
    parser.add_argument(
        "--confirm-recreate",
        action="store_true",
        default=False,
        help="Second opt-in required for --recreate.  Both flags must be present.",
    )
    args = parser.parse_args()

    settings = get_settings()
    index_name: str = args.index_name or settings.PINECONE_INDEX_NAME

    from pinecone import Pinecone  # noqa: PLC0415
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)

    exists = _describe_existing(pc, index_name)

    # ------------------------------------------------------------------
    # Normal (non-destructive) path
    # ------------------------------------------------------------------
    if not args.recreate:
        if exists:
            desc = pc.describe_index(index_name)
            live_model = getattr(getattr(desc, "embed", None), "model", "unknown")
            live_dim = getattr(getattr(desc, "embed", None), "dimension", "unknown")
            print(
                f"Index '{index_name}' already exists — skipping creation.\n"
                f"  live model={live_model}  dimension={live_dim}\n"
                f"  Expected: model={settings.PINECONE_EMBED_MODEL}  "
                f"dimension={settings.PINECONE_EMBED_DIMENSION}\n"
                "To recreate it, run with --recreate --confirm-recreate (DESTRUCTIVE)."
            )
        else:
            _create(pc, index_name, settings, args.cloud, args.region)
        return

    # ------------------------------------------------------------------
    # Destructive recreate — requires BOTH flags
    # ------------------------------------------------------------------
    if not args.confirm_recreate:
        print(
            "ERROR: --recreate requires --confirm-recreate as a second opt-in.\n"
            "Both flags must be present to prevent accidental data loss.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"\n*** WARNING: --recreate will DELETE '{index_name}' and all its data. ***\n"
        "This is irreversible.  Proceeding in 5 seconds …"
    )
    time.sleep(5)

    if exists:
        _delete(pc, index_name)
    else:
        print(f"  Index '{index_name}' does not exist — skipping delete step.")

    _create(pc, index_name, settings, args.cloud, args.region)


if __name__ == "__main__":
    main()

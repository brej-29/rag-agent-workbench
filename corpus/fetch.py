"""Download each source once into corpus/raw/ and describe it for the manifest."""
import datetime
import hashlib
from pathlib import Path
from typing import Optional

import httpx

from corpus.sources import Source

RAW_DIR = Path(__file__).resolve().parent / "raw"


def raw_path(source: Source, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"{source.key}.{'json' if source.accept == 'application/json' else 'xhtml'}"


def fetch(source: Source, raw_dir: Path = RAW_DIR, client: Optional[httpx.Client] = None) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_path(source, raw_dir)
    if not path.exists():
        client = client or httpx.Client(timeout=120, follow_redirects=True)
        resp = client.get(source.url, headers={"Accept": source.accept, "Accept-Language": "eng",
                                               "User-Agent": "rag-agent-workbench corpus builder"})
        resp.raise_for_status()
        if not resp.content:
            raise RuntimeError(f"{source.key}: empty response from {source.url}")
        path.write_bytes(resp.content)
    data = path.read_bytes()
    return {"key": source.key, "url": source.url, "file": path.name, "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data), "retrieved": datetime.date.fromtimestamp(path.stat().st_mtime).isoformat(),
            "licence": source.licence}

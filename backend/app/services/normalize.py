import re
from hashlib import sha256
from typing import Optional


_MIN_DOC_CHARS = 200


def normalize_text(text: str) -> str:
    """Normalize text by collapsing whitespace and stripping edges."""
    # Replace multiple whitespace (including newlines) with a single space
    normalized = re.sub(r"\s+", " ", text or "").strip()
    return normalized


def is_valid_document(text: str, min_chars: int = _MIN_DOC_CHARS) -> bool:
    """Return True if the text is long enough to be considered useful."""
    return len(text) >= min_chars


def make_doc_id(source: str, title: str, url: Optional[str] = None) -> str:
    """Create a stable SHA256-based document id from source, title, and URL."""
    base = f"{source}|{title}|{url or ''}"
    digest = sha256(base.encode("utf-8")).hexdigest()
    return digest
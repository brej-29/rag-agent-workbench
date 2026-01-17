from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Tuple

try:
    from docling.document_converter import DocumentConverter
except ImportError:  # pragma: no cover - optional dependency
    DocumentConverter = None  # type: ignore[assignment]


def convert_uploaded_file_to_text(uploaded_file) -> Tuple[str, Dict[str, Any]]:
    """Convert an uploaded Streamlit file to text/markdown.

    - For .txt and .md, returns raw UTF-8 text.
    - For other supported formats (PDF/Office/HTML), uses Docling when installed.
    - Raises a RuntimeError with a user-friendly message when Docling is required
      but not installed.
    """
    filename = uploaded_file.name
    ext = Path(filename).suffix.lower().lstrip(".")
    size_bytes = getattr(uploaded_file, "size", None)
    content_type = getattr(uploaded_file, "type", None)

    metadata: Dict[str, Any] = {
        "filename": filename,
        "ext": ext,
        "size_bytes": size_bytes,
        "content_type": content_type,
    }

    # Plain text / markdown: read directly.
    if ext in {"txt", "md"}:
        raw_bytes = uploaded_file.read()
        text = raw_bytes.decode("utf-8", errors="ignore")
        metadata["converted_by"] = "raw"
        return text, metadata

    # Rich formats: require Docling.
    if DocumentConverter is None:
        raise RuntimeError(
            "Docling is not installed; conversion for this file type is unavailable. "
            "Install docling (e.g. `pip install docling`) or upload a .md/.txt file."
        )

    # Persist to a temporary file so Docling can read it from disk.
    with NamedTemporaryFile(delete=True, suffix=f".{ext}") as tmp:
        # Streamlit's UploadedFile exposes getbuffer() for zero-copy writes.
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()

        converter = DocumentConverter()
        result = converter.convert(tmp.name)

        try:
            text = result.document.export_to_markdown()
        except Exception:  # noqa: BLE001
            # Fallback to plain text if markdown export is not available.
            text = result.document.export_to_text()

    metadata["converted_by"] = "docling"
    return text, metadata
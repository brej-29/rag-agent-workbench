from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
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
            "Docling is required to convert PDFs/Office docs. Install docling "
            "(e.g. `pip install docling`) or upload a .txt/.md file instead."
        )

    # Persist to a temporary file so Docling can read it from disk.
    # On Windows, NamedTemporaryFile keeps the file handle open which prevents
    # libraries from reopening the file by path, so we use a closed file in a
    # temporary directory instead.
    tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    # Preserve extension where possible; fall back to .bin if missing.
    suffix = ext or "bin"
    file_path = os.path.join(tmp_dir, f"upload.{suffix}")

    text = ""
    try:
        # Write file contents and ensure the handle is closed before Docling reads.
        data = uploaded_file.getbuffer()
        with open(file_path, "wb") as f:
            f.write(data)

        converter = DocumentConverter()

        # Defensive retry around PermissionError, which can occur on Windows if a
        # temp file lock is detected.
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                result = converter.convert(file_path)
                doc = result.document
                try:
                    text = doc.export_to_markdown()
                except Exception:  # noqa: BLE001
                    text = ""
                if not text:
                    text = doc.export_to_text()
                break
            except PermissionError as exc:  # pragma: no cover - platform-specific
                last_exc = exc
                if attempt == 0:
                    # Give the OS a brief moment to release any lingering handles.
                    time.sleep(0.2)
                    continue
                raise RuntimeError(
                    "Windows temp-file lock detected while converting the uploaded "
                    "file with Docling. Ensure no other process is locking the "
                    "temp directory and retry the upload."
                ) from exc

        if last_exc is not None and not text:
            # If we somehow exited without text and had a PermissionError earlier.
            raise last_exc

    finally:
        # Best-effort cleanup of the temp file and directory, with a small retry
        # to handle transient locks on Windows.
        for _ in range(2):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                break
            except PermissionError:  # pragma: no cover - platform-specific
                time.sleep(0.2)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    metadata["converted_by"] = "docling"
    return text, metadata
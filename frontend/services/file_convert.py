from __future__ import annotations

import io
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import streamlit as st

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None  # type: ignore[assignment]

try:
    from docling.document_converter import DocumentConverter
except ImportError:  # pragma: no cover - optional dependency
    DocumentConverter = None  # type: ignore[assignment]


@st.cache_resource
def get_docling_converter():
    """Return a cached Docling converter with PDF options tuned for speed.

    - Disables OCR and table structure extraction to avoid RapidOCR overhead.
    - Forces backend text extraction for PDFs.
    """
    if DocumentConverter is None:
        raise RuntimeError(
            "Docling is not installed; conversion for this file type is unavailable. "
            "Docling is required to convert PDFs/Office docs. Install docling "
            "(e.g. `pip install docling`) or upload a .txt/.md file instead."
        )

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption

    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = False
    pdf_opts.do_table_structure = False
    pdf_opts.force_backend_text = True
    pdf_opts.generate_page_images = False
    pdf_opts.generate_picture_images = False
    pdf_opts.generate_table_images = False
    pdf_opts.generate_parsed_pages = False
    pdf_opts.document_timeout = 45  # seconds

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    )
    return converter


def convert_uploaded_file_to_text(
    uploaded_file,
    use_high_fidelity: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Convert an uploaded Streamlit file to text/markdown.

    Behaviour:
    - For .txt and .md, returns raw UTF-8 text.
    - For .pdf:
      - If `use_high_fidelity` is False (default), try a fast path via `pypdf` first.
        If extracted text looks good, return it immediately.
      - Otherwise, or if fast extraction is insufficient, fall back to Docling with
        OCR disabled and backend text extraction enabled.
    - For other formats (DOCX/PPTX/XLSX/HTML), use Docling.

    Raises:
      RuntimeError with a user-friendly message when Docling is required but not
      installed.
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

    # PDF: try a fast text-only path first, then fall back to Docling.
    if ext == "pdf":
        data = uploaded_file.getvalue()

        if not use_high_fidelity and PdfReader is not None:
            try:
                reader = PdfReader(io.BytesIO(data))
                pages_text = [
                    page.extract_text() or "" for page in reader.pages  # type: ignore[union-attr]
                ]
                text_fast = "\n".join(pages_text)
                cleaned = text_fast.strip()
                alpha_count = sum(1 for c in cleaned if c.isalpha())

                # Heuristic: consider it good enough if there's a reasonable amount
                # of text and alphabetic characters.
                if len(cleaned) >= 800 or (len(cleaned) >= 300 and alpha_count >= 50):
                    metadata["converted_by"] = "pypdf-fast"
                    return cleaned, metadata
            except Exception:
                # Fall back to Docling if pypdf extraction fails.
                pass

        # Docling fallback for PDFs.
        if DocumentConverter is None:
            raise RuntimeError(
                "Docling is not installed; conversion for this PDF is unavailable. "
                "Docling is required to convert PDFs/Office docs. Install docling "
                "(e.g. `pip install docling`) or upload a .txt/.md file instead."
            )

        from docling.datamodel.base_models import DocumentStream

        converter = get_docling_converter()
        source = DocumentStream(name=filename, stream=io.BytesIO(data))
        result = converter.convert(source)
        doc = result.document
        try:
            text = doc.export_to_markdown()
        except Exception:  # noqa: BLE001
            text = ""
        if not text:
            text = doc.export_to_text()

        metadata["converted_by"] = "docling"
        return text, metadata

    # Other rich formats (DOCX/PPTX/XLSX/HTML) via Docling.
    if DocumentConverter is None:
        raise RuntimeError(
            "Docling is not installed; conversion for this file type is unavailable. "
            "Docling is required to convert PDFs/Office docs. Install docling "
            "(e.g. `pip install docling`) or upload a .txt/.md file instead."
        )

    converter = get_docling_converter()

    # Persist to a temporary file so Docling can read it from disk. Use a closed
    # file in a temporary directory to avoid Windows temp-file locking.
    tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    suffix = ext or "bin"
    file_path = os.path.join(tmp_dir, f"upload.{suffix}")

    text = ""
    try:
        data = uploaded_file.getbuffer()
        with open(file_path, "wb") as f:
            f.write(data)

        result = converter.convert(file_path)
        doc = result.document
        try:
            text = doc.export_to_markdown()
        except Exception:  # noqa: BLE001
            text = ""
        if not text:
            text = doc.export_to_text()
    finally:
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
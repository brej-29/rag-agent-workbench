# Optional dependency:
#   pip install docling
#
# This script converts local documents (PDF, Markdown, and other formats
# supported by Docling) to text/markdown and uploads them to the backend via
# /documents/upload-text. Docling is used when available; for .txt/.md files,
# the script can fall back to raw text if Docling is not installed.

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

try:
    from docling.document_converter import DocumentConverter
except ImportError:  # pragma: no cover - optional dependency
    DocumentConverter = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a local document using Docling (when available) and "
            "upload the extracted text to the RAG backend via /documents/upload-text."
        )
    )
    parser.add_argument(
        "--file",
        "--pdf-path",
        "--path",
        dest="file_path",
        type=str,
        required=True,
        help="Path to the local file (PDF, Markdown, DOCX, HTML, TXT, etc.).",
    )
    parser.add_argument(
        "--backend-url",
        "--backend",
        dest="backend_url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the running backend (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="dev",
        help="Target Pinecone namespace (default: dev).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title for the document; defaults to the filename.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="local-file",
        help="Source label stored in metadata (default: local-file).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key for the backend (sent as X-API-Key).",
    )
    return parser.parse_args()


def _docling_available() -> bool:
    return DocumentConverter is not None


def convert_file_to_text(file_path: Path) -> str:
    """Convert a file to markdown/text.

    - If Docling is installed, it is used for all supported formats.
    - If Docling is not installed:
      - .txt and .md files are read as raw text.
      - Other formats raise a RuntimeError with installation instructions.
    """
    suffix = file_path.suffix.lower()

    if _docling_available():
        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        return result.document.export_to_markdown()

    # Docling is not available.
    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8", errors="ignore")

    raise RuntimeError(
        f"Docling is required to convert '{file_path}'. Install it with:\n"
        "  pip install docling"
    )


def upload_text(
    backend_url: str,
    title: str,
    source: str,
    text: str,
    namespace: str,
    metadata: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    url = f"{backend_url.rstrip('/')}/documents/upload-text"
    payload = {
        "title": title,
        "source": source,
        "text": text,
        "namespace": namespace,
        "metadata": metadata or {},
    }
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


def main() -> int:
    args = parse_args()
    file_path = Path(args.file_path).expanduser().resolve()
    if not file_path.is_file():
        raise SystemExit(f"File not found: {file_path}")

    title = args.title or file_path.name

    print(f"Converting file at {file_path}...")
    try:
        text = convert_file_to_text(file_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Error converting file: {exc}")
        return 1

    print(
        f"Uploading converted text to backend at {args.backend_url} "
        f"namespace='{args.namespace}'...",
    )
    response = upload_text(
        backend_url=args.backend_url,
        title=title,
        source=args.source,
        text=text,
        namespace=args.namespace,
        metadata={"original_path": str(file_path), "extension": file_path.suffix.lower()},
        api_key=args.api_key,
    )

    print("Upload response:")
    print(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
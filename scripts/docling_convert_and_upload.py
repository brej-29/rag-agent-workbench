# pip install docling

import argparse
import json
from typing import Any, Dict

import httpx
from docling.document_converter import DocumentConverter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a local PDF using Docling and upload the extracted text "
            "to the RAG backend via /documents/upload-text."
        )
    )
    parser.add_argument(
        "--pdf-path",
        type=str,
        required=True,
        help="Path to the local PDF file.",
    )
    parser.add_argument(
        "--backend-url",
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
        help="Optional title for the document; defaults to the PDF filename.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="docling",
        help="Source label stored in metadata (default: docling).",
    )
    return parser.parse_args()


def convert_pdf_to_markdown(pdf_path: str) -> str:
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()


def upload_text(
    backend_url: str,
    title: str,
    source: str,
    text: str,
    namespace: str,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    url = f"{backend_url.rstrip('/')}/documents/upload-text"
    payload = {
        "title": title,
        "source": source,
        "text": text,
        "namespace": namespace,
        "metadata": metadata or {},
    }
    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def main() -> int:
    args = parse_args()
    title = args.title or args.pdf_path.rsplit("/", 1)[-1]

    print(f"Converting PDF at {args.pdf_path} with Docling...")
    markdown_text = convert_pdf_to_markdown(args.pdf_path)

    print(
        f"Uploading converted text to backend at {args.backend_url} "
        f"namespace='{args.namespace}'...",
    )
    response = upload_text(
        backend_url=args.backend_url,
        title=title,
        source=args.source,
        text=markdown_text,
        namespace=args.namespace,
        metadata={"original_path": args.pdf_path},
    )

    print("Upload response:")
    print(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
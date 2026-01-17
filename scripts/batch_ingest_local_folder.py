# Optional dependency:
#   pip install docling
#
# Batch-ingest a local folder of documents into the backend by converting each
# supported file to markdown/text (using Docling when available) and uploading
# it via /documents/upload-text.

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from docling_convert_and_upload import convert_file_to_text, upload_text  # type: ignore[import]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively ingest a folder of local documents using Docling (when available) "
            "and upload them to the backend via /documents/upload-text."
        )
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Root folder containing documents to ingest.",
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
        "--source",
        type=str,
        default="local-folder",
        help="Source label stored in metadata (default: local-folder).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key for the backend (sent as X-API-Key).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=200,
        help="Maximum number of files to ingest (default: 200).",
    )
    return parser.parse_args()


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".html",
    ".htm",
    ".md",
    ".markdown",
    ".adoc",
    ".txt",
}


def find_files(root: Path, max_files: int) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        files.append(path)
        if len(files) >= max_files:
            break
    return files


def main() -> int:
    args = parse_args()
    root = Path(args.folder).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Folder not found: {root}")

    files = find_files(root, args.max_files)
    if not files:
        print(f"No supported files found in {root}")
        return 0

    print(f"Found {len(files)} file(s) to ingest in {root} (max {args.max_files}).")

    successes = 0
    failures: List[Dict[str, Any]] = []

    for idx, file_path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] Converting {file_path}...")
        try:
            text = convert_file_to_text(file_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  Conversion failed: {exc}")
            failures.append({"path": str(file_path), "error": str(exc)})
            continue

        try:
            response = upload_text(
                backend_url=args.backend_url,
                title=file_path.name,
                source=args.source,
                text=text,
                namespace=args.namespace,
                metadata={
                    "original_path": str(file_path),
                    "extension": file_path.suffix.lower(),
                },
                api_key=args.api_key,
            )
            successes += 1
            print(f"  Uploaded successfully: {json.dumps(response, indent=2)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  Upload failed: {exc}")
            failures.append({"path": str(file_path), "error": str(exc)})

    print()
    print(f"Ingestion complete. Successes: {successes}, Failures: {len(failures)}")
    if failures:
        print("Failures:")
        for item in failures:
            print(f"- {item['path']}: {item['error']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
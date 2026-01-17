# Development helper to validate Docling temp file handling outside Streamlit.
#
# Usage:
#   python scripts/dev_test_docling_temp.py --file path/to/document.pdf
#
# This script uses the same temp-directory pattern as the frontend's
# `convert_uploaded_file_to_text` to exercise Docling on Windows and Linux.

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    raise SystemExit(
        "Docling is not installed. Install it with:\n"
        "  pip install docling"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dev test for Docling conversion using a temp directory."
    )
    parser.add_argument(
        "--file",
        required=True,
        type=str,
        help="Path to a document (PDF/Office/HTML) to convert.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src_path = Path(args.file).expanduser().resolve()
    if not src_path.is_file():
        print(f"File not found: {src_path}")
        return 1

    tmp_dir = tempfile.mkdtemp(prefix="rag_dev_docling_")
    suffix = src_path.suffix or ".bin"
    tmp_file = os.path.join(tmp_dir, f"upload{suffix}")

    try:
        # Copy to temp directory
        with open(src_path, "rb") as f_in, open(tmp_file, "wb") as f_out:
            f_out.write(f_in.read())

        converter = DocumentConverter()

        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                result = converter.convert(tmp_file)
                doc = result.document
                try:
                    text = doc.export_to_markdown()
                except Exception:  # noqa: BLE001
                    text = ""
                if not text:
                    text = doc.export_to_text()
                print("Conversion succeeded.")
                print("First 500 characters:")
                print("-" * 80)
                print(text[:500])
                print("-" * 80)
                return 0
            except PermissionError as exc:
                last_exc = exc
                if attempt == 0:
                    print("PermissionError detected; retrying after brief sleep...")
                    time.sleep(0.2)
                    continue
                print("PermissionError persists after retry:")
                raise
        if last_exc is not None:
            raise last_exc
    finally:
        # Cleanup
        for _ in range(2):
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
                break
            except PermissionError:
                time.sleep(0.2)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
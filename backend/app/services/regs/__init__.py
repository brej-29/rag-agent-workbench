"""Regulatory provision corpus (built by corpus/build.py)."""
import json
from pathlib import Path
from typing import List

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "regs"


def load_provisions(path: Path = DATA_DIR / "regs-v1.jsonl") -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

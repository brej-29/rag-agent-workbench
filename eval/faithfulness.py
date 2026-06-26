"""
Offline faithfulness evaluation — thin wrapper around the runtime checker.

Reuses the same verify_citations and judge_faithfulness implementations from
backend/app/services/faithfulness.py so offline metrics stay bit-for-bit
identical to what runs in production.

NOT wired into CI or `make eval` — it incurs LLM calls.
Invoke on-demand from scripts or notebooks:

    from eval.faithfulness import verify_citations, judge_faithfulness
    from app.services.llm.groq_llm import get_llm

    unverified = verify_citations(answer_text, sources)
    verdict = judge_faithfulness(answer_text, context_str, get_llm())
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"

for _p in (str(_REPO_ROOT), str(_BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_BACKEND_DIR / ".env", override=False)
    _load_dotenv(_REPO_ROOT / ".env", override=False)
except ImportError:
    pass

from app.services.faithfulness import (  # noqa: E402
    FaithfulnessVerdict,
    judge_faithfulness,
    verify_citations,
)

__all__ = ["FaithfulnessVerdict", "judge_faithfulness", "verify_citations"]

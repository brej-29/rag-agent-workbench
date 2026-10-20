import json
from pathlib import Path

from app.schemas.chat import Profile
from app.services.regs.checklist import article_index

ROWS = [json.loads(l) for l in (Path(__file__).resolve().parents[2] / "eval/regs_profiles.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]
PROFILES = [r for r in ROWS if not r.get("_meta")]


def test_golden_profiles_are_valid_and_point_at_real_articles():
    index = article_index()
    assert len(PROFILES) == 12 and len({r["name"] for r in PROFILES}) == 12
    for r in PROFILES:
        Profile(**r["profile"])
        listed = r["binds"] + r["depends"] + r["not"]
        assert len(listed) == len(set(listed)), r["name"]
        assert [a for a in listed if a not in index] == [], r["name"]

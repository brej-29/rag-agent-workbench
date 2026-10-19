from eval.run_regs import label_hits

def test_article_level_labels_match_their_paragraphs_once():
    ranked = ["x/art-26/par-2", "x/art-26/par-1", "y/art-33/par-1", "z/art-1"]
    assert label_hits(ranked, ["x/art-26", "y/art-33/par-1"]) == ["x/art-26", "y/art-33/par-1", "z/art-1"]


def test_every_golden_label_exists_in_the_corpus():
    """Labels must point at real provisions (an article/annex label must have at least one paragraph)."""
    import json
    from pathlib import Path

    import pytest
    from app.services.regs import DATA_DIR, load_provisions

    if not (DATA_DIR / "regs-v1.jsonl").exists():
        pytest.skip("corpus not built")
    ids = [p["id"] for p in load_provisions()]
    rows = [json.loads(l) for l in Path("eval/regs_golden.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    missing = [lab for r in rows if not r.get("_meta") for lab in r["relevant"]
               if not any(i == lab or i.startswith(lab + "/") for i in ids)]
    assert missing == []

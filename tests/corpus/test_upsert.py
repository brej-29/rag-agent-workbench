from corpus.upsert import estimate, to_record

P = {"id": "eu-ai-act/art-26/par-1", "ref": "AI Act Art. 26(1)", "title": "Obligations of deployers", "text": "Deployers shall...",
     "framework": "EU AI Act", "kind": "article_paragraph", "binding": "regulation", "applies_from": "2027-12-02",
     "applies_note": None, "path": ["Chapter III", "Section 3", "Article 26"], "guidance": None, "amended_by": []}

def test_record_is_flat_without_nulls_and_dates_are_ints():
    r = to_record(P)
    assert r["_id"] == P["id"] and r["content"].startswith("AI Act Art. 26(1) Obligations of deployers")
    assert r["applies_from"] == 20271202 and "applies_note" not in r and "guidance" not in r
    assert r["path"] == "Chapter III > Section 3 > Article 26"

def test_estimate_reports_tokens_and_storage():
    e = estimate([P] * 100)
    assert e["records"] == 100 and e["est_embed_tokens"] > 0 and e["est_storage_mb"] > 0

from corpus.build import split_long

def test_long_records_split_on_sentence_boundaries_with_part_ids():
    rec = {"id": "x/art-1", "ref": "X Art. 1", "text": ("Sentence one is here. " * 400).strip()}
    parts = split_long(rec, max_chars=1000)
    assert len(parts) > 1 and all(len(p["text"]) <= 1000 for p in parts)
    assert parts[0]["id"] == "x/art-1/part-1" and parts[1]["ref"] == "X Art. 1 (part 2)"
    assert " ".join(p["text"] for p in parts) == rec["text"]

def test_short_records_untouched():
    rec = {"id": "x", "ref": "X", "text": "short"}
    assert split_long(rec) == [rec]


def test_built_corpus_ids_are_ascii_and_unique():
    """Pinecone rejects non-ASCII vector ids; catch them here, not mid-upload."""
    import pytest
    from app.services.regs import DATA_DIR, load_provisions

    if not (DATA_DIR / "regs-v1.jsonl").exists():
        pytest.skip("corpus not built")
    ids = [p["id"] for p in load_provisions()]
    assert [i for i in ids if not i.isascii()] == []
    assert len(ids) == len(set(ids))

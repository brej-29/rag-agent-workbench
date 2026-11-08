from app.services.regs.lexical import BM25, tokenize

def test_tokenize_normalises_article_references():
    assert tokenize("What does Article 26(1) require?") == ["art", "26", "1", "require"]

def test_reference_queries_rank_the_cited_provision_first():
    idx = BM25.from_provisions([
        {"id": "a/art-26/par-1", "ref": "AI Act Art. 26(1)", "title": "Obligations of deployers", "text": "Deployers shall take measures."},
        {"id": "a/art-16", "ref": "AI Act Art. 16", "title": "Obligations of providers", "text": "Providers shall ensure compliance."},
        {"id": "g/art-33/par-1", "ref": "GDPR Art. 33(1)", "title": "Notification of a personal data breach", "text": "The controller shall notify the breach within 72 hours."},
    ])
    assert idx.search("Article 26 deployers", k=1)[0][0] == "a/art-26/par-1"
    assert idx.search("72 hours breach notification", k=1)[0][0] == "g/art-33/par-1"
    assert idx.search("zebra", k=3) == []


def test_zero_padded_control_numbers_match_how_people_write_them():
    assert tokenize("SP 800-53 AC-02") == tokenize("sp 800-53 ac-2")

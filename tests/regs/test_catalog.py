from app.services.regs.ask import regs_corpus
from app.services.regs.checklist import article_index, catalog_problems, load_catalog
from app.services.regs.facts import load_facts


def test_catalog_has_no_problems():
    corpus, _ = regs_corpus()
    assert catalog_problems(load_catalog(), load_facts(), corpus, article_index()) == []


def test_catalog_covers_the_five_laws():
    arts = [o["article"] for o in load_catalog()["obligations"]]
    counts = {p: sum(a.startswith(p + "/") for a in arts) for p in ("eu-ai-act", "gdpr", "nis2", "dora", "cra")}
    assert counts == {"eu-ai-act": 32, "gdpr": 34, "nis2": 4, "dora": 19, "cra": 10}


def test_catalog_problems_catches_each_kind_of_mistake():
    corpus = {"x/art-1/par-1": {"id": "x/art-1/par-1", "text": "Providers shall keep logs."}}
    index = {"x/art-1": [corpus["x/art-1/par-1"]]}
    facts = [{"id": "size", "type": "choice", "values": ["small", "large"], "basis": "x/art-1"}]
    bad = {"defs": {"big": {"size": "huge"}},
           "obligations": [{"article": "x/art-1", "when": {"use": "nope"}, "basis": "x/art-1/par-1", "quote": "Deployers shall"},
                           {"article": "x/art-9", "when": {"colour": "red"}, "basis": "x/art-9/par-1", "quote": "x"},
                           {"article": "x/art-1", "when": {"size": "small"}, "basis": "x/art-1/par-1", "quote": "Providers shall"}],
           "practices": []}
    problems = " | ".join(catalog_problems(bad, facts, corpus, index))
    for expected in ("size has no value 'huge'", "unknown def nope", "quote not found", "x/art-9: not in the corpus",
                     "unknown fact colour", "x/art-1: listed twice"):
        assert expected in problems

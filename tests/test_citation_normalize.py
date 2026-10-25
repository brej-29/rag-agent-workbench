"""gpt-oss sometimes cites with full-width lenticular brackets (【1】, 【1†L3-L5】); the citation check and
the UI only understand [n], so answers are normalised before verification."""
from app.services.faithfulness import normalize_citations, verify_citations


def test_lenticular_citations_become_square_brackets():
    text = "Notify within 72 hours【1】 and tell the people affected【2†L4-L9】. See also [3]."
    assert normalize_citations(text) == "Notify within 72 hours[1] and tell the people affected[2]. See also [3]."


def test_normalised_citations_are_verified():
    assert verify_citations(normalize_citations("x【7】"), [{}, {}]) == [7]


def test_other_text_is_untouched():
    assert normalize_citations("Brackets 【note】 and [a] stay.") == "Brackets 【note】 and [a] stay."

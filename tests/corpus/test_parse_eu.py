from corpus.parse_eu import parse_eu
from corpus.sources import SOURCES

AI = next(s for s in SOURCES if s.key == "eu-ai-act")
RECITALS = next(s for s in SOURCES if s.key == "eu-ai-act-recitals")
GDPR = next(s for s in SOURCES if s.key == "gdpr")

OJ = """<div class="eli-subdivision" id="cpt_III"><div class="eli-subdivision" id="cpt_III.sct_3">
<div class="eli-subdivision" id="art_26"><p class="oj-ti-art">Article 26</p>
<div class="eli-title" id="art_26.tit_1"><p class="oj-sti-art">Obligations of deployers</p></div>
<div id="026.001"><p class="oj-normal">1. Deployers shall take measures.</p></div>
<div id="026.002"><p class="oj-normal">2. Deployers shall assign oversight:</p>
<table><tr><td><p class="oj-normal">(a)</p></td><td><p class="oj-normal">to competent persons;</p></td></tr></table></div>
</div></div></div>
<div class="eli-subdivision" id="art_99"><p class="oj-ti-art">Article 99</p>
<div class="eli-title" id="art_99.tit_1"><p class="oj-sti-art">Entry into force</p></div>
<p class="oj-normal">It shall apply from 25 May 2018.</p></div>
<div class="eli-subdivision" id="rct_12"><table><tr><td><p class="oj-normal">(12)</p></td>
<td><p class="oj-normal">The notion of AI system should be clearly defined.</p></td></tr></table></div>
<div class="eli-container" id="anx_III"><p class="oj-doc-ti">ANNEX III</p><p class="oj-doc-ti">High-risk AI systems</p>
<p class="oj-normal">The AI systems listed in any of the following areas:</p>
<table><tr><td><p class="oj-normal">1.</p></td><td><p class="oj-normal">Biometrics:</p></td></tr></table>
<table><tr><td><p class="oj-normal">2.</p></td><td><p class="oj-normal">Critical infrastructure.</p></td></tr></table></div>"""

CONSOLIDATED = """<p class="hd-modifiers">►M1 Regulation (EU) 2026/1744 of the European Parliament</p>
<div class="eli-subdivision" id="art_5"><p class="title-article-norm">Article 5</p>
<div class="eli-title" id="art_5.tit_1"><p class="stitle-article-norm">Prohibited AI practices</p></div>
<div class="norm"><span class="no-parag">1. </span><div class="norm inline-element">The following practices are prohibited.</div></div>
<p class="arrow">▼M1</p>
<div class="norm"><span class="no-parag">1a. </span><div class="norm inline-element">New prohibition.</div></div></div>"""


def _by_id(records):
    return {r["id"]: r for r in records}


def test_oj_articles_split_into_paragraphs_with_points_and_path():
    recs = _by_id(parse_eu(OJ, AI))
    p2 = recs["eu-ai-act/art-26/par-2"]
    assert p2["ref"] == "AI Act Art. 26(2)" and p2["kind"] == "article_paragraph"
    assert p2["text"] == "2. Deployers shall assign oversight: (a) to competent persons;"
    assert p2["path"] == ["Chapter III", "Section 3", "Article 26"]
    assert (p2["chapter"], p2["section"], p2["article"], p2["paragraph"]) == ("III", "3", "26", "2")
    assert p2["title"] == "Obligations of deployers" and p2["licence"] == AI.licence


def test_article_without_numbered_paragraphs_is_one_record():
    rec = _by_id(parse_eu(OJ, GDPR))["gdpr/art-99"]
    assert rec["kind"] == "article" and rec["text"] == "It shall apply from 25 May 2018." and rec["paragraph"] is None


def test_recitals_only_for_recital_sources_and_text_sources():
    recs = _by_id(parse_eu(OJ, RECITALS))
    assert set(recs) == {"eu-ai-act/rct-12"}
    assert recs["eu-ai-act/rct-12"]["text"] == "The notion of AI system should be clearly defined."
    assert "gdpr/rct-12" in _by_id(parse_eu(OJ, GDPR))


def test_annex_split_into_points():
    recs = _by_id(parse_eu(OJ, AI))
    assert recs["eu-ai-act/anx-III/pt-1"]["ref"] == "AI Act Annex III, point 1"
    assert recs["eu-ai-act/anx-III/pt-2"]["text"].endswith("Critical infrastructure.")
    assert recs["eu-ai-act/anx-III"]["text"].startswith("ANNEX III High-risk AI systems")


def test_consolidated_markup_strips_markers_and_records_amendments():
    recs = _by_id(parse_eu(CONSOLIDATED, AI))
    assert recs["eu-ai-act/art-5/par-1"]["amended_by"] == []
    new = recs["eu-ai-act/art-5/par-1a"]
    assert new["text"] == "1a. New prohibition." and new["amended_by"] == ["Regulation (EU) 2026/1744"]


REAL_CONSOLIDATED = """<p class="hd-modifiers">Amended by:</p><table><tr>
<td><p class="arrow"><a title="32026R1744">►M1</a></p></td>
<td><p class="title-fam-member-star"><a title="32026R1744"> REGULATION (EU) 2026/1744 OF THE EUROPEAN PARLIAMENT</a></p></td></tr></table>
<div class="eli-subdivision" id="art_113"><p class="title-article-norm">Article 113</p>
<div class="eli-title" id="art_113.tit_1"><p class="stitle-article-norm">Entry into force and application</p></div>
<p class="norm">It shall apply from 2 August 2026. However:</p><p class="arrow">▼M1</p>
<p class="norm">(a) Chapters I and II shall apply from 2 February 2025;</p></div>
<div class="eli-container" id="anx_III"><p class="title-annex-1">ANNEX III</p>
<p class="title-annex-2">High-risk AI systems referred to in Article 6(2)</p>
<p class="norm">The AI systems listed in any of the following areas:</p>
<div class="grid-container grid-list"><div class="list grid-list-column-1"><span>1. </span></div>
<div class="grid-list-column-2"><p class="norm">Biometrics:</p></div></div>
<div class="grid-container grid-list"><div class="list grid-list-column-1"><span>2. </span></div>
<div class="grid-list-column-2"><p class="norm">Critical infrastructure.</p></div></div></div>"""


def test_real_consolidated_header_maps_amendment_codes():
    rec = {r["id"]: r for r in parse_eu(REAL_CONSOLIDATED, AI)}["eu-ai-act/art-113"]
    assert rec["amended_by"] == ["Regulation (EU) 2026/1744"]
    assert "▼" not in rec["text"] and rec["text"].startswith("It shall apply from 2 August 2026.")


def test_consolidated_annex_grid_points():
    recs = {r["id"]: r for r in parse_eu(REAL_CONSOLIDATED, AI)}
    assert recs["eu-ai-act/anx-III/pt-2"]["text"] == "2. Critical infrastructure."
    assert recs["eu-ai-act/anx-III"]["title"] == "High-risk AI systems referred to in Article 6(2)"
    assert recs["eu-ai-act/anx-III"]["text"].startswith("ANNEX III High-risk AI systems referred to in Article 6(2)")


SECTIONED = """<div class="eli-container" id="anx_VIII"><p class="title-annex-1">ANNEX VIII</p>
<p class="title-annex-2">Information to be submitted upon registration</p>
<p class="title-gr-seq-level-1">Section A — Information to be submitted by providers</p>
<div class="grid-container grid-list"><div class="list grid-list-column-1"><span>1. </span></div>
<div class="grid-list-column-2"><p class="norm">The name of the provider;</p></div></div>
<p class="title-gr-seq-level-1">Section B. Information to be submitted by deployers</p>
<div class="grid-container grid-list"><div class="list grid-list-column-1"><span>1. </span></div>
<div class="grid-list-column-2"><p class="norm">The name of the deployer;</p></div></div>
<div><p class="norm">2. A plain div point.</p></div></div>"""


def test_sectioned_annex_points_are_unique_and_cite_their_section():
    recs = {r["id"]: r for r in parse_eu(SECTIONED, AI)}
    assert recs["eu-ai-act/anx-VIII/sec-A/pt-1"]["ref"] == "AI Act Annex VIII, Section A, point 1"
    assert recs["eu-ai-act/anx-VIII/sec-B/pt-1"]["text"] == "1. The name of the deployer;"
    assert recs["eu-ai-act/anx-VIII/sec-B/pt-2"]["text"] == "2. A plain div point."


def test_quoted_amendment_paragraph_labels_are_ascii():
    html = """<div class="eli-subdivision" id="art_105"><p class="title-article-norm">Article 105</p>
<div class="norm"><span class="no-parag">‘5. </span><div class="norm inline-element">For AI systems...</div></div></div>"""
    (rec,) = parse_eu(html, AI)
    assert rec["id"] == "eu-ai-act/art-105/par-5" and rec["id"].isascii()


DECIMAL_ANNEX = """<html><body><div id="anx_1">
<p class="oj-doc-ti">ANNEX</p><p class="oj-doc-ti">Technical and methodological requirements referred to in Article 2</p>
<p class="oj-ti-grseq-1">1. Policy on the security of network and information systems (Article 21(2), point (a) of Directive (EU) 2022/2555)</p>
<p class="oj-ti-grseq-1">1.1. Policy on the security of network and information systems</p>
<table><tr><td>1.1.1. For the purpose of Article 21(2), point (a), the policy shall set out the approach.</td></tr></table>
<table><tr><td>1.1.2. The policy shall be reviewed at planned intervals.</td></tr></table>
<p class="oj-ti-grseq-1">1.2. Roles, responsibilities and authorities</p>
<table><tr><td>1.2.1. Roles shall be assigned.</td></tr></table>
<p class="oj-ti-grseq-1">7. Policies and procedures to assess effectiveness (Article 21(2), point (f) of Directive (EU) 2022/2555)</p>
<table><tr><td>7.1. For the purpose of Article 21(2), point (f), the entities shall establish a policy.</td></tr></table>
<table><tr><td>7.2. The policy shall take into account results of the risk assessment.</td></tr></table>
<p class="oj-ti-grseq-1">11. Access control (Article 21(2), points (i) and (j), of Directive (EU) 2022/2555)</p>
<p class="oj-ti-grseq-1">11.1. Access control policy</p>
<table><tr><td>11.1.1. An access control policy shall be established.</td></tr></table>
</div></body></html>"""


def test_decimal_numbered_annex_becomes_one_record_per_point():
    from corpus.sources import Source
    src = Source("nis2-ir", "nis2-ir", "NIS2 IR", "NIS2", "Implementing Regulation (EU) 2024/2690", "u", "eu-xhtml",
                 "l", "regulation")
    recs = {r["id"]: r for r in parse_eu(DECIMAL_ANNEX, src)}
    assert list(recs) == ["nis2-ir/anx-1", "nis2-ir/anx-1/pt-1.1", "nis2-ir/anx-1/pt-1.2", "nis2-ir/anx-1/pt-7.1",
                          "nis2-ir/anx-1/pt-7.2", "nis2-ir/anx-1/pt-11.1"]
    # Sections whose points have no headings of their own (7., 9. in 2024/2690): each N.M table is a point.
    p7 = recs["nis2-ir/anx-1/pt-7.2"]
    assert p7["title"] == "Policies and procedures to assess effectiveness" and p7["text"].startswith("7.2. The policy")
    assert "(Article 21(2), point (f)" in p7["path"][1]
    p = recs["nis2-ir/anx-1/pt-1.1"]
    assert p["kind"] == "annex_point" and p["ref"] == "NIS2 IR Annex, point 1.1"
    assert p["title"] == "Policy on the security of network and information systems"
    assert p["text"].startswith("1.1.1. For the purpose") and "1.1.2. The policy shall be reviewed" in p["text"]
    assert "(Article 21(2), point (a)" in p["path"][1]
    assert "(Article 21(2), points (i) and (j)" in recs["nis2-ir/anx-1/pt-11.1"]["path"][1]

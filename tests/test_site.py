"""The landing site (site/build.py) may only show dates from timeline.json and quotes from the corpus."""
import importlib.util
import json
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("site_build", ROOT / "site" / "build.py")  # "site" shadows a stdlib module
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)

CORPUS = {r["id"]: r for r in map(json.loads, (ROOT / "backend/app/data/regs/regs-v1.jsonl")
                                  .read_text(encoding="utf-8").splitlines())}
DATES = {m["from"] for m in json.loads((ROOT / "eval/results/timeline.json").read_text(encoding="utf-8"))["milestones"]}
MONTHS = "January February March April May June July August September October November December".split()
DATE_RE = re.compile(r"\b(\d{1,2}) (" + "|".join(MONTHS) + r") (\d{4})\b")


class Page(HTMLParser):
    """Splits a page into quotes (text inside <blockquote><p>, with its data-source) and everything else."""

    def __init__(self):
        super().__init__()
        self.quotes, self.rest, self.source, self.in_quote_p = [], [], None, False

    def handle_starttag(self, tag, attrs):
        if tag == "blockquote":
            self.source = dict(attrs)["data-source"]
        elif tag == "p" and self.source:
            self.in_quote_p, self.quotes = True, self.quotes + [(self.source, "")]

    def handle_endtag(self, tag):
        if tag == "blockquote":
            self.source = None
        elif tag == "p":
            self.in_quote_p = False

    def handle_data(self, data):
        if self.in_quote_p:
            src, text = self.quotes[-1]
            self.quotes[-1] = (src, text + data)
        else:
            self.rest.append(data)


def norm(s):
    return " ".join(s.split())


def test_site_pages_use_only_verified_quotes_and_dates(tmp_path):
    names = build.build(tmp_path)
    assert {"index.html", "faq.html", "timeline.html", "eu-ai-act.html", "gdpr.html", "nis2.html", "dora.html",
            "cyber-resilience-act.html", "nist.html"} <= set(names)
    titles, descs, n_quotes = set(), set(), 0
    for name in names:
        html = (tmp_path / name).read_text(encoding="utf-8")
        assert "—" not in html, f"em dash in {name}"
        assert build.DISCLAIMER in html, name
        titles.add(re.search(r"<title>(.*?)</title>", html).group(1))
        descs.add(re.search(r'<meta name="description" content="(.*?)">', html).group(1))
        assert '<link rel="canonical"' in html and 'property="og:image"' in html, name
        page = Page()
        page.feed(html)
        for source, text in page.quotes:
            assert norm(text) in norm(CORPUS[source]["text"]), f"{name}: quote not in {source}: {text!r}"
        n_quotes += len(page.quotes)
        for day, month, year in DATE_RE.findall(" ".join(page.rest)):
            iso = f"{year}-{MONTHS.index(month) + 1:02d}-{int(day):02d}"
            assert iso in DATES, f"{name}: date {day} {month} {year} is not in timeline.json"
        for iso in re.findall(r'datetime="([\d-]+)"', html):
            assert iso in DATES, f"{name}: <time> {iso} is not in timeline.json"
    assert len(titles) == len(names) and len(descs) == len(names), "titles and descriptions must be unique"
    assert n_quotes >= 20
    sitemap = (tmp_path / "sitemap.xml").read_text(encoding="utf-8")
    assert sitemap.count("<loc>") == len(names)
    assert "Sitemap:" in (tmp_path / "robots.txt").read_text(encoding="utf-8")
    faq = (tmp_path / "faq.html").read_text(encoding="utf-8")
    ld = json.loads(re.search(r'<script type="application/ld\+json">(.*?)</script>', faq).group(1))
    assert ld["@type"] == "FAQPage" and len(ld["mainEntity"]) == len(build.FAQ)

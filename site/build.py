"""Build the static landing site into site/dist/. Stdlib only: `python site/build.py`.

Every date comes from eval/results/timeline.json (a copy of applicability.yaml, where each date is
checked against a verbatim quote of the law). Every quote is checked against regs-v1.jsonl here and
again in tests/test_site.py. Plain-language lines and milestone labels are imported from the app.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from datetime import date
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "frontend"))
from services.eval_data import load_results  # noqa: E402  (stdlib-only modules of the app)
from services.home_html import COLORS, DEMO, MILESTONES, QUESTIONS, proof_panels, stats  # noqa: E402
from services.presenters import RULEBOOK_LINES, rulebook_stack_html  # noqa: E402

SITE = os.environ.get("SITE_URL", "https://brej-29.github.io/rag-agent-workbench").rstrip("/")
VERIFY = os.environ.get("GOOGLE_SITE_VERIFICATION", "")  # Search Console HTML-tag token; public, not a secret
APP = "https://rag-agent-workbench.streamlit.app"
REPO = "https://github.com/brej-29/rag-agent-workbench"
NAME = "Compliance Workbench"
DISCLAIMER = "Information from official EU and NIST text, not legal advice."
CORPUS = ROOT / "backend/app/data/regs/regs-v1.jsonl"
TIMELINE = ROOT / "eval/results/timeline.json"
HERE = Path(__file__).resolve().parent

ROWS = {r["id"]: r for r in map(json.loads, CORPUS.read_text(encoding="utf-8").splitlines())}
# Same milestones the app's home timeline shows (it labels 13 of the 14; the unlabelled one is skipped there too).
MS = [m for m in json.loads(TIMELINE.read_text(encoding="utf-8"))["milestones"] if (m["id"], m["condition"]) in MILESTONES]
LINES = {name: line for name, _, line in RULEBOOK_LINES}
RESULTS = load_results()
# Dates on the pages that are not law dates: the day the replayed answer was recorded (frontend/services/home_html.py).
NON_LAW_DATES = {"2026-11-08"}
PREFIX_FW = {"eu-ai-act": "AI Act", "gdpr": "GDPR", "nis2": "NIS2", "nis2-ir": "NIS2", "dora": "DORA", "cra": "CRA"}
MONTHS = "January February March April May June July August September October November December".split()


def norm(s: str) -> str:
    return " ".join(s.split())


def long_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def quote(row_id: str, text: str) -> str:
    """A verbatim quote of one provision; the build fails if the text is not in that provision."""
    row = ROWS[row_id]
    if norm(text) not in norm(row["text"]):
        raise ValueError(f"quote not found in {row_id}: {text!r}")
    tab = COLORS.get(PREFIX_FW.get(row_id.split("/")[0], ""), "#59627D")
    return (f'<blockquote data-source="{escape(row_id)}" cite="{escape(row["source_url"])}" style="--tab:{tab}">'
            f'<p>{escape(text)}</p><footer><cite><a href="{escape(row["source_url"])}" rel="nofollow">'
            f'{escape(row["ref"])}</a></cite></footer></blockquote>')


# Rulebooks -------------------------------------------------------------------------------------------------
# slug, display name, plain line key (RULEBOOK_LINES), corpus id prefixes, tab colour, page title, description
RULEBOOKS = [
    ("eu-ai-act", "EU AI Act", "EU AI Act", ["eu-ai-act"], "#2F45D6",
     "EU AI Act: what it covers and when each part applies",
     "What the EU AI Act (Regulation (EU) 2024/1689) covers and the dates its rules apply, from AI literacy "
     "and banned practices to high-risk AI, each quoted from Article 113."),
    ("gdpr", "GDPR", "GDPR", ["gdpr"], "#237B78",
     "GDPR: what it covers and when it applies",
     "What the GDPR (Regulation (EU) 2016/679) covers, its articles at a glance and the date it applies, "
     "quoted from the official text."),
    ("nis2", "NIS2 and its Implementing Regulation", "NIS2 and its Implementing Regulation", ["nis2", "nis2-ir"],
     "#7651A8", "NIS2 and Implementing Regulation 2024/2690: scope and dates",
     "What the NIS2 Directive (EU) 2022/2555 and Implementing Regulation (EU) 2024/2690 cover, with their "
     "start dates quoted from the official text."),
    ("dora", "DORA", "DORA", ["dora"], "#8B4D72",
     "DORA: digital operational resilience for financial firms",
     "What the Digital Operational Resilience Act (Regulation (EU) 2022/2554) covers and when it applies, "
     "quoted from the official text."),
    ("cyber-resilience-act", "Cyber Resilience Act", "Cyber Resilience Act", ["cra"], "#9A6518",
     "Cyber Resilience Act: what it covers and when it applies",
     "What the Cyber Resilience Act (Regulation (EU) 2024/2847) covers and its three start dates, "
     "quoted from Article 71."),
    ("nist", "NIST SP 800-53, CSF 2.0 and AI RMF", "NIST SP 800-53, CSF 2.0, AI RMF",
     ["nist-800-53", "nist-csf", "nist-ai-rmf"], "#59627D",
     "NIST SP 800-53, CSF 2.0 and AI RMF in one place",
     "The NIST SP 800-53 Rev. 5 control families, the CSF 2.0 functions and the AI RMF functions, "
     "searchable with cited answers."),
]
BINDING = {"regulation": "EU regulation, applies directly in every Member State",
           "directive": "EU directive, reaches companies through each Member State's own law",
           "voluntary_framework": "Voluntary US framework, no legal start date"}


def rows_for(prefix: str):
    return [r for r in ROWS.values() if r["id"].split("/")[0] == prefix]


def milestones_for(prefixes):
    return [m for m in MS if m["basis"].split("/")[0] in prefixes]


def milestone_html(m, with_quote: bool = True) -> str:
    """One start date as a rail item. data-from lets a few lines of script mark which ones have passed."""
    fw, label = MILESTONES[(m["id"], m["condition"])]
    return (f'<li class="r" data-from="{m["from"]}" style="--tab:{COLORS[fw]}"><div class="when">'
            f'<time datetime="{m["from"]}">{long_date(m["from"])}</time><span class="chip">{escape(fw)}</span></div>'
            f'<div class="what"><h3>{escape(label)}</h3>{quote(m["basis"], m["quote"]) if with_quote else ""}</div></li>')


def contents_html(prefix: str) -> str:
    rows = rows_for(prefix)
    if prefix.startswith("nist-"):
        heads = {}
        for r in rows:
            p = r["path"]
            key = p[1] if prefix == "nist-csf" and len(p) > 1 else p[0]
            heads.setdefault(p[0], {}).setdefault(key, None)
        out = []
        for top, subs in heads.items():
            extra = [s for s in subs if s != top]
            out.append(f"<li><strong>{escape(top.title() if top.isupper() else top)}</strong>"
                       + (f": {escape(', '.join(extra))}" if extra else "") + "</li>")
        return f'<ul class="toc flat">{"".join(out)}</ul>'
    groups = {}
    for r in rows:
        if r["kind"] in ("article", "article_paragraph") and r["article"]:
            title = (r["title"] or "").strip("'’ ")
            if "�" not in title:
                groups.setdefault(r["chapter"], {}).setdefault(r["article"], title)
    out = []
    for chapter, arts in groups.items():
        items = "".join(f"<li><span>Article {escape(a)}</span> {escape(t)}</li>" for a, t in arts.items())
        out.append((f"<h3>Chapter {escape(chapter)}</h3>" if chapter else "") + f'<ul class="toc">{items}</ul>')
    return "".join(out)


def instrument_rows(prefixes) -> str:
    seen = {}
    for p in prefixes:
        r = rows_for(p)[0]
        seen.setdefault(r["instrument"], (r["binding"], r["source_url"], r["framework"]))
    return "".join(f'<li><a href="{escape(url)}" rel="nofollow">{escape(inst)}</a>. {escape(BINDING[b])}.</li>'
                   for inst, (b, url, _) in seen.items())


# FAQ: question, answer, rulebook slug, [(provision id, verbatim quote), ...] -------------------------------
FAQ = [
    ("gdpr-breach-72-hours", "When must a GDPR personal data breach be reported?",
     "The controller must tell the supervisory authority without undue delay and, where feasible, within 72 "
     "hours of becoming aware of the breach, unless it is unlikely to result in a risk to people. A later "
     "notification must give the reasons for the delay.", "gdpr",
     [("gdpr/art-33/par-1", "In the case of a personal data breach, the controller shall without undue delay and, "
       "where feasible, not later than 72 hours after having become aware of it, notify the personal data breach to "
       "the supervisory authority competent in accordance with Article 55, unless the personal data breach is "
       "unlikely to result in a risk to the rights and freedoms of natural persons. Where the notification to the "
       "supervisory authority is not made within 72 hours, it shall be accompanied by reasons for the delay.")]),
    ("gdpr-tell-people", "When must people be told about a GDPR personal data breach?",
     "When the breach is likely to result in a high risk to their rights and freedoms, the controller must tell "
     "the people affected without undue delay.", "gdpr",
     [("gdpr/art-34/par-1", "When the personal data breach is likely to result in a high risk to the rights and "
       "freedoms of natural persons, the controller shall communicate the personal data breach to the data subject "
       "without undue delay.")]),
    ("gdpr-outside-eu", "Does the GDPR apply to companies outside the EU?",
     "It can. It covers processing by a controller or processor not established in the EU when the processing "
     "relates to offering goods or services to people in the EU, or to monitoring their behaviour in the EU.",
     "gdpr",
     [("gdpr/art-3/par-2", "This Regulation applies to the processing of personal data of data subjects who are in "
       "the Union by a controller or processor not established in the Union, where the processing activities are "
       "related to: (a) the offering of goods or services, irrespective of whether a payment of the data subject is "
       "required, to such data subjects in the Union; or (b) the monitoring of their behaviour as far as their "
       "behaviour takes place within the Union.")]),
    ("gdpr-fines", "How high can GDPR fines be?",
     "For the infringements listed in Article 83(5), such as breaking the basic principles or people's rights, "
     "fines go up to 20 000 000 EUR or, for a company, up to 4 % of its total worldwide annual turnover of the "
     "preceding financial year, whichever is higher.", "gdpr",
     [("gdpr/art-83/par-5", "Infringements of the following provisions shall, in accordance with paragraph 2, be "
       "subject to administrative fines up to 20 000 000 EUR, or in the case of an undertaking, up to 4 % of the "
       "total worldwide annual turnover of the preceding financial year, whichever is higher: (a) the basic principles for processing, including conditions for consent, pursuant to "
       "Articles 5, 6, 7 and 9; (b) the data subjects' rights pursuant to Articles 12 to 22")]),
    ("ai-act-high-risk-dates", "When do the AI Act high-risk rules apply?",
     "From 2 December 2027 for high-risk AI systems listed in Annex III, such as uses in hiring and credit, and "
     "from 2 August 2028 for high-risk AI built into products covered by the EU laws in Annex I.", "eu-ai-act",
     [("eu-ai-act/art-113", "(i) 2 December 2027 as regards AI systems classified as high-risk pursuant to Article "
       "6(2) and Annex III; and (ii) 2 August 2028 as regards AI systems classified as high-risk pursuant to "
       "Article 6(1) and Annex I")]),
    ("ai-act-prohibitions-date", "When did the AI Act bans on prohibited practices start?",
     "Chapters I and II, which hold AI literacy and the prohibited practices of Article 5, apply from 2 February "
     "2025. The prohibitions added later, Article 5(1) points (ba) and (bb) and Article 5(1a) and (1b), apply "
     "from 2 December 2026.", "eu-ai-act",
     [("eu-ai-act/art-113", "(a) Chapters I and II shall apply from 2 February 2025, with the exception of Article "
       "5(1), first subparagraph, points (ba) and (bb), and Article 5(1a) and (1b) which shall apply from 2 "
       "December 2026")]),
    ("ai-act-fines", "What are the fines for prohibited AI practices under the AI Act?",
     "Up to EUR 35 000 000 or, for a company, up to 7 % of its total worldwide annual turnover for the "
     "preceding financial year, whichever is higher.", "eu-ai-act",
     [("eu-ai-act/art-99/par-3", "Non-compliance with the prohibition of the AI practices referred to in Article 5 "
       "shall be subject to administrative fines of up to EUR 35 000 000 or, if the offender is an undertaking, up "
       "to 7 % of its total worldwide annual turnover for the preceding financial year, whichever is higher.")]),
    ("nis2-reporting-deadlines", "What are the NIS2 incident reporting deadlines?",
     "For a significant incident: an early warning within 24 hours of becoming aware of it, an incident "
     "notification within 72 hours, and a final report no later than one month after the incident "
     "notification.", "nis2",
     [("nis2/art-23/par-4", "(a) without undue delay and in any event within 24 hours of becoming aware of the "
       "significant incident, an early warning"),
      ("nis2/art-23/par-4", "(b) without undue delay and in any event within 72 hours of becoming aware of the "
       "significant incident, an incident notification"),
      ("nis2/art-23/par-4", "(d) a final report not later than one month after the submission of the incident "
       "notification under point (b)")]),
    ("nis2-start", "When did NIS2 start to apply?",
     "NIS2 is a directive, so it works through national law. Member States had to apply their national NIS2 "
     "measures from 18 October 2024.", "nis2",
     [("nis2/art-41/par-1", "They shall apply those measures from 18 October 2024.")]),
    ("dora-start", "When does DORA apply?",
     "DORA applies from 17 January 2025.", "dora",
     [("dora/art-64", "It shall apply from 17 January 2025.")]),
    ("cra-start", "When does the Cyber Resilience Act apply?",
     "In full from 11 December 2027. The reporting duties of Article 14 apply earlier, from 11 September 2026, "
     "and Chapter IV on conformity assessment bodies from 11 June 2026.", "cyber-resilience-act",
     [("cra/art-71/par-2", "This Regulation shall apply from 11 December 2027. However, Article 14 shall apply from "
       "11 September 2026 and Chapter IV (Articles 35 to 51) shall apply from 11 June 2026.")]),
    ("cra-vulnerability-reporting", "When must a manufacturer report an actively exploited vulnerability under the CRA?",
     "An early warning within 24 hours of becoming aware of it, then a vulnerability notification within 72 "
     "hours. These duties in Article 14 apply from 11 September 2026.", "cyber-resilience-act",
     [("cra/art-14/par-2", "(a) an early warning notification of an actively exploited vulnerability, without undue "
       "delay and in any event within 24 hours of the manufacturer becoming aware of it"),
      ("cra/art-14/par-2", "(b) unless the relevant information has already been provided, a vulnerability "
       "notification, without undue delay and in any event within 72 hours of the manufacturer becoming aware of "
       "the actively exploited vulnerability"),
      ("cra/art-71/par-2", "Article 14 shall apply from 11 September 2026")]),
]

for _id, _q, _a, _slug, quotes in FAQ:  # fail at import, not at render, if a quote drifts
    for rid, text in quotes:
        quote(rid, text)


CHECK = ('<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="M3.5 8.5l3 3 6-7" fill="none" '
         'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>')


def faq_items(entries) -> str:
    out = []
    for fid, q, a, slug, quotes in entries:
        book = next(b for b in RULEBOOKS if b[0] == slug)
        out.append(f'<section class="qa" id="{fid}" style="--tab:{book[4]}"><h2>{escape(q)}</h2><p class="ans-text">'
                   f'{escape(a)}</p>' + "".join(quote(rid, t) for rid, t in quotes)
                   + f'<p class="links"><a class="btn btn-sm" href="{APP}/ask">Ask this in the app</a>'
                   f'<a href="{slug}.html">Guide: {escape(book[1])}</a></p></section>')
    return "".join(out)


# Page shell -------------------------------------------------------------------------------------------------
# Every button and card that invites action goes to the app. Navigation between the reading pages (header, footer,
# guide tabs, "read the law") stays on the site so search engines can crawl it.
def nav(current: str) -> str:
    links = [("timeline.html", "When the rules start"), ("faq.html", "Questions")]
    items = "".join(f'<a href="{h}"' + (' aria-current="page"' if h == current else "") + f">{t}</a>"
                    for h, t in links)
    return (f'<header class="top"><div class="wrap bar"><a class="wordmark" href="index.html">{NAME}</a>'
            f'<nav aria-label="Main">{items}<a class="btn btn-sm" href="{APP}">Open the app</a></nav></div></header>')


def footer() -> str:
    books = "".join(f'<li><a href="{b[0]}.html">{escape(b[1])}</a></li>' for b in RULEBOOKS)
    return (f'<footer class="foot"><div class="wrap"><p class="disclaimer">{DISCLAIMER}</p>'
            f'<ul class="foot-links">{books}</ul>'
            '<p>EU texts: © European Union, https://eur-lex.europa.eu. Reuse authorised under Commission Decision '
            '2011/833/EU. NIST texts: U.S. Government work. '
            f'<a href="{REPO}">Source code</a>.</p></div></footer>')


SCRIPT = """<script>
(function () {
  var d = document, today = new Date().toISOString().slice(0, 10);
  d.querySelectorAll(".rail").forEach(function (ol) {
    var first = null;
    ol.querySelectorAll("li.r").forEach(function (li) {
      if (li.dataset.from > today) { li.classList.add("future"); if (!first) first = li; }
    });
    if (first) { var m = d.createElement("li"); m.className = "today"; m.textContent = "Today"; ol.insertBefore(m, first); }
  });
  var demo = d.querySelector(".demo");
  if (demo) {
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (e, o) { if (e[0].isIntersecting) { demo.classList.add("in"); o.disconnect(); } },
        { threshold: 0.35 }).observe(demo);
    } else { demo.classList.add("in"); }
  }
})();
</script>"""


def page(path: str, title: str, desc: str, body: str, jsonld=None, top: str = "") -> tuple[str, str]:
    url = f"{SITE}/" if path == "index.html" else f"{SITE}/{path}"
    ld = "".join('<script type="application/ld+json">' + json.dumps(x, ensure_ascii=False).replace("</", "<\\/")
                 + "</script>" for x in (jsonld or []))
    verify = f'\n<meta name="google-site-verification" content="{escape(VERIFY)}">' if VERIFY else ""
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<meta name="description" content="{escape(desc)}">
<link rel="canonical" href="{url}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="{NAME}">
<meta property="og:title" content="{escape(title)}">
<meta property="og:description" content="{escape(desc)}">
<meta property="og:url" content="{url}">
<meta property="og:image" content="{SITE}/og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="The Compliance Workbench home screen with its stack of rulebooks">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{escape(title)}">
<meta name="twitter:description" content="{escape(desc)}">
<meta name="twitter:image" content="{SITE}/og.png">
<meta name="theme-color" content="#F6F7FB">{verify}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;500;600&display=swap">
<link rel="stylesheet" href="style.css">
<script>document.documentElement.classList.add("js")</script>
{ld}
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
{nav(path)}
{top}
<main id="main">
{body}
</main>
{footer()}
{SCRIPT}
</body>
</html>
"""
    return path, html


def sec(inner: str, cls: str = "") -> str:
    return f'<section class="sec {cls}"><div class="wrap">{inner}</div></section>'


def cta(inverse: bool = False) -> str:
    cls = " btn-on-dark" if inverse else ""
    return (f'<p class="actions"><a class="btn{cls}" href="{APP}/checklist">Try a ready-made example</a>'
            f'<a class="btn btn-quiet{cls}" href="{APP}/ask">Ask a question</a></p>')


def band(h1: str, lead: str, extra: str = "", color: str = "", current: str = "") -> str:
    """Page header band with the guide tabs sitting on its lower edge, like index tabs on a rulebook."""
    short = {"eu-ai-act": "EU AI Act", "gdpr": "GDPR", "nis2": "NIS2", "dora": "DORA", "cyber-resilience-act": "CRA",
             "nist": "NIST"}
    tabs = "".join(f'<a href="{b[0]}.html" style="--tab:{b[4]}"' + (' aria-current="page"' if b[0] == current else "")
                   + f">{short[b[0]]}</a>" for b in RULEBOOKS)
    style = f' style="--tab:{color}"' if color else ""
    return (f'<div class="band"{style}><div class="wrap"><h1>{escape(h1)}</h1><p class="lead">{escape(lead)}</p>{extra}'
            f'<nav class="tabs" aria-label="Guides">{tabs}</nav></div></div>')


def stack_html() -> str:
    """The app's own rulebook stack (frontend/services/presenters.py), each tab opening the app."""
    books = re.findall(r'--i:(\d+);--tab:(#\w+)"><span>(.*?)</span><small>(.*?)</small>', rulebook_stack_html())
    assert len(books) == 8, "the app's rulebook stack changed shape"
    items = "".join(f'<a class="book" role="listitem" href="{APP}" style="--i:{i};--tab:{color}"><span>{title}</span>'
                    f"<small>{line}</small></a>" for i, color, title, line in books)
    return f'<div class="stack" role="list" aria-label="Eight rulebooks">{items}</div>'


def demo_html() -> str:
    d = DEMO
    y, m, day = d["status"].rsplit(" ", 1)[1].split("-")
    status = f"In force since {int(day)} {MONTHS[int(m) - 1]} {y}"
    assert d["status"].rsplit(" ", 1)[1] in {m_["from"] for m_ in MS}, "demo date is not a tracked start date"
    steps = "".join(f'<li style="--s:{n}">{CHECK}{escape(t)}</li>' for n, t in enumerate(d["steps"]))
    words = " ".join(f'<span style="--w:{n}">{escape(w)}</span>' for n, w in enumerate(d["answer"].split()))
    return (f'<div class="demo" aria-label="Example answer"><p class="bubble">{escape(d["question"])}</p>'
            f'<ol class="steps-done">{steps}</ol><p class="ans">{words}<sup>1</sup></p>'
            f'<p class="src"><span class="chip" style="--tab:{COLORS.get(d["source"].split()[0], "#59627D")}">1 · {escape(d["source"])}</span><span class="badge">{status}</span></p></div>')


def dots(p) -> str:
    total = p["total"]
    cols = next((c for c in (10, 8, 7, 6) if total % c == 0), 10)
    cells = "".join(f'<i class="{"on" if k < p["filled"] else ""}"></i>' for k in range(total))
    return (f'<figure class="proof tone-{p["tone"]}"><div class="dots" style="--cols:{cols}" role="img" '
            f'aria-label="{p["filled"]} of {total}">{cells}</div><figcaption><b>{escape(p["big"])}</b> '
            f'{escape(p["text"])}</figcaption></figure>')


def home():
    n = stats(RESULTS.get("corpus"), RESULTS.get("timeline") or {}, len(MS))
    sentence = (f"Built from {n[0]['n']} official texts and {n[1]['n']:,} indexed articles, paragraphs and controls, "
                f"with {n[3]['n']} start dates each quoted from the law.")
    rows = "".join(
        f'<li style="--tab:{b[4]}"><div><a class="name" href="{APP}/ask">{escape(b[1])}</a><p>{escape(LINES[b[2]])}</p></div>'
        f'<a class="btn btn-quiet btn-sm" href="{APP}/ask">Ask about it</a></li>' for b in RULEBOOKS)
    qs = "".join(f'<li><a href="{APP}/ask">{escape(f[1])}</a></li>' for f in FAQ[:6])
    tools = [("Ask", "ask", "One question, one cited answer with its date.", "Open Ask"),
             ("Checklist", "checklist", "Which articles bind your product, now and later.", "Build a checklist"),
             ("Gap check", "gapcheck", "What your policy covers and what it misses.", "Check a policy")]
    tool_rows = "".join(f'<li><h3>{t}</h3><p>{d}</p><a class="btn btn-quiet btn-sm" href="{APP}/{s}">{c}</a></li>'
                        for t, s, d, c in tools)
    hero = (f'<section class="hero"><div class="wrap hero-grid"><div class="hero-copy"><h1>Which EU and US rules '
            '<mark>apply to your</mark> AI, data or cyber product?</h1><p class="lead">Plain answers quoted from the '
            'official text, with the date each rule starts to apply. For founders, product teams and engineers, not '
            f'only lawyers.</p>{cta()}<p class="people-ask"><span>People ask</span><a href="{APP}/ask">'
            f'{escape(QUESTIONS[1])}</a></p></div>{stack_html()}</div></section>')
    replay = sec(
        '<div class="split"><div><h2>See an answer, start to finish</h2><p class="lead">Every answer shows how it was '
        'found, quotes the law, and says whether the rule applies today.</p><ul class="points">'
        '<li><strong>It finds the passage first.</strong> The answer is written only from the official text it '
        'retrieved.</li><li><strong>It says no when it should.</strong> If nothing relevant is found, you get a plain '
        'decline, not a guess.</li><li><strong>It checks dates.</strong> Each cited rule shows whether it is in force '
        'on the date you choose.</li></ul><p class="note">Replayed from a real answer on 8 November 2026. Live answers '
        f'take a few seconds.</p><p><a class="btn" href="{APP}/ask">Ask your own question</a></p></div>{demo_html()}</div>',
        "soft")
    how = sec(
        '<h2>How it works</h2><ol class="how"><li><strong>Tell it about you</strong><p>Pick an example such as '
        '“chatbot startup” or answer only what you know.</p></li><li><strong>See what applies</strong><p>Articles '
        'that apply now, ones coming later, and the one question that would settle the rest.</p></li>'
        '<li><strong>Check your policy</strong><p>Paste your policy and see, duty by duty, what it covers, with quotes '
        'from both sides.</p></li></ol>')
    rail = "".join(milestone_html(m, False) for m in MS)
    when = (f'<section class="sec ink"><div class="wrap split split-rail"><div><h2>When the rules start</h2>'
            f'<p class="lead">{len(MS)} start dates, each one taken from the sentence of the law that sets it. Some have '
            f'passed and some are still ahead.</p><p><a class="link-on-dark" href="{APP}/checklist">See which of '
            f'these apply to you</a></p></div><ol class="rail">{rail}</ol></div></section>')
    tools_sec = sec(f'<h2>Three tools, one set of sources</h2><ul class="tools">{tool_rows}</ul>')
    proofs = "".join(dots(p) for p in proof_panels(RESULTS))
    proof_sec = sec(
        '<h2>Can you trust it?</h2><p class="lead">Measured on labelled test questions. In the first panel each dot is '
        f'one percentage point; in the others each dot is one question.</p><div class="proofs">{proofs}</div>'
        f'<p><a href="{APP}/evaluation">See how each number is measured</a></p>', "soft")
    library = sec(
        f'<h2>What is in the library</h2><p class="lead">{sentence}</p><ul class="library">{rows}</ul><p class="note">'
        'ISO/IEC 27001 and 42001 are licensed. Bring your own copy on Gap check. Questions about other rules are '
        'declined instead of guessed.</p>')
    questions = sec(f'<h2>Questions people ask</h2><ul class="qlist">{qs}</ul>'
                    f'<p><a class="btn" href="{APP}/ask">Ask your own question</a></p>')
    close = ('<section class="sec close"><div class="wrap"><h2>Find out which rules apply to you</h2><p class="lead">Pick '
             'a ready-made example or ask about your own product. Every answer quotes the article and shows its '
             f'date.</p>{cta(True)}<p class="note">The first visit after a quiet spell can take a few seconds while the '
             'app wakes up.</p></div></section>')
    body = hero + replay + how + when + tools_sec + proof_sec + library + questions + close
    ld = [{"@context": "https://schema.org", "@type": "SoftwareApplication", "name": NAME, "url": APP,
           "applicationCategory": "BusinessApplication", "operatingSystem": "Web browser",
           "description": "Plain answers about the EU AI Act, GDPR, NIS2, DORA, the Cyber Resilience Act and NIST "
                          "frameworks, quoted from the official text with the date each rule starts to apply.",
           "image": f"{SITE}/og.png", "sameAs": [REPO]}]
    return page("index.html", "Compliance Workbench: EU AI Act, GDPR, NIS2, DORA and CRA answers",
                "Which EU and US rules apply to your AI, data or cyber product? Plain answers quoted from the "
                "official text of the EU AI Act, GDPR, NIS2, DORA, CRA and NIST, with start dates.", body, ld)


def rulebook(b):
    slug, name, line_key, prefixes, color, title, desc = b
    ms = milestones_for(prefixes)
    dates = (f'<ol class="rail">{"".join(milestone_html(m) for m in ms)}</ol>' if ms else
             "<p>These are voluntary frameworks, so there is no legal start date. Use them as a reference for "
             "security and AI risk practices.</p>")
    contents = "".join((f"<h3>{escape(rows_for(p)[0]['instrument'])}</h3>" if len(prefixes) > 1 else "")
                       + contents_html(p) for p in prefixes)
    faqs = [f for f in FAQ if f[3] == slug]
    top = band(name, LINES[line_key], f'<ul class="facts">{instrument_rows(prefixes)}</ul>{cta(True)}', color, slug)
    body = (sec(f"<h2>When it applies</h2>{dates}")
            + (sec(f"<h2>Common questions</h2>{faq_items(faqs)}", "soft") if faqs else "")
            + sec(f'<h2>What the text covers</h2><div class="contents">{contents}</div>'))
    return page(f"{slug}.html", title, desc, body, top=top)


def timeline():
    top = band("When the rules start", "Every start date the app tracks, in order. Each one quotes the sentence of the "
               "law that sets it.", cta(True), "", "")
    body = sec(f'<h2>Every start date</h2><ol class="rail">{"".join(milestone_html(m) for m in MS)}</ol>'
               "<p>NIST SP 800-53, CSF 2.0 and the AI RMF are voluntary, so they have no legal start date.</p>")
    return page("timeline.html", "When the EU AI Act, GDPR, NIS2, DORA and CRA rules start",
                "Every start date for the EU AI Act, GDPR, NIS2, DORA and the Cyber Resilience Act, from 25 May 2018 "
                "to 2 August 2028, each quoted from the article that sets it.", body, top=top)


def faq():
    ld = [{"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": [
        {"@type": "Question", "name": q, "acceptedAnswer": {
            "@type": "Answer", "text": f"{a} Source: {', '.join(dict.fromkeys(ROWS[r]['ref'] for r, _ in qs))}."}}
        for _, q, a, _, qs in FAQ]}]
    top = band("Questions about EU AI, data and cyber rules", "Short answers, each with the words of the law and its "
               "article.", "", "", "")
    return page("faq.html", "EU AI Act, GDPR, NIS2 and CRA questions answered from the law",
                "When must a GDPR breach be reported? When do the AI Act high-risk rules apply? Short answers "
                "quoting the GDPR, AI Act, NIS2, DORA and CRA with article references.", sec(faq_items(FAQ)), ld,
                top=top)


def build(out: Path = HERE / "dist") -> list[str]:
    pages = [home(), timeline(), faq()] + [rulebook(b) for b in RULEBOOKS]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for path, html in pages:
        (out / path).write_text(html, encoding="utf-8", newline="\n")
    shutil.copy(HERE / "style.css", out / "style.css")
    shutil.copy(HERE / "static/og.png", out / "og.png")
    urls = "".join(f"<url><loc>{SITE}/{'' if p == 'index.html' else p}</loc></url>" for p, _ in pages)
    (out / "sitemap.xml").write_text('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/'
                                     f'schemas/sitemap/0.9">{urls}</urlset>\n', encoding="utf-8")
    (out / "robots.txt").write_text(f"User-agent: *\nAllow: /\nSitemap: {SITE}/sitemap.xml\n", encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return [p for p, _ in pages]


if __name__ == "__main__":
    print("\n".join(build()))

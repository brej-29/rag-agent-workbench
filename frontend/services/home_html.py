"""Home page blocks: HTML built only from published snapshots (eval/results) and one replayed real answer.
Pure functions, so tests can check every number and date that reaches the page."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from .presenters import esc

COLORS = {"AI Act": "#2F45D6", "GDPR": "#237B78", "NIS2": "#7651A8", "DORA": "#8B4D72", "CRA": "#9A6518"}
_SHORT = {"eu-ai-act": "AI Act", "gdpr": "GDPR", "nis2": "NIS2", "nis2-ir": "NIS2 IR", "dora": "DORA", "cra": "CRA"}

# (applicability rule id, condition) -> (framework, what starts to apply). Dates come from timeline.json,
# which copies applicability.yaml, where each date is checked against a verbatim quote of the law.
MILESTONES = {
    ("gdpr", None): ("GDPR", "Applies in full"),
    ("nis2", None): ("NIS2", "National cybersecurity rules apply"),
    ("nis2-ir", None): ("NIS2", "Technical rules for digital providers"),
    ("dora", None): ("DORA", "Applies to financial firms"),
    ("ai-act-chapters-1-2", None): ("AI Act", "AI literacy and banned practices"),
    ("ai-act-gpai-governance", None): ("AI Act", "General-purpose AI models, governance and penalties"),
    ("cra-chapter-4", None): ("CRA", "Rules for conformity assessment bodies"),
    ("ai-act-general", None): ("AI Act", "Most remaining rules"),
    ("cra-article-14", None): ("CRA", "Reporting exploited vulnerabilities and incidents"),
    ("ai-act-new-prohibitions", None): ("AI Act", "Newly added prohibitions"),
    ("ai-act-high-risk", "annex_iii"): ("AI Act", "High-risk uses such as hiring and credit"),
    ("cra-general", None): ("CRA", "Applies in full"),
    ("ai-act-high-risk", "annex_i"): ("AI Act", "High-risk AI inside regulated products"),
}

QUESTIONS = [
    "Does the AI Act cover my CV-screening tool?",
    "When must a personal data breach be reported?",
    "What are the incident reporting deadlines across EU rules?",
    "What must a connected camera maker do under the CRA?",
]


def nice_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.day} {d:%b %Y}"


def until(iso: str, today: date) -> str:
    days = (date.fromisoformat(iso) - today).days
    if days <= 0:
        return "In force"
    if days < 60:
        return f"In {days} day{'s' if days != 1 else ''}"
    return f"In {round(days / 30.44)} months"


def basis_ref(basis: str) -> str:
    """'gdpr/art-99/par-2' -> 'GDPR Art. 99(2)'."""
    parts = basis.split("/")
    ref = f"{_SHORT.get(parts[0], parts[0])} Art. {parts[1].removeprefix('art-')}" if len(parts) > 1 else basis
    return ref + (f"({parts[2].removeprefix('par-')})" if len(parts) > 2 else "")


def timeline(snapshot: Dict[str, Any], today: date) -> List[Dict[str, Any]]:
    out = []
    for m in snapshot.get("milestones") or []:
        key = (m["id"], m.get("condition"))
        if key in MILESTONES:
            fw, label = MILESTONES[key]
            out.append({"framework": fw, "label": label, "from": m["from"], "date": nice_date(m["from"]),
                        "past": m["from"] <= today.isoformat(), "until": until(m["from"], today),
                        "quote": m.get("quote") or "", "basis": basis_ref(m["basis"])})
    return out


def timeline_summary(items: List[Dict[str, Any]]) -> str:
    past = sum(i["past"] for i in items)
    nxt = next((i for i in items if not i["past"]), None)
    head = f"{past} of {len(items)} start dates have passed."
    if not nxt:
        return head
    return f"{head} Next: {nxt['framework']}, {nxt['label'][0].lower()}{nxt['label'][1:]}, on {nxt['date']}."


def timeline_html(items: List[Dict[str, Any]]) -> str:
    rows, marked = [], False
    for n, i in enumerate(items):
        if not i["past"] and not marked:
            rows.append('<li class="cw-today" aria-label="Today"><span>Today</span></li>')
            marked = True
        quote = f'<blockquote>“{esc(i["quote"])}”<cite>{esc(i["basis"])}</cite></blockquote>' if i["quote"] else ""
        rows.append(
            f'<li class="cw-ms {"is-past" if i["past"] else "is-future"}" tabindex="0" '
            f'style="--tab:{COLORS[i["framework"]]};--i:{n}">'
            f'<span class="cw-ms-dot" aria-hidden="true"></span>'
            f'<time datetime="{i["from"]}">{esc(i["date"])}</time>'
            f'<b>{esc(i["framework"])}</b><p>{esc(i["label"])}</p><small>{esc(i["until"])}</small>{quote}</li>')
    return f'<ol class="cw-timeline">{"".join(rows)}</ol>'


def stats(corpus: Optional[List[Dict[str, Any]]], snapshot: Dict[str, Any], n_dates: int) -> List[Dict[str, Any]]:
    out = []
    if corpus:
        out += [{"n": len(corpus), "label": "official texts, EU and NIST"},
                {"n": sum(c["provisions"] for c in corpus), "label": "articles, paragraphs and controls indexed"}]
    if snapshot.get("obligations"):
        out.append({"n": snapshot["obligations"], "label": "duty-bearing articles in the checklist"})
    if n_dates:
        out.append({"n": n_dates, "label": "start dates tracked, each quoted from the law"})
    return out


def stats_html(items: List[Dict[str, Any]]) -> str:
    cells = "".join(f'<div class="cw-stat" style="--to:{i["n"]};--i:{k}"><b aria-hidden="true"></b>'
                    f'<span class="cw-sr">{i["n"]:,}</span><p>{esc(i["label"])}</p></div>'
                    for k, i in enumerate(items))
    return f'<div class="cw-stats cw-reveal">{cells}</div>'


def ticker_html() -> str:
    qs = "".join(f'<i style="--q:{n}"><span>{esc(q)}</span></i>' for n, q in enumerate(QUESTIONS))
    return f'<p class="cw-ticker"><span>People ask</span><span class="cw-ticker-q">{qs}</span></p>'


# A real Ask run against the API on 2026-11-08 (6 passages retrieved, 6 cleared the floor, cited GDPR Art. 33(1)).
DEMO = {
    "question": "When must a personal data breach be reported?",
    "steps": ["Understood as a question about the rules", "Retrieved 6 passages from the official texts",
              "6 of 6 passages cleared the relevance floor", "Wrote the answer and checked each citation"],
    "answer": ("The controller must notify the supervisory authority without undue delay and, where feasible, "
               "not later than 72 hours after becoming aware of the personal data breach."),
    "source": "GDPR Art. 33(1)", "status": "in force since 2018-05-25",
}


def demo_html() -> str:
    d = DEMO
    steps = "".join(f'<li style="--s:{n}">{esc(t)}</li>' for n, t in enumerate(d["steps"]))
    words = " ".join(f'<span style="--w:{n}">{esc(w)}</span>' for n, w in enumerate(d["answer"].split()))
    return f"""<div class="cw-demo cw-reveal" aria-label="Example answer">
  <div class="cw-demo-q">{esc(d["question"])}</div>
  <ol class="cw-demo-steps">{steps}</ol>
  <p class="cw-demo-a">{words} <sup class="cw-demo-cite">1</sup></p>
  <div class="cw-demo-src"><span class="cw-chip">1 · {esc(d["source"])}</span>
    <span class="cw-badge">{esc(d["status"])}</span></div>
</div>"""


def proof_panels(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Measured numbers as (filled, total) dot grids with one plain sentence each."""
    out = []
    r = results.get("regs_retrieval")
    if r:
        pct = round(r["aggregate"]["recall"] * 100)
        out.append({"big": f"{pct}%", "filled": pct, "total": 100, "tone": "sage",
                    "text": f"of the right articles were in the top {r['k']} results, averaged over {r['n']} test "
                            "questions."})
    a = results.get("regs_abstain")
    if a:
        out.append({"big": f"{a['correct_abstain']} of {a['unanswerable']}", "filled": a["correct_abstain"],
                    "total": a["unanswerable"], "tone": "stamp",
                    "text": "questions about rules it does not cover (such as HIPAA) were declined, not guessed."})
        ok = a["answerable"] - a["false_abstain"]
        out.append({"big": f"{a['false_abstain']} of {a['answerable']}", "filled": ok, "total": a["answerable"],
                    "tone": "sage", "text": "questions it does cover were wrongly declined. Green dots were answered."})
    rt = results.get("router")
    if rt:
        hit = round(rt["accuracy"] * rt["n"])
        out.append({"big": f"{rt['accuracy']:.0%}", "filled": hit, "total": rt["n"], "tone": "primary",
                    "text": f"of {rt['n']} requests went to the right tool: ask, checklist or gap check."})
    return out


def proof_html(panels: List[Dict[str, Any]]) -> str:
    cards = []
    for p in panels:
        dots = "".join(f'<span class="{"on" if k < p["filled"] else ""}" style="--d:{k}"></span>'
                       for k in range(p["total"]))
        cards.append(f'<figure class="cw-proof-card cw-reveal tone-{p["tone"]}">'
                     f'<div class="cw-dots" role="img" aria-label="{p["filled"]} of {p["total"]}">{dots}</div>'
                     f'<figcaption><b>{esc(p["big"])}</b> {esc(p["text"])}</figcaption></figure>')
    return f'<div class="cw-proof-grid">{"".join(cards)}</div>'


# Adds .is-in to each .cw-reveal block as it scrolls into view; without JS everything shows in its final state.
REVEAL_JS = """<script>
(() => {
  const root = document.documentElement;
  root.classList.add("cw-js");
  try { CSS.registerProperty({ name: "--n", syntax: "<integer>", inherits: false, initialValue: "0" }); } catch (e) {}
  const io = window.__cwIO || (window.__cwIO = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) { e.target.classList.add("is-in"); window.__cwIO.unobserve(e.target); }
  }), { threshold: 0.18 }));
  const scan = () => document.querySelectorAll(".cw-reveal:not(.is-in)").forEach((el) => io.observe(el));
  scan(); setTimeout(scan, 400); setTimeout(scan, 1500);
})();
</script>"""

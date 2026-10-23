"""EU Official Journal XHTML (original and consolidated markup) → provision records."""
import re
from typing import Dict, List, Optional

from bs4 import BeautifulSoup, Tag

from corpus.sources import Source

_MARKER = re.compile(r"[▼►]\s*[A-Z]\d*|◄")
_MODIFIER = re.compile(r"►\s*(M\d+)\s+((?:Regulation|Directive|Decision)\s+\((?:EU|EC|EEC)\)\s+(?:No\s+)?\d+/\d+)",
                       re.IGNORECASE)
_OJ_PARA = re.compile(r"^\d{3}\.(\d{3}[a-z]?)$")
_TITLE_CLASSES = {"oj-ti-art", "oj-sti-art", "title-article-norm", "stitle-article-norm"}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", _MARKER.sub(" ", text)).strip()


def _text(el: Tag) -> str:
    return _clean(el.get_text(" ", strip=True))


def _modifiers(soup: BeautifulSoup) -> Dict[str, str]:
    # Consolidated headers print the act in capitals ("REGULATION (EU) 2026/1744"); store it in citation case.
    return {code: act[0].upper() + act[1:].lower().replace("(eu)", "(EU)").replace("(ec)", "(EC)").replace("(eec)", "(EEC)")
            for code, act in _MODIFIER.findall(soup.get_text(" "))}


def _codes(raw: str) -> List[str]:
    return re.findall(r"[▼►]\s*(M\d+)", raw)


def _location(el: Tag) -> Dict[str, Optional[str]]:
    chapter = section = None
    for parent in el.parents:
        pid = parent.get("id", "") if isinstance(parent, Tag) else ""
        m = re.fullmatch(r"cpt_([IVXLC]+)(?:\.sct_(\d+))?", pid)
        if m:
            chapter = chapter or m.group(1)
            section = section or m.group(2)
    return {"chapter": chapter, "section": section}


def _base(src: Source) -> dict:
    return {"framework": src.framework, "instrument": src.instrument, "binding": src.binding,
            "licence": src.licence, "source_url": src.url, "amended_by": [], "guidance": None}


def _paragraphs(art: Tag):
    """Yield (label, element, preceding_marker_text) for numbered paragraphs, both markups."""
    prev = ""
    for child in art.find_all(recursive=False):
        cls = set(child.get("class") or [])
        m = _OJ_PARA.match(child.get("id", ""))
        if m:
            yield m.group(1).lstrip("0") or "0", child, prev
        elif "norm" in cls and child.find("span", class_="no-parag", recursive=False):
            label = child.find("span", class_="no-parag", recursive=False).get_text(strip=True).rstrip(".")
            # Amending articles quote inserted paragraphs ("‘5."): ids must stay ASCII.
            yield re.sub(r"[^0-9A-Za-z]", "", label), child, prev
        prev = child.get_text(" ", strip=True) if len(child.get_text(strip=True)) < 12 else ""


def _article(art: Tag, src: Source, mods: Dict[str, str]) -> List[dict]:
    num = art["id"].split("_", 1)[1]
    title_el = art.find(class_=["oj-sti-art", "stitle-article-norm"])
    loc = _location(art)
    path = [p for p in (f"Chapter {loc['chapter']}" if loc["chapter"] else None,
                        f"Section {loc['section']}" if loc["section"] else None) if p] + [f"Article {num}"]
    common = {**_base(src), **loc, "article": num, "title": _text(title_el) if title_el else "", "path": path}
    out = []
    for label, el, prev in _paragraphs(art):
        raw = el.get_text(" ") + " " + prev
        out.append({**common, "id": f"{src.prefix}/art-{num}/par-{label}", "ref": f"{src.short} Art. {num}({label})",
                    "kind": "article_paragraph", "paragraph": label, "text": _text(el),
                    "amended_by": sorted({mods[c] for c in _codes(raw) if c in mods})})
    if not out:
        body = [c for c in art.find_all(recursive=False) if not (set(c.get("class") or []) & _TITLE_CLASSES)
                and not c.get("id", "").endswith(".tit_1")]
        raw = " ".join(c.get_text(" ") for c in body)
        out.append({**common, "id": f"{src.prefix}/art-{num}", "ref": f"{src.short} Art. {num}", "kind": "article",
                    "paragraph": None, "text": _clean(raw),
                    "amended_by": sorted({mods[c] for c in _codes(raw) if c in mods})})
    return out


_DEC_SECTION = re.compile(r"^(\d+)\.\s+\S")
_DEC_POINT = re.compile(r"^(\d+\.\d+)\.\s+(.+)$")


def _decimal_points(anx: Tag, src: Source, common: dict, num: str) -> List[dict]:
    """Annexes numbered 1. / 1.1. / 1.1.1. in plain headings and tables (e.g. Implementing Regulation
    2024/2690): one record per second-level point, its sub-points as text, its section heading in the path."""
    out, section, point = [], None, None
    for block in anx.find_all(["p", "table"], recursive=False):
        text = _text(block)
        if block.name == "p" and "oj-ti-grseq-1" in (block.get("class") or []):
            m = _DEC_POINT.match(text)
            if m:
                point = {**common, "id": f"{src.prefix}/anx-{num}/pt-{m.group(1)}",
                         "ref": f"{src.short} Annex, point {m.group(1)}", "kind": "annex_point",
                         "title": m.group(2), "text": "", "path": [f"Annex {num}", section, f"{m.group(1)} {m.group(2)}"]}
                out.append(point)
                continue
            if _DEC_SECTION.match(text):
                section, point = text, None
                continue
        if point is not None and text:
            point["text"] = f"{point['text']} {text}".strip()
    return out


def _annex(anx: Tag, src: Source) -> List[dict]:
    num = anx["id"].split("_", 1)[1]
    common = {**_base(src), "chapter": None, "section": None, "article": None, "paragraph": None,
              "path": [f"Annex {num}"]}
    heads = [_text(p) for p in anx.find_all("p", class_=["oj-doc-ti", "title-annex-1", "title-annex-2"])]
    title = " ".join(h for h in heads if not h.upper().startswith("ANNEX"))
    out = [{**common, "id": f"{src.prefix}/anx-{num}", "ref": f"{src.short} Annex {num}", "kind": "annex",
            "title": title, "text": _text(anx)}]
    decimal = _decimal_points(anx, src, common, num)
    if decimal:
        sections = [_text(p) for p in anx.find_all("p", class_="oj-ti-grseq-1", recursive=False)
                    if _DEC_SECTION.match(_text(p)) and not _DEC_POINT.match(_text(p))]
        out[0]["text"] = " ".join(heads + sections)
        return out + decimal
    # Top-level numbered points: tables (OJ markup), grid containers or plain divs (consolidated markup).
    # Points restart in each section ("Section A", "Section B"), so the section is part of the id.
    section = None
    for block in anx.find_all(["p", "table", "div"], recursive=False):
        if block.name == "p":
            m = re.match(r"(?:Section|SECTION|Part|PART)\s+([A-Z0-9]+)\b", _text(block))
            if m and "oj-normal" not in (block.get("class") or []) and "norm" not in (block.get("class") or []):
                section = m.group(1)
            continue
        if block.name == "table":
            first = block.find("td")
            label = first.get_text(strip=True).rstrip(".") if first else ""
        else:
            m = re.match(r"(\d+)\.\s", _text(block))
            label = m.group(1) if m else ""
        if re.fullmatch(r"\d+", label):
            sec_id, sec_ref = (f"/sec-{section}", f", Section {section}") if section else ("", "")
            out.append({**common, "id": f"{src.prefix}/anx-{num}{sec_id}/pt-{label}",
                        "ref": f"{src.short} Annex {num}{sec_ref}, point {label}", "kind": "annex_point",
                        "title": title, "text": _text(block)})
    if len(out) > 1:
        out[0]["text"] = " ".join(heads + [_text(p) for p in anx.find_all("p", class_=["oj-normal", "norm"], recursive=False)])
    return out


def _recital(rct: Tag, src: Source) -> dict:
    num = rct["id"].split("_", 1)[1]
    cells = rct.find_all("td")
    text = _text(cells[-1]) if len(cells) > 1 else re.sub(r"^\(\d+\)\s*", "", _text(rct))
    return {**_base(src), "id": f"{src.prefix}/rct-{num}", "ref": f"{src.short} Recital {num}", "kind": "recital",
            "title": "", "text": text, "path": [f"Recital {num}"], "chapter": None, "section": None,
            "article": None, "paragraph": None}


def parse_eu(html: str, source: Source) -> List[dict]:
    soup = BeautifulSoup(html, "html.parser")
    mods = _modifiers(soup)
    out: List[dict] = []
    if source.role == "text":
        for art in soup.find_all("div", id=re.compile(r"^art_\d+[a-z]?$")):
            out += _article(art, source, mods)
        for anx in soup.find_all("div", id=re.compile(r"^anx_(?:[IVXLC]+|\d+)[a-z]?$")):
            out += _annex(anx, source)
    for rct in soup.find_all("div", id=re.compile(r"^rct_\d+$")):
        out.append(_recital(rct, source))
    return out

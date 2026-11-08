"""NIST OSCAL catalogs (SP 800-53, CSF 2.0) and the AI RMF Playbook → provision records."""
import re
from typing import Dict, List

from corpus.sources import Source

_PARAM = re.compile(r"\{\{\s*insert:\s*param,\s*([\w.\-]+)\s*\}\}")
_KIND = {"SP800-53": "control", "SP800-53-enhancement": "control_enhancement",
         "category": "category", "subcategory": "subcategory"}


def _prop(obj: dict, name: str):
    return next((p["value"] for p in obj.get("props", []) if p.get("name") == name), None)


def _param_labels(control: dict) -> Dict[str, str]:
    out = {}
    for p in control.get("params", []):
        choices = (p.get("select") or {}).get("choice")
        out[p["id"]] = p.get("label") or ("; ".join(choices) if choices else "organization-defined value")
    return out


def _prose(part: dict, params: Dict[str, str]) -> List[str]:
    label = _prop(part, "label")
    prose = _PARAM.sub(lambda m: f"[{params.get(m.group(1), 'organization-defined value')}]", part.get("prose", ""))
    lines = [" ".join(x for x in (label, prose) if x)] if (label or prose) else []
    for child in part.get("parts", []):
        lines += _prose(child, params)
    return lines


def _part(control: dict, name: str, params: Dict[str, str]) -> str:
    return " ".join(line for p in control.get("parts", []) if p.get("name") == name for line in _prose(p, params))


def _walk(controls: List[dict], src: Source, path: List[str], params_up: Dict[str, str]) -> List[dict]:
    out = []
    for c in controls:
        if _prop(c, "status") == "withdrawn":
            continue
        params = {**params_up, **_param_labels(c)}
        label = _prop(c, "label") or c["id"]
        kind = _KIND.get(c.get("class", ""), "control")
        heading = label if c["title"] in label else f"{label} {c['title']}"
        cite = label if " " not in label else c["id"]  # CSF category labels embed the name: cite by id
        out.append({"id": f"{src.prefix}/{c['id']}", "ref": f"{src.short} {cite}", "kind": kind,
                    "framework": src.framework, "instrument": src.instrument, "binding": src.binding,
                    "licence": src.licence, "source_url": src.url, "title": c["title"],
                    "text": _part(c, "statement", params), "guidance": _part(c, "guidance", params) or None,
                    "path": path + [heading], "chapter": None, "section": None, "article": None,
                    "paragraph": None, "amended_by": []})
        out += _walk(c.get("controls", []), src, path + [heading], params)
    return out


def parse_oscal(catalog: dict, source: Source) -> List[dict]:
    out = []
    for group in catalog["groups"]:
        out += _walk(group.get("controls", []), source, [group["title"]], {})
    return out


def parse_playbook(entries: List[dict], source: Source) -> List[dict]:
    return [{"id": f"{source.prefix}/{e['title'].lower().replace(' ', '-')}", "ref": f"{source.short} {e['title']}",
             "kind": "subcategory", "framework": source.framework, "instrument": source.instrument,
             "binding": source.binding, "licence": source.licence, "source_url": source.url,
             "title": e["title"], "text": e["description"].strip(), "guidance": (e.get("section_about") or "").strip() or None,
             "path": [e["type"], e["category"], e["title"]], "chapter": None, "section": None, "article": None,
             "paragraph": None, "amended_by": []} for e in entries]

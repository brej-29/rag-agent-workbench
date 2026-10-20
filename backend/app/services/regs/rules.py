"""Three-valued (yes / no / unknown) evaluation of applicability conditions over profile facts (spec §11.2).

Condition grammar: {"all": [...]}, {"any": [...]}, {"not": cond}, {"use": def_name}, or a single fact test
{fact: value | [values]} (true when the fact's value, or any element of a list-valued fact, is in the values).
"""
from typing import Any, Dict, Optional, Tuple

Explained = Tuple[Optional[bool], Dict[str, Any]]


def _test(fact: str, wanted: Any, profile: Dict[str, Any]) -> Explained:
    value = profile.get(fact)
    if value is None:
        return None, {fact: None}
    allowed = wanted if isinstance(wanted, list) else [wanted]
    if isinstance(value, list):
        return any(v in allowed for v in value), {fact: value}
    return value in allowed, {fact: value}


def evaluate(cond: Dict[str, Any], profile: Dict[str, Any], defs: Dict[str, Any]) -> Explained:
    """Return (outcome, the facts that decided it); unknown facts map to None."""
    if "use" in cond:
        return evaluate(defs[cond["use"]], profile, defs)
    if "not" in cond:
        value, facts = evaluate(cond["not"], profile, defs)
        return (None if value is None else not value), facts
    for op, decisive in (("all", False), ("any", True)):
        if op in cond:
            results = [evaluate(c, profile, defs) for c in cond[op]]
            hit = next((r for r in results if r[0] is decisive), None)
            if hit:
                return decisive, hit[1]
            unknown = [facts for value, facts in results if value is None]
            if unknown:
                return None, {k: v for facts in unknown for k, v in facts.items()}
            return (not decisive), {k: v for _, facts in results for k, v in facts.items()}
    (fact, wanted), = cond.items()
    return _test(fact, wanted, profile)

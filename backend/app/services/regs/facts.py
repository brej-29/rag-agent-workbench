"""Profile facts (facts.yaml): the questions the checklist decides on."""
from functools import lru_cache
from typing import Any, Dict, List

import yaml

from app.services.regs import DATA_DIR


@lru_cache(maxsize=1)
def load_facts() -> List[dict]:
    return yaml.safe_load((DATA_DIR / "facts.yaml").read_text(encoding="utf-8"))


def normalize_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Drop unknowns; fill facts implied by others (a deployer uses an AI system, a controller processes data)."""
    p = {k: v for k, v in (profile or {}).items() if v is not None and v != "unknown"}
    if p.get("ai_roles") and "ai_system" not in p:
        p["ai_system"] = True
    if p.get("data_roles") and "personal_data" not in p:
        p["personal_data"] = True
    if p.get("gpai_systemic_risk") and "gpai_model" not in p:
        p["gpai_model"] = True
    return p

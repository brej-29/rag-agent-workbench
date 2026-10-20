from datetime import date

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.chat import ChatRequest, ChatResponse, Profile, SourceHit


def test_regs_settings_defaults():
    s = Settings()
    assert (s.REGS_NAMESPACE, s.REGS_MIN_SCORE, s.REGS_TOP_K, s.REGS_CANDIDATES) == ("regs-v1", 0.30, 6, 30)


def test_request_accepts_as_of_frameworks_and_profile():
    r = ChatRequest(query="q", as_of="2026-10-20", frameworks=["GDPR"],
                    profile={"high_risk": "annex_iii", "roles": ["deployer"]})
    assert r.as_of == date(2026, 10, 20) and r.frameworks == ["GDPR"] and r.profile.high_risk == "annex_iii"


def test_profile_rejects_unknown_values():
    with pytest.raises(ValidationError):
        Profile(high_risk="very")
    with pytest.raises(ValidationError):
        Profile(roles=["wizard"])


def test_regs_fields_default_empty():
    s = SourceHit(source="EU AI Act", title="t", chunk_text="c")
    assert s.ref == "" and s.status is None and s.applies_from is None
    r = ChatResponse(answer="a", trace={})
    assert r.warnings == [] and r.profile_updates == {} and r.intent is None and r.as_of is None

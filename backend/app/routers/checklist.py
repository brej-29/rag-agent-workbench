"""Deterministic checklist endpoints (no LLM): which EU-law articles bind a profile on a date."""
from datetime import date
from typing import Any, Dict, List

from fastapi import APIRouter, Request

from app.core.rate_limit import limiter
from app.schemas.checklist import ChecklistRequest
from app.services.regs.checklist import build_checklist
from app.services.regs.facts import load_facts

router = APIRouter(tags=["checklist"])


@router.get("/checklist/facts", summary="Profile facts the checklist uses, with questions and allowed values")
async def checklist_facts() -> List[Dict[str, Any]]:
    return load_facts()


@router.post("/checklist", summary="Which articles of the AI Act, GDPR, NIS2, DORA and CRA bind this profile")
@limiter.limit("60/minute")
async def checklist(request: Request, payload: ChecklistRequest) -> Dict[str, Any]:  # noqa: ARG001
    profile = payload.profile.model_dump(exclude_none=True) if payload.profile else {}
    return build_checklist(profile, payload.as_of or date.today(), with_text=True)

"""Stateless private-workspace endpoints (spec §15): nothing sent here is stored, indexed or logged."""
from datetime import date
from typing import Any, AsyncGenerator, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.core.rate_limit import limiter
from app.schemas.workspace import GapCheckRequest, ParseRequest
from app.services.chat.streaming import _sse
from app.services.llm.groq_llm import get_llm
from app.services.regs import gapcheck
from app.services.regs.standards import STANDARDS, coverage, detect_standard, parse_clauses

router = APIRouter(tags=["workspace"])


@router.post("/workspace/parse", summary="Split a policy into passages or a licensed standard into clauses (not stored)")
@limiter.limit("30/minute")
async def parse(request: Request, payload: ParseRequest) -> Dict[str, Any]:  # noqa: ARG001
    if payload.kind == "policy":
        return {"kind": "policy", "title": payload.title, "chars": len(payload.text),
                "passages": gapcheck.split_passages(payload.text)}
    standard = payload.standard or detect_standard(payload.text)
    if not standard:
        raise HTTPException(422, "Could not tell which standard this is. Choose ISO/IEC 27001 or ISO/IEC 42001.")
    clauses = parse_clauses(payload.text, standard)
    return {"kind": "standard", "title": payload.title, "standard": standard, "name": STANDARDS[standard],
            "clauses": clauses, "coverage": coverage(clauses, standard)}


def _plan(payload: GapCheckRequest, refuse_big: bool) -> dict:
    profile = payload.profile.model_dump(exclude_none=True) if payload.profile else {}
    try:
        p = gapcheck.plan(payload.policy_text, payload.articles, profile, payload.as_of or date.today())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    msg = gapcheck.budget_error(p) if refuse_big else None
    if msg:
        raise HTTPException(422, msg)
    return p


@router.post("/gapcheck/plan", summary="What a gap check would test and roughly what it costs (no model call)")
@limiter.limit("30/minute")
async def gapcheck_plan(request: Request, payload: GapCheckRequest) -> Dict[str, Any]:  # noqa: ARG001
    return gapcheck.plan_summary(_plan(payload, refuse_big=False))


@router.post("/gapcheck", summary="Check a policy against the selected articles, duty by duty")
@limiter.limit("10/minute")
async def gapcheck_run(request: Request, payload: GapCheckRequest) -> Dict[str, Any]:  # noqa: ARG001
    p, llm = _plan(payload, refuse_big=True), get_llm()
    results = [await run_in_threadpool(gapcheck.check_unit, u, llm) for u in p["units"]]
    return {"plan": gapcheck.plan_summary(p), "results": results, "summary": gapcheck.summarize(results)}


@router.post("/gapcheck/stream", summary="Gap check streamed per article (SSE: plan, article…, done)")
@limiter.limit("10/minute")
async def gapcheck_stream(request: Request, payload: GapCheckRequest) -> StreamingResponse:  # noqa: ARG001
    p = _plan(payload, refuse_big=True)

    async def events() -> AsyncGenerator[str, None]:
        llm, results = get_llm(), []
        yield _sse("plan", gapcheck.plan_summary(p))
        for unit in p["units"]:
            result = await run_in_threadpool(gapcheck.check_unit, unit, llm)
            results.append(result)
            yield _sse("article", result)
        yield _sse("done", {"summary": gapcheck.summarize(results)})

    return StreamingResponse(events(), media_type="text/event-stream")

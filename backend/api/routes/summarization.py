"""Синхронная генерация и просмотр истории запросов."""

import json
from datetime import UTC, date, datetime, time
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select

from backend.agents.state import CriticCriterion, ErrorCode, ExecutionStatus, PipelineMode, ValidationCode
from backend.api.deps import CurrentUser, DBSession
from backend.api.runtime import SummaryRuntime, during_request, get_runtime, summary_result
from backend.api.schemas.summarization import (
    BatchInput,
    CallTrace,
    Configuration,
    EnsembleInput,
    EnsembleResult,
    GroupResult,
    ReasoningStep,
    RequestDetails,
    RequestPreview,
    RequestReasoning,
    SummarizeInput,
    SummaryResult,
    VariantsInput,
)
from backend.db.models import ReasoningLog, SummarizationRequest, User
from backend.db.llm_calls import LLMCallStore
from backend.db.reasoning import ReasoningLogStore
from backend.utils.text_processing import detect_language

router = APIRouter(prefix="/api/v1", tags=["Суммаризация"])
Runtime = Annotated[SummaryRuntime, Depends(get_runtime)]
_CODES = set(ErrorCode) | set(ValidationCode) | {"validation_warnings_present"}


def _rag_language(item: SummarizeInput) -> str | None:
    return item.rag_language or detect_language(item.text)


@router.post("/summarize")
async def summarize(payload: SummarizeInput, request: Request, user: CurrentUser, runtime: Runtime) -> SummaryResult:
    request_id = UUID(request.state.request_id)
    candidate = await during_request(
        request, lambda: runtime.generate(payload.text, _configuration(payload), user.id, request_id,
                                          category=payload.category, language=_rag_language(payload), queue_review=True,
                                          summary_language=payload.summary_language),
        runtime.settings.api_group_timeout_seconds,
    )
    if candidate.error:
        raise HTTPException(candidate.error.http_status, candidate.error.model_dump(mode="json"))
    return candidate.result


def _configuration(payload: SummarizeInput) -> Configuration:
    return Configuration.model_validate(payload.model_dump(exclude={"text", "category", "rag_language"}))


async def _group(request: Request, payloads: list[SummarizeInput], user: User, runtime: SummaryRuntime) -> GroupResult:
    async def run():
        return GroupResult(results=[
            await runtime.generate(item.text, _configuration(item), user.id, uuid4(), category=item.category,
                                   language=_rag_language(item), queue_review=True, summary_language=item.summary_language)
            for item in payloads
        ])

    return await during_request(request, run, runtime.settings.api_group_timeout_seconds)


@router.post("/summarize/batch")
async def summarize_batch(payload: BatchInput, request: Request, user: CurrentUser, runtime: Runtime) -> GroupResult:
    return await _group(request, payload.items, user, runtime)


@router.post("/summarize/multi")
async def summarize_multi(payload: VariantsInput, request: Request, user: CurrentUser, runtime: Runtime) -> GroupResult:
    items = [SummarizeInput(text=payload.text, category=payload.category, rag_language=payload.rag_language,
                            **config.model_dump())
             for config in payload.variants]
    return await _group(request, items, user, runtime)


@router.post("/summarize/ensemble")
async def summarize_ensemble(
    payload: EnsembleInput, request: Request, user: CurrentUser, runtime: Runtime,
) -> EnsembleResult:
    items = [SummarizeInput(text=payload.text, category=payload.category, max_sentences=payload.max_sentences,
                            rag_language=payload.rag_language, summary_language=payload.summary_language,
                            mode="full", **config.model_dump()) for config in payload.candidates]
    outcome = await _group(request, items, user, runtime)
    eligible = (
        candidate.result for candidate in outcome.results
        if not candidate.error and candidate.result is not None and candidate.result.status == "verified"
        and not candidate.result.factual_review_required and candidate.result.critic_score is not None
    )
    winner = max(eligible, key=lambda summary: summary.critic_score, default=None)
    return EnsembleResult(results=outcome.results, criterion="verified_then_highest_critic_score", winner=winner)


async def _owned_request(request_id: UUID, user: CurrentUser, session: DBSession) -> SummarizationRequest:
    row = await session.get(SummarizationRequest, request_id)
    if row is None:
        raise HTTPException(404, "Запрос не найден.")
    if row.user_id != user.id and user.role != "admin":
        raise HTTPException(403, "Доступ к этому запросу запрещён.")
    return row


OwnedRequest = Annotated[SummarizationRequest, Depends(_owned_request)]


@router.get("/requests")
async def request_list(
    user: CurrentUser, session: DBSession, mode: PipelineMode | None = None, status: ExecutionStatus | None = None,
    date_from: date | None = None, date_to: date | None = None, limit: Annotated[int, Query(ge=1, le=100)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[RequestPreview]:
    table = SummarizationRequest
    if date_from and date_to and date_to < date_from:
        raise HTTPException(422, "Начальная дата должна предшествовать конечной.")
    statement = select(table).where(table.user_id == user.id)
    if mode:
        statement = statement.where(table.mode == mode)
    if status:
        statement = statement.where(table.status == status)
    if date_from:
        statement = statement.where(table.created_at >= datetime.combine(date_from, time.min, UTC))
    if date_to:
        statement = statement.where(table.created_at <= datetime.combine(date_to, time.max, UTC))
    rows = await session.scalars(statement.order_by(table.created_at.desc(), table.id.desc()).offset(offset).limit(limit))
    return [RequestPreview(request_id=row.id, mode=row.mode, status=row.status,
                           preview=(row.input_text or "")[:120], created_at=row.created_at) for row in rows]


@router.get("/requests/{request_id}")
async def request_detail(row: OwnedRequest) -> RequestDetails:
    result = summary_result(row.input_text, {
        **(row.parameters or {}), "request_id": row.id, "mode": row.mode,
        "status": row.status, "final_summary": row.summary,
    })
    return RequestDetails(**result.model_dump(), user_id=row.user_id, created_at=row.created_at)


def _reasoning_step(row: ReasoningLog) -> ReasoningStep:
    try:
        output = json.loads(row.output_summary or "{}")
    except ValueError:
        output = {}
    output = output if isinstance(output, dict) else {}
    criteria = output.get("critic_criteria_scores")
    criteria = criteria if isinstance(criteria, dict) else {}
    params = row.parameters or {}
    return ReasoningStep(
        id=row.id, step=params.get("step"), iteration=params.get("iteration"), agent=row.agent_name,
        status=row.status, model=row.model_name, started_at=row.started_at, finished_at=row.finished_at,
        duration_ms=params.get("duration_ms"), validation_passed=output.get("validation_passed"),
        factual_review_required=output.get("factual_review_required"), critic_score=output.get("critic_score"),
        critic_passed=output.get("critic_passed"), critical_factual_error=output.get("critical_factual_error"),
        missing_key_fact=output.get("missing_key_fact"), critic_fallback_used=output.get("critic_fallback_used"),
        editor_changed=output.get("editor_changed"),
        critic_criteria_scores={key: score for key, score in criteria.items()
                                if key in set(CriticCriterion) and isinstance(score, (int, float))},
        errors=[code for code in row.errors or [] if code in _CODES],
        warnings=[code for code in row.warnings or [] if code in _CODES],
    )


@router.get("/requests/{request_id}/reasoning_log")
async def request_reasoning(row: OwnedRequest, runtime: Runtime) -> RequestReasoning:
    steps = await ReasoningLogStore(runtime.database.sessions).list_for_request(row.id)
    return RequestReasoning(request_id=row.id, user_id=row.user_id, status=row.status,
                            steps=[_reasoning_step(step) for step in steps],
                            revisions=(row.parameters or {}).get("revisions", []))


@router.get("/requests/{request_id}/llm_calls")
async def request_calls(row: OwnedRequest, runtime: Runtime) -> list[CallTrace]:
    return [CallTrace(agent=call.agent_name, provider=call.provider, model=call.model_name,
                      status=call.status, duration_ms=call.latency_ms, tokens=call.tokens_used,
                      created_at=call.created_at, language_retry=(call.parameters or {}).get("language_retry", False),
                      target_language=(call.parameters or {}).get("target_language"),
                      actual_language=(call.parameters or {}).get("actual_language"),
                      language_mismatch=(call.parameters or {}).get("error_type") == "language_mismatch")
            for call in await LLMCallStore(runtime.database.sessions).list_for_request(row.id)]

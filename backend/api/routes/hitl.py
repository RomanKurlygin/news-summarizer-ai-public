"""Защищённая очередь редакторов и публикация проверенного корпуса."""

import asyncio
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from backend.api.deps import DBSession, require_editor
from backend.api.routes.summarization import Runtime
from backend.db.models import Feedback, KnowledgeBase, UncertainExample, User
from backend.utils.corpus_text import content_hash
from backend.utils.text_processing import detect_language, split_sentences

router = APIRouter(prefix="/api/v1/hitl", tags=["Ручная проверка"], dependencies=[Depends(require_editor)])
EditorUser = Annotated[User, Depends(require_editor)]
ReviewStatus = Literal["pending", "approved", "rejected"]


class ReviewItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    request_id: UUID
    input_text: str
    model_output: str
    critic_score: float | None
    status: ReviewStatus
    reasons: list[str]
    created_at: datetime


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["approved", "rejected"]
    corrected_summary: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=600)] | None = None
    comment: Annotated[str, StringConstraints(strip_whitespace=True, max_length=1000)] | None = None

    @model_validator(mode="after")
    def validate_summary(self):
        if self.decision == "approved" and not self.corrected_summary:
            raise ValueError("Для подтверждения нужен итоговый текст.")
        if self.corrected_summary is not None and not 1 <= len(split_sentences(self.corrected_summary)) <= 3:
            raise ValueError("Резюме должно содержать от одного до трёх предложений.")
        return self


class ReviewDecision(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    editor_id: UUID | None
    decision: str
    corrected_summary: str | None
    comment: str | None
    created_at: datetime


@router.get("/queue")
async def review_queue(session: DBSession, status: ReviewStatus = "pending",
                       limit: Annotated[int, Query(ge=1, le=50)] = 15,
                       offset: Annotated[int, Query(ge=0)] = 0) -> list[ReviewItem]:
    row = UncertainExample
    values = await session.scalars(select(row).where(row.status == status)
                                   .order_by(row.created_at.desc(), row.id.desc()).limit(limit).offset(offset))
    return [ReviewItem.model_validate({**{key: getattr(item, key) for key in ReviewItem.model_fields},
                                      "reasons": item.reasons or []}) for item in values]


@router.get("/queue/{item_id}/decisions")
async def review_history(item_id: UUID, session: DBSession) -> list[ReviewDecision]:
    if await session.get(UncertainExample, item_id) is None:
        raise HTTPException(404, "Проверка не найдена.")
    values = await session.scalars(select(Feedback).where(Feedback.uncertain_example_id == item_id)
                                   .order_by(Feedback.created_at, Feedback.id))
    return [ReviewDecision.model_validate(value) for value in values]


async def _add_to_corpus(item: UncertainExample, summary: str, session, runtime) -> str:
    from backend.retrieval.documents import IndexIntegrityError

    try:
        from scripts.build_indexes import load_holdout

        guard = await asyncio.to_thread(load_holdout, runtime.settings.holdout_path)
    except (ImportError, IndexIntegrityError, OSError, ValueError):
        return "holdout_unavailable"
    digest = content_hash(item.input_text)
    if {digest, content_hash(summary)} & guard.hashes:
        return "heldout"
    exists = await session.scalar(select(KnowledgeBase.id).where(KnowledgeBase.content_hash == digest))
    if exists is not None:
        return "already_in_corpus"
    session.add(KnowledgeBase(source_text=item.input_text, target_text=summary, category="general",
                              language=detect_language(item.input_text) or "und", source_dataset="human_review",
                              source_document_id=str(item.id), source_split="train", content_hash=digest))
    return "awaiting_index"


@router.post("/queue/{item_id}/decisions", status_code=201)
async def record_decision(item_id: UUID, payload: Decision, session: DBSession,
                          editor: EditorUser, runtime: Runtime) -> dict:
    training = "not_applicable"
    try:
        async with session.begin():
            item = await session.scalar(update(UncertainExample).where(
                UncertainExample.id == item_id, UncertainExample.status == "pending",
            ).values(status=payload.decision).returning(UncertainExample))
            if item is None:
                if await session.get(UncertainExample, item_id) is None:
                    raise HTTPException(404, "Проверка не найдена.")
                raise HTTPException(409, "Решение по этому примеру уже сохранено.")
            feedback = Feedback(uncertain_example_id=item.id, editor_id=editor.id, decision=payload.decision,
                                corrected_summary=payload.corrected_summary, comment=payload.comment)
            session.add(feedback)
            if payload.decision == "approved":
                training = await _add_to_corpus(item, payload.corrected_summary, session, runtime)
            await session.flush()
    except IntegrityError:
        raise HTTPException(409, "Данные изменились параллельно. Обновите очередь и повторите решение.") from None
    return {"feedback": ReviewDecision.model_validate(feedback), "training_status": training}


@router.post("/publish")
async def publish_corpus(runtime: Runtime) -> dict:
    try:
        result = await runtime.publish_reviewed_corpus()
    except (ImportError, OSError, RuntimeError, ValueError):
        raise HTTPException(503, "Не удалось опубликовать индексы. Проверьте корпус и поисковый сервис.") from None
    return {"version": result["version"], "documents": result["document_count"]}

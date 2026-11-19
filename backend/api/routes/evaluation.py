"""Синхронное измерение качества на небольшом отложенном наборе."""

import asyncio
from time import perf_counter
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query, Request
from kombu.exceptions import OperationalError
from sqlalchemy import select, update

from backend.api.deps import CurrentUser, DBSession
from backend.api.routes.summarization import Runtime
from backend.api.runtime import SummaryRuntime, during_request
from backend.api.schemas.evaluation import EvaluateInput, EvaluationItem, EvaluationOutput, ExperimentHistoryItem
from backend.api.schemas.summarization import MAX_NEWS_CHARS, Configuration
from backend.db.models import EvaluationResult, ExperimentJob, ModelComparison, User
from backend.evaluation.experiments import create_job, job_report
from backend.evaluation.metrics_calculator import MetricsUnavailable, average_metrics
from backend.evaluation.protocol import ExperimentPlan
from backend.worker.celery_app import COMPARE_TASK, EVALUATE_TASK, celery_client
from scripts.corpus import DataError, News, load_prepared

router = APIRouter(prefix="/api/v1/evaluate", tags=["Оценка качества"])


def _selected(runtime: SummaryRuntime, keys: list[str] | None) -> list[News]:
    try:
        rows = load_prepared(runtime.settings.holdout_path, News)
        if not rows or any(row.split != "test" for row in rows) or len({row.id for row in rows}) != len(rows):
            raise DataError("Некорректный контрольный набор.")
    except (ValueError, OSError):
        raise HTTPException(503, "Контрольный набор недоступен.") from None
    if keys is None:
        return rows[:1]
    by_id = {str(row.id): row for row in rows}
    by_id.update({row.source_document_id: row for row in rows if row.source_document_id})
    if not all(key in by_id for key in keys):
        raise HTTPException(404, "Контрольная новость не найдена.")
    selected = [by_id[key] for key in keys]
    if len({row.id for row in selected}) != len(selected):
        raise HTTPException(422, "Контрольные новости должны быть уникальны.")
    return selected


def _evaluation(row: EvaluationResult) -> EvaluationOutput:
    return EvaluationOutput.model_validate({
        **row.metrics_json, "id": row.id, "user_id": row.user_id, "created_at": row.created_at,
        "model_name": row.model_name, "sample_size": row.sample_size, "evaluation_time": row.evaluation_time,
    })


async def _evaluate(runtime: SummaryRuntime, payload: EvaluateInput, user: User) -> EvaluationOutput:
    started = perf_counter()
    rows = await asyncio.to_thread(_selected, runtime, payload.test_ids)
    if any(len(row.source_text) > MAX_NEWS_CHARS for row in rows):
        raise HTTPException(422, "Контрольная новость превышает допустимый лимит.")
    config = Configuration.model_validate(payload.model_dump(exclude={"test_ids"}))
    items, pairs, pair_positions, models = [], [], [], []
    for row in rows:
        request_id = uuid4()
        candidate = await runtime.generate(
            row.source_text, config, user.id, request_id, language=row.language,
        )
        result = candidate.result
        if result is not None and result.final_summary is not None and candidate.error is None:
            pair_positions.append(len(items))
            pairs.append((result.final_summary, row.target_text))
            models.append(result.model)
        items.append(EvaluationItem(test_id=row.id, source_document_id=row.source_document_id,
                                    request_id=request_id, status=result.status if result else "failed",
                                    error=candidate.error))
    if not pairs:
        error = next((item.error for item in items if item.error), None)
        raise HTTPException(error.http_status if error else 502, {
            "message": "Нет результатов для расчёта метрик.",
            "items": [item.model_dump(mode="json") for item in items],
        })
    try:
        scores = await asyncio.to_thread(runtime.metrics.calculate, pairs)
        metadata = runtime.metrics.metadata()
    except MetricsUnavailable:
        raise HTTPException(503, "Проверьте библиотеки оценки, WordNet и локальную модель BERTScore.") from None
    if len(set(models)) != 1 or models[0] is None:
        raise HTTPException(502, "Провайдер не подтвердил одну фактически использованную модель.")
    for index, score in zip(pair_positions, scores, strict=True):
        items[index].metrics = score
    report = EvaluationResult(
        user_id=user.id, model_name=models[0], mode=config.mode, sample_size=len(rows),
        evaluation_time=round(perf_counter() - started, 3),
        metrics_json={
            "configuration": config.model_dump(mode="json"),
            "items": [item.model_dump(mode="json") for item in items], "scored_count": len(scores),
            "metrics": average_metrics(scores).model_dump(), "metric_metadata": metadata,
        },
    )
    async with runtime.database.sessions.begin() as session:
        session.add(report)
        await session.flush()
        await session.refresh(report)
    return _evaluation(report)


@router.post("")
async def evaluate(payload: EvaluateInput, request: Request, user: CurrentUser, runtime: Runtime) -> EvaluationOutput:
    return await during_request(request, lambda: _evaluate(runtime, payload, user),
                                runtime.settings.api_group_timeout_seconds)


@router.get("/history")
async def evaluation_history(
    user: CurrentUser, session: DBSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20, offset: Annotated[int, Query(ge=0)] = 0,
) -> list[EvaluationOutput | ExperimentHistoryItem]:
    query = select(EvaluationResult).where(EvaluationResult.user_id.is_not(None))
    if user.role != "admin":
        query = query.where(EvaluationResult.user_id == user.id)
    rows = await session.scalars(query.order_by(EvaluationResult.created_at.desc(), EvaluationResult.id.desc())
                                 .offset(offset).limit(limit))
    return [_evaluation(row) if row.experiment_id is None else ExperimentHistoryItem(
        id=row.id, experiment_id=row.experiment_id, user_id=row.user_id, model_name=row.model_name,
        provider=row.metrics_json["provider"], mode=row.mode, sample_size=row.sample_size,
        created_at=row.created_at, evaluation_time=row.evaluation_time, aggregate=row.metrics_json["aggregate"],
    ) for row in rows]


def _publish(settings, experiment_id: UUID, *, comparison: bool) -> None:
    with celery_client(settings) as client:
        client.send_task(COMPARE_TASK if comparison else EVALUATE_TASK, args=[str(experiment_id)],
                         task_id=str(experiment_id), retry=False)


@router.post("/batch", status_code=202)
async def batch(payload: ExperimentPlan, user: CurrentUser, runtime: Runtime) -> dict:
    job = await create_job(runtime.database.sessions, payload, user.id)
    try:
        await asyncio.to_thread(_publish, runtime.settings, job.id,
                                comparison=len(payload.models) * len(payload.modes) > 1)
    except (OperationalError, OSError):
        async with runtime.database.sessions.begin() as session:
            await session.execute(update(ExperimentJob).where(
                ExperimentJob.id == job.id, ExperimentJob.status == "queued",
            ).values(status="broker_unavailable", error="broker_unavailable"))
        raise HTTPException(503, {"code": "broker_unavailable", "experiment_id": str(job.id),
                                  "message": "Брокер Redis недоступен."}) from None
    return {"experiment_id": str(job.id), "status": "queued"}


async def _owned_job(experiment_id: UUID, user: User, session) -> ExperimentJob:
    job = await session.get(ExperimentJob, experiment_id)
    if job is None:
        raise HTTPException(404, "Эксперимент не найден.")
    if job.user_id != user.id and user.role != "admin":
        raise HTTPException(403, "Доступ к этому эксперименту запрещён.")
    return job


@router.get("/experiments/{experiment_id}")
async def experiment_status(experiment_id: UUID, user: CurrentUser, session: DBSession) -> dict:
    return await job_report(session, await _owned_job(experiment_id, user, session))


@router.get("/compare")
async def compare(
    user: CurrentUser, session: DBSession, experiment_id: UUID | None = None,
    sample_size: Annotated[int | None, Query(ge=1, le=100)] = None,
    models: Annotated[str | None, Query(max_length=1024)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[dict]:
    query = select(ModelComparison).join(ExperimentJob, ModelComparison.experiment_id == ExperimentJob.id)
    query = query.where(ExperimentJob.status == "completed")
    if experiment_id is not None:
        await _owned_job(experiment_id, user, session)
        query = query.where(ExperimentJob.id == experiment_id)
    elif user.role != "admin":
        query = query.where(ExperimentJob.user_id == user.id)
    if sample_size is not None:
        query = query.where(ModelComparison.sample_size == sample_size)
    query = query.order_by(ModelComparison.created_at.desc(), ModelComparison.id)
    requested = {name.strip() for name in models.split(",") if name.strip()} if models else set()
    rows = await session.stream_scalars(query)
    comparisons = []
    async for row in rows:
        if requested and not requested.issubset(row.models_compared):
            continue
        comparisons.append({"id": str(row.id), "experiment_id": str(row.experiment_id),
                            "sample_size": row.sample_size, **row.results_json})
        if len(comparisons) >= limit:
            break
    await rows.close()
    return comparisons

"""Обработчики Celery для существующих записей экспериментальных задач."""

import asyncio
from uuid import UUID

import httpx

from backend.api.runtime import SummaryRuntime
from backend.core.config import Settings
from backend.db.cache import SummaryCache
from backend.db.session import Database
from backend.evaluation.experiments import ExperimentBusy, ExperimentRunner
from backend.evaluation.protocol import ExperimentError
from backend.worker.celery_app import COMPARE_TASK, EVALUATE_TASK, PURGE_CACHE_TASK, app


async def execute(settings: Settings, experiment_id: UUID) -> None:
    if settings.database_url is None:
        raise ValueError("Для worker требуется DATABASE_URL.")
    database = Database(settings.database_url.get_secret_value(), settings)
    try:
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds, trust_env=False) as http:
            pinned = settings.model_copy(update={"embedding_local_files_only": True})
            await ExperimentRunner(SummaryRuntime(pinned, database, http)).run(experiment_id)
    finally:
        await database.close()


def _execute(task, experiment_id: str) -> str:
    identifier = UUID(experiment_id)
    try:
        asyncio.run(execute(Settings(), identifier))
    except ExperimentBusy as exc:
        raise task.retry(exc=exc, countdown=30, max_retries=90)
    except Exception as exc:
        code = exc.code if isinstance(exc, ExperimentError) else type(exc).__name__
        raise RuntimeError(code) from None
    return str(identifier)


@app.task(bind=True, name=EVALUATE_TASK)
def evaluate_model_task(self, experiment_id: str) -> str:
    return _execute(self, experiment_id)


@app.task(bind=True, name=COMPARE_TASK)
def compare_models_task(self, experiment_id: str) -> str:
    return _execute(self, experiment_id)


async def purge_cache(settings: Settings) -> int:
    if settings.database_url is None:
        raise ValueError("Для очистки кэша требуется DATABASE_URL.")
    database = Database(settings.database_url.get_secret_value(), settings)
    try:
        return await SummaryCache(database.sessions).purge()
    finally:
        await database.close()


@app.task(name=PURGE_CACHE_TASK)
def purge_cache_task() -> int:
    return asyncio.run(purge_cache(Settings()))

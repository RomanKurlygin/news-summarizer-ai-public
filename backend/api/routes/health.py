"""Проверка готовности базы данных."""

import asyncio
import json
import logging

from asyncpg import PostgresError
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
import httpx
from kombu.exceptions import OperationalError
from redis.asyncio import Redis
from redis.exceptions import RedisError

from backend.api.deps import CurrentUser
from backend.db.models import SummarizationRequest
from backend.db.session import Database
from backend.worker.celery_app import celery_client

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", responses={503: {"description": "Database unavailable"}})
async def health(request: Request) -> JSONResponse:
    database: Database | None = request.app.state.database
    if database is None:
        return JSONResponse(
            content={"status": "unavailable", "database": "not_configured"},
            status_code=503,
        )

    try:
        async with asyncio.timeout(request.app.state.settings.db_health_timeout):
            async with database.engine.connect() as connection:
                await connection.execute(select(SummarizationRequest.id).limit(0))
    except (SQLAlchemyError, PostgresError, OSError, TimeoutError) as exc:
        logger.warning(
            "Database readiness check failed (%s)",
            type(exc).__name__,
            extra={"request_id": request.state.request_id},
        )
        return JSONResponse(
            content={"status": "unavailable", "database": "unavailable"},
            status_code=503,
        )

    return JSONResponse(content={"status": "ok", "database": "ok"})


async def _http_health(http: httpx.AsyncClient, url: str) -> str:
    try:
        return "ok" if (await http.get(url, timeout=1.5)).status_code == 200 else "unavailable"
    except httpx.HTTPError:
        return "unavailable"


def _snapshot_health(settings) -> str:
    directory = settings.rag_index_dir
    try:
        pointer = json.loads((directory / "active.json").read_bytes())
        version = pointer["version"]
        if not isinstance(version, str) or len(version) != 32 or any(char not in "0123456789abcdef" for char in version):
            return "unavailable"
        if pointer["mode"] == settings.qdrant_mode and (directory / "versions" / version / "bm25.json").is_file():
            if settings.qdrant_mode == "server" or (directory / "versions" / version / "qdrant").is_dir():
                return "snapshot_available"
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return "unavailable"


async def _redis_health(settings) -> str:
    try:
        async with Redis.from_url(settings.redis_url.get_secret_value(), socket_connect_timeout=1,
                                  socket_timeout=1) as cache:
            async with asyncio.timeout(1.5):
                return "ok" if await cache.ping() else "unavailable"
    except (RedisError, OSError, TimeoutError, ValueError):
        return "unavailable"


def _worker_health(settings) -> str:
    try:
        with celery_client(settings) as client:
            return "ok" if client.control.ping(timeout=1) else "unavailable"
    except (OperationalError, OSError, ValueError):
        return "unavailable"


@router.get("/health/dependencies")
async def dependencies(request: Request, user: CurrentUser) -> dict:
    settings = request.app.state.settings
    http = request.app.state.http
    result = await asyncio.gather(
        health(request), _http_health(http, str(settings.ollama_base_url).rstrip("/") + "/api/version"),
        _redis_health(settings), asyncio.to_thread(_snapshot_health, settings),
    )
    database, ollama, redis, snapshot = result
    qdrant = (snapshot if settings.qdrant_mode == "local" else
              await _http_health(http, str(settings.qdrant_url).rstrip("/") + "/readyz"))
    worker = await asyncio.to_thread(_worker_health, settings) if redis == "ok" else "not_checked"
    data: dict = {"api": "ok", "database": "ok" if database.status_code == 200 else "unavailable",
                  "ollama": ollama, "qdrant": qdrant, "rag_snapshot": snapshot, "redis": redis, "worker": worker}
    if user.role == "admin":
        data["cache"] = {"enabled": bool(settings.baseline_cache_ttl_seconds)}
        runtime = request.app.state.summary_runtime
        if runtime is not None and database.status_code == 200:
            data["cache"].update(await runtime.cache.statistics())
    return data

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
import pytest
from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.main import create_app
from backend.core.config import Settings
from backend.db.init_db import create_test_schema
from backend.db.models import Glossary
from backend.db.session import get_db_session

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def running_api(application) -> AsyncIterator[httpx.AsyncClient]:
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://testserver"
        ) as client:
            yield client


async def test_database_health_is_ready_after_schema_creation(tmp_path):
    application = create_app(
        Settings(
            _env_file=None,
            database_url=f"sqlite+aiosqlite:///{tmp_path / 'ready.sqlite3'}",
            log_dir=tmp_path / "logs",
        )
    )

    @application.get("/_test/glossary")
    async def glossary(session: Annotated[AsyncSession, Depends(get_db_session)]) -> str:
        term = await session.scalar(select(Glossary.term))
        return term

    async with running_api(application) as client:
        database = application.state.database
        await create_test_schema(database.engine)
        async with database.sessions.begin() as session:
            session.add(Glossary(term="BM25", definition="Lexical retrieval"))

        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "database": "ok"}
        assert (await client.get("/_test/glossary")).json() == "BM25"
        assert application.state.database is database

    assert application.state.database is None


@pytest.mark.parametrize("configured", [False, True], ids=["not-configured", "missing-schema"])
async def test_database_health_is_unavailable_but_liveness_works(tmp_path, configured):
    url = f"sqlite+aiosqlite:///{tmp_path / 'empty.sqlite3'}" if configured else None
    application = create_app(
        Settings(
            _env_file=None,
            database_url=url,
            app_name="NewsSummarizer AI",
            log_dir=tmp_path / "logs",
        )
    )

    async with running_api(application) as client:
        response = await client.get("/health")
        assert response.status_code == 503
        assert response.json() == {
            "status": "unavailable",
            "database": "unavailable" if configured else "not_configured",
        }

        liveness = await client.get("/healthz")
        assert liveness.status_code == 200
        assert liveness.json() == {"status": "ok", "service": "NewsSummarizer AI"}

import asyncio

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, select

from backend.core.config import Settings
from backend.core.constants import PROJECT_ROOT
from backend.db.models import Base, Glossary
from backend.db.session import Database


@pytest.mark.asyncio
async def test_initial_migration_is_repeatable_and_matches_models(tmp_path, monkeypatch):
    url = f"sqlite+aiosqlite:///{tmp_path / 'migration.sqlite3'}"
    monkeypatch.setenv("DATABASE_URL", url)
    alembic = Config(str(PROJECT_ROOT / "alembic.ini"))
    await asyncio.to_thread(command.upgrade, alembic, "head")

    database = Database(url, Settings(_env_file=None))
    try:
        async with database.engine.connect() as connection:
            tables = await connection.run_sync(lambda sync: set(inspect(sync).get_table_names()))
        assert tables == set(Base.metadata.tables) | {"alembic_version"}

        async with database.sessions.begin() as session:
            session.add(Glossary(term="RRF", definition="Reciprocal Rank Fusion"))

        await asyncio.to_thread(command.upgrade, alembic, "head")
        await asyncio.to_thread(command.check, alembic)

        async with database.sessions() as session:
            assert (await session.scalars(select(Glossary.term))).all() == ["RRF"]
    finally:
        await database.close()

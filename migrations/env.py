"""Миграции Alembic для асинхронной базы приложения."""

import asyncio

from alembic import context
from sqlalchemy.engine import Connection

from backend.core.config import Settings
from backend.db.models import Base
from backend.db.session import create_db_engine

settings = Settings()
if settings.database_url is None:
    raise RuntimeError("Set DATABASE_URL before running Alembic")
database_url = settings.database_url.get_secret_value()


def run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=Base.metadata,
        transactional_ddl=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_online() -> None:
    engine = create_db_engine(database_url, settings)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    context.configure(
        url=database_url,
        target_metadata=Base.metadata,
        literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(run_online())

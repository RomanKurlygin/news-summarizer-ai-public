"""Движок и сессии SQLAlchemy для одного процесса приложения."""

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy import event
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from backend.core.config import Settings
from backend.core.exceptions import ServiceUnavailableError


def create_db_engine(url: str, settings: Settings) -> AsyncEngine:
    try:
        parsed_url = make_url(url)
    except ArgumentError:
        raise ValueError("Invalid database URL") from None

    if parsed_url.drivername == "postgresql+asyncpg":
        return create_async_engine(
            parsed_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_pre_ping=True,
            hide_parameters=True,
        )
    if parsed_url.drivername != "sqlite+aiosqlite":
        raise ValueError("Use postgresql+asyncpg or sqlite+aiosqlite for database URLs")

    engine = create_async_engine(parsed_url, hide_parameters=True)

    @event.listens_for(engine.sync_engine, "connect")
    def configure_sqlite(connection, _record) -> None:
        # SQLAlchemy открывает транзакции явно, в том числе для SQLite DDL.
        connection.isolation_level = None
        cursor = connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    @event.listens_for(engine.sync_engine, "begin")
    def begin_sqlite(connection) -> None:
        connection.exec_driver_sql("BEGIN")

    return engine


class Database:
    def __init__(self, url: str, settings: Settings) -> None:
        self.engine = create_db_engine(url, settings)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def close(self) -> None:
        await self.engine.dispose()


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Выдать сессию запроса. Запись выполняется в явной транзакции session.begin()."""
    database: Database | None = request.app.state.database
    if database is None:
        raise ServiceUnavailableError("Database is not configured")

    async with database.sessions() as session:
        yield session

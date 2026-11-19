"""Краткоживущий кэш результатов baseline в SQLite или PostgreSQL."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from backend.db.models import CacheEntry


class SummaryCache:
    def __init__(self, sessions):
        self.sessions = sessions

    async def take(self, key: str) -> dict | None:
        async with self.sessions.begin() as session:
            return await session.scalar(update(CacheEntry).where(
                CacheEntry.input_key == key, CacheEntry.expires_at > datetime.now(UTC),
            ).values(hits=CacheEntry.hits + 1).returning(CacheEntry.output))

    async def store(self, key: str, output: dict, ttl: int) -> None:
        now = datetime.now(UTC)
        async with self.sessions.begin() as session:
            insert = postgres_insert if session.get_bind().dialect.name == "postgresql" else sqlite_insert
            statement = insert(CacheEntry).values(input_key=key, output=output, expires_at=now + timedelta(seconds=ttl))
            await session.execute(statement.on_conflict_do_update(
                index_elements=[CacheEntry.input_key], where=CacheEntry.expires_at <= now,
                set_={"output": statement.excluded.output, "expires_at": statement.excluded.expires_at, "hits": 0},
            ))

    async def purge(self) -> int:
        async with self.sessions.begin() as session:
            result = await session.execute(delete(CacheEntry).where(CacheEntry.expires_at <= datetime.now(UTC)))
            return result.rowcount

    async def statistics(self) -> dict[str, int]:
        async with self.sessions() as session:
            rows, hits = (await session.execute(select(func.count(), func.coalesce(func.sum(CacheEntry.hits), 0))
                .where(CacheEntry.expires_at > datetime.now(UTC)))).one()
            return {"entries": rows, "hits": hits}

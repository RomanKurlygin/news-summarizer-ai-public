"""Применение миграций и создание изолированной тестовой схемы."""

from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncEngine

from backend.core.constants import PROJECT_ROOT
from backend.db.models import Base


async def create_test_schema(engine: AsyncEngine) -> None:
    """Создать таблицы из metadata в SQLite для тестов."""
    if engine.dialect.name != "sqlite":
        raise ValueError("Use Alembic migrations for PostgreSQL")

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


if __name__ == "__main__":
    command.upgrade(Config(str(PROJECT_ROOT / "alembic.ini")), "head")

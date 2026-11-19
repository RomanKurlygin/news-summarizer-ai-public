"""Работа с учётными записями при регистрации и входе."""

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from backend.core.exceptions import AppError
from backend.db.models import User
from backend.utils.security import hash_password, verify_password


class RegistrationConflictError(AppError):
    """Логин или почта уже заняты."""


async def register_user(
    session: AsyncSession, *, username: str, email: str, password: str
) -> User:
    password_hash = await run_in_threadpool(hash_password, password)
    user = User(username=username, email=email, hashed_password=password_hash, role="user")
    try:
        async with session.begin():
            session.add(user)
    except IntegrityError:
        # Ограничения БД проверяют уникальность даже при параллельной регистрации.
        conflict = await session.scalar(
            select(User.id).where((User.username == username) | (User.email == email)).limit(1)
        )
        if conflict is None:
            raise
        raise RegistrationConflictError() from None

    return user


async def authenticate_user(
    session: AsyncSession, *, username: str, password: str
) -> User | None:
    async with session.begin():
        user = await session.scalar(select(User).where(User.username == username))

    password_hash = user.hashed_password if user is not None else None
    matches = await run_in_threadpool(verify_password, password, password_hash)
    if user is None or not matches or not user.is_active:
        return None
    return user

"""Зависимости FastAPI для аутентификации и проверки ролей."""

from typing import Annotated
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import User
from backend.db.session import get_db_session
from backend.utils.security import decode_access_token

DBSession = Annotated[AsyncSession, Depends(get_db_session)]
bearer_scheme = HTTPBearer(auto_error=False, scheme_name="BearerAuth", bearerFormat="JWT")


def invalid_credentials() -> HTTPException:
    return HTTPException(
        status_code=401, detail="Invalid credentials", headers={"WWW-Authenticate": "Bearer"}
    )


async def _get_token_subject(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> UUID:
    if credentials is None:
        raise invalid_credentials()
    try:
        return decode_access_token(credentials.credentials, request.app.state.settings)
    except (jwt.InvalidTokenError, ValueError):
        raise invalid_credentials() from None


async def get_current_user(
    user_id: Annotated[UUID, Depends(_get_token_subject)], session: DBSession
) -> User:
    # Завершаем транзакцию чтения до записи журналов из других сессий.
    async with session.begin():
        user = await session.get(User, user_id)
    if user is None or not user.is_active:
        raise invalid_credentials()
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def require_editor(user: Annotated[User, Depends(get_current_user)]) -> User:
    if user.role not in {"editor", "admin"}:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return user


async def require_admin(user: Annotated[User, Depends(get_current_user)]) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return user

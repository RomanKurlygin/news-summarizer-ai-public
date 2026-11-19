"""Регистрация, вход и получение собственного профиля."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from backend.api.deps import DBSession, get_current_user, invalid_credentials
from backend.api.schemas.auth import AccessToken, LoginRequest, RegistrationRequest, UserProfile
from backend.db.models import User
from backend.services.auth_service import (
    RegistrationConflictError,
    authenticate_user,
    register_user,
)
from backend.utils.security import create_access_token, get_signing_key

router = APIRouter(prefix="/api/v1/auth", tags=["Аутентификация"])


@router.post("/register", status_code=201, responses={409: {"description": "Конфликт регистрации"}})
async def register(payload: RegistrationRequest, session: DBSession) -> UserProfile:
    """Создать учётную запись с ролью user."""
    try:
        user = await register_user(
            session,
            username=payload.username,
            email=payload.email,
            password=payload.password.get_secret_value(),
        )
    except RegistrationConflictError:
        raise HTTPException(status_code=409, detail="Registration conflict") from None
    return UserProfile.model_validate(user)


@router.post("/login", responses={401: {"description": "Некорректные учётные данные"}})
async def login(
    payload: LoginRequest, request: Request, response: Response, session: DBSession
) -> AccessToken:
    """Принять username и password в JSON и выдать JWT."""
    settings = request.app.state.settings
    key = get_signing_key(settings)
    user = await authenticate_user(
        session, username=payload.username, password=payload.password.get_secret_value()
    )
    if user is None:
        raise invalid_credentials()

    response.headers["Cache-Control"] = "no-store"
    return AccessToken(access_token=create_access_token(user.id, settings, key))


@router.get("/me", responses={401: {"description": "Требуется токен Bearer"}})
async def me(user: Annotated[User, Depends(get_current_user)]) -> UserProfile:
    """Вернуть профиль владельца действующего токена."""
    return UserProfile.model_validate(user)

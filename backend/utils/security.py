"""Хэширование паролей и подпись JWT."""

from datetime import UTC, datetime, timedelta
from uuid import UUID

import bcrypt
import jwt

from backend.core.config import Settings
from backend.core.exceptions import ServiceUnavailableError

BCRYPT_MAX_PASSWORD_BYTES = 72
_BCRYPT_ROUNDS = 12
_MINIMUM_KEY_BYTES = {"HS256": 32, "HS384": 48, "HS512": 64}

# Случайный bcrypt-хэш для проверки при отсутствии пользователя.
_DUMMY_PASSWORD_HASH = b"$2b$12$OmMzQOcF78O0ARfaWZSdRul6mim9oMjYDbsmdENeCv67XW/PEn5Ae"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)
    ).decode("ascii")


def verify_password(password: str, password_hash: str | None) -> bool:
    password_bytes = password.encode("utf-8")
    if len(password_bytes) > BCRYPT_MAX_PASSWORD_BYTES:
        return False

    try:
        hash_bytes = password_hash.encode("ascii") if password_hash else _DUMMY_PASSWORD_HASH
        return bcrypt.checkpw(password_bytes, hash_bytes)
    except ValueError:
        bcrypt.checkpw(password_bytes, _DUMMY_PASSWORD_HASH)
        return False


def get_signing_key(settings: Settings) -> str:
    if settings.jwt_secret_key is None:
        raise ServiceUnavailableError("JWT signing key is not configured")

    key = settings.jwt_secret_key.get_secret_value()
    if len(key.encode("utf-8")) < _MINIMUM_KEY_BYTES[settings.jwt_algorithm]:
        raise ServiceUnavailableError("JWT signing key is too short")
    return key


def create_access_token(user_id: UUID, settings: Settings, signing_key: str) -> str:
    now = datetime.now(UTC)
    expires_at = now + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    return jwt.encode(
        {"sub": str(user_id), "iat": now, "exp": expires_at},
        signing_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_access_token(token: str, settings: Settings) -> UUID:
    claims = jwt.decode(
        token,
        get_signing_key(settings),
        algorithms=[settings.jwt_algorithm],
        options={"require": ["sub", "iat", "exp"]},
    )
    return UUID(claims["sub"])

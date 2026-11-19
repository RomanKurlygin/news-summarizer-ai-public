from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from secrets import token_urlsafe
from typing import Annotated

import bcrypt
import httpx
import jwt
import pytest
import pytest_asyncio
from fastapi import Depends
from sqlalchemy import delete, select, update

from backend.api.deps import require_admin, require_editor
from backend.api.main import create_app
from backend.core.config import Settings
from backend.db.init_db import create_test_schema
from backend.db.models import User
from backend.db.session import Database
from backend.utils.logging import LOG_FILENAME

pytestmark = pytest.mark.asyncio

AUTH = "/api/v1/auth"
REGISTRATION = {
    "username": "LocalReader",
    "email": "Reader@example.com",
    "password": "fixture-only-password-01",
}
CREDENTIALS = {field: REGISTRATION[field] for field in ("username", "password")}


@pytest_asyncio.fixture
async def auth_api(tmp_path) -> AsyncIterator[tuple[httpx.AsyncClient, Settings, Database]]:
    settings = Settings(
        _env_file=None,
        app_env="production",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'auth.sqlite3'}",
        jwt_secret_key=token_urlsafe(64),
        jwt_algorithm="HS256",
        jwt_access_token_expire_minutes=7,
        log_dir=tmp_path / "logs",
        log_level="INFO",
    )
    application = create_app(settings)

    # Эти маршруты существуют только на экземпляре приложения теста.
    @application.get("/_test/editor")
    async def editor(user: Annotated[User, Depends(require_editor)]) -> str:
        return user.role

    @application.get("/_test/admin")
    async def admin(user: Annotated[User, Depends(require_admin)]) -> str:
        return user.role

    async with application.router.lifespan_context(application):
        database = application.state.database
        await create_test_schema(database.engine)
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client, settings, database


async def register_and_login(client):
    registration = await client.post(f"{AUTH}/register", json=REGISTRATION)
    assert registration.status_code == 201
    login = await client.post(f"{AUTH}/login", json=CREDENTIALS)
    assert login.status_code == 200
    return registration.json(), login.json()["access_token"]


def authorization(token):
    return {"Authorization": f"Bearer {token}"}


async def test_registration_persists_bcrypt_and_safe_profile(auth_api):
    client, _, database = auth_api
    response = await client.post(f"{AUTH}/register", json=REGISTRATION)

    assert response.status_code == 201
    profile = response.json()
    assert profile == {
        "id": profile["id"],
        "username": "localreader",
        "email": "reader@example.com",
        "role": "user",
        "is_active": True,
    }
    async with database.sessions() as session:
        user = await session.scalar(select(User).where(User.username == profile["username"]))
        assert user.hashed_password.startswith("$2b$12$")
        assert bcrypt.checkpw(
            REGISTRATION["password"].encode("utf-8"), user.hashed_password.encode("ascii")
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"username": "other-reader", "email": "READER@example.com"},
        {"username": "LOCALREADER", "email": "other@example.com"},
    ],
    ids=["email", "username"],
)
async def test_registration_rejects_duplicates(auth_api, overrides):
    client, _, _ = auth_api
    assert (await client.post(f"{AUTH}/register", json=REGISTRATION)).status_code == 201

    response = await client.post(f"{AUTH}/register", json=REGISTRATION | overrides)
    assert response.status_code == 409
    assert response.json() == {"detail": "Registration conflict"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"email": "not-an-email"}, {"password": "weak"}, {"password": ""},
        {"password": " " * 15}, {"password": "я" * 37},
        {"role": "admin"}, {"role": "editor"},
    ],
    ids=["email", "weak", "empty", "blank", "bcrypt-byte-limit", "admin", "editor"],
)
async def test_registration_rejects_invalid_or_privileged_input(auth_api, overrides):
    client, _, _ = auth_api
    response = await client.post(f"{AUTH}/register", json=REGISTRATION | overrides)
    assert response.status_code == 422
    assert response.json() == {"detail": "Request validation failed"}


async def test_login_returns_jwt_and_profile(auth_api):
    client, settings, _ = auth_api
    profile, token = await register_and_login(client)
    payload = jwt.decode(
        token, settings.jwt_secret_key.get_secret_value(), algorithms=["HS256"]
    )
    assert payload["sub"] == profile["id"]
    assert payload["exp"] - payload["iat"] == settings.jwt_access_token_expire_minutes * 60

    response = await client.get(f"{AUTH}/me", headers=authorization(token))
    assert response.status_code == 200
    assert response.json() == profile


async def test_invalid_login_does_not_disclose_account_state(auth_api):
    client, _, database = auth_api
    assert (await client.post(f"{AUTH}/register", json=REGISTRATION)).status_code == 201

    wrong_password = await client.post(
        f"{AUTH}/login", json=CREDENTIALS | {"password": "incorrect"}
    )
    unknown_user = await client.post(
        f"{AUTH}/login", json=CREDENTIALS | {"username": "unknown-reader"}
    )
    async with database.sessions.begin() as session:
        await session.execute(update(User).values(is_active=False))
    inactive_user = await client.post(f"{AUTH}/login", json=CREDENTIALS)

    for response in (wrong_password, unknown_user, inactive_user):
        assert response.status_code == 401
        assert response.json() == {"detail": "Invalid credentials"}
        assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "kind",
    ["missing", "malformed", "signature", "algorithm", "expired", "no-expiration"],
)
async def test_profile_rejects_invalid_or_missing_token(auth_api, kind):
    client, settings, _ = auth_api
    response = await client.post(f"{AUTH}/register", json=REGISTRATION)
    assert response.status_code == 201
    now = datetime.now(UTC)
    payload = {"sub": response.json()["id"], "iat": now, "exp": now + timedelta(minutes=1)}
    key = settings.jwt_secret_key.get_secret_value()
    headers = {}

    if kind == "malformed":
        headers = authorization("invalid.jwt.token")
    elif kind != "missing":
        algorithm = "HS384" if kind == "algorithm" else "HS256"
        if kind == "signature":
            key = token_urlsafe(64)
        if kind == "expired":
            payload["exp"] = now - timedelta(seconds=1)
        if kind == "no-expiration":
            del payload["exp"]
        headers = authorization(jwt.encode(payload, key, algorithm=algorithm))

    response = await client.get(f"{AUTH}/me", headers=headers)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize("removed", [False, True], ids=["inactive", "deleted"])
async def test_profile_checks_latest_account_state(auth_api, removed):
    client, _, database = auth_api
    _, token = await register_and_login(client)
    async with database.sessions.begin() as session:
        statement = delete(User) if removed else update(User).values(is_active=False)
        await session.execute(statement)

    assert (await client.get(f"{AUTH}/me", headers=authorization(token))).status_code == 401


@pytest.mark.parametrize(
    ("role", "editor_status", "admin_status"),
    [("user", 403, 403), ("editor", 200, 403), ("admin", 200, 200)],
)
async def test_role_dependencies_use_current_database_role(
    auth_api, role, editor_status, admin_status
):
    client, _, database = auth_api
    _, token = await register_and_login(client)
    async with database.sessions.begin() as session:
        await session.execute(update(User).values(role=role))

    headers = authorization(token)
    assert (await client.get("/_test/editor", headers=headers)).status_code == editor_status
    assert (await client.get("/_test/admin", headers=headers)).status_code == admin_status


async def test_login_fails_closed_without_configured_key(auth_api):
    client, settings, _ = auth_api
    assert (await client.post(f"{AUTH}/register", json=REGISTRATION)).status_code == 201
    settings.jwt_secret_key = None

    response = await client.post(f"{AUTH}/login", json=CREDENTIALS)
    assert response.status_code == 503
    assert response.json() == {"detail": "Service unavailable"}


async def test_openapi_documents_json_login_and_bearer(auth_api):
    client, _, _ = auth_api
    schema = (await client.get("/openapi.json")).json()

    login = schema["paths"][f"{AUTH}/login"]["post"]
    assert "application/json" in login["requestBody"]["content"]
    assert schema["paths"][f"{AUTH}/me"]["get"]["security"] == [{"BearerAuth": []}]
    assert schema["components"]["securitySchemes"]["BearerAuth"] == {
        "type": "http", "scheme": "bearer", "bearerFormat": "JWT",
    }


async def test_auth_omits_password_and_token_from_logs(auth_api):
    client, settings, _ = auth_api
    registered = await client.post(f"{AUTH}/register", json=REGISTRATION)
    login = await client.post(f"{AUTH}/login", json=CREDENTIALS)
    assert registered.status_code == 201
    assert login.status_code == 200
    assert login.headers["Cache-Control"] == "no-store"
    token = login.json()["access_token"]
    assert login.json()["token_type"] == "bearer"
    response = await client.get(f"{AUTH}/me", headers=authorization(token))
    assert response.status_code == 200

    logs = (settings.log_dir / LOG_FILENAME).read_text(encoding="utf-8")
    assert "HTTP response 200" in logs
    assert REGISTRATION["password"] not in logs
    assert token not in logs
    assert settings.jwt_secret_key.get_secret_value() not in logs

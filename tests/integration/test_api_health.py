from uuid import UUID

import httpx
import pytest
import pytest_asyncio

from backend.api.main import create_app
from backend.core.config import Settings
from backend.core.constants import REQUEST_ID_HEADER
from backend.core.exceptions import ServiceUnavailableError
from backend.utils.logging import LOG_FILENAME

ALLOWED_ORIGIN = "http://localhost:8501"
TEST_SECRET = "synthetic-credential-for-test"

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def client(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"JWT_SECRET_KEY={TEST_SECRET}\n", encoding="utf-8")
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    application = create_app(
        Settings(
            _env_file=dotenv,
            app_name="NewsSummarizer AI",
            cors_origins=[ALLOWED_ORIGIN],
            log_dir=tmp_path / "logs",
            log_level="INFO",
        )
    )

    @application.get("/_test/validate")
    async def validate(amount: int) -> int:
        return amount

    @application.get("/_test/unavailable")
    async def unavailable() -> None:
        raise ServiceUnavailableError(TEST_SECRET)

    @application.get("/_test/fail")
    async def fail() -> None:
        raise RuntimeError(TEST_SECRET)

    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as test_client:
            yield test_client


async def test_root_and_healthz(client):
    root = await client.get("/")
    assert root.status_code == 200
    assert root.json() == {"service": "NewsSummarizer AI", "version": "0.1.0"}

    health = await client.get("/healthz")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "service": "NewsSummarizer AI"}
    request_id = health.headers[REQUEST_ID_HEADER]
    assert UUID(hex=request_id).version == 4
    assert root.headers[REQUEST_ID_HEADER] != request_id


async def test_swagger_and_openapi_are_available(client):
    docs = await client.get("/docs")
    assert docs.status_code == 200
    assert "Swagger UI" in docs.text

    schema = (await client.get("/openapi.json")).json()
    assert schema["info"] == {"title": "NewsSummarizer AI", "version": "0.1.0"}
    assert "get" in schema["paths"]["/healthz"]


async def test_cors_uses_configured_origins(client):
    response = await client.options(
        "/healthz",
        headers={"Origin": ALLOWED_ORIGIN, "Access-Control-Request-Method": "GET"},
    )
    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == ALLOWED_ORIGIN

    response = await client.get("/healthz", headers={"Origin": "https://other.example"})
    assert "Access-Control-Allow-Origin" not in response.headers


@pytest.mark.parametrize(
    ("method", "path", "status", "detail"),
    [
        ("GET", "/missing", 404, "Not Found"),
        ("POST", "/healthz", 405, "Method Not Allowed"),
        ("GET", f"/_test/validate?amount={TEST_SECRET}", 422, "Request validation failed"),
        ("GET", "/_test/unavailable", 503, "Service unavailable"),
        ("GET", "/_test/fail", 500, "Internal server error"),
    ],
)
async def test_api_errors_have_safe_json(client, method, path, status, detail):
    response = await client.request(method, path, headers={"Origin": ALLOWED_ORIGIN})

    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert REQUEST_ID_HEADER in response.headers
    assert response.headers["Access-Control-Allow-Origin"] == ALLOWED_ORIGIN


async def test_request_logs_do_not_include_secrets(client, tmp_path):
    response = await client.get(
        "/_test/fail",
        params={"token": TEST_SECRET},
        headers={"Authorization": f"Bearer {TEST_SECRET}"},
    )

    assert response.status_code == 500
    file_output = (tmp_path / "logs" / LOG_FILENAME).read_text(encoding="utf-8")
    assert response.headers[REQUEST_ID_HEADER] in file_output
    assert "Unhandled request failure (RuntimeError)" in file_output
    assert TEST_SECRET not in file_output
    assert "JWT_SECRET_KEY" not in file_output

import httpx
import pytest

from streamlit_app.api_client import APIClient, APIError
from streamlit_app.presentation import archive_rows, history_rows


def test_client_sends_bearer_only_to_configured_api():
    def respond(request):
        assert request.headers["Authorization"] == "Bearer fixture-token"
        assert request.url.host == "testserver" and request.url.params["mode"] == "full"
        return httpx.Response(200, json=[{"request_id": "owned"}])

    client = APIClient("http://testserver", "fixture-token", transport=httpx.MockTransport(respond))
    assert client.get("/api/v1/requests", {"mode": "full"}) == [{"request_id": "owned"}]
    with pytest.raises(APIError, match="настроенному FastAPI"):
        client.get("https://different-origin.example/api/v1/requests")


@pytest.mark.parametrize(("status", "text"), [(401, "Сеанс"), (403, "Недостаточно"), (503, "Сервис")])
def test_client_never_displays_untrusted_backend_details(status, text):
    transport = httpx.MockTransport(lambda _: httpx.Response(status, json={"detail": "private server detail"}))
    with pytest.raises(APIError, match=text) as error:
        APIClient("http://testserver", transport=transport).get("/api/v1/requests")
    assert error.value.status == status and "private" not in str(error.value)


def test_client_handles_api_connection_failure():
    def fail(request):
        raise httpx.ConnectError("hidden host configuration", request=request)

    with pytest.raises(APIError, match="Не удалось подключиться к FastAPI"):
        APIClient("http://testserver", transport=httpx.MockTransport(fail)).get("/healthz")


def test_archive_and_history_preserve_missing_fact_metrics():
    entry = {"mode": "baseline", "processed": 20, "successful": 19, "verified_rate": None}
    assert archive_rows({"rows": [entry]})[0]["verified_rate"] is None
    rows = history_rows([{"sample_size": 20, "scored_count": 19, "id": "ae4def12", "model_name": "fixture",
                         "mode": "baseline", "metrics": {"rouge_l": 0.2}},
                        {"sample_size": 20, "id": "448a", "mode": "+rag", "aggregate": {"metrics": None}}])
    row, = rows
    assert row["success_rate"] == 0.95 and row["verified_rate"] is None

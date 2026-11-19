import asyncio
import json
from secrets import token_urlsafe
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio
from fastapi import HTTPException, Request
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from backend.api.main import create_app
from backend.api.runtime import during_request
from backend.core.config import Settings
from backend.data.llm_types import LLMError, LLMReply
from backend.db.init_db import create_test_schema
from backend.db.models import EvaluationResult, LLMCall, SummarizationRequest, User
from backend.retrieval.documents import Document, Holdout
from backend.utils.security import create_access_token, get_signing_key
from scripts.corpus import News

pytestmark = pytest.mark.asyncio
API = "/api/v1"
NEWS = "Konta won the match on Friday and hit 11 aces during the tournament."
SUMMARY = "Konta recorded 11 aces in Friday's win."
UNSUPPORTED = "Johanna Konta won with 11 aces."
REFERENCE = "The reference is only for metrics and must stay out of every prompt."
RUSSIAN_NEWS = "В Казани открыли новую городскую библиотеку. В ней оборудовали 50 рабочих мест и компьютерный класс."
RUSSIAN_SUMMARY = "Новая библиотека в Казани предлагает 50 рабочих мест и компьютерный класс."
ENGLISH_SUMMARY = "A new library with 50 workstations and a computer lab has opened in Kazan."


def judge(score=9, *, critical=False):
    return json.dumps({
        "factual_accuracy": score, "completeness": score, "coherence": score,
        "readability": score, "conciseness": score, "overall_score": score,
        "reasoning": "Internal judge message.", "suggestions": [],
        "critical_factual_error": critical, "missing_key_fact": False,
    })


class Provider:
    def __init__(self, responses, prompts):
        self.responses = iter(responses)
        self.prompts = prompts

    async def complete(self, prompt, model, options):
        self.prompts.append((prompt, model, options.task))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return LLMReply(response, model + "-resolved")


@pytest_asyncio.fixture
async def api(tmp_path):
    fixture = News(id=uuid4(), source_text=NEWS, target_text=REFERENCE, language="en",
                   category="sport", source_dataset="fixture", source_document_id="fixture-1", split="test")
    holdout = tmp_path / "holdout.json"
    holdout.write_text("[" + fixture.model_dump_json() + "]", encoding="utf-8")
    settings = Settings(
        _env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path / 'api.sqlite'}",
        jwt_secret_key=token_urlsafe(64), log_dir=tmp_path / "logs", holdout_path=holdout, llm_max_attempts=1,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        database = app.state.database
        await create_test_schema(database.engine)
        roles = {name: User(username=name, email=f"{name}@example.com", hashed_password="unused",
                            role="admin" if name == "admin" else "user") for name in ("reader", "other", "admin")}
        roles["inactive"] = User(username="inactive", email="inactive@example.com", hashed_password="unused",
                                 role="user", is_active=False)
        async with database.sessions.begin() as session:
            session.add_all(roles.values())
        key = get_signing_key(settings)
        headers = {name: {"Authorization": f"Bearer {create_access_token(user.id, settings, key)}"}
                   for name, user in roles.items()}
        runtime = app.state.summary_runtime
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client, headers, database, runtime, fixture, roles


def configure(runtime, responses, monkeypatch, *, rag=True):
    prompts = []
    runtime.providers = {"ollama": Provider(responses, prompts)}
    if rag:
        async def enrich(state, *, category, language):
            assert REFERENCE != state.original_text
            return state.model_copy(update={"rag_document_ids": [UUID(int=42)]})

        monkeypatch.setattr(runtime, "enrich", enrich)
    return prompts


@pytest.mark.parametrize(("mode", "source", "options", "expected", "target", "rag"), [
    ("baseline", RUSSIAN_NEWS, {}, RUSSIAN_SUMMARY, "ru", None),
    ("baseline", RUSSIAN_NEWS, {"summary_language": "ru"}, RUSSIAN_SUMMARY, "ru", None),
    ("baseline", NEWS, {"summary_language": "auto"}, SUMMARY, "en", None),
    ("full", RUSSIAN_NEWS, {"summary_language": "en", "rag_language": "ru"}, ENGLISH_SUMMARY, "en", "ru"),
    ("full", RUSSIAN_NEWS, {"summary_language": "ru", "language": "en"}, RUSSIAN_SUMMARY, "ru", "en"),
    ("full", RUSSIAN_NEWS, {"summary_language": "auto"}, RUSSIAN_SUMMARY, "ru", "ru"),
])
async def test_summary_language_is_independent_of_retrieval(api, monkeypatch, mode, source, options, expected, target, rag):
    client, headers, _, runtime, _, _ = api
    rag_filters = []

    async def enrich(state, *, category, language):
        rag_filters.append(language)
        return state

    prompts = configure(runtime, [expected, judge()], monkeypatch)
    monkeypatch.setattr(runtime, "enrich", enrich)
    response = await client.post(API + "/summarize", headers=headers["reader"],
                                 json={"text": source, "mode": mode, **options})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == ("verified" if mode == "full" else "draft")
    assert data["final_summary"] == expected and data["language_retries"] == 0
    assert data["source_language"] == ("ru" if source == RUSSIAN_NEWS else "en")
    assert data["target_language"] == data["actual_summary_language"] == target
    assert data["rag_language"] == rag and rag_filters == ([rag] if rag else [])
    assert ("на русском языке" if target == "ru" else "in English") in prompts[0][0]
    assert json.loads(prompts[0][0].partition("Входные данные в JSON:\n")[2])["news"] == source
    assert not data["factual_findings"]


async def test_russian_editor_can_use_one_audited_language_retry(api, monkeypatch):
    client, headers, _, runtime, _, _ = api
    initial = "В новой библиотеке Казани оборудовали компьютерный класс."
    prompts = configure(runtime, [initial, judge(5), ENGLISH_SUMMARY, RUSSIAN_SUMMARY, judge(9)], monkeypatch)
    response = await client.post(API + "/summarize", headers=headers["reader"], json={
        "text": RUSSIAN_NEWS, "mode": "full", "summary_language": "ru",
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "verified" and result["final_summary"] == RUSSIAN_SUMMARY
    assert result["iterations"] == result["language_retries"] == 1
    editor_prompts = [prompt for prompt, _, agent in prompts if agent == "Editor"]
    assert len(editor_prompts) == 2 and all("на русском языке" in prompt for prompt in editor_prompts)
    path = API + f"/requests/{result['request_id']}/llm_calls"
    calls = [row for row in (await client.get(path, headers=headers["reader"])).json() if row["agent"] == "Editor"]
    assert [(row["status"], row["actual_language"], row["target_language"], row["language_retry"], row["language_mismatch"])
            for row in calls] == [("failed", "en", "ru", False, True), ("ok", "ru", "ru", True, False)]


async def test_language_failure_is_reported_without_returning_wrong_language(api, monkeypatch):
    client, headers, _, runtime, _, _ = api
    prompts = configure(runtime, [ENGLISH_SUMMARY, ENGLISH_SUMMARY], monkeypatch)
    response = await client.post(API + "/summarize", headers=headers["reader"], json={
        "text": RUSSIAN_NEWS, "mode": "baseline", "summary_language": "auto",
    })
    assert response.status_code == 502 and response.json()["detail"]["code"] == "language_mismatch"
    request_id = response.json()["detail"]["request_id"]
    path = f"{API}/requests/{request_id}"
    saved = (await client.get(path, headers=headers["reader"])).json()
    assert saved["status"] == "failed" and saved["final_summary"] is None
    assert saved["language_retries"] == 1 and len(prompts) == 2
    steps = (await client.get(path + "/reasoning_log", headers=headers["reader"])).json()["steps"]
    assert steps[0]["errors"] == ["language_mismatch"]


async def test_baseline_uses_only_summarizer_and_persists_request(api, monkeypatch):
    client, headers, database, runtime, _, users = api
    prompts = configure(runtime, [SUMMARY], monkeypatch)
    response = await client.post(API + "/summarize", headers=headers["reader"], json={
        "text": NEWS, "mode": "baseline", "model": "local:test", "max_sentences": 1,
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert UUID(result["request_id"]) == UUID(response.headers["X-Request-ID"])
    assert result["text"] == NEWS and result["model"] == "local:test-resolved"
    assert result["status"] == "draft" and result["initial_summary"] == result["final_summary"] == SUMMARY
    assert result["summary_sentence_count"] == result["max_sentences"] == 1
    assert result["critic_passed"] is result["validation_passed"] is None and result["rag_ids"] == []
    assert [task for _, _, task in prompts] == ["Summarizer"]
    assert '"max_sentences": 1' in prompts[0][0]
    path = API + "/requests/" + result["request_id"]
    for role, expected in (("reader", 200), ("admin", 200), ("other", 403), ("inactive", 401)):
        detail = await client.get(path, headers=headers[role])
        assert detail.status_code == expected
        if expected == 200:
            assert detail.json()["user_id"] == str(users["reader"].id)
        log = await client.get(path + "/reasoning_log", headers=headers[role])
        assert log.status_code == expected
        if expected == 200:
            assert [step["agent"] for step in log.json()["steps"]] == ["Summarizer", "Supervisor"]
    assert (await client.get(f"{API}/requests/{uuid4()}", headers=headers["reader"])).status_code == 404
    async with database.sessions() as session:
        row = await session.get(SummarizationRequest, UUID(result["request_id"]))
        call = await session.scalar(select(LLMCall).where(LLMCall.request_id == row.id))
        assert row.summary == SUMMARY and call.agent_name == "Summarizer"


async def test_baseline_rejects_excess_sentences_instead_of_cutting_off_facts(api, monkeypatch):
    client, headers, _, runtime, _, _ = api
    configure(runtime, ["A library opened. It has a computer lab. It offers 50 workstations."], monkeypatch)
    response = await client.post(API + "/summarize", headers=headers["reader"], json={
        "text": "The city opened a library with a computer lab and 50 workstations.",
        "mode": "baseline", "max_sentences": 2,
    })
    assert response.status_code == 502
    error = response.json()["detail"]
    assert error["code"] == "sentence_limit"
    saved = await client.get(f"{API}/requests/{error['request_id']}", headers=headers["reader"])
    assert saved.json()["status"] == "failed" and saved.json()["final_summary"] is None


async def test_full_edits_overlong_summary_and_passes_same_limit(api, monkeypatch):
    client, headers, _, runtime, _, _ = api
    draft = "Новая библиотека открылась в Казани. Она предлагает 50 рабочих мест. В ней есть компьютерный класс."
    prompts = configure(runtime, [draft, judge(10), RUSSIAN_SUMMARY, judge(9)], monkeypatch)
    response = await client.post(API + "/summarize", headers=headers["reader"], json={
        "text": RUSSIAN_NEWS, "mode": "full", "max_sentences": 2,
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["initial_summary"] == draft and result["final_summary"] == RUSSIAN_SUMMARY
    assert result["status"] == "verified" and result["iterations"] == result["summary_sentence_count"] == 1
    assert result["actual_summary_language"] == "ru"
    assert [json.loads(prompt.split("Входные данные в JSON:\n", 1)[1])["max_sentences"]
            for prompt, _, agent in prompts if agent in {"Summarizer", "Editor"}] == [2, 2]


async def test_full_can_finish_without_retrieved_documents(api, monkeypatch):
    client, headers, _, runtime, _, _ = api
    prompts = configure(runtime, [RUSSIAN_SUMMARY, judge()], monkeypatch)

    async def empty(state, *, category, language):
        assert language == "ru" and category == "unrepresented"
        return state.model_copy(update={"rag_search_mode": "no_results"})

    monkeypatch.setattr(runtime, "enrich", empty)
    response = await client.post(API + "/summarize", headers=headers["reader"], json={
        "text": RUSSIAN_NEWS, "mode": "full", "category": "unrepresented",
    })
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "verified" and result["rag_search_mode"] == "no_results" and not result["rag_ids"]
    assert "style_examples" not in json.loads(prompts[0][0].split("Входные данные в JSON:\n", 1)[1])


async def test_baseline_cache_never_reuses_another_users_result_or_outdated_model(api):
    from backend.data.ollama import OllamaProvider
    from sqlalchemy import update
    from backend.db.models import CacheEntry
    from datetime import UTC, datetime, timedelta

    client, headers, database, runtime, _, _ = api
    runtime.settings.baseline_cache_ttl_seconds = 300
    revision = "initial-digest"
    count = 0

    def respond(request):
        nonlocal count
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:1.5b", "digest": revision}]})
        assert request.url.path == "/api/generate"
        count += 1
        return httpx.Response(200, json={"response": SUMMARY, "done": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        runtime.providers["ollama"] = OllamaProvider(http, "http://ollama")
        data = {"text": NEWS, "mode": "baseline"}
        first = (await client.post(API + "/summarize", headers=headers["reader"], json=data)).json()
        second = (await client.post(API + "/summarize", headers=headers["reader"], json=data)).json()
        assert count == 1 and not first["cache_hit"] and second["cache_hit"]
        assert first["request_id"] != second["request_id"] and second["final_summary"] == SUMMARY
        path = f"{API}/requests/{second['request_id']}"
        assert (await client.get(path + "/llm_calls", headers=headers["reader"])).json() == []
        steps = (await client.get(path + "/reasoning_log", headers=headers["reader"])).json()["steps"]
        assert [row["agent"] for row in steps] == ["Cache"]
        assert (await client.get(path, headers=headers["other"])).status_code == 403
        assert not (await client.post(API + "/summarize", headers=headers["other"], json=data)).json()["cache_hit"]
        assert not (await client.post(API + "/summarize", headers=headers["reader"],
                                      json=data | {"max_sentences": 1})).json()["cache_hit"]
        revision = "updated-digest"
        assert not (await client.post(API + "/summarize", headers=headers["reader"], json=data)).json()["cache_hit"]
    assert count == 4 and await runtime.cache.statistics() == {"entries": 4, "hits": 1}
    async with database.sessions.begin() as session:
        await session.execute(update(CacheEntry).values(expires_at=datetime.now(UTC) - timedelta(seconds=1)))
    assert await runtime.cache.purge() == 4
    assert await runtime.cache.statistics() == {"entries": 0, "hits": 0}


async def test_monitor_reports_missing_services_and_only_admin_gets_cache_stats(api, monkeypatch):
    from backend.api.routes import health

    client, headers, _, _, _, _ = api

    async def ollama(_http, _url):
        return "ok"

    async def offline(_settings):
        return "unavailable"

    monkeypatch.setattr(health, "_http_health", ollama)
    monkeypatch.setattr(health, "_redis_health", offline)
    monkeypatch.setattr(health, "_snapshot_health", lambda _: "snapshot_available")
    url = "/health/dependencies"
    assert (await client.get(url)).status_code == 401
    data = (await client.get(url, headers=headers["reader"])).json()
    assert data == {"api": "ok", "database": "ok", "ollama": "ok", "redis": "unavailable", "worker": "not_checked",
                    "qdrant": "snapshot_available", "rag_snapshot": "snapshot_available"}
    assert (await client.get(url, headers=headers["admin"])).json()["cache"] == {
        "enabled": False, "entries": 0, "hits": 0,
    }


async def test_monitor_distinguishes_qdrant_service_from_rag_snapshot(api, monkeypatch, tmp_path):
    from contextlib import asynccontextmanager, nullcontext
    from backend.api.routes import health

    client, headers, _, runtime, _, _ = api
    runtime.settings.qdrant_mode = "server"
    runtime.settings.rag_index_dir = tmp_path
    version = "f" * 32
    manifest = tmp_path / "active.json"
    snapshot = tmp_path / "versions" / version
    snapshot.mkdir(parents=True)
    (snapshot / "bm25.json").touch()
    manifest.write_text(json.dumps({"version": version, "mode": "local"}), encoding="utf-8")
    async def http_probe(url, *, timeout):
        assert timeout == 1.5
        return httpx.Response(200 if httpx.URL(url).path in {"/api/version", "/readyz"} else 404)

    monkeypatch.setattr(runtime.providers["ollama"].client, "get", http_probe)

    class RedisProbe:
        async def ping(self):
            return True

    @asynccontextmanager
    async def redis_connection(*_args, **_kwargs):
        yield RedisProbe()

    class Control:
        def ping(self, *, timeout):
            assert timeout == 1
            return [{"test-worker": {"ok": "pong"}}]

    class CeleryProbe:
        control = Control()

    monkeypatch.setattr(health.Redis, "from_url", redis_connection)
    monkeypatch.setattr(health, "celery_client", lambda _: nullcontext(CeleryProbe()))
    path = "/health/dependencies"
    first = (await client.get(path, headers=headers["reader"])).json()
    assert first["qdrant"] == "ok" and first["rag_snapshot"] == "unavailable"
    assert first["worker"] == first["redis"] == "ok"
    manifest.write_text(json.dumps({"version": version, "mode": "server"}), encoding="utf-8")
    assert (await client.get(path, headers=headers["reader"])).json()["rag_snapshot"] == "snapshot_available"


async def test_monitor_handles_malformed_local_snapshot_and_service_errors(api, monkeypatch, tmp_path):
    from contextlib import asynccontextmanager, contextmanager
    from kombu.exceptions import OperationalError as BrokerError
    from redis.exceptions import RedisError
    from backend.api.routes import health

    client, headers, _, runtime, _, _ = api
    runtime.settings.rag_index_dir = tmp_path
    manifest = tmp_path / "active.json"
    manifest.write_text('{"version":"../outside","mode":"local"}', encoding="utf-8")

    async def no_server(_url, **_kwargs):
        raise httpx.ConnectError("private endpoint details")

    class RedisProbe:
        async def ping(self):
            raise RedisError("private broker details")

    @asynccontextmanager
    async def redis_connection(*_args, **_kwargs):
        yield RedisProbe()

    @contextmanager
    def offline_worker(_settings):
        raise BrokerError("private worker address")
        yield

    monkeypatch.setattr(runtime.providers["ollama"].client, "get", no_server)
    monkeypatch.setattr(health.Redis, "from_url", redis_connection)
    monkeypatch.setattr(health, "celery_client", offline_worker)
    path = "/health/dependencies"
    result = await client.get(path, headers=headers["reader"])
    assert result.status_code == 200 and "private" not in result.text
    assert result.json()["qdrant"] == result.json()["ollama"] == result.json()["redis"] == "unavailable"
    manifest.write_text("invalid JSON", encoding="utf-8")
    assert health._snapshot_health(runtime.settings) == "unavailable"
    assert health._worker_health(runtime.settings) == "unavailable"


@pytest.mark.parametrize("tags", [None, {"models": []}])
async def test_unavailable_model_digest_bypasses_active_cache(api, tags):
    from backend.data.ollama import OllamaProvider

    client, headers, _, runtime, _, _ = api
    runtime.settings.baseline_cache_ttl_seconds = 300
    generations = 0

    def respond(request):
        nonlocal generations
        if request.url.path == "/api/tags":
            return httpx.Response(503 if tags is None else 200, json=tags)
        generations += 1
        return httpx.Response(200, json={"response": SUMMARY, "done": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        runtime.providers["ollama"] = OllamaProvider(http, "http://ollama")
        for _ in range(2):
            response = await client.post(API + "/summarize", headers=headers["reader"],
                                         json={"text": NEWS, "mode": "baseline"})
            assert response.status_code == 200 and not response.json()["cache_hit"]
    assert generations == 2 and await runtime.cache.statistics() == {"entries": 0, "hits": 0}


@pytest.mark.parametrize("path", ["/summarize", "/summarize/batch", "/summarize/multi",
                                 "/summarize/ensemble", "/evaluate"])
async def test_post_routes_require_jwt(api, path):
    client, *_ = api
    response = await client.post(API + path, json={})
    assert response.status_code == 401


@pytest.mark.parametrize("patch", [
    {"text": "   "}, {"text": "a" * 8001}, {"mode": "invalid"}, {"max_sentences": 0},
    {"max_sentences": "2"}, {"provider": "gemini"}, {"model": "llama:cloud"},
    {"provider": "transformers", "model": "C:/local/path"},
])
async def test_input_is_validated(api, patch):
    client, headers, *_ = api
    response = await client.post(API + "/summarize", headers=headers["reader"], json={"text": NEWS} | patch)
    assert response.status_code == 422


@pytest.mark.parametrize(("first", "grade", "edited", "expected"), [
    (SUMMARY, judge(9), None, "verified"),
    (UNSUPPORTED, judge(9), UNSUPPORTED, "needs_review"),
    (SUMMARY, judge(9, critical=True), SUMMARY, "unchanged"),
    ("Konta recorded 90 aces in Friday's win.", judge(9), SUMMARY, "verified"),
    ("Konta won. She recorded 11 aces.", judge(9), SUMMARY, "verified"),
])
async def test_full_verdict_and_reasoning(api, monkeypatch, first, grade, edited, expected):
    client, headers, _, runtime, _, _ = api
    responses = [first, grade] + ([edited] if edited else []) + ([judge()] if edited and edited != first else [])
    configure(runtime, responses, monkeypatch)
    response = await client.post(API + "/summarize", headers=headers["reader"], json={
        "text": NEWS, "category": "sport", "max_sentences": 1,
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["rag_ids"] == [str(UUID(int=42))]
    assert result["status"] == expected and result["factual_review_required"] is (expected == "needs_review")
    log = await client.get(f"{API}/requests/{result['request_id']}/reasoning_log", headers=headers["reader"])
    steps = log.json()["steps"]
    assert [step["agent"] for step in steps[:4]] == ["RAG", "Summarizer", "Validator", "Critic"]
    assert steps[2]["validation_passed"] is (first in (SUMMARY, UNSUPPORTED))
    assert steps[3]["critic_criteria_scores"]["factual_accuracy"] == 9
    assert "Internal judge message" not in log.text
    if edited and edited != first:
        assert log.json()["revisions"] == [{"iteration": 1, "summary": SUMMARY}]
        assert [step["agent"] for step in steps[4:]] == ["Editor", "Validator", "Critic", "Supervisor"]


@pytest.mark.parametrize(("code", "status"), [
    ("model_unavailable", 503), ("provider_unavailable", 503), ("timeout", 504), ("invalid_response", 502),
])
async def test_llm_failure_has_safe_http_status_and_persisted_history(api, monkeypatch, code, status):
    client, headers, _, runtime, _, _ = api
    configure(runtime, [LLMError(code)], monkeypatch)
    response = await client.post(API + "/summarize", headers=headers["reader"], json={"text": NEWS, "mode": "baseline"})
    assert response.status_code == status
    error = response.json()["detail"]
    assert error["code"] == code
    logged = await client.get(f"{API}/requests/{error['request_id']}", headers=headers["reader"])
    assert logged.json()["status"] == "failed"


async def test_index_failure_is_503_and_does_not_call_llm(api, monkeypatch):
    client, headers, _, runtime, _, _ = api
    prompts = configure(runtime, [], monkeypatch, rag=False)

    async def unavailable(*_args, **_kwargs):
        raise RuntimeError("internal index location")

    monkeypatch.setattr(runtime, "enrich", unavailable)
    response = await client.post(API + "/summarize", headers=headers["reader"], json={"text": NEWS})
    assert response.status_code == 503 and response.json()["detail"]["code"] == "rag_failed"
    assert not prompts and "internal index location" not in response.text


async def test_database_failure_does_not_expose_details(api, monkeypatch):
    client, headers, _, runtime, _, _ = api

    async def unavailable(*_args, **_kwargs):
        raise OperationalError("query", {}, RuntimeError("private database details"))

    monkeypatch.setattr(runtime, "generate", unavailable)
    response = await client.post(API + "/summarize", headers=headers["reader"], json={"text": NEWS})
    assert response.status_code == 503 and "private database details" not in response.text


async def test_generation_deadline_cancels_and_preserves_audit(api, monkeypatch):
    client, headers, database, runtime, _, _ = api
    configure(runtime, [], monkeypatch)
    runtime.settings.api_run_timeout_seconds = 1
    entered = asyncio.Event()

    async def blocked(*_args):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(runtime.providers["ollama"], "complete", blocked)
    response = await client.post(API + "/summarize", headers=headers["reader"], json={
        "text": NEWS, "mode": "baseline",
    })
    assert response.status_code == 504 and entered.is_set()
    request_id = response.json()["detail"]["request_id"]
    logged = await client.get(f"{API}/requests/{request_id}", headers=headers["reader"])
    assert logged.json()["status"] == "cancelled"
    async with database.sessions() as session:
        call = await session.scalar(select(LLMCall).where(LLMCall.request_id == UUID(request_id)))
        assert call.status == call.parameters["error_type"] == "cancelled"


@pytest.mark.parametrize("route", ["batch", "multi"])
async def test_group_preserves_results_and_candidate_errors(api, monkeypatch, route):
    client, headers, _, runtime, _, _ = api
    configure(runtime, [SUMMARY, LLMError("model_unavailable")], monkeypatch)
    variants = [{"mode": "baseline", "model": "ready"}, {"mode": "baseline", "model": "missing"}]
    payload = ({"items": [variant | {"text": NEWS} for variant in variants]} if route == "batch"
               else {"text": NEWS, "variants": variants})
    response = await client.post(f"{API}/summarize/{route}", headers=headers["reader"], json=payload)
    assert response.status_code == 200, response.text
    good, bad = response.json()["results"]
    assert good["result"]["final_summary"] == SUMMARY and good["result"]["model"] == "ready-resolved"
    assert bad["error"]["code"] == "model_unavailable"
    assert bad["error"]["request_id"] == bad["result"]["request_id"] != good["result"]["request_id"]


async def test_ensemble_filters_high_scoring_unsupported_candidate(api, monkeypatch):
    client, headers, _, runtime, _, _ = api
    configure(runtime, [UNSUPPORTED, judge(10), UNSUPPORTED, SUMMARY, judge(7), LLMError("model_unavailable")],
              monkeypatch)
    payload = {"text": NEWS, "candidates": [{"model": tag} for tag in ("overconfident", "valid", "missing")]}
    response = await client.post(API + "/summarize/ensemble", headers=headers["reader"], json=payload)
    assert response.status_code == 200, response.text
    first, winner, third = response.json()["results"]
    assert first["result"]["status"] == "needs_review" and first["result"]["critic_score"] == 10
    assert third["error"]["code"] == "model_unavailable"
    assert response.json()["winner"] == winner["result"]


async def test_evaluate_persists_metrics_without_exposing_reference(api, monkeypatch):
    pytest.importorskip("sacrebleu")
    pytest.importorskip("bert_score")
    client, headers, database, runtime, fixture, users = api
    prompts = configure(runtime, [SUMMARY, judge()], monkeypatch)
    seen = []

    def calculate(pairs):
        seen.extend(pairs)
        from backend.evaluation.metrics_calculator import SummaryMetrics
        return [SummaryMetrics.model_validate({key: 0.25 for key in SummaryMetrics.model_fields})]

    monkeypatch.setattr(runtime.metrics, "calculate", calculate)
    payload = {"test_ids": [fixture.source_document_id], "model": "test", "mode": "full"}
    response = await client.post(API + "/evaluate", headers=headers["reader"], json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["sample_size"] == data["scored_count"] == 1 and data["model_name"] == "test-resolved"
    assert [task for _, _, task in prompts] == ["Summarizer", "Critic"]
    assert seen == [(SUMMARY, REFERENCE)] and all(REFERENCE not in prompt for prompt, _, _ in prompts)
    assert REFERENCE not in response.text
    history = API + "/evaluate/history"
    assert (await client.get(history, headers=headers["reader"])).json() == [data]
    assert (await client.get(history, headers=headers["admin"])).json() == [data]
    assert (await client.get(history, headers=headers["other"])).json() == []
    assert (await client.get(history)).status_code == 401
    async with database.sessions() as session:
        report = await session.get(EvaluationResult, UUID(data["id"]))
        assert report.user_id == users["reader"].id and report.metrics_json["items"] == data["items"]
    duplicate = payload | {"test_ids": [fixture.source_document_id, str(fixture.id)]}
    assert (await client.post(API + "/evaluate", headers=headers["reader"], json=duplicate)).status_code == 422


async def test_disconnection_cancels_and_joins_worker():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def work():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def receive():
        await started.wait()
        return {"type": "http.disconnect"}

    request = Request({"type": "http"}, receive=receive)
    with pytest.raises(HTTPException) as error:
        await during_request(request, work, 1)
    assert error.value.status_code == 499 and cancelled.is_set()


async def test_admin_glossary_crud_and_atomic_csv(api):
    client, headers, *_ = api
    endpoint = API + "/admin/glossary"
    entry = {"term": "BM25", "definition": "A relevance ranking function.", "language": "en"}
    assert (await client.post(endpoint, headers=headers["reader"], json=entry)).status_code == 403
    created = await client.post(endpoint, headers=headers["admin"], json=entry)
    assert created.status_code == 201, created.text
    path = f"{endpoint}/{created.json()['id']}"
    assert (await client.get(path, headers=headers["admin"])).json()["term"] == "BM25"
    updated = await client.put(path, headers=headers["admin"], json=entry | {"aliases": ["Okapi BM25"]})
    assert updated.status_code == 200 and updated.json()["aliases"] == ["Okapi BM25"]
    csv_headers = headers["admin"] | {"Content-Type": "text/csv"}
    csv = "term,definition,language\nRRF,Rank fusion,en\nBM25,Duplicate,en\n"
    conflict = await client.post(endpoint + "/import", headers=csv_headers, content=csv.encode())
    assert conflict.status_code == 409
    assert len((await client.get(endpoint, headers=headers["admin"])).json()) == 1
    response = await client.post(endpoint + "/import", headers=csv_headers, content=csv.split("BM25")[0].encode())
    assert response.status_code == 201 and response.json() == {"imported": 1}
    assert (await client.delete(path, headers=headers["admin"])).status_code == 204
    assert (await client.get(path, headers=headers["admin"])).status_code == 404


async def test_admin_knowledge_base_rejects_holdout_pairs(api, monkeypatch):
    pytest.importorskip("qdrant_client")
    client, headers, _, _, fixture, _ = api
    from scripts import build_indexes

    holdout = Holdout.from_documents([
        Document.model_validate({**fixture.model_dump(exclude={"split"}), "source_split": "test"}),
    ])
    monkeypatch.setattr(build_indexes, "load_holdout", lambda _: holdout)
    endpoint = API + "/admin/knowledge-base"
    payload = {"source_text": "Independent training fixture.", "target_text": "Fixture summary.", "language": "en"}
    assert (await client.post(endpoint, headers=headers["admin"], json=payload)).status_code == 201
    for changed in ({"source_text": NEWS}, {"target_text": REFERENCE}):
        assert (await client.post(endpoint, headers=headers["admin"], json=payload | changed)).status_code == 422


async def test_batch_experiment_jwt_history_and_comparison(api, monkeypatch):
    from backend.api.routes import evaluation as routes
    from backend.db.models import ExperimentJob, ModelComparison

    client, headers, database, _, _, users = api
    published = []

    def publish(_settings, experiment_id, *, comparison):
        published.append((experiment_id, comparison))

    monkeypatch.setattr(routes, "_publish", publish)
    path = API + "/evaluate"
    payload = {"models": [{"model": "one"}, {"model": "two"}], "modes": ["baseline"], "sample_sizes": [20]}
    assert (await client.post(path + "/batch", json=payload)).status_code == 401
    response = await client.post(path + "/batch", headers=headers["reader"], json=payload)
    assert response.status_code == 202
    experiment_id = UUID(response.json()["experiment_id"])
    assert published == [(experiment_id, True)]
    resource = f"{path}/experiments/{experiment_id}"
    own = await client.get(resource, headers=headers["reader"])
    assert own.json()["status"] == "queued" and own.json()["total"] == 40
    assert (await client.get(resource, headers=headers["other"])).status_code == 403
    assert (await client.get(resource, headers=headers["admin"])).status_code == 200
    async with database.sessions.begin() as session:
        job = await session.get(ExperimentJob, experiment_id)
        job.status = "completed"
        session.add(ModelComparison(experiment_id=experiment_id, models_compared=["one", "two"], sample_size=20,
                                    results_json={"runs": [], "winner_evaluation_id": None}))
        session.add(EvaluationResult(user_id=users["reader"].id, experiment_id=experiment_id, model_name="one",
                                     mode="baseline", sample_size=20, evaluation_time=3.2,
                                     metrics_json={"provider": "ollama", "aggregate": {"success_count": 20}}))
    assert (await client.get(path + "/compare?sample_size=20&models=one,two",
                             headers=headers["reader"])).json()[0]["experiment_id"] == str(experiment_id)
    assert (await client.get(path + "/compare", headers=headers["other"])).json() == []
    assert (await client.get(path + "/history", headers=headers["reader"])).json()[0]["aggregate"] == {
        "success_count": 20,
    }


async def test_batch_unavailable_broker_returns_503(api, monkeypatch):
    from kombu.exceptions import OperationalError
    from backend.api.routes import evaluation as routes

    def unavailable(*args, **kwargs):
        raise OperationalError("unexposed broker configuration")

    monkeypatch.setattr(routes, "_publish", unavailable)
    client, headers, *_ = api
    path = API + "/evaluate"
    payload = {"models": [{"model": "one"}], "modes": ["+rag"], "sample_sizes": [20]}
    response = await client.post(path + "/batch", headers=headers["reader"], json=payload)
    assert response.status_code == 503 and "unexposed" not in response.text
    experiment_id = response.json()["detail"]["experiment_id"]
    state = await client.get(f"{path}/experiments/{experiment_id}", headers=headers["reader"])
    assert state.json()["status"] == "broker_unavailable"


async def test_evaluate_keeps_failed_example_out_of_metrics_and_in_history(api, monkeypatch):
    pytest.importorskip("sacrebleu")
    pytest.importorskip("bert_score")
    from backend.evaluation.metrics_calculator import SummaryMetrics

    client, headers, _, runtime, fixture, _ = api
    other = fixture.model_copy(update={"id": uuid4(), "source_document_id": "fixture-2",
                                      "source_text": "A different article about a library.",
                                      "target_text": "A separate held-out reference."})
    runtime.settings.holdout_path.write_text(
        json.dumps([row.model_dump(mode="json") for row in (fixture, other)]), encoding="utf-8",
    )
    configure(runtime, [SUMMARY, LLMError("model_unavailable")], monkeypatch)

    def score(pairs):
        assert pairs == [(SUMMARY, REFERENCE)]
        return [SummaryMetrics.model_validate(dict.fromkeys(SummaryMetrics.model_fields, 0.4))]

    monkeypatch.setattr(runtime.metrics, "calculate", score)
    base = API + "/evaluate"
    payload = {"test_ids": ["fixture-1", "fixture-2"], "mode": "baseline", "model": "test"}
    response = await client.post(base, headers=headers["reader"], json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sample_size"] == 2 and data["scored_count"] == 1
    assert data["metrics"]["rouge_l"] == 0.4 and data["items"][1]["metrics"] is None
    assert data["items"][1]["error"]["code"] == "model_unavailable"
    assert (await client.get(base + "/history", headers=headers["reader"])).json() == [data]

    missing = await client.post(base, headers=headers["reader"], json=payload | {"test_ids": ["unknown"]})
    assert missing.status_code == 404
    configure(runtime, [LLMError("model_unavailable")], monkeypatch)
    failed = await client.post(base, headers=headers["reader"], json=payload | {"test_ids": ["fixture-2"]})
    assert failed.status_code == 503 and failed.json()["detail"]["items"][0]["status"] == "failed"


async def test_evaluate_rejects_train_split_and_does_not_persist_unavailable_metrics(api, monkeypatch):
    from backend.evaluation.metrics_calculator import MetricsUnavailable

    client, headers, _, runtime, fixture, _ = api
    path = API + "/evaluate"
    payload = {"mode": "baseline", "model": "test"}
    prompts = configure(runtime, [SUMMARY], monkeypatch)
    train = fixture.model_copy(update={"split": "train"})
    runtime.settings.holdout_path.write_text("[" + train.model_dump_json() + "]", encoding="utf-8")
    rejected = await client.post(path, headers=headers["reader"], json=payload)
    assert rejected.status_code == 503 and not prompts

    runtime.settings.holdout_path.write_text("[" + fixture.model_dump_json() + "]", encoding="utf-8")
    def unavailable(_pairs):
        raise MetricsUnavailable("metric artifacts missing")

    monkeypatch.setattr(runtime.metrics, "calculate", unavailable)
    rejected = await client.post(path, headers=headers["reader"], json=payload)
    assert rejected.status_code == 503 and "WordNet" in rejected.json()["detail"]
    assert len(prompts) == 1
    assert (await client.get(path + "/history", headers=headers["reader"])).json() == []


ATTRIBUTION_SOURCE = (
    "Officials said the graves held unidentified people and newborns. Human Rights Watch says investigators "
    "must verify whether people killed in January protests were buried at the site. "
    "The ministry says burials took place on March 19."
)
ATTRIBUTION_ERROR = "The ministry claims all bodies buried at the site were killed in the January protests."


async def test_regression_xsum_32197742_routes_wrong_attribution_to_review(api, monkeypatch):
    client, headers, database, runtime, fixture, _ = api
    pytest.importorskip("qdrant_client")
    from backend.db.models import Feedback, KnowledgeBase
    from scripts import build_indexes

    holdout = Holdout.from_documents([
        Document.model_validate({**fixture.model_dump(exclude={"split"}), "source_split": "test"}),
    ])
    monkeypatch.setattr(build_indexes, "load_holdout", lambda _: holdout)
    configure(runtime, [ATTRIBUTION_ERROR, judge(10), ATTRIBUTION_ERROR], monkeypatch)
    response = await client.post(API + "/summarize", headers=headers["reader"],
                                 json={"text": ATTRIBUTION_SOURCE, "mode": "full"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "needs_review" and result["critic_score"] == 10
    assert any(item["code"] == "unsupported_attribution" for item in result["factual_findings"])
    path = API + "/hitl/queue"
    assert (await client.get(path, headers=headers["reader"])).status_code == 403
    listed = await client.get(path, headers=headers["admin"])
    item, = listed.json()
    assert item["request_id"] == result["request_id"] and item["reasons"] == ["unsupported_attribution"]
    endpoint = f"{path}/{item['id']}/decisions"
    assert (await client.post(endpoint, headers=headers["other"], json={"decision": "rejected"})).status_code == 403
    corrected = "Human Rights Watch called for checks into whether people killed during protests were buried at the site."
    payload = {"decision": "approved", "corrected_summary": corrected, "comment": "Исправлена атрибуция."}
    approved = await client.post(endpoint, json=payload, headers=headers["admin"])
    assert approved.status_code == 201, approved.text
    assert approved.json()["training_status"] == "awaiting_index"
    assert (await client.post(endpoint, json=payload, headers=headers["admin"])).status_code == 409
    decisions = await client.get(endpoint, headers=headers["admin"])
    assert [item["corrected_summary"] for item in decisions.json()] == [corrected]
    async with database.sessions() as session:
        records = (await session.scalars(select(KnowledgeBase))).all()
        assert [(entry.target_text, entry.source_split) for entry in records] == [(corrected, "train")]
        assert len((await session.scalars(select(Feedback))).all()) == 1


async def test_hitl_feedback_excludes_heldout_pair_from_training(api, monkeypatch):
    pytest.importorskip("qdrant_client")
    from backend.db.models import KnowledgeBase, UncertainExample
    from scripts import build_indexes

    client, headers, database, _, fixture, users = api
    holdout = Holdout.from_documents([
        Document.model_validate({**fixture.model_dump(exclude={"split"}), "source_split": "test"}),
    ])
    monkeypatch.setattr(build_indexes, "load_holdout", lambda _: holdout)
    async with database.sessions.begin() as session:
        request = SummarizationRequest(user_id=users["reader"].id, input_text=NEWS,
                                       summary=SUMMARY, status="needs_review")
        session.add(request)
        await session.flush()
        item = UncertainExample(request_id=request.id, input_text=NEWS, model_output=SUMMARY)
        session.add(item)
        await session.flush()
    endpoint = f"{API}/hitl/queue/{item.id}/decisions"
    approved = await client.post(endpoint, headers=headers["admin"],
                                 json={"decision": "approved", "corrected_summary": SUMMARY})
    assert approved.status_code == 201 and approved.json()["training_status"] == "heldout"
    async with database.sessions() as session:
        assert (await session.scalars(select(KnowledgeBase))).all() == []


async def test_editor_can_publish_but_user_cannot_and_failure_is_safe(api, monkeypatch):
    from backend.retrieval.documents import IndexIntegrityError

    client, headers, database, runtime, _, _ = api
    editor = User(username="editor", email="editor@example.com", hashed_password="unused", role="editor")
    async with database.sessions.begin() as session:
        session.add(editor)
    token = create_access_token(editor.id, runtime.settings, get_signing_key(runtime.settings))
    authorization = {"Authorization": f"Bearer {token}"}
    endpoint = API + "/hitl/publish"

    async def unavailable():
        raise IndexIntegrityError("internal filesystem path")

    monkeypatch.setattr(runtime, "publish_reviewed_corpus", unavailable)
    assert (await client.post(endpoint, headers=headers["reader"])).status_code == 403
    failure = await client.post(endpoint, headers=authorization)
    assert failure.status_code == 503 and "internal filesystem path" not in failure.text

    async def published():
        return {"version": "snapshot-2", "document_count": 1001}

    monkeypatch.setattr(runtime, "publish_reviewed_corpus", published)
    response = await client.post(endpoint, headers=authorization)
    assert response.status_code == 200
    assert response.json() == {"version": "snapshot-2", "documents": 1001}


async def test_editor_can_record_decision_when_holdout_is_missing(api, tmp_path):
    from backend.db.models import KnowledgeBase, UncertainExample

    client, headers, database, runtime, _, users = api
    runtime.settings.holdout_path = tmp_path / "missing-holdout.json"
    async with database.sessions.begin() as session:
        request = SummarizationRequest(user_id=users["reader"].id, input_text=NEWS,
                                       summary=SUMMARY, status="needs_review")
        session.add(request)
        await session.flush()
        item = UncertainExample(request_id=request.id, input_text=NEWS, model_output=SUMMARY)
        session.add(item)
        await session.flush()
    editor = headers["admin"]
    endpoint = f"{API}/hitl/queue/{item.id}/decisions"
    unknown = f"{API}/hitl/queue/{uuid4()}/decisions"
    payload = {"decision": "approved", "corrected_summary": SUMMARY}
    assert (await client.post(endpoint, headers=editor, json={"decision": "approved"})).status_code == 422
    assert (await client.post(endpoint, headers=editor, json=payload | {
        "corrected_summary": "Sentence one. Sentence two. Sentence three. Sentence four.",
    })).status_code == 422
    assert (await client.get(unknown, headers=editor)).status_code == 404
    assert (await client.post(unknown, headers=editor, json=payload)).status_code == 404
    response = await client.post(endpoint, headers=editor, json=payload)
    assert response.status_code == 201 and response.json()["training_status"] == "holdout_unavailable"
    assert (await client.get(endpoint, headers=editor)).json()[0]["decision"] == "approved"
    async with database.sessions() as session:
        assert (await session.scalars(select(KnowledgeBase))).all() == []


async def test_history_filters_and_llm_trace_only_contain_owned_data(api, monkeypatch):
    client, headers, _, runtime, _, _ = api
    configure(runtime, [SUMMARY, SUMMARY], monkeypatch)
    own, foreign = [(await client.post(API + "/summarize", headers=headers[role],
                                       json={"text": NEWS, "mode": "baseline"})).json()
                    for role in ("reader", "other")]
    route = API + "/requests"
    options = {"mode": "baseline", "status": "draft", "date_from": "2000-01-01", "date_to": "2100-01-01"}
    answer = await client.get(route, headers=headers["reader"], params=options)
    assert [item["request_id"] for item in answer.json()] == [own["request_id"]]
    assert (await client.get(route, headers=headers["reader"], params=options | {"offset": 1})).json() == []
    assert (await client.get(route, headers=headers["reader"], params=options | {"mode": "full"})).json() == []
    assert (await client.get(route, headers=headers["reader"], params=options | {"date_to": "1999-01-01"})).status_code == 422
    trace = f"{route}/{own['request_id']}/llm_calls"
    assert (await client.get(trace, headers=headers["other"])).status_code == 403
    log = await client.get(trace, headers=headers["reader"])
    assert len(log.json()) == 1 and log.json()[0]["agent"] == "Summarizer"
    assert "prompt" not in log.text and foreign["request_id"] not in answer.text


async def test_archived_comparison_projects_only_aggregate_metrics(api, tmp_path):
    client, headers, _, runtime, _, _ = api
    path = tmp_path / "archive.json"
    runtime.settings.comparison_archive_path = path
    endpoint = API + "/evaluate/archive"
    assert (await client.get(endpoint, headers=headers["reader"])).status_code == 404
    result = {"baseline_id": str(uuid4()), "experiment_id": str(uuid4()),
              "rows": [{"mode": "baseline", "processed": 20, "successful": 19,
                        "rouge_l": 0.18, "verified_rate": None, "source_text": "private corpus text"}]}
    path.write_text(json.dumps(result), encoding="utf-8")
    assert (await client.get(endpoint)).status_code == 401
    reply = await client.get(endpoint, headers=headers["reader"])
    assert reply.status_code == 200 and reply.json()["rows"][0]["rouge_l"] == 0.18
    assert "private corpus text" not in reply.text and reply.json()["rows"][0]["verified_rate"] is None
    result["rows"][0]["successful"] = 21
    path.write_text(json.dumps(result), encoding="utf-8")
    assert (await client.get(endpoint, headers=headers["reader"])).status_code == 422


@pytest.mark.parametrize("draft", [
    "Human Rights Watch says investigators must verify whether January protest victims were buried at the site.",
    "The ministry says the burials happened on March 19.",
    "The ministry says protesters might have been buried at the site.",
])
async def test_attribution_rule_keeps_faithful_reporting(draft):
    from backend.agents.state import AgentState
    from backend.agents.validator import Validator

    state = await Validator(check_source_claims=True).process(
        AgentState(original_text=ATTRIBUTION_SOURCE, current_summary=draft),
    )
    assert all(item.code != "unsupported_attribution" for item in state.factual_findings)


async def test_model_catalog_exposes_local_models_and_no_remote_configuration(api, monkeypatch):
    from backend.api.routes import ui_data
    from backend.data.ollama import OllamaProvider

    client, headers, _, runtime, _, _ = api
    def tags(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [
            {"name": "qwen2.5:1.5b"}, {"name": "version:cloud"}, {"name": "alias", "remote_host": "remote-server"},
            {"name": "broken//model"},
        ]})

    monkeypatch.setattr(ui_data, "_cached_transformers", lambda _: [])
    async with httpx.AsyncClient(transport=httpx.MockTransport(tags)) as http:
        runtime.providers["ollama"] = OllamaProvider(http, "http://ollama")
        assert (await client.get(API + "/models")).status_code == 401
        response = await client.get(API + "/models", headers=headers["reader"])
    assert response.json() == {"ollama": ["qwen2.5:1.5b"], "transformers": []}


async def test_model_catalog_handles_ollama_outage_and_only_offers_cached_weights(api, monkeypatch, tmp_path):
    hub = pytest.importorskip("huggingface_hub")
    from backend.data.ollama import OllamaProvider
    from backend.data.transformers_llm import MODEL, REVISION

    client, headers, _, runtime, _, _ = api
    runtime.settings.hf_model_id = MODEL
    runtime.settings.hf_model_revision = None

    def missing(model, **options):
        assert model == MODEL and options["revision"] == REVISION and options["local_files_only"]
        raise FileNotFoundError("no local snapshot")

    def offline(request):
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(offline)) as http:
        runtime.providers["ollama"] = OllamaProvider(http, "http://ollama")
        route = API + "/models"
        monkeypatch.setattr(hub, "snapshot_download", missing)
        assert (await client.get(route, headers=headers["reader"])).json() == {"ollama": [], "transformers": []}
        monkeypatch.setattr(hub, "snapshot_download", lambda *_args, **_options: str(tmp_path))
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        assert (await client.get(route, headers=headers["reader"])).json()["transformers"] == []
        (tmp_path / "model.safetensors").touch()
        assert (await client.get(route, headers=headers["reader"])).json()["transformers"] == [MODEL]

"""Сетевой тест Uvicorn, Ollama, Qdrant local и метрик на отложенной новости."""

import asyncio
import json
import os
from pathlib import Path
from secrets import token_urlsafe
import socket
from uuid import UUID

from alembic import command
from alembic.config import Config
import httpx
import pytest
import uvicorn

from backend.api.main import create_app
from backend.core.config import Settings
from backend.core.constants import PROJECT_ROOT
from backend.db.models import EvaluationResult
from scripts.corpus import News, load_prepared


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("RUN_LIVE_STAGE12") != "1", reason="Требуются Ollama, MiniLM, WordNet и локальные индексы")
async def test_real_api_over_uvicorn(tmp_path, monkeypatch):
    url = f"sqlite+aiosqlite:///{tmp_path / 'stage12.sqlite3'}"
    settings = Settings(
        _env_file=None, database_url=url, jwt_secret_key=token_urlsafe(64), log_dir=tmp_path / "logs",
        ollama_base_url="http://127.0.0.1:11434", embedding_local_files_only=True,
        summarizer_temperature=0, critic_temperature=0, editor_temperature=0,
    )
    monkeypatch.setenv("DATABASE_URL", url)
    await asyncio.to_thread(command.upgrade, Config(str(PROJECT_ROOT / "alembic.ini")), "head")
    item = next(row for row in load_prepared(settings.holdout_path, News)
                if row.source_document_id == "34227252" and row.split == "test")
    app = create_app(settings)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    probe = []

    async def ollama_request(request):
        data = json.loads(request.content)
        values = json.loads(data["prompt"])
        assert request.url.host == "127.0.0.1" and request.url.path == "/api/generate"
        assert values["news"] == item.source_text and item.target_text not in request.content.decode()
        probe.append({"structured": "format" in data, "style_examples": len(values.get("style_examples", []))})

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        base = f"http://127.0.0.1:{listener.getsockname()[1]}"
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(10):
                while not server.started:
                    if task.done():
                        task.result()
                        pytest.fail("Uvicorn не запустился.")
                    await asyncio.sleep(0.02)
            provider = app.state.summary_runtime.providers["ollama"]
            provider.client.event_hooks["request"].append(ollama_request)
            async with httpx.AsyncClient(base_url=base, timeout=150, trust_env=False) as client:
                docs = await client.get("/docs")
                spec = (await client.get("/openapi.json")).json()
                assert docs.status_code == 200 and "SwaggerUIBundle" in docs.text
                assert spec["components"]["securitySchemes"]["BearerAuth"]["scheme"] == "bearer"
                auth, summary_path = "/api/v1/auth", "/api/v1/summarize"
                login = {"username": "stage12-reader", "password": token_urlsafe(28)}
                registration = await client.post(auth + "/register", json=login | {"email": "fixture@example.net"})
                assert registration.status_code == 201
                token = await client.post(auth + "/login", json=login)
                assert token.status_code == 200
                headers = {"Authorization": "Bearer " + token.json()["access_token"]}
                assert (await client.get(auth + "/me", headers=headers)).status_code == 200
                assert (await client.post(summary_path, json={"text": item.source_text})).status_code == 401
                report = {"test_id": item.source_document_id, "docs": docs.status_code}
                for mode in ("baseline", "full"):
                    response = await client.post(summary_path, headers=headers, json={
                        "text": item.source_text, "provider": "ollama", "model": "qwen2.5:1.5b", "mode": mode,
                    })
                    assert response.status_code == 200, response.text
                    summary = response.json()
                    assert summary["provider"] == "ollama" and summary["model"] == "qwen2.5:1.5b"
                    assert summary["final_summary"]
                    assert summary["status"] == "draft" if mode == "baseline" else summary["status"] in {
                        "verified", "needs_review", "unchanged", "iteration_limit",
                    }
                    request_path = f"/api/v1/requests/{summary['request_id']}"
                    stored = await client.get(request_path, headers=headers)
                    log = await client.get(request_path + "/reasoning_log", headers=headers)
                    assert stored.status_code == log.status_code == 200
                    assert stored.json()["user_id"] == registration.json()["id"]
                    assert stored.json()["final_summary"] == summary["final_summary"]
                    agents = [step["agent"] for step in log.json()["steps"]]
                    if mode == "full":
                        assert agents[:4] == ["RAG", "Summarizer", "Validator", "Critic"] and summary["rag_ids"]
                    else:
                        assert agents == ["Summarizer", "Supervisor"]
                    report[mode] = {
                        "http_status": response.status_code, **{key: val for key, val in summary.items() if key != "text"},
                        "request_status": stored.status_code, "log_status": log.status_code, "agents": agents,
                    }
                evaluated = await client.post("/api/v1/evaluate", headers=headers, json={
                    "test_ids": [item.source_document_id], "model": "qwen2.5:1.5b", "mode": "baseline",
                })
                assert evaluated.status_code == 200, evaluated.text
                evaluation = evaluated.json()
                history = await client.get("/api/v1/evaluate/history", headers=headers)
                assert history.status_code == 200 and history.json() == [evaluation]
                assert evaluation["scored_count"] == 1
                async with app.state.database.sessions() as session:
                    row = await session.get(EvaluationResult, UUID(evaluation["id"]))
                    assert row.metrics_json["metrics"] == evaluation["metrics"]
                assert any(item["structured"] for item in probe) and any(item["style_examples"] for item in probe)
                report["evaluation"] = {"http_status": evaluated.status_code, **evaluation}
                report["history_status"] = history.status_code
                report["ollama_calls"] = len(probe)
                path = os.getenv("STAGE12_REPORT_PATH")
                if path:
                    output = Path(path)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            server.should_exit = True
            await task

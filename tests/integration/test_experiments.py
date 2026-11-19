import asyncio
import csv
import json
from threading import Event
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import select

from backend.api.runtime import SummaryRuntime
from backend.api.schemas.summarization import ModelChoice
from backend.core.config import Settings
from backend.data.llm_types import LLMReply
from backend.db.init_db import create_test_schema
from backend.db.models import EvaluationResult, ExperimentJob, Glossary, ModelComparison, SummarizationRequest
from backend.db.session import Database
from backend.evaluation.experiments import ExperimentBusy, ExperimentRunner, create_job, generate_example, job_report
from backend.evaluation.metrics_calculator import SummaryMetrics
from backend.evaluation.protocol import (
    ExperimentError, ExperimentPlan, Protocol, model_versions, prepare_protocol, select_samples,
)
from backend.retrieval.glossary import glossary_snapshot
from backend.utils.rag_context import StylePattern
from backend.worker.celery_app import COMPARE_TASK, EVALUATE_TASK, celery_client
from scripts.corpus import News, evaluation_ids
from scripts.run_eval import export

pytestmark = pytest.mark.asyncio
MODEL = ModelChoice(model="fixture-model")
SUMMARY = "A local library reopened following repairs."


@pytest.fixture
def corpus(tmp_path):
    rows = [News(
        id=UUID(int=index + 1), source_text=f"The library reopened after repairs. Bulletin {index} is dated Friday.",
        target_text=f"Held-out reference text {index}.", source_document_id=f"fixture-{index}",
        language="en", category="general", source_dataset="test-fixture", split="test",
    ) for index in range(200)]
    path = tmp_path / "news_test.json"
    path.write_text(json.dumps([row.model_dump(mode="json") for row in rows]), encoding="utf-8")
    path.with_name("evaluation_ids.json").write_text(json.dumps(evaluation_ids(rows)), encoding="utf-8")
    return path, rows


async def test_fixed_nested_sizes_and_reproducibility(corpus):
    path, rows = corpus
    samples, guard, digest = select_samples(path, [20, 50, 100])
    assert list(map(len, samples.values())) == [20, 50, 100]
    assert samples[20] == samples[50][:20] == samples[100][:20] == rows[:20]
    assert len(guard.ids) == 200 and len(guard.hashes) == 400
    assert select_samples(path, [50])[2] == digest
    assert select_samples(path, [20], 2)[0] == {20: rows[:2]}
    path.with_name("evaluation_ids.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ExperimentError, match="invalid_evaluation_ids"):
        select_samples(path, [20])


class Provider:
    def __init__(self):
        self.calls = []
        self.revision = "fixture-revision"

    async def model_version(self, _):
        return self.revision

    async def complete(self, prompt, model, options):
        payload = json.loads(prompt.split("Входные данные в JSON:\n", 1)[1])
        assert "Held-out reference text" not in prompt
        assert options.seed == 42
        self.calls.append((model, options.task, options.temperature, payload))
        if options.task == "Critic":
            factual_error = "Atlantis" in payload["summary"]
            value = 10 if factual_error else 8
            return LLMReply(json.dumps({
                **dict.fromkeys(("factual_accuracy", "completeness", "coherence", "readability", "conciseness",
                                 "overall_score"), value),
                "reasoning": "Fixture", "suggestions": [], "critical_factual_error": factual_error,
                "missing_key_fact": False,
            }), model)
        if options.temperature == 0.2 or options.task == "Editor":
            return LLMReply("A local library reopened in Atlantis.", model)
        return LLMReply(SUMMARY, model)


class Metrics:
    def calculate(self, pairs):
        assert all("Held-out reference text" in ref for _, ref in pairs)
        return [SummaryMetrics.model_validate(dict.fromkeys(SummaryMetrics.model_fields, 0.5)) for _ in pairs]

    def metadata(self):
        return {"bertscore_model": "fixture", "libraries": {"bert-score": "fixture"}}


@pytest_asyncio.fixture
async def runtime(tmp_path, corpus):
    settings = Settings(_env_file=None, holdout_path=corpus[0], llm_max_attempts=1,
                        database_url=f"sqlite+aiosqlite:///{tmp_path / 'experiments.sqlite'}")
    db = Database(settings.database_url.get_secret_value(), settings)
    await create_test_schema(db.engine)
    async with db.sessions.begin() as session:
        session.add(Glossary(term="library", definition="A collection of books.", language="en"))
    try:
        async with httpx.AsyncClient() as http:
            runtime = SummaryRuntime(settings, db, http)
            runtime.providers = {"ollama": Provider()}
            runtime.metrics = Metrics()
            yield runtime
    finally:
        await db.close()


@pytest.mark.parametrize(("mode", "expected", "dict_enabled", "rag_enabled"), [
    ("baseline", ["Summarizer"], False, False),
    ("+dict", ["Summarizer"], True, False),
    ("+rag", ["Summarizer"], False, True),
    ("+agents", ["Summarizer", "Critic"], False, False),
    ("full", ["Summarizer", "Critic"], True, True),
    ("+ensemble", ["Summarizer", "Critic", "Summarizer", "Critic", "Editor"], False, False),
])
async def test_pipeline_modes_use_real_agents(runtime, corpus, monkeypatch, mode, expected, dict_enabled, rag_enabled):
    async def enrich(state, **_):
        return state.model_copy(update={"rag_document_ids": [UUID(int=123)],
                                        "rag_context": [StylePattern.SINGLE_BRIEF]})

    monkeypatch.setattr(runtime, "enrich", enrich)
    sample, holdout, _ = select_samples(corpus[0], [20], 2)
    protocol = Protocol({}, sample, holdout, None, await glossary_snapshot(runtime.database.sessions), [MODEL])
    plan = ExperimentPlan(models=[MODEL], modes=[mode], sample_sizes=[20])
    item, pair = await generate_example(runtime, corpus[1][0], MODEL, mode, plan, protocol, None)
    assert pair == (SUMMARY, corpus[1][0].target_text)
    calls = runtime.providers["ollama"].calls
    assert [call[1] for call in calls] == expected
    assert bool(calls[0][3].get("glossary")) is dict_enabled
    assert bool(calls[0][3].get("style_examples")) is rag_enabled
    if mode == "+ensemble":
        assert item["winner_index"] == 0 and item["candidates"][1]["critic_score"] == 10
        assert item["candidates"][1]["status"] != "verified"
    if mode == "+agents":
        assert calls[1][0] == plan.judge.model


async def test_checkpoint_resume_timeout_compare_and_export(runtime, corpus, monkeypatch, tmp_path):
    runner = ExperimentRunner(runtime)
    plan = ExperimentPlan(models=[MODEL, ModelChoice(model="other-model")], modes=["baseline"],
                          sample_sizes=[20], parallelism=2)
    job = await create_job(runtime.database.sessions, plan, diagnostic=3)
    original = runner._checkpoint

    async def interrupt(*args):
        await original(*args)
        raise asyncio.CancelledError

    monkeypatch.setattr(runner, "_checkpoint", interrupt)
    with pytest.raises(asyncio.CancelledError):
        await runner.run(job.id)
    before = len(runtime.providers["ollama"].calls)
    assert before == 2
    monkeypatch.setattr(runner, "_checkpoint", original)
    normal = runtime.run_pipeline

    async def timeout_example(text, **kwargs):
        if kwargs["model"] == "other-model" and "Bulletin 1 " in text:
            raise TimeoutError
        return await normal(text, **kwargs)

    monkeypatch.setattr(runtime, "run_pipeline", timeout_example)
    await runner.run(job.id)
    await runner.run(job.id)
    async with runtime.database.sessions() as session:
        stored = await session.get(ExperimentJob, job.id)
        report = await job_report(session, stored)
        requests = list(await session.scalars(select(SummarizationRequest)))
    assert report["status"] == "completed" and report["progress"] == report["total"] == 6
    assert len(requests) == len(runtime.providers["ollama"].calls) == 5
    first, second = report["evaluations"]
    assert first["aggregate"]["success_count"] == 3 and first["aggregate"]["mean_critic"] is None
    assert second["aggregate"]["success_count"] == 2 and second["aggregate"]["errors"] == {"timeout": 1}
    assert report["comparisons"][0]["winner_evaluation_id"] is None
    json_path, csv_path = export(report, tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["evaluations"][1]["items"][1]["metrics"] is None
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["success_count"] for row in rows] == ["3", "2"]


async def test_expired_lease_and_protocol_mismatch(runtime, monkeypatch):
    plan = ExperimentPlan(models=[MODEL], modes=["baseline"], sample_sizes=[20])
    runner = ExperimentRunner(runtime)
    job = await create_job(runtime.database.sessions, plan, diagnostic=2)
    await runner._claim(job.id, uuid4())
    with pytest.raises(ExperimentBusy):
        await runner.run(job.id)

    async def stop(*args):
        raise ExperimentError("fixture_stop")

    monkeypatch.setattr(runner, "_evaluate", stop)
    async with runtime.database.sessions.begin() as session:
        row = await session.get(ExperimentJob, job.id)
        row.lease_until = row.created_at
    with pytest.raises(ExperimentError, match="fixture_stop"):
        await runner.run(job.id)
    runtime.providers["ollama"].revision = "another-revision"
    with pytest.raises(ExperimentError, match="protocol_changed"):
        await runner.run(job.id)
    async with runtime.database.sessions() as session:
        assert (await session.get(ExperimentJob, job.id)).error == "protocol_changed"


async def test_completed_comparison_does_not_duplicate_and_uses_tiebreaker(runtime, monkeypatch):
    runner = ExperimentRunner(runtime)
    plan = ExperimentPlan(models=[MODEL, ModelChoice(model="other-model")], modes=["baseline"], sample_sizes=[20])
    first = await create_job(runtime.database.sessions, plan, diagnostic=2)
    second = await create_job(runtime.database.sessions, plan, diagnostic=2)
    await asyncio.gather(runner.run(first.id), runner.run(second.id))
    await runner.run(first.id)
    async with runtime.database.sessions() as session:
        rows = list(await session.scalars(select(ModelComparison)))
        evaluations = list(await session.scalars(select(EvaluationResult)))
    assert len(rows) == 2 and len(evaluations) == 4
    assert len(runtime.providers["ollama"].calls) == 8
    for row in rows:
        assert row.results_json["winner_evaluation_id"]
        assert {item["evaluation_id"] for item in row.results_json["runs"]} == {
            str(item.id) for item in evaluations if item.experiment_id == row.experiment_id
        }


async def test_beat_cache_task_removes_only_expired_rows(runtime, monkeypatch):
    from backend.worker.celery_app import PURGE_CACHE_TASK, app
    from backend.worker.tasks import purge_cache_task

    assert app.conf.beat_schedule["purge-expired-cache"]["task"] == PURGE_CACHE_TASK
    monkeypatch.setenv("DATABASE_URL", runtime.settings.database_url.get_secret_value())
    await runtime.cache.store("expired-key", {"final_summary": "Expired"}, -5)
    await runtime.cache.store("active-key", {"final_summary": "Active"}, 60)
    assert await asyncio.to_thread(purge_cache_task.run) == 1
    assert await runtime.cache.take("expired-key") is None
    assert await runtime.cache.take("active-key") == {"final_summary": "Active"}
    assert await asyncio.to_thread(purge_cache_task.run) == 0


async def test_worker_retries_busy_job_and_sanitizes_permanent_error(monkeypatch):
    from unittest.mock import Mock
    from celery.exceptions import Retry
    from backend.worker import tasks

    async def bad_protocol(_settings, _identifier):
        raise ExperimentError("protocol_changed")

    task = Mock()
    monkeypatch.setattr(tasks, "execute", bad_protocol)
    identifier = str(uuid4())
    with pytest.raises(RuntimeError, match="^protocol_changed$"):
        await asyncio.to_thread(tasks._execute, task, identifier)
    task.retry.assert_not_called()

    async def busy(_settings, _identifier):
        raise ExperimentBusy()

    monkeypatch.setattr(tasks, "execute", busy)
    task.retry.return_value = Retry()
    with pytest.raises(Retry):
        await asyncio.to_thread(tasks._execute, task, identifier)
    assert task.retry.call_args.kwargs["countdown"] == 30


async def test_reuse_nested_results_in_order_and_after_interruption(runtime, monkeypatch):
    from backend.evaluation import experiments

    calls = []

    async def generate(_runtime, row, model, mode, *_args):
        calls.append((model.model, mode, row.id))
        failed = row.source_document_id in {"fixture-0", "fixture-50"}
        item = {
            "test_id": str(row.id), "status": "failed" if failed else "draft",
            "error": "timeout" if failed else None, "request_id": str(uuid4()),
            "duration_s": 3 if failed else 2, "llm_attempts": 2 if failed else 1,
        }
        return item, None if failed else (SUMMARY, row.target_text)

    monkeypatch.setattr(experiments, "generate_example", generate)
    runner = ExperimentRunner(runtime)
    plan = ExperimentPlan(models=[MODEL], modes=["baseline"], sample_sizes=[100, 20, 50], parallelism=2)
    job = await create_job(runtime.database.sessions, plan)
    checkpoint = runner._checkpoint

    async def interrupt(job, token, row_id, protocol_hash, model, mode, sample, items, count):
        row = await checkpoint(job, token, row_id, protocol_hash, model, mode, sample, items, count)
        if len(sample) == 50 and len(items) == 2:
            raise asyncio.CancelledError
        return row

    monkeypatch.setattr(runner, "_checkpoint", interrupt)
    with pytest.raises(asyncio.CancelledError):
        await runner.run(job.id)
    assert len(calls) == 20
    monkeypatch.setattr(runner, "_checkpoint", checkpoint)
    await runner.run(job.id)
    assert len(calls) == len(set(calls)) == 100

    async with runtime.database.sessions() as session:
        report = await job_report(session, await session.get(ExperimentJob, job.id))
    assert report["progress"] == report["total"] == 170
    small, middle, large = report["evaluations"]
    assert [row["sample_size"] for row in (small, middle, large)] == [20, 50, 100]
    assert [row["reused_count"] for row in (small, middle, large)] == [0, 20, 50]
    assert [row["generated_count"] for row in (small, middle, large)] == [20, 30, 50]
    assert [row["aggregate"]["error_count"] for row in (small, middle, large)] == [1, 1, 2]
    assert [row["aggregate"]["success_count"] for row in (small, middle, large)] == [19, 49, 98]
    for row in (middle, large):
        assert row["sample_ids"][:20] == small["sample_ids"]
        assert row["items"][0]["request_id"] == small["items"][0]["request_id"]
        assert row["items"][0]["reused_from_evaluation_id"] == small["id"]
        assert row["items"][0]["metrics"] is None and row["items"][0]["error"] == "timeout"
    assert large["items"][20]["reused_from_evaluation_id"] == middle["id"]
    assert large["aggregate"]["mean_llm_attempts"] == pytest.approx(1.02)
    assert large["aggregate"]["mean_seconds"] == pytest.approx(2.02)
    assert large["aggregate"]["metrics"]["rouge_l"] == 0.5
    assert large["aggregate"]["factual_notice_count"] is None


async def test_rag_and_glossary_protocol_pins_index_and_excludes_reference(runtime, tmp_path):
    pytest.importorskip("qdrant_client")
    runtime.settings.rag_index_dir = tmp_path
    manifest = {
        "version": "a" * 32, "corpus_sha256": "b" * 64, "holdout_sha256": "c" * 64,
        "document_count": 1, "holdout_count": 200, "model_name": "fixture", "model_revision": "d" * 40,
        "dimension": 384, "mode": "local", "collection": "test",
    }
    active = tmp_path / "active.json"
    active.write_text(json.dumps(manifest), encoding="utf-8")
    async with runtime.database.sessions.begin() as session:
        session.add(Glossary(term="Friday", language="en", definition="Text: Held-out reference text 0."))
    plan = ExperimentPlan(models=[MODEL], modes=["+dict", "+rag"], sample_sizes=[20])
    protocol = await prepare_protocol(runtime, plan, 2)
    assert protocol.metadata["index"]["model_revision"] == "d" * 40
    assert protocol.metadata["glossary_count"] == 1
    assert [term.term for term in protocol.terms] == ["library"]
    active.write_text(json.dumps(manifest | {"model_revision": "main"}), encoding="utf-8")
    with pytest.raises(ExperimentError, match="unpinned_embedding_revision"):
        await prepare_protocol(runtime, plan, 2)


@pytest.mark.parametrize("status", [200, 503])
async def test_ollama_version_validation(runtime, status):
    from backend.data.ollama import OllamaProvider

    def tags(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(status, json={"models": [{"name": "fixture-model:latest", "digest": "digest"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(tags)) as http:
        runtime.providers = {"ollama": OllamaProvider(http, "http://127.0.0.1:11434")}
        if status == 200:
            assert await model_versions(runtime, [MODEL]) == {"ollama:fixture-model": "digest"}
            with pytest.raises(ExperimentError, match="model_version_unavailable"):
                await model_versions(runtime, [ModelChoice(model="missing")])
        else:
            with pytest.raises(ExperimentError, match="model_version_unavailable"):
                await model_versions(runtime, [MODEL])


async def test_model_change_prevents_checkpointing_mixed_generation(runtime, monkeypatch):
    provider = runtime.providers["ollama"]
    complete = provider.complete

    async def complete_then_change_version(*args):
        reply = await complete(*args)
        provider.revision = "updated-after-first-generation"
        return reply

    monkeypatch.setattr(provider, "complete", complete_then_change_version)
    plan = ExperimentPlan(models=[MODEL], modes=["baseline"], sample_sizes=[20])
    job = await create_job(runtime.database.sessions, plan, diagnostic=2)
    with pytest.raises(ExperimentError, match="model_changed"):
        await ExperimentRunner(runtime).run(job.id)
    async with runtime.database.sessions() as session:
        stored = await session.get(ExperimentJob, job.id)
        assert stored.status == "failed" and stored.progress == 0
        assert (await job_report(session, stored))["evaluations"] == []


async def test_transformers_protocol_pins_locally_resolved_revision(runtime, monkeypatch, tmp_path):
    hub = pytest.importorskip("huggingface_hub")
    from backend.data.transformers_llm import MODEL as HF_MODEL, REVISION, TransformersProvider

    def snapshot(model, *, revision, cache_dir, local_files_only):
        assert model == HF_MODEL and revision == REVISION and local_files_only
        return tmp_path / "snapshots" / REVISION

    monkeypatch.setattr(hub, "snapshot_download", snapshot)
    adapter = TransformersProvider(runtime.settings)
    runtime.providers = {"transformers": adapter}
    assert await model_versions(runtime, [ModelChoice(model=HF_MODEL, provider="transformers")]) == {
        f"transformers:{HF_MODEL}": REVISION,
    }
    assert adapter.pinned_revisions == {HF_MODEL: REVISION}


async def test_worker_executes_job_with_owned_database_resources(runtime, monkeypatch):
    from backend.worker import tasks

    def create_runtime(settings, database, http):
        assert settings.embedding_local_files_only and database is not runtime.database
        result = SummaryRuntime(settings, database, http)
        result.providers = runtime.providers
        result.metrics = runtime.metrics
        return result

    monkeypatch.setattr(tasks, "SummaryRuntime", create_runtime)
    plan = ExperimentPlan(models=[MODEL], modes=["baseline"], sample_sizes=[20])
    job = await create_job(runtime.database.sessions, plan, diagnostic=2)
    await tasks.execute(runtime.settings, job.id)
    async with runtime.database.sessions() as session:
        report = await job_report(session, await session.get(ExperimentJob, job.id))
    assert report["status"] == "completed" and report["progress"] == 2


async def test_celery_worker_consumes_json_messages_from_memory_broker(monkeypatch):
    from celery.contrib.testing.worker import start_worker
    from backend.worker import tasks

    received = []
    completed = Event()

    async def execute(_settings, identifier):
        received.append(identifier)
        if len(received) == 2:
            completed.set()

    monkeypatch.setattr(tasks, "execute", execute)
    identifier = uuid4()

    def transfer():
        with celery_client(Settings(_env_file=None, celery_broker_url="memory://")) as app:
            with start_worker(app, perform_ping_check=False, pool="solo"):
                for name in (EVALUATE_TASK, COMPARE_TASK):
                    app.send_task(name, args=[str(identifier)])
                assert completed.wait(5)

    await asyncio.to_thread(transfer)
    assert received == [identifier, identifier]

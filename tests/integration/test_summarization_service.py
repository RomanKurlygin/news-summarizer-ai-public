import asyncio
import hashlib
import json
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select

from backend.core.config import Settings
from backend.data.llm_types import LLMError, LLMReply
from backend.db.models import Base, SummarizationRequest
from backend.db.session import Database
from backend.services.summarization import SummarizationService
from backend.utils.rag_context import StylePattern

pytestmark = pytest.mark.asyncio
NEWS = "15 марта 2026 года в Казани открыли два читальных зала. Вход бесплатный."
CORRECT = "В Казани 15 марта 2026 года заработали два бесплатных читальных зала."
WRONG = "В Казани 20 марта 2026 года заработали два бесплатных читальных зала."
RAG_ID = UUID(int=42)


def verdict(score, critical=False):
    return json.dumps({
        "factual_accuracy": score, "completeness": score, "coherence": score, "readability": score,
        "conciseness": score, "overall_score": score, "reasoning": "Сверены даты и числа.",
        "suggestions": ["Исправить дату."] if critical else [],
        "critical_factual_error": critical, "missing_key_fact": False,
    }, ensure_ascii=False)


class Scripted:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    async def complete(self, prompt, model, options):
        self.calls.append((prompt, model, options))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return LLMReply(response, "resolved-test-model", 23, 11)


async def enrich(state):
    return state.model_copy(update={
        "rag_document_ids": [RAG_ID],
        "rag_context": [StylePattern.SINGLE_BRIEF],
    })


@pytest_asyncio.fixture
async def database(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'llm-service.sqlite3'}"
    db = Database(url, Settings(_env_file=None))
    async with db.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield db
    finally:
        await db.close()


async def request_row(database, request_id):
    async with database.sessions() as session:
        return await session.scalar(select(SummarizationRequest).where(SummarizationRequest.id == request_id))


async def test_full_run_revises_and_saves_both_audits(database):
    adapter = Scripted(WRONG, verdict(6, critical=True), CORRECT, verdict(9))
    config = Settings(_env_file=None, summarizer_temperature=0.21, critic_temperature=0, editor_temperature=0.12)
    service = SummarizationService(config, database.sessions, {"ollama": adapter}, rag=enrich)
    report = await service.run(NEWS, provider="ollama", model="explicit-test-model")
    request_id = UUID(report["request_id"])
    assert report["status"] == "verified" and report["critic_score"] == 9
    assert report["initial_summary"] == WRONG and report["final_summary"] == CORRECT
    assert report["revisions"] == [{"iteration": 1, "summary": CORRECT}]
    assert report["iterations"] == 1 and report["rag_ids"] == [str(RAG_ID)]
    assert report["models"] == [{"provider": "ollama", "model": "resolved-test-model"}]
    assert all(call[1] == "explicit-test-model" for call in adapter.calls)
    assert [call[2].temperature for call in adapter.calls] == [0.21, 0, 0.12, 0]
    assert [call[2].schema is not None for call in adapter.calls] == [False, True, False, True]
    assert "style_examples" in adapter.calls[0][0]
    row = await request_row(database, request_id)
    assert row.status == "verified" and row.summary == CORRECT and row.input_text == NEWS
    assert row.parameters["revisions"] == report["revisions"]

    actions = await service.reasoning.list_for_request(request_id)
    assert [step.agent_name for step in actions] == [
        "RAG", "Summarizer", "Validator", "Critic", "Editor", "Validator", "Critic", "Supervisor",
    ]
    calls = await service.calls.list_for_request(request_id)
    assert [call.agent_name for call in calls] == ["Summarizer", "Critic", "Editor", "Critic"]
    for index, (row, step, transport) in enumerate(zip(
        calls, [step for step in actions if step.agent_name in {"Summarizer", "Critic", "Editor"}],
        adapter.calls, strict=True,
    ), 1):
        assert row.request_id == step.request_id == request_id
        assert row.model_name == step.model_name == "resolved-test-model"
        assert row.parameters["sequence"] == index
        assert row.status == "ok" and row.tokens_used == 34 and row.latency_ms >= 0
        assert row.prompt_hash == hashlib.sha256(transport[0].encode()).hexdigest()
        assert row.response_hash and len(row.response_hash) == 64
        assert NEWS not in str(row.parameters) + step.input_summary + step.output_summary


async def test_baseline_skips_rag_and_validators(database):
    async def unexpected(_):
        pytest.fail("baseline ran the retriever")

    adapter = Scripted(CORRECT)
    service = SummarizationService(Settings(_env_file=None), database.sessions, {"ollama": adapter}, rag=unexpected)
    report = await service.run(NEWS, mode="baseline")
    assert report["status"] == "draft" and report["final_summary"] == CORRECT
    assert report["critic_score"] is None and report["validation_passed"] is None
    assert report["rag_ids"] == [] and report["llm_attempts"] == 1
    data = json.loads(adapter.calls[0][0].split("Входные данные в JSON:\n", 1)[1])
    assert data == {"news": NEWS}


@pytest.mark.parametrize("bad_output", ['not JSON', '{"factual_accuracy":100}'])
async def test_full_preserves_failed_critic_response_hash_and_never_verifies(database, bad_output):
    service = SummarizationService(Settings(_env_file=None), database.sessions, {
        "ollama": Scripted(CORRECT, bad_output),
    }, rag=enrich)
    report = await service.run(NEWS)
    assert report["status"] == "failed" and report["final_summary"] == CORRECT
    assert report["critic_score"] == 5 and report["critic_fallback_used"] is True
    assert report["iterations"] == 0
    call = (await service.calls.list_for_request(UUID(report["request_id"])))[1]
    assert call.status == "failed" and call.parameters["error_type"] == "invalid_response"
    assert call.response_hash == hashlib.sha256(bad_output.encode()).hexdigest()


async def test_baseline_rejects_a_json_envelope(database):
    service = SummarizationService(Settings(_env_file=None), database.sessions,
                                   {"ollama": Scripted('{"summary":"test prose"}')})
    report = await service.run(NEWS, mode="baseline")
    assert report["status"] == "failed" and report["final_summary"] is None
    assert (await service.calls.list_for_request(UUID(report["request_id"])))[0].parameters["error_type"] == "invalid_response"


async def test_full_rejects_new_names_even_if_critic_overrates_draft(database):
    news = "Johanna Konta won her match and will play Saisai Zheng on Friday."
    invented = "Roger Federer defeated Tommy Robredo at the Australian Open."
    service = SummarizationService(Settings(_env_file=None), database.sessions, {
        "ollama": Scripted(invented, verdict(8), invented),
    }, rag=enrich)
    report = await service.run(news)
    assert report["status"] == "unchanged" and report["iterations"] == 1
    assert report["critic_passed"] is True and report["validation_passed"] is False
    actions = await service.reasoning.list_for_request(UUID(report["request_id"]))
    assert "unsupported_name" in next(step.errors for step in actions if step.agent_name == "Validator")


async def test_full_allows_supported_glossary_alias(database):
    async def with_glossary(state):
        state = await enrich(state)
        return state.model_copy(update={"glossary_context": ["Мария Волкова | Maria Volkova"]})

    news = "Мария Волкова рассказала об открытии музея."
    result = "Maria Volkova reported the museum opening."
    service = SummarizationService(Settings(_env_file=None), database.sessions, {
        "ollama": Scripted(result, verdict(8)),
    }, rag=with_glossary)
    report = await service.run(news)
    assert report["validation_passed"] is True and report["status"] == "verified"


async def test_full_requires_review_for_unconfirmed_given_name_despite_high_score(database):
    news = "Konta hit 11 aces and beat the American wildcard."
    draft = "Johanna Konta beat the wildcard with 11 aces."
    adapter = Scripted(draft, verdict(9.8), draft)
    service = SummarizationService(Settings(_env_file=None), database.sessions, {"ollama": adapter}, rag=enrich)

    report = await service.run(news)

    assert report["status"] == "needs_review" and report["iterations"] == 1
    assert report["critic_passed"] and report["validation_passed"] and report["factual_review_required"]
    assert report["factual_findings"] == [{
        "code": "unconfirmed_entity", "severity": "review", "category": "person", "subject": "Johanna Konta",
    }]
    row = await request_row(database, UUID(report["request_id"]))
    assert row.status == "needs_review" and row.parameters["factual_review_required"]
    assert row.parameters["factual_findings"] == report["factual_findings"]
    steps = await service.reasoning.list_for_request(row.id)
    validator = next(step for step in steps if step.agent_name == "Validator")
    assert "unconfirmed_entity" in validator.warnings
    assert all("Johanna" not in content for content in [*validator.warnings, validator.output_summary])


async def test_full_rechecks_corrected_count_despite_high_judge_scores(database):
    news = "Teams found three parcels in a warehouse after a search."
    mistaken = "A search turned up 9 parcels."
    corrected = "A search turned up 3 parcels."
    service = SummarizationService(Settings(_env_file=None), database.sessions, {
        "ollama": Scripted(mistaken, verdict(10), corrected, verdict(10)),
    }, rag=enrich)

    report = await service.run(news)

    assert report["status"] == "verified" and report["iterations"] == 1
    assert report["final_summary"] == corrected and not report["factual_findings"]
    steps = await service.reasoning.list_for_request(UUID(report["request_id"]))
    assert [step.agent_name for step in steps] == [
        "RAG", "Summarizer", "Validator", "Critic", "Editor", "Validator", "Critic", "Supervisor",
    ]
    assert "unsupported_number" in steps[2].errors


async def test_service_failure_is_audited_without_prompt_or_exception_text(database):
    service = SummarizationService(Settings(_env_file=None, llm_retry_delay_seconds=0), database.sessions, {
        "ollama": Scripted(LLMError("provider_unavailable", retryable=True), RuntimeError("test-only-private-detail")),
    })
    report = await service.run(NEWS, mode="baseline")
    assert report["status"] == "failed" and report["final_summary"] is None
    calls = await service.calls.list_for_request(UUID(report["request_id"]))
    assert [row.parameters["error_type"] for row in calls] == ["provider_unavailable", "provider_failure"]
    assert all(row.status == "failed" and row.response_hash is None for row in calls)
    assert all("test-only-private-detail" not in str(row.parameters) for row in calls)


@pytest.mark.parametrize("configuration", ["missing", "failed", "wrong_request"])
async def test_rag_failure_prevents_llm_requests(database, configuration):
    async def broken(state):
        if configuration == "wrong_request":
            return state.model_copy(update={"request_id": uuid4()})
        raise RuntimeError("temporary test issue")

    service = SummarizationService(Settings(_env_file=None), database.sessions, {},
                                   rag=None if configuration == "missing" else broken)
    report = await service.run(NEWS)
    assert report["status"] in {"failed", "incomplete_configuration"}
    assert report["error_codes"] == ["rag_failed"] and report["llm_attempts"] == 0


async def test_cancellation_finalizes_request_and_audits_attempt(database):
    entered = asyncio.Event()

    class Blocking:
        async def complete(self, *args):
            entered.set()
            await asyncio.Future()

    service = SummarizationService(Settings(_env_file=None), database.sessions, {"ollama": Blocking()})
    rid = uuid4()
    task = asyncio.create_task(service.run(NEWS, mode="baseline", request_id=rid))
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    saved = await request_row(database, rid)
    assert saved.status == "cancelled"
    assert [(event.status, event.parameters["error_type"]) for event in await service.calls.list_for_request(rid)] == [
        ("cancelled", "cancelled"),
    ]
    assert (await service.reasoning.list_for_request(rid))[-1].status == "cancelled"


async def test_concurrent_requests_use_separate_audit_and_summaries(database):
    both = asyncio.Event()
    arrived = 0

    class Concurrent:
        async def complete(self, prompt, model, options):
            nonlocal arrived
            arrived += 1
            if arrived == 2:
                both.set()
            await asyncio.wait_for(both.wait(), timeout=3)
            return LLMReply("First result." if "First input" in prompt else "Second result.", model)

    service = SummarizationService(Settings(_env_file=None), database.sessions, {"ollama": Concurrent()})
    reports = await asyncio.gather(
        service.run("First input", mode="baseline", model="first:test"),
        service.run("Second input", mode="baseline", model="second:test"),
    )
    assert [report["final_summary"] for report in reports] == ["First result.", "Second result."]
    for index, report in enumerate(reports):
        rid = UUID(report["request_id"])
        (call,) = await service.calls.list_for_request(rid)
        assert call.model_name == ["first:test", "second:test"][index]
        assert call.request_id == rid and report["llm_attempts"] == 1

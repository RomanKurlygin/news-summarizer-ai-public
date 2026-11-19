import json
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from backend.agents.state import AgentState, ErrorCode
from backend.agents.supervisor import Supervisor
from backend.core.config import Settings
from backend.db.init_db import create_test_schema
from backend.db.models import ReasoningLog, SummarizationRequest
from backend.db.reasoning import ReasoningLogStore
from backend.db.session import Database

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def database(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'reasoning.sqlite3'}"
    database = Database(url, Settings(_env_file=None))
    try:
        await create_test_schema(database.engine)
        yield database
    finally:
        await database.close()


async def test_fake_summarizer_demo_and_reasoning_log(database, fake_llm):
    state = AgentState(
        request_id=UUID("00000000-0000-4000-8000-000000000005"),
        original_text="21 апреля 2026 года в Казани открыли три библиотеки. "
                      "В городе продолжается обновление библиотечной сети.",
    )
    summary = "21 апреля 2026 года в Казани открыли три библиотеки."
    store = ReasoningLogStore(database.sessions)
    supervisor = Supervisor(lambda: fake_llm(summary), store.save_run, validator_factory=None)

    result = await supervisor.run(state)
    records = await store.list_for_request(result.request_id)
    async with database.sessions() as session:
        saved = await session.get(SummarizationRequest, result.request_id)

    assert result.current_summary == saved.summary == summary
    assert result.execution_status == saved.status == "incomplete_configuration"
    assert [row.agent_name for row in records] == ["Summarizer", "Supervisor"]
    step = records[0]
    assert step.request_id == saved.id
    assert step.started_at <= step.finished_at
    assert step.status == "ok" and step.errors == []
    assert step.parameters["duration_ms"] >= 0
    assert json.loads(step.output_summary)["summary_chars"] == len(summary)
    assert await store.list_for_request(uuid4()) == []

    fields = {"request_id", "current_summary", "validation_passed",
              "critic_score", "iteration", "execution_status"}
    print("FakeLLMClient, test scenario")
    print("AgentState before:", state.model_dump_json(include=fields))
    print("AgentState after:", result.model_dump_json(include=fields))
    print("ReasoningLog:", json.dumps({
        "request_id": str(step.request_id), "agent_name": step.agent_name,
        "status": step.status, "started_at": step.started_at.isoformat(),
        "finished_at": step.finished_at.isoformat(),
        "input_summary": json.loads(step.input_summary),
        "output_summary": json.loads(step.output_summary),
        "parameters": step.parameters, "errors": step.errors,
    }, ensure_ascii=False))


@pytest.mark.parametrize("fails", [False, True])
async def test_reasoning_log_excludes_secrets_from_input_output_and_errors(database, fake_llm, fails):
    secret = "synthetic-example-secret"
    store = ReasoningLogStore(database.sessions)
    client = fake_llm(secret, error=TimeoutError(secret) if fails else None)
    result = await Supervisor(lambda: client, store.save_run).run(
        AgentState(original_text=secret, rag_context=[secret], glossary_context=[secret])
    )

    records = await store.list_for_request(result.request_id)
    for row in records:
        fields = [
            row.input_summary, row.output_summary, row.parameters, row.errors, row.warnings,
        ]
        assert secret not in json.dumps(fields)
    if fails:
        assert result.execution_status == "failed"
        assert records[0].errors == [ErrorCode.LLM_FAILED]
    else:
        assert result.current_summary == secret


async def test_invalid_owner_rolls_back_request_and_reasoning_steps(database, fake_llm):
    store = ReasoningLogStore(database.sessions)
    supervisor = Supervisor(lambda: fake_llm("Черновик"), store.save_run)

    with pytest.raises(IntegrityError):
        await supervisor.run(AgentState(original_text="Новость"), user_id=uuid4())

    async with database.sessions() as session:
        assert await session.scalar(select(func.count()).select_from(SummarizationRequest)) == 0
        assert await session.scalar(select(func.count()).select_from(ReasoningLog)) == 0

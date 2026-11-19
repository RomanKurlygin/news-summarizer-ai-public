from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import func, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from backend.core.config import Settings
from backend.db.init_db import create_test_schema
from backend.db.models import (
    CacheEntry,
    EvaluationResult,
    Feedback,
    Glossary,
    KnowledgeBase,
    LLMCall,
    ModelComparison,
    ReasoningLog,
    SummarizationRequest,
    UncertainExample,
    User,
)
from backend.db.session import Database

pytestmark = pytest.mark.asyncio

TABLES = {
    "users", "knowledge_base", "glossary", "summarization_requests",
    "uncertain_examples", "cache", "reasoning_logs", "llm_calls",
    "evaluation_results", "model_comparisons", "experiment_jobs", "feedback",
}

USER = dict(
    email="reader@example.test", username="reader", hashed_password="fixture-only-hash"
)


@pytest_asyncio.fixture
async def database(tmp_path):
    settings = Settings(
        _env_file=None, test_database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.sqlite3'}"
    )
    database = Database(settings.test_database_url.get_secret_value(), settings)
    try:
        await create_test_schema(database.engine)
        yield database
    finally:
        await database.close()


async def test_create_all_tables(database):
    async with database.engine.connect() as connection:
        actual = await connection.run_sync(lambda sync: set(inspect(sync).get_table_names()))

    assert actual == TABLES


@pytest.mark.parametrize(
    ("model", "fields"),
    [
        (User, USER),
        (KnowledgeBase, dict(source_text="Новость", target_text="Резюме", language="ru")),
        (Glossary, dict(term="RAG", definition="Генерация с поиском")),
        (CacheEntry, dict(input_key="fixture-key", output={"text": "Резюме"})),
        (
            EvaluationResult,
            dict(
                model_name="fixture-model", mode="baseline", sample_size=20,
                metrics_json={"rouge": {"f1": 0.4}}, evaluation_time=1.5,
            ),
        ),
        (
            ModelComparison,
            dict(
                models_compared=["fixture-a", "fixture-b"], sample_size=20,
                results_json={"fixture-a": {"rouge": 0.4}}, winner="fixture-a",
            ),
        ),
    ],
    ids=lambda value: value.__name__ if isinstance(value, type) else None,
)
async def test_persist_records(database, model, fields):
    record = model(**fields)
    async with database.sessions.begin() as session:
        session.add(record)

    assert isinstance(record.id, UUID)
    async with database.sessions() as session:
        stored = await session.get(model, record.id)

        assert stored is not None
        assert stored.created_at is not None
        assert stored.updated_at is not None
        assert {field: getattr(stored, field) for field in fields} == fields


async def test_request_persists_audit_and_feedback(database):
    user = User(**USER, role="editor")
    request = SummarizationRequest(
        input_text="Новость",
        reasoning_logs=[
            ReasoningLog(
                agent_name="Critic", status="ok", parameters={"temperature": 0},
                warnings=["review"], errors=[],
            )
        ],
        llm_calls=[
            LLMCall(
                agent_name="Critic", status="ok", model_name="fixture-model",
                provider="fixture", prompt_hash="a" * 64, response_hash="b" * 64,
                latency_ms=12, tokens_used=5, parameters={"temperature": 0},
            )
        ],
    )
    async with database.sessions.begin() as session:
        session.add(user)
        await session.flush()
        request.user_id = user.id
        session.add(request)
        await session.flush()
        item = UncertainExample(
            request_id=request.id, input_text=request.input_text, model_output="Черновик",
            critic_score=6.5, reasons=["Уточнить факт"],
        )
        session.add(item)
        await session.flush()
        feedback = Feedback(
            uncertain_example_id=item.id, editor_id=user.id,
            decision="approved", corrected_summary="Исправленное резюме",
        )
        session.add(feedback)

    async with database.sessions() as session:
        stored = await session.scalar(
            select(SummarizationRequest)
            .where(SummarizationRequest.id == request.id)
            .options(
                selectinload(SummarizationRequest.reasoning_logs),
                selectinload(SummarizationRequest.llm_calls),
            )
        )
        log, = stored.reasoning_logs
        call, = stored.llm_calls
        assert log.request_id == call.request_id == stored.id
        assert log.warnings == ["review"]
        assert call.latency_ms == 12
        assert call.parameters == {"temperature": 0}

        review = await session.get(Feedback, feedback.id)
        assert review.editor_id == user.id
        assert review.uncertain_example_id == item.id
        assert review.corrected_summary == "Исправленное резюме"


@pytest.mark.parametrize(
    ("model", "fields"),
    [
        (SummarizationRequest, dict(user_id=uuid4(), input_text="Новость")),
        (ReasoningLog, dict(request_id=uuid4(), agent_name="Validator", status="ok")),
        (
            LLMCall,
            dict(
                request_id=uuid4(), agent_name="Critic", model_name="fixture-model",
                provider="fixture", prompt_hash="a" * 64, status="ok",
            ),
        ),
        (Feedback, dict(uncertain_example_id=uuid4(), decision="approved")),
    ],
    ids=["owner", "reasoning_log", "llm_call", "feedback"],
)
async def test_foreign_keys_reject_orphans(database, model, fields):
    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
        async with database.sessions.begin() as session:
            session.add(model(**fields))


@pytest.mark.parametrize(
    ("model", "first", "second"),
    [
        (User, USER, USER | {"username": "other"}),
        (User, USER, USER | {"email": "other@example.test"}),
        (Glossary, dict(term="RAG", definition="First"), dict(term="RAG", definition="Second")),
        (
            CacheEntry,
            dict(input_key="same", output={"value": 1}),
            dict(input_key="same", output={"value": 2}),
        ),
    ],
    ids=["email", "username", "glossary", "cache"],
)
async def test_unique_constraints(database, model, first, second):
    async with database.sessions.begin() as session:
        session.add(model(**first))

    with pytest.raises(IntegrityError, match="UNIQUE"):
        async with database.sessions.begin() as session:
            session.add(model(**second))


async def test_transaction_rollback_keeps_committed_rows_and_session_usable(database):
    async with database.sessions() as session:
        async with session.begin():
            session.add(User(**USER))

        with pytest.raises(IntegrityError):
            async with session.begin():
                session.add(User(**(USER | {"email": "rollback@example.test", "username": "rollback"})))
                await session.flush()
                session.add(User(**USER))
                await session.flush()

        assert await session.scalar(select(func.count()).select_from(User)) == 1

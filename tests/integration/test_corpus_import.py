import asyncio

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import event, func, select, text

from backend.core.config import Settings
from backend.core.constants import PROJECT_ROOT
from backend.db.models import Base, Glossary, KnowledgeBase
from backend.db.session import Database
from scripts.build_glossary import extract_terms
from scripts.corpus import DataError, content_hash, prepare_corpus
from scripts.corpus_db import import_corpus
from scripts.verify_dataset import VerifiedCorpus, verify_import

pytestmark = pytest.mark.asyncio


@pytest.fixture
def corpus():
    train, test, _ = prepare_corpus([
        {"document": "The Regional Health Board (RHB) announced a public health meeting.",
         "summary": "Residents will meet about health programs.", "id": "t1", "split": "train"},
        {"document": "The Climate Council discussed carbon emissions with researchers.",
         "summary": "Delegates debated emissions limits.", "id": "t2", "split": "train"},
        {"document": "The last film of the season premiered at a festival.",
         "summary": "A festival screened the final movie.", "id": "e1", "split": "test"},
    ], train_size=2, test_size=1, seed=42, dataset="integration-fixture", language="en")
    return VerifiedCorpus(train, test, extract_terms(train, 4), {})


@pytest_asyncio.fixture
async def database(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'import.sqlite3'}"
    db = Database(url, Settings(_env_file=None))
    async with db.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield db
    finally:
        await db.close()


async def counts(database):
    async with database.sessions() as session:
        return tuple([await session.scalar(select(func.count()).select_from(model))
                      for model in (KnowledgeBase, Glossary)])


async def test_import_persists_only_train_and_glossary_provenance_and_is_repeatable(database, corpus):
    first = await import_corpus(database.sessions, corpus.train, corpus.terms, corpus.test)
    assert (first["inserted_news"], first["inserted_glossary"]) == (2, 4)
    second = await import_corpus(database.sessions, corpus.train, corpus.terms, corpus.test)
    assert (second["inserted_news"], second["inserted_glossary"]) == (0, 0)
    assert (second["skipped_news"], second["skipped_glossary"]) == (2, 4)
    assert await counts(database) == (2, 4)

    async with database.sessions() as session:
        entries = (await session.scalars(select(KnowledgeBase))).all()
        assert {row.id for row in entries} == {row.id for row in corpus.train}
        assert {row.content_hash for row in entries} == {content_hash(row.source_text) for row in corpus.train}
        assert {row.source_document_id for row in entries} == {"t1", "t2"}
        assert {row.source_dataset for row in entries} == {"integration-fixture"}
        assert {row.source_split for row in entries} == {"train"}
        assert {row.target_text for row in entries} == {row.target_text for row in corpus.train}
        record = await session.scalar(select(Glossary).where(Glossary.term == "Regional Health Board"))
        assert record.aliases == ["RHB"]
        assert record.source["news_id"] in {str(row.id) for row in corpus.train}
        assert record.source["method"] == "expansion"


async def test_dry_verification_detects_missing_and_does_not_write(database, corpus):
    url = database.engine.url.render_as_string()
    with pytest.raises(DataError, match="Часть корпуса отсутствует"):
        await verify_import(url, corpus, Settings(_env_file=None))
    assert await counts(database) == (0, 0)
    await import_corpus(database.sessions, corpus.train, corpus.terms, corpus.test)
    report = await verify_import(url, corpus, Settings(_env_file=None))
    assert report["repeat_would_insert"] == report["database_test_overlaps"] == 0


async def test_existing_news_that_matches_test_blocks_import(database, corpus):
    test = corpus.test[0]
    async with database.sessions.begin() as session:
        session.add(KnowledgeBase(source_text=test.target_text.upper(), target_text="Legacy record."))
    with pytest.raises(DataError, match="существующей базе знаний"):
        await import_corpus(database.sessions, corpus.train, corpus.terms, corpus.test)
    assert await counts(database) == (1, 0)


async def test_test_input_and_test_glossary_provenance_are_rejected(database, corpus):
    with pytest.raises(DataError, match="отдельные"):
        await import_corpus(database.sessions, corpus.test, corpus.terms, corpus.test)
    term = corpus.terms[0]
    corrupted = term.model_copy(update={"source": term.source.model_copy(update={"news_id": corpus.test[0].id})})
    with pytest.raises(DataError, match="внешней ссылки"):
        await import_corpus(database.sessions, corpus.train, [corrupted], corpus.test)
    assert await counts(database) == (0, 0)


async def test_legacy_data_is_deduplicated_without_overwriting(database, corpus):
    news = corpus.train[0]
    term = corpus.terms[0]
    async with database.sessions.begin() as session:
        session.add(KnowledgeBase(source_text=news.source_text.upper(), target_text=news.target_text))
        session.add(Glossary(term=term.term.upper(), language=term.language, definition="Hand-edited definition."))
    result = await import_corpus(database.sessions, corpus.train, corpus.terms, corpus.test)
    assert result["skipped_news"] == result["skipped_glossary"] == 1
    assert await counts(database) == (2, 4)
    async with database.sessions() as session:
        legacy = await session.scalar(select(Glossary).where(Glossary.term == term.term.upper()))
        assert legacy.definition == "Hand-edited definition."


async def test_transaction_rolls_back_news_and_glossary_together(database, corpus):
    writes = []

    @event.listens_for(database.engine.sync_engine, "before_cursor_execute")
    def fail_after_glossary(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO "):
            writes.append(statement.split()[2])
        if statement.startswith("INSERT INTO knowledge_base"):
            raise RuntimeError("simulated write failure")

    with pytest.raises(RuntimeError, match="simulated write failure"):
        await import_corpus(database.sessions, corpus.train, corpus.terms, corpus.test)
    assert writes[-1] == "knowledge_base" and set(writes[:-1]) == {"glossary"}
    assert await counts(database) == (0, 0)


async def test_migration_keeps_rows_created_under_0001(tmp_path, monkeypatch):
    url = f"sqlite+aiosqlite:///{tmp_path / 'legacy.sqlite3'}"
    monkeypatch.setenv("DATABASE_URL", url)
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    await asyncio.to_thread(command.upgrade, config, "0001")
    database = Database(url, Settings(_env_file=None))
    try:
        async with database.engine.begin() as connection:
            await connection.execute(text("INSERT INTO knowledge_base (id, source_text, target_text) VALUES "
                                           "('12345678123456781234567812345678', 'Legacy news', 'Legacy reference')"))
            await connection.execute(text("INSERT INTO glossary (id, term, definition) VALUES "
                                           "('87654321876543218765432187654321', 'Legacy term', 'Legacy definition')"))
        await asyncio.to_thread(command.upgrade, config, "head")
        async with database.sessions() as session:
            legacy_news = await session.scalar(select(KnowledgeBase))
            legacy_term = await session.scalar(select(Glossary))
            assert legacy_news.source_text == "Legacy news" and legacy_news.source_dataset is None
            assert legacy_term.definition == "Legacy definition" and legacy_term.source is None
            assert legacy_term.review_required is False
    finally:
        await database.close()

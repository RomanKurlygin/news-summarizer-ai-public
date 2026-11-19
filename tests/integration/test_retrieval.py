import json
from dataclasses import replace
from uuid import UUID
from unittest.mock import AsyncMock

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("rank_bm25")
pytest.importorskip("qdrant_client")

from qdrant_client import AsyncQdrantClient

from backend.agents.state import AgentState
from backend.agents.summarizer import Summarizer
from backend.core.config import Settings
from backend.retrieval.bm25_index import BM25Index, tokenize
from backend.retrieval.documents import Document, Hit, Holdout, IndexIntegrityError
from backend.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from backend.retrieval.index_bundle import build_indexes, open_indexes
from backend.retrieval.vector_index import VectorIndex, VectorIndexError
from backend.utils.corpus_text import content_hash
from backend.utils.rag_context import StylePattern, limit_examples


def make_document(number, source, reference, language="en", category="transport", split="train"):
    return Document(id=UUID(int=number), source_text=source, target_text=reference, language=language,
                    category=category, source_dataset="fixture", source_document_id=str(number), source_split=split)


@pytest.fixture
def documents():
    return [
        make_document(1, "Rail delays follow a disruption on the London railway.", "London rail journeys were delayed."),
        make_document(2, "Commuters wait after strikes stop trains.", "Train strikes affected passengers."),
        make_document(3, "В Минске прошли соревнования по плаванию.", "В Минске завершился турнир.", "ru", "sport"),
        make_document(4, "Heavy rain interrupts the county tournament.", "Rain interrupted play.", category="sport"),
    ]


@pytest.fixture
def test_document():
    return make_document(15, "Held-out rail news describes private test station.", "Reserved evaluation target.", split="test")


class FixtureEmbeddings:
    model_name = "fixture"
    revision = None
    dimension = 3

    def __init__(self):
        self.batches = []
        self.fail = False

    async def embed_many(self, texts):
        self.batches.append(texts)
        if self.fail:
            raise RuntimeError("benchmark fixture error")
        vectors = [[1, 0, 0] if "Rail delays" in text else [0.9, 0.2, 0] if "rail" in text.casefold()
                   or "trains" in text else [0, 0, 1] for text in texts]
        return np.asarray(vectors)

    async def embed(self, text):
        return (await self.embed_many([text]))[0]


def test_bm25_tokenize_filter_save_load_rebuild(documents, tmp_path, monkeypatch):
    assert tokenize("РАЗВИТИЕ Ёжик — Rail-42 _ １２") == ["развитие", "ёжик", "rail", "42", "12"]
    index = BM25Index(documents)
    assert index.search("rail delays")[0].id == documents[0].id
    assert index.search("ПЛАВАНИЮ", language="ru", category="sport") == index.search("плаванию")
    assert not index.search("плаванию", language="en")
    assert not index.search("rail", category="sport")
    assert not index.search("rail", language="ru", category="transport")
    assert not index.search("nonsense")
    assert not index.search(" ")
    assert documents[0].id not in {hit.id for hit in index.search(documents[0].source_text, exclude_self=True)}
    path = tmp_path / "bm25.json"
    index.save(path)
    with monkeypatch.context() as context:
        context.setattr("backend.retrieval.bm25_index.tokenize", lambda _: pytest.fail("loading retokenizes text"))
        restored = BM25Index.load(path)
    assert restored.search("rail delays") == index.search("rail delays")
    assert restored.get(documents[0].id) == documents[0]
    restored.rebuild(documents[1:])
    assert restored.get(documents[0].id) is None
    with pytest.raises(IndexIntegrityError, match="повторяющиеся"):
        restored.rebuild([documents[1]] * 2)
    assert not BM25Index().search("rail")
    with pytest.raises(ValueError, match="top_k"):
        restored.search("rail", 0)


def test_invalid_bm25_snapshot_is_rejected(tmp_path):
    path = tmp_path / "bm25.json"
    path.write_text('{"format": 2, "documents": [], "tokens": [null]}', encoding="utf-8")
    with pytest.raises(IndexIntegrityError, match="снимок BM25"):
        BM25Index.load(path)


def test_rrf_uses_ranks_and_counts_each_id_once_per_channel(documents):
    first, second, third = [doc.id for doc in documents[:3]]
    lexical = [Hit(first, 30.0), Hit(first, -100.0), Hit(second, 0.0)]
    vector = [Hit(second, 0.02), Hit(third, 0.01)]
    ranking = reciprocal_rank_fusion(lexical, vector)
    assert [hit.id for hit in ranking] == [second, first, third]
    assert ranking[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert (ranking[0].bm25_rank, ranking[0].qdrant_rank) == (2, 1)
    assert reciprocal_rank_fusion([], []) == []
    with pytest.raises(ValueError, match="RRF"):
        reciprocal_rank_fusion([], [], k=0)


@pytest.mark.asyncio
async def test_real_qdrant_local_crud_and_filters(tmp_path, documents):
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    index = VectorIndex(client, 3)
    try:
        assert not await index.exists()
        await index.ensure_collection()
        await index.ensure_collection()
        assert await index.exists()
        vectors = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 0, 1], [0, 1, 0]])
        await index.upsert(documents, vectors, batch_size=1)
        await index.upsert(documents, vectors)
        assert await index.count() == 4
        assert len(await index.search([1, 0, 0])) == 4
        assert [hit.id for hit in await index.search([1, 0, 0], language="ru", category="sport")] == [documents[2].id]
        assert not await index.search([1, 0, 0], category="other")
        assert (await index.search([1, 0, 0], 1))[0].score == pytest.approx(1)
        assert (await index.search([1, 0, 0], 1, excluded_hash=content_hash(documents[0].source_text)))[0].id == documents[1].id
        assert not await index.get([])
        assert (await index.get([documents[0].id]))[0].source_dataset == "fixture"
        updated = documents[0].model_copy(update={"category": "politics"})
        await index.upsert([updated], [[0, 1, 0]])
        assert (await index.get([updated.id]))[0].category == "politics"
        await index.delete([])
        await index.delete([documents[0].id])
        assert await index.count() == 3
        assert documents[0].id not in await index.all_documents()
        with pytest.raises(ValueError, match="размерность"):
            await index.search([1, 0])
        with pytest.raises(ValueError, match="Число документов"):
            await index.upsert(documents, [[1, 0, 0]])
        with pytest.raises(ValueError, match="top_k"):
            await index.search([1, 0, 0], 0)
        with pytest.raises(IndexIntegrityError, match="Размерность"):
            await VectorIndex(client, 2).check_collection()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_qdrant_reports_backend_failure(monkeypatch):
    client = AsyncQdrantClient(":memory:")
    try:
        monkeypatch.setattr(client, "query_points", AsyncMock(side_effect=TimeoutError("secret upstream info")))
        with pytest.raises(VectorIndexError, match="TimeoutError") as fault:
            await VectorIndex(client, 3).search([1, 0, 0])
        assert "secret upstream info" not in str(fault.value)
    finally:
        await client.close()


@pytest.fixture
def index_settings(tmp_path):
    return Settings(_env_file=None, rag_index_dir=tmp_path, qdrant_mode="local")


@pytest.mark.asyncio
async def test_build_reuse_and_search_context_with_real_qdrant(index_settings, documents, test_document, fake_llm):
    embeddings = FixtureEmbeddings()
    holdout = Holdout.from_documents([test_document])
    created = await build_indexes(documents, holdout, index_settings, embeddings, batch_size=2)
    assert len(embeddings.batches) == 2 and not created["reused"]
    reused = await build_indexes(reversed(documents), holdout, index_settings, embeddings)
    assert len(embeddings.batches) == 2 and reused["version"] == created["version"]
    assert reused["reused"] and reused["test_overlaps"] == 0
    assert created["bm25_count"] == created["qdrant_count"] == 4 and created["ids_match"]

    async with open_indexes(index_settings, holdout, embeddings) as (retriever, manifest):
        assert manifest.version == created["version"]
        assert not await retriever.search("")
        assert not await retriever.search("rail", category="missing")
        assert not await retriever.semantic(" ")
        assert not await retriever.vector.get([test_document.id])
        assert retriever.bm25.get(test_document.id) is None
        assert [row.document.id for row in await retriever.search(test_document.source_text, language="en", category="transport")] == [
            documents[0].id, documents[1].id,
        ]
        results = await retriever.search(documents[0].source_text)
        assert documents[0].id not in {hit.document.id for hit in results}
        incoming = AgentState(original_text=test_document.source_text)
        enriched = await retriever.enrich_state(incoming, max_examples=2, language="en", category="transport")
        assert not incoming.rag_context and enriched.original_text == incoming.original_text
        assert enriched.rag_context == [StylePattern.SINGLE_BRIEF]
        assert enriched.rag_document_ids == [documents[0].id, documents[1].id]
        client = fake_llm("This is a test summary.")
        result = await Summarizer(client).process(enriched)
        assert result.current_summary == "This is a test summary."
        prompt = client.calls[0][0]
        assert json.loads(prompt.split("Входные данные в JSON:\n", 1)[1])["style_examples"] == enriched.rag_context
        assert test_document.target_text not in prompt
        assert documents[0].source_text not in prompt and documents[0].target_text not in prompt
        # Изменился только эталон ближайшей учебной новости, поэтому новая подсказка
        # должна описывать форму этого эталона и по-прежнему не содержать его фактов.
        hits = await retriever.search(incoming.original_text, language="en", category="transport")
        hit = hits[0]
        reference = "Ada Lovelace opened a museum in Oslo. Nine guests attended."
        alternate = hit.document.model_copy(update={"target_text": reference})
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(retriever, "search_with_mode", AsyncMock(return_value=(
                [replace(hit, document=alternate)], "same_language",
            )))
            alternative = await retriever.enrich_state(incoming, language="en", category="transport")
        assert alternative.rag_context == [StylePattern.MULTI_SENTENCE]
        style_client = fake_llm("This is a test summary.")
        await Summarizer(style_client).process(alternative)
        style_prompt = style_client.calls[0][0]
        assert all(fact not in style_prompt for fact in ("Ada Lovelace", "Oslo", "Nine guests"))
        assert (await retriever.enrich_state(incoming, max_chars=10)).rag_context == []
        with pytest.raises(ValueError, match="глубины"):
            await retriever.search("rail", candidate_k=0)
        with pytest.raises(ValueError, match="top_k"):
            await retriever.semantic("rail", top_k=0)


@pytest.mark.asyncio
async def test_multilingual_fallback_respects_native_order_category_score_and_holdout(index_settings, documents, test_document):
    model = FixtureEmbeddings()
    holdout = Holdout.from_documents([test_document])
    await build_indexes(documents, holdout, index_settings, model)
    async with open_indexes(index_settings, holdout, model) as (retriever, _):
        query = "Новости railway и новых поездов."
        strictly, mode = await retriever.search_with_mode(query, language="ru", category="transport")
        assert strictly == [] and mode == "no_results"
        calls = len(model.batches)
        found, mode = await retriever.search_with_mode(query, language="ru", category="transport", cross_language=True)
        assert len(model.batches) == calls + 1
        assert mode == "cross_language" and len(found) == 2
        assert {hit.document.id for hit in found} == {documents[0].id, documents[1].id}
        assert all(hit.bm25_rank is None and hit.cosine >= 0.30 for hit in found)
        mixed, mode = await retriever.search_with_mode("Плавание и турнир.", language="ru", cross_language=True)
        assert mode == "mixed" and [hit.document.id for hit in mixed] == [documents[2].id, documents[3].id]
        native, mode = await retriever.search_with_mode("Плавание и турнир.", 1, language="ru", cross_language=True)
        assert mode == "same_language" and len(native) == 1
        none, mode = await retriever.search_with_mode("Плавание.", language="ru", category="transport", cross_language=True)
        assert mode == "no_results" and not none


@pytest.mark.asyncio
async def test_failed_reindex_leaves_old_pair_and_success_switches_both(index_settings, documents, test_document):
    embeddings = FixtureEmbeddings()
    holdout = Holdout.from_documents([test_document])
    old = await build_indexes(documents, holdout, index_settings, embeddings)
    pointer = index_settings.rag_index_dir / "active.json"
    old_pointer = pointer.read_bytes()
    embeddings.fail = True
    with pytest.raises(RuntimeError, match="fixture error"):
        await build_indexes(documents[:2], holdout, index_settings, embeddings)
    assert pointer.read_bytes() == old_pointer
    async with open_indexes(index_settings, holdout, embeddings) as (retriever, _):
        assert len(retriever.bm25.documents) == await retriever.vector.count() == 4
    embeddings.fail = False
    new = await build_indexes(documents[:2], holdout, index_settings, embeddings)
    assert new["version"] != old["version"]
    async with open_indexes(index_settings, holdout, embeddings) as (retriever, manifest):
        assert manifest.version == new["version"]
        assert len(retriever.bm25.documents) == await retriever.vector.count() == 2
        assert await retriever.vector.get([documents[2].id]) == []


@pytest.mark.asyncio
async def test_runtime_uses_pinned_rag_snapshot_after_active_index_switch(
    index_settings, documents, test_document, monkeypatch,
):
    import httpx
    from backend.api.runtime import SummaryRuntime
    from backend.retrieval.index_bundle import IndexManifest

    embeddings = FixtureEmbeddings()
    holdout = Holdout.from_documents([test_document])
    await build_indexes(documents, holdout, index_settings, embeddings)
    pointer = index_settings.rag_index_dir / "active.json"
    pinned = IndexManifest.model_validate_json(pointer.read_bytes())
    await build_indexes(documents[1:], holdout, index_settings, embeddings)
    assert IndexManifest.model_validate_json(pointer.read_bytes()).version != pinned.version
    monkeypatch.setattr("backend.data.embeddings.Embeddings", lambda _settings: embeddings)
    monkeypatch.setattr("scripts.build_indexes.load_holdout", lambda _path: holdout)
    state = AgentState(original_text=test_document.source_text)
    async with httpx.AsyncClient() as http:
        runtime = SummaryRuntime(index_settings, None, http)
        result = await runtime.enrich(state, language="en", category="transport", manifest=pinned, holdout=holdout)
        current = await runtime.enrich(state, language="en", category="transport")
    assert result.rag_document_ids == [documents[0].id, documents[1].id]
    assert result.rag_context == [StylePattern.SINGLE_BRIEF]
    assert current.rag_document_ids == [documents[1].id]


@pytest.mark.asyncio
async def test_publish_reviewed_pair_updates_bm25_and_qdrant(index_settings, documents, test_document, monkeypatch):
    import httpx
    from backend.api.runtime import SummaryRuntime
    from backend.db.init_db import create_test_schema
    from backend.db.models import KnowledgeBase
    from backend.db.session import Database

    settings = index_settings.model_copy(update={
        "database_url": Settings(_env_file=None, database_url=(
            f"sqlite+aiosqlite:///{index_settings.rag_index_dir / 'review.sqlite3'}"
        )).database_url,
    })
    database = Database(settings.database_url.get_secret_value(), settings)
    try:
        await create_test_schema(database.engine)
        item = documents[0]
        async with database.sessions.begin() as session:
            session.add(KnowledgeBase(id=item.id, source_text=item.source_text, target_text=item.target_text,
                                      category=item.category, language=item.language, source_dataset="human_review",
                                      source_split="train"))
        embeddings = FixtureEmbeddings()
        holdout = Holdout.from_documents([test_document])
        monkeypatch.setattr("backend.data.embeddings.Embeddings", lambda _settings: embeddings)
        monkeypatch.setattr("scripts.build_indexes.load_holdout", lambda _path: holdout)
        async with httpx.AsyncClient() as http:
            runtime = SummaryRuntime(settings, database, http)
            first = await runtime.publish_reviewed_corpus()
            second = await runtime.publish_reviewed_corpus()
        assert first["ids_match"] and first["document_count"] == 1 and second["reused"]
        async with open_indexes(settings, holdout, embeddings) as (retriever, manifest):
            assert manifest.version == first["version"]
            assert retriever.bm25.get(item.id).target_text == item.target_text
            assert (await retriever.vector.get([item.id]))[0].knowledge_id == item.id
    finally:
        await database.close()


@pytest.mark.parametrize("field", ["id", "source_text", "target_text", "source_split"])
@pytest.mark.asyncio
async def test_build_rejects_holdout_fields_before_indexing(index_settings, documents, test_document, field):
    candidate = documents[0].model_copy(update={field: getattr(test_document, field)})
    holdout = Holdout.from_documents([test_document])
    embeddings = FixtureEmbeddings()
    with pytest.raises(IndexIntegrityError, match="test|train"):
        await build_indexes([candidate], holdout, index_settings, embeddings)
    assert not embeddings.batches and not (index_settings.rag_index_dir / "active.json").exists()


@pytest.mark.asyncio
async def test_detects_tampering_in_vector_and_bm25(index_settings, documents, test_document):
    embeddings = FixtureEmbeddings()
    holdout = Holdout.from_documents([test_document])
    created = await build_indexes(documents, holdout, index_settings, embeddings)
    directory = index_settings.rag_index_dir / "versions" / created["version"]
    async with open_indexes(index_settings, holdout, embeddings) as (retriever, _):
        await retriever.vector.upsert([test_document], [[1, 0, 0]])
    with pytest.raises(IndexIntegrityError, match="Qdrant.*test"):
        async with open_indexes(index_settings, holdout, embeddings):
            pytest.fail("contaminated vector bundle opened")
    path = directory / "bm25.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(IndexIntegrityError, match="Контрольная сумма"):
        async with open_indexes(index_settings, holdout, embeddings):
            pytest.fail("corrupted BM25 snapshot opened")


@pytest.mark.asyncio
async def test_index_must_use_current_holdout(index_settings, documents, test_document):
    embeddings = FixtureEmbeddings()
    old = Holdout.from_documents([test_document])
    new = Holdout.from_documents([test_document.model_copy(update={"id": UUID(int=55)})])
    await build_indexes(documents, old, index_settings, embeddings)
    with pytest.raises(IndexIntegrityError, match="контрольный набор"):
        async with open_indexes(index_settings, new, embeddings):
            pytest.fail("stale holdout accepted")


def test_context_budget_skips_oversized_examples_and_caps_count():
    examples = ["x" * 12, "a", "", "b", "c"]
    assert limit_examples(examples, max_examples=3, max_chars=10) == ["a", "b"]
    assert limit_examples(examples, max_examples=1) == ["x" * 12]
    with pytest.raises(ValueError, match="неотрицательными"):
        limit_examples(examples, max_chars=-1)

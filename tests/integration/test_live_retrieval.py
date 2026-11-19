import os
from uuid import UUID

import pytest

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.live_embeddings,
    pytest.mark.skipif(os.environ.get("RUN_LIVE_EMBEDDINGS") != "1", reason="requires cached MiniLM weights"),
]


async def test_cached_minilm_and_persistent_qdrant(tmp_path):
    import numpy as np
    import torch

    from backend.core.config import Settings
    from backend.data.embeddings import Embeddings
    from backend.retrieval.documents import Document, Holdout
    from backend.retrieval.index_bundle import build_indexes, open_indexes

    settings = Settings(
        _env_file=None, embedding_local_files_only=True, embedding_device="cpu",
        embedding_batch_size=2, rag_index_dir=tmp_path, qdrant_mode="local",
    )
    embeddings = Embeddings(settings)
    assert torch.version.cuda is None
    docs = [
        Document(
            id=UUID(int=1), source_text="Railway workers went on strike and several trains were cancelled.",
            target_text="Several train services stopped after a strike.", language="en", category="transport",
            source_split="train",
        ),
        Document(
            id=UUID(int=2), source_text="Пекарь рассказал, как приготовить пирог с яблоками и корицей.",
            target_text="Опубликован рецепт яблочного пирога.", language="ru", category="food",
            source_split="train",
        ),
    ]
    test = Document(
        id=UUID(int=3), source_text="A new gallery opens on Saturday.", target_text="A gallery opens this weekend.",
        language="en", category="culture", source_split="test",
    )
    holdout = Holdout.from_documents([test])
    vectors = await embeddings.embed_many([doc.source_text for doc in docs])
    assert vectors.shape == (2, 384)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(2), atol=1e-6)
    report = await build_indexes(docs, holdout, settings, embeddings)
    assert report["qdrant_count"] == report["bm25_count"] == 2

    async with open_indexes(settings, holdout, embeddings) as (retriever, _):
        hits = await retriever.semantic("Train passengers could not travel because of a railway strike.")
        assert [hit.id for hit in hits] == [docs[0].id, docs[1].id]
        assert hits[0].score > hits[1].score
        rus = await retriever.semantic("Как испечь яблочный пирог?", language="ru")
        assert [hit.id for hit in rus] == [docs[1].id]
        assert not await retriever.vector.get([test.id])

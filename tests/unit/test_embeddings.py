import asyncio
import threading

import pytest

np = pytest.importorskip("numpy")

from backend.core.config import Settings
from backend.core.constants import DEFAULT_EMBEDDING_REVISION
from backend.data import embeddings as module
from backend.data.embeddings import EmbeddingError, Embeddings

pytestmark = pytest.mark.asyncio


class RecordingModel:
    def __init__(self, size=2, vectors=None):
        self.size = size
        self.vectors = vectors
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return self.size

    def encode(self, texts, **params):
        self.calls.append((texts, params, threading.get_ident()))
        return self.vectors if self.vectors is not None else [[3, 4]] * len(texts)


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, embedding_size=2, embedding_batch_size=7, embedding_cache_dir=tmp_path)


async def test_lazy_loading_reuses_model_for_parallel_encodes(monkeypatch, settings):
    loads = []
    model = RecordingModel()

    def load(*args):
        loads.append((args, threading.get_ident()))
        return model

    monkeypatch.setattr(module, "_load_model", load)
    first, second = Embeddings(settings), Embeddings(settings)
    assert loads == []
    assert first.revision == DEFAULT_EMBEDDING_REVISION
    batch, single = await asyncio.gather(first.embed_many(["English", "Русский"]), second.embed("News"))
    assert len(loads) == 1
    assert loads[0][1] != threading.get_ident()
    assert loads[0][0][2] == "cpu"
    assert model.calls[0][1]["batch_size"] == model.calls[1][1]["batch_size"] == 7
    assert {tuple(call[0]) for call in model.calls} == {("English", "Русский"), ("News",)}
    assert all(call[1]["normalize_embeddings"] and call[2] != threading.get_ident() for call in model.calls)
    assert single.shape == (2,) and batch.shape == (2, 2)
    np.testing.assert_allclose(batch, [[0.6, 0.8], [0.6, 0.8]])
    np.testing.assert_allclose(single, [0.6, 0.8])


async def test_empty_input_does_not_load_model(monkeypatch, settings):
    monkeypatch.setattr(module, "_load_model", lambda *args: pytest.fail("empty text loaded model"))
    embeddings = Embeddings(settings)
    assert (await embeddings.embed_many([])).shape == (0, 2)
    for text in ("", " \n", None):
        with pytest.raises(ValueError, match="непустой"):
            await embeddings.embed(text)


@pytest.mark.parametrize("size,vectors,message", [
    (3, None, "Размерность модели"),
    (2, [[1, 2, 3]], "Некорректная форма"),
    (2, [[float("nan"), 0]], "Некорректная форма"),
    (2, [[0, 0]], "Некорректная норма"),
])
async def test_invalid_embeddings(monkeypatch, settings, size, vectors, message):
    monkeypatch.setattr(module, "_load_model", lambda *args: RecordingModel(size, vectors))
    with pytest.raises(EmbeddingError, match=message):
        await Embeddings(settings).embed("Example")


async def test_model_loading_failure_can_be_retried(monkeypatch, settings):
    invocations = 0

    def load(*args):
        nonlocal invocations
        invocations += 1
        if invocations == 1:
            raise OSError("upstream detail")
        return RecordingModel()

    monkeypatch.setattr(module, "_load_model", load)
    embeddings = Embeddings(settings)
    with pytest.raises(EmbeddingError, match="OSError") as fault:
        await embeddings.embed("Example")
    assert "upstream detail" not in str(fault.value)
    assert (await embeddings.embed("Example")).shape == (2,)
    assert invocations == 2

"""Ленивая модель SentenceTransformers для асинхронных сценариев поиска."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from backend.core.config import Settings
from backend.core.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_REVISION


class EmbeddingError(RuntimeError):
    """Модель недоступна или вернула некорректные векторы."""


def _load_model(model: str, revision: str | None, device: str, cache: str, offline: bool) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model, revision=revision, device=device, cache_folder=cache, local_files_only=offline,
        model_kwargs={"use_safetensors": True}, trust_remote_code=False,
    )


@dataclass
class _SharedModel:
    model: Any
    encode_lock: Lock = field(default_factory=Lock)


_models: dict[tuple[str, str | None, str, str], _SharedModel] = {}
_models_lock = Lock()


class Embeddings:
    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.embedding_model
        self.revision = settings.embedding_revision
        if not self.revision and self.model_name == DEFAULT_EMBEDDING_MODEL:
            self.revision = DEFAULT_EMBEDDING_REVISION
        self.dimension = settings.embedding_size
        self.batch_size = settings.embedding_batch_size
        self.device = settings.embedding_device
        self.cache = str(Path(settings.embedding_cache_dir).resolve())
        self.offline = settings.embedding_local_files_only

    async def embed(self, text: str) -> np.ndarray:
        return (await self.embed_many([text]))[0]

    async def embed_many(self, texts: list[str]) -> np.ndarray:
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Для эмбеддинга требуется непустой текст.")
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return await asyncio.to_thread(self._encode, list(texts))

    def _encode(self, texts: list[str]) -> np.ndarray:
        key = (self.model_name, self.revision, self.device, self.cache)
        try:
            with _models_lock:
                shared = _models.get(key)
                if shared is None:
                    shared = _SharedModel(_load_model(*key, self.offline))
                    _models[key] = shared
            with shared.encode_lock:
                reported = shared.model.get_sentence_embedding_dimension()
                if reported != self.dimension:
                    raise EmbeddingError(f"Размерность модели {reported}, ожидалась {self.dimension}.")
                vectors = np.asarray(shared.model.encode(
                    texts, batch_size=self.batch_size, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False,
                ), dtype=np.float32)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Сбой SentenceTransformers ({type(exc).__name__}).") from exc
        if vectors.shape != (len(texts), self.dimension) or not np.isfinite(vectors).all():
            raise EmbeddingError("Некорректная форма или значения эмбеддингов.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms <= 0).any() or not np.isfinite(norms).all():
            raise EmbeddingError("Некорректная норма эмбеддингов.")
        return vectors / norms

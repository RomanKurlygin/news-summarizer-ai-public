"""Индекс Qdrant с UUID базы знаний в качестве идентификаторов точек."""

from collections.abc import Awaitable, Iterable, Sequence
from typing import TypeVar
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ValidationError
from qdrant_client import AsyncQdrantClient, models

from backend.retrieval.documents import Document, Hit, IndexIntegrityError

COLLECTION = "news_knowledge_base"
T = TypeVar("T")


class VectorIndexError(RuntimeError):
    """Операция с Qdrant не завершилась."""


class VectorDocument(BaseModel):
    knowledge_id: UUID
    language: str
    category: str
    source_dataset: str | None
    source_document_id: str | None
    source_url: str | None
    source_hash: str
    target_hash: str

    @classmethod
    def from_document(cls, doc: Document) -> "VectorDocument":
        source_hash, target_hash = doc.hashes()
        return cls(
            knowledge_id=doc.id, language=doc.language, category=doc.category,
            source_dataset=doc.source_dataset, source_document_id=doc.source_document_id,
            source_url=doc.source_url, source_hash=source_hash, target_hash=target_hash,
        )


async def _call(operation: Awaitable[T]) -> T:
    try:
        return await operation
    except Exception as exc:
        raise VectorIndexError(f"Операция Qdrant не выполнена ({type(exc).__name__}).") from exc


class VectorIndex:
    def __init__(self, client: AsyncQdrantClient, dimension: int, collection: str = COLLECTION) -> None:
        if dimension < 1:
            raise ValueError("Размерность Qdrant должна быть положительной.")
        self.client = client
        self.dimension = dimension
        self.collection = collection

    async def exists(self) -> bool:
        return await _call(self.client.collection_exists(self.collection))

    async def ensure_collection(self, *, server_indexes: bool = False) -> None:
        if not await self.exists():
            await _call(self.client.create_collection(
                self.collection, vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE)
            ))
        await self.check_collection()
        if server_indexes:
            for field in ("category", "language"):
                await _call(self.client.create_payload_index(
                    self.collection, field, models.PayloadSchemaType.KEYWORD, wait=True,
                ))

    async def check_collection(self) -> None:
        collection = await _call(self.client.get_collection(self.collection))
        vectors = collection.config.params.vectors
        if (not isinstance(vectors, models.VectorParams)
                or vectors.size != self.dimension or vectors.distance != models.Distance.COSINE):
            raise IndexIntegrityError("Размерность или метрика коллекции Qdrant не соответствует cosine-индексу.")

    def _vector(self, vector: Sequence[float] | np.ndarray) -> list[float]:
        values = np.asarray(vector, dtype=float)
        if values.shape != (self.dimension,) or not np.isfinite(values).all() or not values.any():
            raise ValueError("Неверная размерность, нулевой или нечисловой вектор Qdrant.")
        return values.tolist()

    async def upsert(self, documents: Sequence[Document], vectors: Sequence[Sequence[float]] | np.ndarray,
                     *, batch_size: int = 64) -> None:
        if len(documents) != len(vectors) or batch_size < 1:
            raise ValueError("Число документов и векторов должно совпадать, размер пакета должен быть положительным.")
        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            points = [
                models.PointStruct(id=str(doc.id), vector=self._vector(vector),
                                   payload=VectorDocument.from_document(doc).model_dump(mode="json"))
                for doc, vector in zip(documents[start:end], vectors[start:end], strict=True)
            ]
            await _call(self.client.upsert(self.collection, points, wait=True))

    async def search(self, vector: Sequence[float] | np.ndarray, top_k: int = 5, *,
                     language: str | None = None, category: str | None = None,
                     excluded_hash: str | None = None, excluded_language: str | None = None) -> list[Hit]:
        if top_k < 1:
            raise ValueError("top_k должен быть положительным.")
        must = [models.FieldCondition(key=key, match=models.MatchValue(value=value))
                for key, value in (("category", category), ("language", language)) if value is not None]
        must_not = [models.FieldCondition(key=key, match=models.MatchValue(value=excluded_hash))
                    for key in ("source_hash", "target_hash")] if excluded_hash else []
        if excluded_language:
            must_not.append(models.FieldCondition(key="language", match=models.MatchValue(value=excluded_language)))
        response = await _call(self.client.query_points(
            self.collection, query=self._vector(vector), limit=top_k,
            query_filter=models.Filter(must=must, must_not=must_not) if must or must_not else None,
            with_payload=False,
        ))
        return sorted((Hit(UUID(str(row.id)), row.score) for row in response.points), key=lambda hit: (-hit.score, hit.id))

    @staticmethod
    def _payload(row: models.Record) -> VectorDocument:
        try:
            data = VectorDocument.model_validate(row.payload)
            if str(data.knowledge_id) != str(row.id):
                raise ValueError("Идентификатор payload расходится с идентификатором точки.")
            return data
        except (ValidationError, ValueError) as exc:
            raise IndexIntegrityError("Повреждён payload Qdrant.") from exc

    async def get(self, identifiers: Iterable[UUID]) -> list[VectorDocument]:
        ids = [str(value) for value in identifiers]
        if not ids:
            return []
        rows = await _call(self.client.retrieve(self.collection, ids, with_payload=True, with_vectors=False))
        return [self._payload(row) for row in rows]

    async def all_documents(self) -> dict[UUID, VectorDocument]:
        result = {}
        offset = None
        while True:
            rows, offset = await _call(self.client.scroll(
                self.collection, limit=256, offset=offset, with_payload=True, with_vectors=False,
            ))
            for row in rows:
                doc = self._payload(row)
                result[doc.knowledge_id] = doc
            if offset is None:
                return result

    async def count(self) -> int:
        return (await _call(self.client.count(self.collection, exact=True))).count

    async def delete(self, identifiers: Iterable[UUID]) -> None:
        ids = [str(value) for value in identifiers]
        if ids:
            await _call(self.client.delete(self.collection, models.PointIdsList(points=ids), wait=True))

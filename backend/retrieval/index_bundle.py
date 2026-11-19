"""Версии BM25 и Qdrant с единым атомарным указателем."""

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import AsyncQdrantClient

from backend.core.config import Settings
from backend.data.embeddings import Embeddings
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.documents import Document, Holdout, IndexIntegrityError
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.vector_index import VectorDocument, VectorIndex, VectorIndexError

_LOGGER = logging.getLogger(__name__)


class IndexManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    format: Literal[1] = 1
    version: str = Field(pattern=r"^[a-f0-9]{32}$")
    corpus_sha256: str
    holdout_sha256: str
    bm25_sha256: str | None = None
    document_count: int = Field(gt=0)
    holdout_count: int = Field(gt=0)
    model_name: str
    model_revision: str | None
    dimension: int = Field(gt=0)
    mode: Literal["local", "server"]
    collection: str


def _corpus_digest(documents: Iterable[Document]) -> str:
    digest = hashlib.sha256()
    for doc in sorted(documents, key=lambda doc: doc.id):
        digest.update(doc.model_dump_json().encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _holdout_digest(holdout: Holdout) -> str:
    data = json.dumps([sorted(map(str, holdout.ids)), sorted(holdout.hashes)])
    return hashlib.sha256(data.encode()).hexdigest()


def _bm25_digest(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        raise IndexIntegrityError("Снимок BM25 отсутствует или недоступен.") from exc


def _manifest(path: Path) -> IndexManifest:
    try:
        return IndexManifest.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise IndexIntegrityError("Снимок индексов отсутствует или манифест повреждён.") from exc


def _publish(path: Path, manifest: IndexManifest) -> None:
    with NamedTemporaryFile(dir=path.parent, prefix=".active-", suffix=".tmp", delete=False) as temporary:
        temporary.write(manifest.model_dump_json(indent=2).encode())
    staged = Path(temporary.name)
    try:
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def _collection(settings: Settings, version: str) -> str:
    if settings.qdrant_mode == "local":
        return settings.qdrant_collection
    return f"{settings.qdrant_collection}__{version}"


@asynccontextmanager
async def _vector(settings: Settings, manifest: IndexManifest, directory: Path) -> AsyncIterator[VectorIndex]:
    try:
        if manifest.mode == "local":
            client = AsyncQdrantClient(path=str(directory / "qdrant"))
        else:
            key = settings.qdrant_api_key
            client = AsyncQdrantClient(url=str(settings.qdrant_url), api_key=key.get_secret_value() if key else None)
    except (OSError, RuntimeError, ValueError) as exc:
        raise VectorIndexError(f"Не удалось открыть Qdrant ({type(exc).__name__}).") from exc
    try:
        yield VectorIndex(client, manifest.dimension, manifest.collection)
    finally:
        await client.close()


async def _verify(bm25: BM25Index, vector: VectorIndex, manifest: IndexManifest, holdout: Holdout) -> None:
    documents = bm25.documents
    if len(documents) != manifest.document_count or _corpus_digest(documents) != manifest.corpus_sha256:
        raise IndexIntegrityError("Снимок BM25 расходится с манифестом.")
    holdout.validate(documents)
    await vector.check_collection()
    records = await vector.all_documents()
    if any(doc_id in holdout.ids or {doc.source_hash, doc.target_hash} & holdout.hashes
           for doc_id, doc in records.items()):
        raise IndexIntegrityError("В Qdrant обнаружен идентификатор или текст из test.")
    expected = {doc.id: VectorDocument.from_document(doc) for doc in documents}
    if await vector.count() != manifest.document_count or records != expected:
        raise IndexIntegrityError("Количество, UUID или метаданные в Qdrant и BM25 расходятся.")


@asynccontextmanager
async def open_indexes(settings: Settings, holdout: Holdout, embeddings: Embeddings,
                       *, manifest: IndexManifest | None = None) -> AsyncIterator[tuple[HybridRetriever, IndexManifest]]:
    root = settings.rag_index_dir
    current = manifest or _manifest(root / "active.json")
    if (current.mode != settings.qdrant_mode or current.collection != _collection(settings, current.version)
            or (current.model_name, current.model_revision, current.dimension)
            != (embeddings.model_name, embeddings.revision, embeddings.dimension)
            or current.holdout_count != len(holdout.ids) or current.holdout_sha256 != _holdout_digest(holdout)):
        raise IndexIntegrityError("Настройки поиска или контрольный набор изменились. Перестройте индексы.")
    directory = root / "versions" / current.version
    bm25_path = directory / "bm25.json"
    if not current.bm25_sha256 or await asyncio.to_thread(_bm25_digest, bm25_path) != current.bm25_sha256:
        raise IndexIntegrityError("Контрольная сумма снимка BM25 расходится с манифестом.")
    bm25 = await asyncio.to_thread(BM25Index.load, bm25_path)
    async with _vector(settings, current, directory) as vector:
        await _verify(bm25, vector, current, holdout)
        yield HybridRetriever(bm25, vector, embeddings, holdout), current


async def build_indexes(documents: Iterable[Document], holdout: Holdout, settings: Settings,
                        embeddings: Embeddings, *, batch_size: int = 64) -> dict:
    started = perf_counter()
    docs = tuple(sorted(documents, key=lambda doc: doc.id))
    if not docs or batch_size < 1:
        raise ValueError("Нужен непустой корпус и положительный размер пакета.")
    holdout.validate(docs)
    if len({doc.id for doc in docs}) != len(docs):
        raise IndexIntegrityError("В снимке базы знаний повторяется UUID.")
    root = settings.rag_index_dir
    version = uuid4().hex
    candidate = IndexManifest(
        version=version, corpus_sha256=_corpus_digest(docs), holdout_sha256=_holdout_digest(holdout),
        document_count=len(docs), holdout_count=len(holdout.ids), model_name=embeddings.model_name,
        model_revision=embeddings.revision, dimension=embeddings.dimension, mode=settings.qdrant_mode,
        collection=_collection(settings, version),
    )
    pointer = root / "active.json"
    if pointer.exists():
        previous = _manifest(pointer)
        if (previous.bm25_sha256 and previous.corpus_sha256 == candidate.corpus_sha256
                and previous.holdout_sha256 == candidate.holdout_sha256
                and previous.mode == candidate.mode and previous.collection == _collection(settings, previous.version)
                and (previous.model_name, previous.model_revision, previous.dimension)
                == (candidate.model_name, candidate.model_revision, candidate.dimension)):
            async with open_indexes(settings, holdout, embeddings, manifest=previous):
                return _report(previous, started, reused=True)

    directory = root / "versions" / version
    directory.mkdir(parents=True)
    bm25 = await asyncio.to_thread(BM25Index, docs)
    bm25_path = directory / "bm25.json"
    await asyncio.to_thread(bm25.save, bm25_path)
    candidate = candidate.model_copy(update={"bm25_sha256": await asyncio.to_thread(_bm25_digest, bm25_path)})
    async with _vector(settings, candidate, directory) as vector:
        await vector.ensure_collection(server_indexes=settings.qdrant_mode == "server")
        for start in range(0, len(docs), batch_size):
            batch = docs[start:start + batch_size]
            vectors = await embeddings.embed_many([doc.source_text for doc in batch])
            await vector.upsert(batch, vectors, batch_size=batch_size)
            _LOGGER.info("Проиндексировано %d/%d новостей.", start + len(batch), len(docs))
    async with open_indexes(settings, holdout, embeddings, manifest=candidate):
        pass
    (directory / "manifest.json").write_text(candidate.model_dump_json(indent=2), encoding="utf-8")
    _publish(pointer, candidate)
    return _report(candidate, started, reused=False)


def _report(manifest: IndexManifest, started: float, *, reused: bool) -> dict:
    return {
        **manifest.model_dump(), "bm25_count": manifest.document_count, "qdrant_count": manifest.document_count,
        "ids_match": True, "test_overlaps": 0, "reused": reused,
        "seconds": round(perf_counter() - started, 3), "errors": 0,
    }

"""Построить или проверить индексы базы знаний после импорта корпуса."""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from time import perf_counter

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from backend.core.config import Settings
from backend.core.constants import PROJECT_ROOT
from backend.data.embeddings import EmbeddingError, Embeddings
from backend.db.models import KnowledgeBase
from backend.db.session import Database
from backend.retrieval.documents import Document, Holdout, IndexIntegrityError
from backend.retrieval.index_bundle import build_indexes, open_indexes
from backend.retrieval.vector_index import VectorIndexError
from scripts.corpus import TEST_SIZE, TRAIN_SIZE, News, load_prepared

TEST_FILE = PROJECT_ROOT / "data" / "test" / "news_test.json"


def load_holdout(path: Path) -> Holdout:
    test = load_prepared(path, News)
    if len(test) < TEST_SIZE:
        raise IndexIntegrityError(f"Нужно минимум {TEST_SIZE} тестовых пар. Получено {len(test)}.")
    return Holdout.from_documents(
        Document.model_validate({**row.model_dump(exclude={"split"}), "source_split": row.split}) for row in test
    )


async def read_database(settings: Settings) -> list[Document]:
    if settings.database_url is None:
        raise IndexIntegrityError("Задайте DATABASE_URL базы знаний.")
    database = Database(settings.database_url.get_secret_value(), settings)
    try:
        async with database.sessions() as session:
            rows = (await session.scalars(select(KnowledgeBase).order_by(KnowledgeBase.id))).all()
            return [Document(
                id=row.id, source_text=row.source_text, target_text=row.target_text, category=row.category,
                language=row.language, source_dataset=row.source_dataset, source_document_id=row.source_document_id,
                source_url=row.source_url, source_split=row.source_split,
            ) for row in rows]
    finally:
        await database.close()


async def _run(args: argparse.Namespace, settings: Settings) -> dict:
    holdout = load_holdout(args.test_file)
    embeddings = Embeddings(settings)
    if args.verify_only:
        async with open_indexes(settings, holdout, embeddings) as (_, manifest):
            return {**manifest.model_dump(), "bm25_count": manifest.document_count,
                    "qdrant_count": manifest.document_count, "ids_match": True, "test_overlaps": 0, "errors": 0}
    documents = await read_database(settings)
    if len(documents) < args.min_documents:
        raise IndexIntegrityError(f"В базе {len(documents)} строк. Нужно минимум {args.min_documents}.")
    return await build_indexes(documents, holdout, settings, embeddings, batch_size=args.batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, default=TEST_FILE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-documents", type=int, default=TRAIN_SIZE)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.min_documents < 1 or args.batch_size < 1:
        parser.error("--min-documents и --batch-size должны быть положительными")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    started = perf_counter()
    try:
        report = asyncio.run(_run(args, Settings()))
    except (EmbeddingError, IndexIntegrityError, VectorIndexError, OSError, SQLAlchemyError, ValueError) as exc:
        message = str(exc) if isinstance(exc, (EmbeddingError, IndexIntegrityError, VectorIndexError)) else type(exc).__name__
        parser.exit(1, f"Индексация остановлена. Ошибок: 1. Время: {perf_counter() - started:.3f} с. {message}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Миграции и загрузка подготовленного корпуса, если он смонтирован в Compose."""

import asyncio
import logging
from pathlib import Path

from alembic import command
from alembic.config import Config

from backend.core.config import Settings
from backend.data.embeddings import Embeddings
from backend.db.session import Database
from backend.retrieval.index_bundle import build_indexes
from scripts.build_indexes import load_holdout, read_database
from scripts.corpus_db import import_corpus
from scripts.verify_dataset import verify_dataset


async def prepare(settings: Settings, corpus_dir: Path) -> None:
    required = [corpus_dir / name for name in (
        "import/news_train.json", "import/glossary.json", "test/news_test.json",
    )]
    present = [path.is_file() for path in required]
    if not any(present):
        print("Корпус не смонтирован. Доступен baseline. Для full подготовьте учебные и контрольные данные.")
        return
    if not all(present):
        raise ValueError("Смонтирован неполный корпус. Нужны обе учебные выгрузки и отдельный test.")
    corpus = await asyncio.to_thread(verify_dataset, corpus_dir)
    database = Database(settings.database_url.get_secret_value(), settings)
    try:
        await import_corpus(database.sessions, corpus.train, corpus.terms, corpus.test)
    finally:
        await database.close()
    guard = await asyncio.to_thread(load_holdout, settings.holdout_path)
    documents = await read_database(settings)
    result = await build_indexes(documents, guard, settings, Embeddings(settings))
    print(f"Опубликован индекс {result['version']} с {result['document_count']} документами.")
    wordnet = settings.metric_nltk_dir / "corpora" / "wordnet"
    if not wordnet.is_dir() and not wordnet.with_suffix(".zip").is_file():
        import nltk

        await asyncio.to_thread(nltk.download, "wordnet", download_dir=str(settings.metric_nltk_dir),
                                quiet=True, raise_on_error=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    if settings.database_url is None:
        raise SystemExit("Задайте DATABASE_URL.")
    command.upgrade(Config("alembic.ini"), "head")
    asyncio.run(prepare(settings, settings.holdout_path.parent.parent))


if __name__ == "__main__":
    main()

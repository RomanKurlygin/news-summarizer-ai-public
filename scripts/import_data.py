"""Импортировать проверенные учебные новости и справочник в мигрированную БД."""

import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from backend.core.config import Settings
from backend.db.session import Database
from scripts.corpus import DataError, News, Term, load_prepared
from scripts.corpus_db import import_corpus
from scripts.verify_dataset import VerifiedCorpus, verify_dataset


async def _save(args: argparse.Namespace, corpus: VerifiedCorpus, settings: Settings) -> dict:
    url = args.database_url or (settings.database_url.get_secret_value() if settings.database_url else None)
    if not url:
        raise DataError("Укажите DATABASE_URL или --database-url.")
    train, terms = corpus.train, corpus.terms
    if args.format == "csv":
        train = load_prepared(args.data_dir / "import/news_train.csv", News)
        terms = load_prepared(args.data_dir / "import/glossary.csv", Term)
        if train != corpus.train or terms != corpus.terms:
            raise DataError("CSV изменились после проверки корпуса.")
    database = Database(url, settings)
    try:
        return await import_corpus(database.sessions, train, terms, corpus.test)
    finally:
        await database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument("--database-url", help="По умолчанию берётся DATABASE_URL")
    args = parser.parse_args()
    try:
        corpus = verify_dataset(args.data_dir)
        statistics = asyncio.run(_save(args, corpus, Settings()))
    except (DataError, OSError) as exc:
        parser.exit(1, f"Импорт отменён: {exc}\n")
    except (SQLAlchemyError, ValueError) as exc:
        parser.exit(1, f"Импорт отменён ({type(exc).__name__}). Проверьте подключение и миграции.\n")
    print(json.dumps(statistics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

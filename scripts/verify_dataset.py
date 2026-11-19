"""Проверить объёмы, происхождение, разделение и готовность импорта корпуса."""

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
from sqlalchemy.exc import SQLAlchemyError

from backend.core.config import Settings
from backend.db.session import Database
from scripts.build_glossary import extract_terms, glossary_stats
from scripts.corpus import (
    GLOSSARY_SIZE, TEST_SIZE, TRAIN_SIZE, DataError, News, Term, evaluation_ids, file_hash,
    intersections, load_prepared, prepare_corpus, read_json,
)
from scripts.corpus_db import import_corpus
from scripts.generate_data import ROWS_URL, XSUM_DATASET, fetch_xsum, xsum_row_url


@dataclass(frozen=True)
class VerifiedCorpus:
    train: list[News]
    test: list[News]
    terms: list[Term]
    report: dict


def verify_dataset(data_dir: Path, *, min_train: int = TRAIN_SIZE, min_test: int = TEST_SIZE,
                   min_glossary: int = GLOSSARY_SIZE, check_source: bool = False) -> VerifiedCorpus:
    manifest = read_json(data_dir / "import/manifest.json")
    snapshot_path = data_dir / "raw/news_candidates.json"
    snapshot = read_json(snapshot_path)
    if not isinstance(manifest, dict) or manifest.get("version") != 1 or not isinstance(snapshot, dict):
        raise DataError("Неизвестный формат манифеста или снимка.")
    if manifest.get("snapshot_sha256") != file_hash(snapshot_path):
        raise DataError("Контрольная сумма исходного снимка не совпадает с манифестом.")
    source = snapshot.get("provenance")
    raw = snapshot.get("rows")
    if (not isinstance(source, dict) or source != manifest.get("source")
            or not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw)):
        raise DataError("Неполные сведения об исходных данных.")
    if source.get("kind") == "huggingface_rows":
        _check_xsum_metadata(source, raw)
    elif (source.get("kind") != "local_file" or not source.get("filename")
          or not re.fullmatch(r"[a-f0-9]{64}", str(source.get("sha256")))):
        raise DataError("Неизвестное происхождение данных.")

    train = load_prepared(data_dir / "import/news_train.json", News)
    test = load_prepared(data_dir / "test/news_test.json", News)
    terms = load_prepared(data_dir / "import/glossary.json", Term)
    counts = (len(train), len(test), len(terms))
    minima = (min_train, min_test, min_glossary)
    if any(count < minimum for count, minimum in zip(counts, minima, strict=True)):
        raise DataError(f"Недостаточно данных. Train/test/glossary: {counts}, требуется {minima}.")
    overlap = intersections(train, test)
    if any(overlap.values()):
        raise DataError(f"Утечка между train и test: {overlap}.")
    settings = manifest.get("settings")
    if not isinstance(settings, dict) or settings.keys() != {"train_size", "test_size", "seed", "dataset", "language"}:
        raise DataError("Отсутствуют параметры отбора.")
    if not all(type(settings[key]) is int for key in ("train_size", "test_size", "seed")) or not all(
        isinstance(settings[key], str) for key in ("dataset", "language")
    ):
        raise DataError("Некорректные параметры отбора.")
    expected_train, expected_test, statistics = prepare_corpus(raw, **settings)
    if (train, test, statistics) != (expected_train, expected_test, manifest.get("statistics")):
        raise DataError("Учебные/тестовые записи или статистика не совпадают с повторным отбором из снимка.")
    if terms != extract_terms(train, len(terms)):
        raise DataError("Записи справочника не совпадают с повторным извлечением только из train.")
    ids = evaluation_ids(test)
    if read_json(data_dir / "test/evaluation_ids.json") != ids:
        raise DataError("ID выборок 20/50/100 не совпадают с зафиксированным порядком test.")
    for filename, schema, expected in (
        ("import/news_train.csv", News, train), ("test/news_test.csv", News, test),
        ("import/glossary.csv", Term, terms),
    ):
        if load_prepared(data_dir / filename, schema) != expected:
            raise DataError(f"CSV и JSON расходятся: {filename}.")

    if check_source:
        if source["kind"] != "huggingface_rows":
            raise DataError("Онлайн-сверка реализована для XSum. Локальные файлы проверяются по снимку.")
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            upstream, _ = fetch_xsum(client, settings["train_size"], settings["test_size"], source["revision"])
        if upstream != raw:
            raise DataError("Исходный снимок расходится со строками XSum, полученными по HTTPS.")

    return VerifiedCorpus(train, test, terms, {
        "checks": "passed", "corpus": statistics, "glossary": glossary_stats(terms),
        "source": source, "snapshot_sha256": manifest["snapshot_sha256"],
        "online_source_matched": check_source, "evaluation_sizes": {key: len(value) for key, value in ids.items()},
    })


def _check_xsum_metadata(source: dict, rows: list[dict]) -> None:
    limits = source.get("row_limits")
    if (source.get("dataset") != XSUM_DATASET or source.get("endpoint") != ROWS_URL
            or not re.fullmatch(r"[a-f0-9]{40}", str(source.get("revision")))
            or not isinstance(limits, dict) or limits.keys() != {"train", "test"}
            or any(type(count) is not int or not 0 < count <= 4000 for count in limits.values())):
        raise DataError("Некорректное описание снимка XSum.")
    expected = [(split, index) for split in ("train", "test") for index in range(limits[split])]
    actual = [(row.get("split"), row.get("row_index")) for row in rows]
    if expected != actual or any(
        row.get("source_dataset") != XSUM_DATASET or row.get("language") != "en"
        or not re.fullmatch(r"[0-9]+", str(row.get("id")))
        or row.get("source_url") != xsum_row_url(split, index)
        for row, (split, index) in zip(rows, expected, strict=True)
    ):
        raise DataError("Снимок XSum не соответствует заявленным разделам и идентификаторам.")


async def verify_import(database_url: str, corpus: VerifiedCorpus, settings: Settings) -> dict:
    database = Database(database_url, settings)
    try:
        statistics = await import_corpus(database.sessions, corpus.train, corpus.terms, corpus.test, dry_run=True)
        if statistics["inserted_news"] or statistics["inserted_glossary"]:
            raise DataError("Часть корпуса отсутствует в БД. Повторный импорт ещё предложит вставки.")
        return {**statistics, "repeat_would_insert": 0, "database_test_overlaps": 0}
    finally:
        await database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--check-source", action="store_true", help="Сверить снимок с XSum по HTTPS")
    parser.add_argument("--check-db", action="store_true", help="Проверить существующую БД без записи")
    parser.add_argument("--database-url", help="URL БД для --check-db, иначе DATABASE_URL из окружения")
    args = parser.parse_args()
    try:
        corpus = verify_dataset(args.data_dir, check_source=args.check_source)
        if args.check_db:
            settings = Settings()
            url = args.database_url or (settings.database_url.get_secret_value() if settings.database_url else None)
            if not url:
                raise DataError("Для --check-db задайте DATABASE_URL или --database-url.")
            corpus.report["database"] = asyncio.run(verify_import(url, corpus, settings))
    except (DataError, OSError) as exc:
        parser.exit(1, f"Проверка не пройдена: {exc}\n")
    except SQLAlchemyError as exc:
        parser.exit(1, f"Ошибка БД: {type(exc).__name__}. Проверьте подключение и миграции.\n")
    print(json.dumps(corpus.report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

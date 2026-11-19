"""Форматы файлов, нормализация и воспроизводимый отбор новостей."""

import csv
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.utils.corpus_text import clean_text, content_hash, normalized

SEED = 42
SAMPLE_SIZES = (20, 50, 100)
TRAIN_SIZE = 1000
TEST_SIZE = 200
GLOSSARY_SIZE = 500
csv.field_size_limit(8 * 1024 * 1024)


class DataError(ValueError):
    """Нарушен формат или условия подготовки данных."""


class News(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    source_text: str = Field(min_length=1)
    target_text: str = Field(min_length=1)
    language: str = Field(pattern=r"^[a-z]{2,3}(?:-[a-z0-9]{2,8})*$", max_length=16)
    category: str = Field(min_length=1, max_length=120)
    source_dataset: str = Field(min_length=1, max_length=160)
    source_document_id: str | None = Field(default=None, min_length=1, max_length=255)
    source_url: str | None = None
    split: str = Field(pattern=r"^(train|test)$")


class TermSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    news_id: UUID
    method: str = Field(min_length=1)
    document_count: int = Field(ge=1)


class Term(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    term: str = Field(min_length=1, max_length=255)
    definition: str = Field(min_length=1)
    category: str = Field(pattern=r"^(person|organization|place|topic)$")
    language: str = Field(pattern=r"^[a-z]{2,3}(?:-[a-z0-9]{2,8})*$", max_length=16)
    aliases: list[str] = Field(default_factory=list)
    source: TermSource
    review_required: bool


def stable_id(kind: str, *parts: str) -> UUID:
    return uuid5(NAMESPACE_URL, "news-summarizer-ai/" + kind + json.dumps(parts, ensure_ascii=False))


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                    errors="backslashreplace", newline="\n")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise DataError(f"Не удалось прочитать JSON {path}: {type(exc).__name__}") from None


def read_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        result = read_json(path)
        if isinstance(result, list) and all(isinstance(row, dict) for row in result):
            return result
        raise DataError(f"{path} должен содержать массив JSON-объектов.")
    if path.suffix.lower() not in {".csv", ".txt", ".tsv"}:
        raise DataError(f"Поддерживаются CSV, JSON и TXT/TSV с заголовком: {path}")
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.DictReader(stream, delimiter="," if path.suffix.lower() == ".csv" else "\t", strict=True)
            fields = rows.fieldnames
            if not fields or not all(fields) or len(fields) != len(set(fields)):
                raise DataError(f"Некорректный заголовок {path}.")
            result = []
            for row in rows:
                if None in row or any(value is None for value in row.values()):
                    raise DataError(f"Несовпадение количества колонок в {path}, строка {rows.line_num}.")
                result.append(row)
            return result
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DataError(f"Не удалось прочитать {path}: {type(exc).__name__}") from None


def export_records(path: Path, records: Iterable[BaseModel]) -> None:
    rows = [record.model_dump(mode="json") for record in records]
    write_json(path.with_suffix(".json"), rows)
    if not rows:
        return
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(val, ensure_ascii=False) if isinstance(val, (list, dict)) else val
                             for key, val in row.items()})


def load_prepared(path: Path, model: type[News] | type[Term]):
    result = []
    for index, row in enumerate(read_records(path), 1):
        try:
            if path.suffix.lower() == ".csv":
                if model is Term:
                    for field in ("aliases", "source"):
                        row[field] = json.loads(row[field])
                else:
                    for field in ("source_document_id", "source_url"):
                        row[field] = row.get(field) or None
            record = model.model_validate(row)
            strings = (record.source_text, record.target_text, record.category, record.source_dataset) if model is News else (
                record.term, record.definition, *record.aliases, record.source.method,
            )
            if any(not value or clean_text(value) != value for value in strings):
                raise ValueError("Нужен непустой нормализованный текст")
            result.append(record)
        except (ValidationError, KeyError, ValueError) as exc:
            raise DataError(f"Некорректная запись {index} в {path.name}: {type(exc).__name__}") from None
    return result


def _clean_news(row: dict, dataset: str, language: str) -> News:
    source_text = clean_text(row.get("source_text", row.get("document", row.get("article"))))
    target_text = clean_text(row.get("target_text", row.get("summary", row.get("highlights"))))
    origin = clean_text(row.get("source_dataset") or dataset)
    document = row.get("source_document_id", row.get("id"))
    document = clean_text(str(document)) if type(document) in {str, int} else ""
    if not source_text or not target_text or not origin or normalized(source_text) == normalized(target_text):
        raise ValueError("Некорректная пара")
    source_url = clean_text(row.get("source_url")) or None
    return News(
        id=stable_id("news", origin.casefold(), document or content_hash(source_text)),
        source_text=source_text, target_text=target_text,
        language=clean_text(row.get("language") or language).lower(),
        category=clean_text(row.get("category") or "general"),
        source_dataset=origin, source_document_id=document or None, source_url=source_url,
        split="train",
    )


def fingerprints(row: News) -> set[str]:
    return {content_hash(row.source_text), content_hash(row.target_text)}


def intersections(train: Iterable[News], test: Iterable[News]) -> dict[str, int]:
    train, test = list(train), list(test)
    return {
        "ids": len({row.id for row in train} & {row.id for row in test}),
        "texts": len(set().union(*(fingerprints(row) for row in train))
                     & set().union(*(fingerprints(row) for row in test))),
    }


def _unique(rows: Iterable[News], stats: Counter, excluded: Iterable[News] = ()) -> list[News]:
    excluded = list(excluded)
    reserved_ids = {row.id for row in excluded}
    reserved_texts = set().union(*(fingerprints(row) for row in excluded))
    used_ids: set[UUID] = set()
    used_texts: set[str] = set()
    result = []
    for row in rows:
        keys = fingerprints(row)
        if row.id in reserved_ids or keys & reserved_texts:
            stats["excluded_overlap"] += 1
        elif row.id in used_ids or keys & used_texts:
            stats["duplicates"] += 1
        else:
            used_ids.add(row.id)
            used_texts.update(keys)
            result.append(row)
    return result


def prepare_corpus(rows: list[dict], *, train_size: int, test_size: int, seed: int,
                   dataset: str, language: str) -> tuple[list[News], list[News], dict]:
    if train_size < 1 or test_size < 1:
        raise DataError("Размеры учебной и тестовой выборок должны быть положительными.")
    splits = {row.get("split") or "" for row in rows if isinstance(row.get("split"), (str, type(None)))}
    official = bool(splits - {""})
    if official and ("" in splits or not splits <= {"train", "test", "validation"}):
        raise DataError("Укажите train/test/validation во всех строках либо оставьте split пустым во всех строках.")
    groups: dict[str, list[News]] = {"train": [], "test": []}
    stats: Counter = Counter(input_rows=len(rows), invalid=0, duplicates=0, excluded_overlap=0, ignored_validation=0)
    for row in rows:
        label = row.get("split")
        if label is not None and not isinstance(label, str):
            stats["invalid"] += 1
            continue
        split = label or "train"
        if split == "validation":
            stats["ignored_validation"] += 1
            continue
        try:
            groups[split].append(_clean_news(row, dataset, language).model_copy(update={"split": split}))
        except (ValidationError, ValueError):
            stats["invalid"] += 1

    rng = random.Random(seed)
    rng.shuffle(groups["train"])
    if official:
        rng.shuffle(groups["test"])
        test = _unique(groups["test"], stats)[:test_size]
        train = _unique(groups["train"], stats, groups["test"])[:train_size]
    else:
        unique = _unique(groups["train"], stats)
        test = [row.model_copy(update={"split": "test"}) for row in unique[:test_size]]
        train = unique[test_size:test_size + train_size]
    if len(train) < train_size or len(test) < test_size:
        raise DataError(f"После очистки доступно {len(train)}/{train_size} train, {len(test)}/{test_size} test. {dict(stats)}")
    return train, test, {**stats, "split_method": "source_splits" if official else "seeded_holdout",
                         "train": len(train), "test": len(test), **intersections(train, test)}


def evaluation_ids(test: list[News]) -> dict[str, list[str]]:
    return {str(size): [str(row.id) for row in test[:size]] for size in SAMPLE_SIZES if len(test) >= size}

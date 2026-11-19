"""Подготовить учебный корпус и изолированный набор оценки."""

import argparse
import json
import time
from pathlib import Path

import httpx

from scripts.corpus import (
    SEED, TEST_SIZE, TRAIN_SIZE, DataError, evaluation_ids, export_records, file_hash,
    prepare_corpus, read_json, read_records, write_json,
)

XSUM_DATASET = "EdinburghNLP/xsum"
XSUM_REVISION = "7d4d486c2f8ef850b1a11aead99b894ff3dd7da9"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET_URL = f"https://huggingface.co/api/datasets/{XSUM_DATASET}"
PAGE_SIZE = 100
MAX_DOWNLOAD = 32 * 1024 * 1024


def xsum_row_url(split: str, index: int) -> str:
    return str(httpx.URL(ROWS_URL, params={
        "dataset": XSUM_DATASET, "config": "default", "split": split, "offset": index, "length": 1,
    }))


def _download(client: httpx.Client, url: str, limit: int, **params) -> tuple[dict, int]:
    for attempt in range(3):
        try:
            with client.stream("GET", url, params=params) as response:
                response.raise_for_status()
                payload = bytearray()
                for chunk in response.iter_bytes():
                    payload.extend(chunk)
                    if len(payload) > limit:
                        raise DataError("Ответ XSum превысил оставшийся лимит загрузки.")
                data = json.loads(payload)
                if not isinstance(data, dict):
                    raise DataError("Некорректный ответ API XSum.")
                return data, len(payload)
        except httpx.HTTPStatusError as exc:
            if attempt == 2 or exc.response.status_code not in {429, 500, 502, 503, 504}:
                raise DataError(f"API XSum вернул HTTP {exc.response.status_code}.") from None
        except httpx.RequestError as exc:
            if attempt == 2:
                raise DataError(f"XSum недоступен: {type(exc).__name__}.") from None
        except ValueError as exc:
            if isinstance(exc, DataError):
                raise
            raise DataError("Ответ API XSum не является корректным JSON.") from None
        time.sleep(2 ** attempt)
    raise AssertionError("Недостижимое состояние повторной загрузки")


def fetch_xsum(client: httpx.Client, train_size: int, test_size: int,
               revision: str) -> tuple[list[dict], dict]:
    limits = {"train": train_size + max(30, train_size // 4), "test": test_size + max(30, test_size // 2)}
    if train_size < 1 or test_size < 1 or sum(limits.values()) > 4000:
        raise DataError("Для загрузки XSum разрешён диапазон от одной записи до 4000 кандидатов суммарно.")
    before, downloaded = _download(client, DATASET_URL, MAX_DOWNLOAD)
    if before.get("sha") != revision:
        raise DataError("Версия XSum на Hub отличается от запрошенной. Используйте локальный снимок или явно укажите новую ревизию.")

    rows = []
    for split, count in limits.items():
        for offset in range(0, count, PAGE_SIZE):
            length = min(PAGE_SIZE, count - offset)
            page, size = _download(
                client, ROWS_URL, MAX_DOWNLOAD - downloaded,
                dataset=XSUM_DATASET, config="default", split=split, offset=offset, length=length,
            )
            downloaded += size
            try:
                if page.get("partial") or len(page["rows"]) != length:
                    raise DataError(f"Неполная страница XSum {split} ({offset}).")
                for index, item in enumerate(page["rows"], offset):
                    row = item["row"]
                    if (item["row_idx"] != index or item.get("truncated_cells")
                            or not isinstance(row["document"], str) or not isinstance(row["summary"], str)
                            or not isinstance(row["id"], str) or not row["id"].isascii() or not row["id"].isdigit()):
                        raise DataError(f"Неполная или некорректная строка XSum {split} ({index}).")
                    rows.append({"document": row["document"], "summary": row["summary"],
                                 "id": row["id"], "split": split, "row_index": index,
                                 "language": "en", "source_dataset": XSUM_DATASET,
                                 "source_url": xsum_row_url(split, index)})
            except (KeyError, TypeError):
                raise DataError(f"Изменилась схема API XSum {split} ({offset}).") from None

    after, size = _download(client, DATASET_URL, MAX_DOWNLOAD - downloaded)
    if after.get("sha") != revision:
        raise DataError("Версия XSum изменилась во время загрузки.")
    return rows, {"kind": "huggingface_rows", "dataset": XSUM_DATASET, "revision": revision,
                  "endpoint": ROWS_URL, "row_limits": limits, "response_bytes": downloaded + size,
                  "license_metadata": after.get("cardData", {}).get("license")}


def generate(data_dir: Path, rows: list[dict], provenance: dict, *, train_size: int = TRAIN_SIZE,
             test_size: int = TEST_SIZE, seed: int = SEED, dataset: str = "", language: str = "en") -> dict:
    settings = dict(train_size=train_size, test_size=test_size, seed=seed, dataset=dataset, language=language)
    train, test, stats = prepare_corpus(rows, **settings)
    snapshot = data_dir / "raw/news_candidates.json"
    write_json(snapshot, {"provenance": provenance, "rows": rows})
    export_records(data_dir / "import/news_train", train)
    export_records(data_dir / "test/news_test", test)
    write_json(data_dir / "test/evaluation_ids.json", evaluation_ids(test))
    manifest = {"version": 1, "source": provenance, "snapshot_sha256": file_hash(snapshot),
                "settings": settings, "statistics": stats}
    write_json(data_dir / "import/manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="Локальный CSV, JSON или табличный TXT/TSV")
    source.add_argument("--snapshot", type=Path, help="Локальный снимок news_candidates.json для повтора без сети")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--source-dataset", default="", help="Имя локального источника, если его нет в строках")
    parser.add_argument("--language", default="en")
    parser.add_argument("--train-count", type=int, default=TRAIN_SIZE)
    parser.add_argument("--test-count", type=int, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--revision", default=XSUM_REVISION)
    args = parser.parse_args()
    try:
        dataset, language = args.source_dataset, args.language
        if args.input:
            rows = read_records(args.input)
            provenance = {"kind": "local_file", "filename": args.input.name, "sha256": file_hash(args.input)}
        elif args.snapshot:
            snapshot = read_json(args.snapshot)
            rows, provenance = snapshot["rows"], snapshot["provenance"]
            if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows) or not isinstance(provenance, dict):
                raise DataError("Некорректный формат снимка данных.")
        else:
            with httpx.Client(timeout=30, follow_redirects=True) as client:
                print("XSum: выборочная загрузка официальных train/test, предел 32 МиБ.", flush=True)
                rows, provenance = fetch_xsum(client, args.train_count, args.test_count, args.revision)
            dataset, language = XSUM_DATASET, "en"
        result = generate(args.output_dir, rows, provenance, train_size=args.train_count, test_size=args.test_count,
                          seed=args.seed, dataset=dataset, language=language)
    except (DataError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Ошибка подготовки данных: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

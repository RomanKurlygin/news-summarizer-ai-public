import csv
from pathlib import Path

import httpx
import pytest

from scripts.build_glossary import extract_terms
from scripts.corpus import (
    DataError, News, Term, clean_text, evaluation_ids, export_records, file_hash, intersections,
    load_prepared, prepare_corpus, read_json, read_records, write_json,
)
from scripts.generate_data import DATASET_URL, ROWS_URL, XSUM_REVISION, fetch_xsum, generate
from scripts.verify_dataset import verify_dataset


def examples(size=8):
    # Синтетическая фикстура только для офлайн-тестов алгоритмов.
    return [{"id": str(i), "document": f"Учебный текст номер {i} для проверки корпуса.",
             "summary": f"Тестовое резюме номер {i}.", "category": "test-fixture"} for i in range(size)]


def prepare(rows, **kwargs):
    return prepare_corpus(rows, **{"train_size": 3, "test_size": 2, "seed": 42,
                                   "dataset": "unit-fixture", "language": "ru", **kwargs})


def test_cleanup_normalizes_unicode_and_whitespace():
    assert clean_text(" \ufeffТекст\nс\tномером １２\u0000. ") == "Текст с номером 12."
    assert clean_text("A\ud800B") == "AB"
    assert clean_text("\t!?\u200b") == ""
    assert clean_text(123) == ""


def test_invalid_rows_and_duplicates_removed_before_selection():
    rows = examples(5)
    rows += [{**rows[0], "id": "another", "document": rows[0]["document"].upper().replace(" ", "\t")},
             {**rows[1], "summary": "Alternate with same ID"}, {**rows[3], "summary": None},
             {**rows[3], "document": ""}]
    train, test, stats = prepare(rows)
    assert len(train) == 3 and len(test) == 2
    assert stats["invalid"] == 2 and stats["duplicates"] == 2


def test_same_seed_reproduces_ids_and_a_different_seed_changes_membership():
    rows = examples(140)
    train, test, _ = prepare(rows, train_size=20, test_size=100)
    assert prepare(rows, train_size=20, test_size=100)[:2] == (train, test)
    assert prepare(rows, train_size=20, test_size=100, seed=7)[1] != test
    nested = evaluation_ids(test)
    assert len(nested["100"]) == 100
    assert nested["20"] == nested["50"][:20] == nested["100"][:20]
    assert nested["50"] == nested["100"][:50]
    assert len(set(nested["100"])) == 100
    assert not any(intersections(train, test).values())


def test_source_splits_hold_out_test_including_unselected_records():
    rows = [{**row, "split": "train"} for row in examples(5)]
    holdout = [{"id": "external-1", "document": "A held-out news document.", "summary": "Separate abstract.", "split": "test"},
               {**rows[2], "id": "external-2", "split": "test"}]
    rows[4]["summary"] = holdout[0]["document"].upper()
    train, test, stats = prepare(rows + holdout + [{**rows[0], "split": "validation"}], test_size=1)
    assert {row.source_document_id for row in train} == {"0", "1", "3"}
    assert all(row.split == "test" and row.source_document_id.startswith("external") for row in test)
    assert stats["excluded_overlap"] == 2
    assert stats["ignored_validation"] == 1
    assert stats["split_method"] == "source_splits"


def test_overlap_cross_checks_summary_article_and_uuid():
    train, test, _ = prepare(examples())
    test[0] = test[0].model_copy(update={"id": train[0].id, "target_text": train[0].source_text.upper()})
    assert intersections(train, test) == {"ids": 1, "texts": 1}


def test_rejects_mixed_splits_and_insufficient_size():
    rows = examples(4)
    with pytest.raises(DataError, match="После очистки"):
        prepare(rows)
    rows[0]["split"] = "test"
    with pytest.raises(DataError, match="Укажите train"):
        prepare(rows)


@pytest.mark.parametrize(("suffix", "content"), [
    (".json", '[{"unclosed": 1}'), (".json", '{}'), (".json", '[true]'),
    (".csv", 'id,id\n0,1\n'), (".csv", 'id,document\n0\n'),
    (".csv", 'id,document\n0,"bad'), (".csv", 'id,document\n0,x,y\n'),
])
def test_malformed_csv_json(tmp_path, suffix, content):
    path = tmp_path / f"bad{suffix}"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DataError):
        read_records(path)


@pytest.mark.parametrize(("suffix", "delimiter"), [(".csv", ","), (".txt", "\t")])
def test_table_preserves_quoted_multiline_content(tmp_path, suffix, delimiter):
    path = tmp_path / f"news{suffix}"
    fields = ["id", "document", "summary"]
    row = {"id": "7", "document": "Текст, включая\nновую строку", "summary": "Сводка"}
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerow(row)
    assert read_records(path) == [row]


def test_glossary_uses_source_evidence_and_attested_aliases():
    raw = {"id": "only", "document": "Dr. Ada Lovelace spoke in the city of London. The Regional Health Board (RHB) "
           "held a public health meeting. The council told residents to stay informed in Manchester.",
           "summary": "The Government Communications Office (GCO) is a summary-only term."}
    train, test, _ = prepare([{**examples(1)[0], "split": "test"}, {**raw, "split": "train"}],
                             train_size=1, test_size=1, language="en")
    items = {row.term: row for row in extract_terms(train, 60)}
    assert items["Ada Lovelace"].category == "person"
    assert items["Regional Health Board"].category == "organization"
    assert items["Regional Health Board"].aliases == ["RHB"]
    assert items["London"].category == "place"
    assert items["Manchester"].review_required
    assert items["public health"].category == "topic"
    assert all(item.definition in train[0].source_text and item.source.news_id == train[0].id for item in items.values())
    assert "Government Communications Office" not in items
    with pytest.raises(DataError, match="только записи train"):
        extract_terms(test)


@pytest.fixture
def prepared_dir(tmp_path):
    source = tmp_path / "real_input_for_test.json"
    rows = [{**row, "document": f"The City Council presented a public health plan, report number {i}.",
             "language": "en"} for i, row in enumerate(examples(7))]
    write_json(source, rows)
    out = tmp_path / "data"
    generate(out, read_records(source), {"kind": "local_file", "filename": source.name, "sha256": file_hash(source)},
             train_size=4, test_size=3, dataset="unit-fixture")
    terms = extract_terms(load_prepared(out / "import/news_train.json", News), 2)
    export_records(out / "import/glossary", terms)
    return out


def verify_small(path: Path):
    return verify_dataset(path, min_train=4, min_test=3, min_glossary=2)


def test_file_verification_replays_source_and_reads_both_formats(prepared_dir):
    corpus = verify_small(prepared_dir)
    assert corpus.report["checks"] == "passed"
    assert corpus.report["source"]["kind"] == "local_file"
    assert corpus.report["corpus"]["ids"] == corpus.report["corpus"]["texts"] == 0
    assert load_prepared(prepared_dir / "import/glossary.csv", Term) == corpus.terms


@pytest.mark.parametrize("fault", ["snapshot", "duplicate", "overlap", "fake_evidence", "ids", "csv"])
def test_verifier_rejects_data_tampering(prepared_dir, fault):
    root = prepared_dir
    files = {"snapshot": "raw/news_candidates.json", "duplicate": "import/news_train.json",
             "overlap": "test/news_test.json", "fake_evidence": "import/glossary.json", "ids": "test/evaluation_ids.json"}
    if fault == "csv":
        path = root / "import/news_train.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("City Council", "County Council", 1), encoding="utf-8")
    else:
        path = root / files[fault]
        data = read_json(path)
        if fault == "snapshot":
            data["rows"][0]["document"] = "Changed upstream record."
        elif fault == "duplicate":
            data[1] = data[0]
        elif fault == "overlap":
            data[0]["source_text"] = read_json(root / "import/news_train.json")[0]["target_text"]
        elif fault == "fake_evidence":
            data[0]["definition"] = "Unsupported explanation."
        else:
            data["20"] = ["fake"]
        write_json(path, data)
    with pytest.raises(DataError):
        verify_small(root)


@pytest.mark.parametrize("fault", [None, "wrong_revision", "partial", "truncated"])
def test_xsum_fetch_uses_bounded_pages_and_checks_completeness(fault, monkeypatch):
    monkeypatch.setattr("scripts.generate_data.PAGE_SIZE", 20)
    requested = []

    def respond(request):
        url = str(request.url).split("?")[0]
        if url == DATASET_URL:
            return httpx.Response(200, json={"sha": "different" if fault == "wrong_revision" else XSUM_REVISION})
        assert url == ROWS_URL
        params = request.url.params
        offset, length = int(params["offset"]), int(params["length"])
        requested.append((params["split"], offset, length))
        return httpx.Response(200, json={"partial": fault == "partial", "rows": [
            {"row_idx": idx, "truncated_cells": ["document"] if fault == "truncated" else [],
             "row": {"id": str(idx), "document": "Article", "summary": "Reference"}}
            for idx in range(offset, offset + length)
        ]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        if fault:
            with pytest.raises(DataError):
                fetch_xsum(client, 2, 2, XSUM_REVISION)
        else:
            rows, source = fetch_xsum(client, 2, 2, XSUM_REVISION)
            assert requested == [("train", 0, 20), ("train", 20, 12), ("test", 0, 20), ("test", 20, 12)]
            assert len(rows) == 64 and source["row_limits"] == {"train": 32, "test": 32}
            assert source["response_bytes"] > 0

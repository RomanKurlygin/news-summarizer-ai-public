"""Лексический поиск BM25 с JSON-снимком документов."""

import json
import re
from collections.abc import Iterable
from pathlib import Path
from uuid import UUID

from rank_bm25 import BM25Okapi

from backend.retrieval.documents import Document, Hit, IndexIntegrityError
from backend.utils.corpus_text import content_hash, normalized

_TOKENS = re.compile(r"[^\W_]+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKENS.findall(normalized(text))


class BM25Index:
    def __init__(self, documents: Iterable[Document] = ()) -> None:
        self.rebuild(documents)

    def rebuild(self, documents: Iterable[Document]) -> None:
        rows = sorted(documents, key=lambda doc: doc.id)
        self._install(rows, [tokenize(doc.source_text) for doc in rows])

    def _install(self, rows: list[Document], tokens: list[list[str]]) -> None:
        by_id = {doc.id: doc for doc in rows}
        if len(by_id) != len(rows):
            raise IndexIntegrityError("Индекс BM25 получил повторяющиеся UUID.")
        if len(tokens) != len(rows) or any(not terms for terms in tokens):
            raise IndexIntegrityError("В документе BM25 отсутствуют токены.")
        self._engine = BM25Okapi(tokens) if rows else None
        self._by_id = by_id
        self._rows = tuple(rows)
        self._tokens = tokens
        self._terms = tuple(map(set, tokens))
        self._hashes = tuple(doc.hashes() for doc in rows)

    @property
    def documents(self) -> tuple[Document, ...]:
        return self._rows

    def get(self, document_id: UUID) -> Document | None:
        return self._by_id.get(document_id)

    def search(self, query: str, top_k: int = 5, *, language: str | None = None,
               category: str | None = None, exclude_self: bool = False) -> list[Hit]:
        if top_k < 1:
            raise ValueError("top_k должен быть положительным.")
        terms = tokenize(query)
        if not terms or self._engine is None:
            return []
        scores = self._engine.get_scores(terms)
        query_tokens = set(terms)
        excluded = content_hash(query) if exclude_self else None
        hits = [
            Hit(doc.id, float(score))
            for doc, tokens, hashes, score in zip(self._rows, self._terms, self._hashes, scores, strict=True)
            if query_tokens.intersection(tokens)
            and excluded not in hashes
            and (language is None or doc.language == language)
            and (category is None or doc.category == category)
        ]
        return sorted(hits, key=lambda hit: (-hit.score, hit.id))[:top_k]

    def save(self, path: Path) -> None:
        data = {"format": 2, "documents": [doc.model_dump(mode="json") for doc in self._rows], "tokens": self._tokens}
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            tokens = data["tokens"]
            if (data["format"] != 2 or not isinstance(data["documents"], list) or not isinstance(tokens, list)
                    or any(not isinstance(row, list) or not all(isinstance(token, str) and token for token in row)
                           for row in tokens)):
                raise ValueError("Неподдерживаемый формат BM25.")
            restored = cls()
            restored._install([Document.model_validate(row) for row in data["documents"]], tokens)
            return restored
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise IndexIntegrityError("Не удалось прочитать снимок BM25.") from exc

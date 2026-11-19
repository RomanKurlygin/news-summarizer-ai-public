"""Общие документы и правила изоляции контрольной выборки."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.utils.corpus_text import clean_text, content_hash


class IndexIntegrityError(RuntimeError):
    """Нарушена согласованность или изоляция индекса."""


class Document(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UUID
    source_text: str
    target_text: str
    language: str = Field(min_length=1)
    category: str = Field(min_length=1)
    source_dataset: str | None = None
    source_document_id: str | None = None
    source_url: str | None = None
    source_split: Literal["train", "test"] | None = None

    @field_validator("source_text", "target_text")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not clean_text(value):
            raise ValueError("Новость и эталон должны содержать текст.")
        return value

    def hashes(self) -> tuple[str, str]:
        return content_hash(self.source_text), content_hash(self.target_text)


@dataclass(frozen=True)
class Hit:
    id: UUID
    score: float


@dataclass(frozen=True)
class Holdout:
    ids: frozenset[UUID]
    hashes: frozenset[str]

    @classmethod
    def from_documents(cls, documents: Iterable[Document]) -> "Holdout":
        rows = list(documents)
        if not rows or any(doc.source_split != "test" for doc in rows):
            raise IndexIntegrityError("Требуется непустой контрольный набор с пометкой test.")
        ids = frozenset(doc.id for doc in rows)
        hashes = frozenset(digest for doc in rows for digest in doc.hashes())
        if len(ids) != len(rows) or len(hashes) != len(rows) * 2:
            raise IndexIntegrityError("Контрольный набор содержит дубликаты.")
        return cls(ids, hashes)

    def validate(self, documents: Iterable[Document]) -> None:
        for doc in documents:
            if doc.id in self.ids or self.hashes.intersection(doc.hashes()):
                raise IndexIntegrityError(f"Найдено пересечение с test для документа {doc.id}.")
            if doc.source_split != "train":
                raise IndexIntegrityError(f"Документ {doc.id} не помечен как train.")

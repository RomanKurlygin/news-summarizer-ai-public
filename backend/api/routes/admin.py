"""Управление базой знаний и справочником с ролью администратора."""

import asyncio
import csv
import io
import json
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.api.deps import DBSession, require_admin
from backend.api.schemas.summarization import Category
from backend.db.models import Glossary, KnowledgeBase
from backend.retrieval.documents import Document, IndexIntegrityError
from backend.utils.corpus_text import content_hash

router = APIRouter(prefix="/api/v1/admin", tags=["Администрирование"],
                   dependencies=[Depends(require_admin)])
Resource = Literal["knowledge-base", "glossary"]
Language = Annotated[str, StringConstraints(pattern=r"^[a-z]{2,3}(?:-[a-z0-9]{2,8})*$", max_length=16)]
EntryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=16000)]
TermText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]


class KnowledgeEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)
    source_text: EntryText
    target_text: EntryText
    category: Category = "general"
    language: Language = "und"


class GlossaryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)
    term: TermText
    definition: EntryText
    category: Category = "general"
    language: Language = "und"
    aliases: list[TermText] | None = Field(default=None, max_length=25)


class KnowledgeView(KnowledgeEntry):
    id: UUID


class GlossaryView(GlossaryEntry):
    id: UUID


Entry = KnowledgeEntry | GlossaryEntry
EntryView = KnowledgeView | GlossaryView
_TABLES = {"knowledge-base": KnowledgeBase, "glossary": Glossary}
_TYPES = {"knowledge-base": KnowledgeEntry, "glossary": GlossaryEntry}


def _view(row: KnowledgeBase | Glossary) -> EntryView:
    schema = KnowledgeView if isinstance(row, KnowledgeBase) else GlossaryView
    return schema.model_validate(row)


async def _values(resource: Resource, entries: list[Entry], request: Request) -> list[dict]:
    if any(not isinstance(entry, _TYPES[resource]) for entry in entries):
        raise HTTPException(422, "Поля записи не соответствуют ресурсу.")
    values = [entry.model_dump() for entry in entries]
    if resource == "knowledge-base":
        try:
            from scripts.build_indexes import load_holdout

            holdout = await asyncio.to_thread(load_holdout, request.app.state.settings.holdout_path)
        except (ImportError, OSError, ValueError, IndexIntegrityError):
            raise HTTPException(503, "Контрольный набор недоступен для проверки импорта.") from None
        try:
            holdout.validate(Document(id=uuid4(), source_split="train", **item) for item in values)
        except (ValidationError, IndexIntegrityError):
            raise HTTPException(422, "Некорректная обучающая запись или пересечение с контрольными данными.") from None
        return [item | {"source_split": "train", "content_hash": content_hash(item["source_text"])}
                for item in values]
    return values


async def _get(resource: Resource, row_id: UUID, session: DBSession):
    row = await session.get(_TABLES[resource], row_id)
    if row is None:
        raise HTTPException(404, "Запись не найдена.")
    return row


@router.get("/{resource}")
async def list_entries(
    resource: Resource, session: DBSession, limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[EntryView]:
    table = _TABLES[resource]
    rows = await session.scalars(select(table).order_by(table.created_at.desc(), table.id).offset(offset).limit(limit))
    return [_view(row) for row in rows]


@router.post("/{resource}", status_code=201)
async def create_entry(resource: Resource, entry: Entry, request: Request, session: DBSession) -> EntryView:
    values, = await _values(resource, [entry], request)
    row = _TABLES[resource](**values)
    try:
        async with session.begin():
            session.add(row)
            await session.flush()
    except IntegrityError:
        raise HTTPException(409, "Запись уже существует или нарушает ограничения базы.") from None
    return _view(row)


@router.post("/{resource}/import", status_code=201)
async def import_entries(
    resource: Resource, request: Request, session: DBSession,
    data: Annotated[bytes, Body(media_type="text/csv")],
) -> dict[str, int]:
    if len(data) > 256 * 1024:
        raise HTTPException(413, "CSV превышает 256 КиБ.")
    try:
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Некорректный заголовок CSV.")
        entries = []
        for row in reader:
            if len(entries) >= 100 or None in row or None in row.values():
                raise ValueError("Некорректные строки или превышен лимит.")
            if resource == "glossary" and "aliases" in row:
                row["aliases"] = json.loads(row["aliases"]) if row["aliases"] else None
            entries.append(_TYPES[resource].model_validate(row))
        if not entries:
            raise ValueError("Нет записей.")
    except (ValueError, csv.Error):
        raise HTTPException(422, "Некорректный CSV или больше 100 записей.") from None
    values = await _values(resource, entries, request)
    try:
        async with session.begin():
            session.add_all([_TABLES[resource](**item) for item in values])
    except IntegrityError:
        raise HTTPException(409, "Конфликт записей. CSV не импортирован.") from None
    return {"imported": len(values)}


@router.get("/{resource}/{row_id}")
async def get_entry(resource: Resource, row_id: UUID, session: DBSession) -> EntryView:
    return _view(await _get(resource, row_id, session))


@router.put("/{resource}/{row_id}")
async def update_entry(
    resource: Resource, row_id: UUID, entry: Entry, request: Request, session: DBSession,
) -> EntryView:
    values, = await _values(resource, [entry], request)
    try:
        async with session.begin():
            row = await _get(resource, row_id, session)
            for field, value in values.items():
                setattr(row, field, value)
    except IntegrityError:
        raise HTTPException(409, "Запись уже существует или нарушает ограничения базы.") from None
    return _view(row)


@router.delete("/{resource}/{row_id}", status_code=204)
async def delete_entry(resource: Resource, row_id: UUID, session: DBSession) -> Response:
    async with session.begin():
        await session.delete(await _get(resource, row_id, session))
    return Response(status_code=204)

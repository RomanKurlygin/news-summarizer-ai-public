"""Транзакционный импорт учебных записей через общие модели проекта."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.db.models import Glossary, KnowledgeBase
from scripts.corpus import DataError, News, Term, content_hash, fingerprints, intersections, normalized


def _origin(dataset: str | None, document: str | None) -> tuple[str, str] | None:
    return (dataset.casefold(), document) if dataset and document else None


async def import_corpus(sessions: async_sessionmaker[AsyncSession], train: list[News], terms: list[Term],
                        test: list[News], *, dry_run: bool = False) -> dict[str, int]:
    train_ids = {row.id for row in train}
    if any(row.split != "train" for row in train) or any(row.split != "test" for row in test):
        raise DataError("Импорт требует отдельные учебный и тестовый разделы.")
    if any(intersections(train, test).values()) or any(term.source.news_id not in train_ids for term in terms):
        raise DataError("Импорт заблокирован из-за пересечения с test или внешней ссылки справочника.")
    if (len(train_ids) != len(train) or len({content_hash(row.source_text) for row in train}) != len(train)
            or len({(row.language, normalized(row.term)) for row in terms}) != len(terms)
            or len({row.id for row in terms}) != len(terms)):
        raise DataError("Повторяющиеся новости или термины во входном корпусе.")

    test_ids = {row.id for row in test}
    test_texts = {key for row in test for key in fingerprints(row)}
    test_origins = {_origin(row.source_dataset, row.source_document_id) for row in test} - {None}
    async with sessions.begin() as session:
        existing_news = (await session.scalars(select(KnowledgeBase))).all()
        existing_terms = (await session.scalars(select(Glossary))).all()
        by_id = {row.id: row for row in existing_news}
        by_content = {content_hash(row.source_text): row for row in existing_news}
        by_origin = {_origin(row.source_dataset, row.source_document_id): row for row in existing_news}
        if any(
            row.id in test_ids or _origin(row.source_dataset, row.source_document_id) in test_origins
            or content_hash(row.source_text) in test_texts or content_hash(row.target_text) in test_texts
            for row in existing_news
        ):
            raise DataError("В существующей базе знаний уже найдены совпадения с test.")

        new_news = []
        for row in train:
            digest = content_hash(row.source_text)
            origin = _origin(row.source_dataset, row.source_document_id)
            matches = [item for item in (by_id.get(row.id), by_content.get(digest), by_origin.get(origin) if origin else None)
                       if item is not None]
            if matches:
                if any(content_hash(item.source_text) != digest
                       or normalized(item.target_text) != normalized(row.target_text) for item in matches):
                    raise DataError(f"Конфликт существующей новости с записью {row.id}.")
                continue
            new_news.append(KnowledgeBase(
                id=row.id, source_text=row.source_text, target_text=row.target_text,
                language=row.language, category=row.category, source_url=row.source_url,
                source_dataset=row.source_dataset, source_document_id=row.source_document_id,
                source_split=row.split, content_hash=digest,
            ))

        term_ids = {row.id: row for row in existing_terms}
        term_keys = {(row.language, normalized(row.term)) for row in existing_terms}
        new_terms = []
        for row in terms:
            key = (row.language, normalized(row.term))
            existing = term_ids.get(row.id)
            if existing is not None and (existing.language, normalized(existing.term)) != key:
                raise DataError(f"Конфликт существующего термина с записью {row.id}.")
            if key in term_keys:
                continue
            new_terms.append(Glossary(
                id=row.id, term=row.term, definition=row.definition, language=row.language,
                category=row.category, aliases=row.aliases, source=row.source.model_dump(mode="json"),
                review_required=row.review_required,
            ))
        if not dry_run:
            session.add_all([*new_news, *new_terms])

    inserted_news, inserted_glossary = len(new_news), len(new_terms)
    return {
        "inserted_news": inserted_news, "inserted_glossary": inserted_glossary,
        "skipped_news": len(train) - inserted_news, "skipped_glossary": len(terms) - inserted_glossary,
        "total_news": len(existing_news) + (0 if dry_run else inserted_news),
        "total_glossary": len(existing_terms) + (0 if dry_run else inserted_glossary),
    }

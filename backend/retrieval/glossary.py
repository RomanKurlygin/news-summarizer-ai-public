"""Контекст справочника для терминов, встречающихся в самой новости."""

import re
from dataclasses import dataclass

from sqlalchemy import select

from backend.agents.state import AgentState
from backend.db.models import Glossary
from backend.utils.corpus_text import content_hash


@dataclass(frozen=True)
class GlossaryTerm:
    term: str
    definition: str
    aliases: tuple[str, ...]
    language: str


async def glossary_snapshot(sessions) -> tuple[GlossaryTerm, ...]:
    async with sessions() as session:
        rows = await session.scalars(select(Glossary).where(Glossary.review_required.is_(False))
                                     .order_by(Glossary.term, Glossary.language, Glossary.category, Glossary.id))
        return tuple(GlossaryTerm(row.term, row.definition, tuple(row.aliases or ()), row.language) for row in rows)


def glossary_context(state: AgentState, terms: tuple[GlossaryTerm, ...], *, language: str | None = None,
                     excluded_hashes: frozenset[str] = frozenset()) -> AgentState:
    text = " ".join(state.original_text.casefold().split())
    context = []
    for row in terms:
        if language and row.language not in {language, "und"}:
            continue
        variants = (row.term, *row.aliases)
        if not any(re.search(r"(?<!\w)" + re.escape(" ".join(term.casefold().split())) + r"(?!\w)", text)
                   for term in variants if term.strip()):
            continue
        # Включаем только проверенные записи и исключаем совпадения с контрольными парами.
        if any(content_hash(value) in excluded_hashes for value in (*variants, row.definition)):
            continue
        entry = " | ".join(variants[:4]) + " — " + row.definition
        if len(entry) <= 240 and len(entry) + sum(map(len, context)) <= 800:
            context.append(entry)
        if len(context) >= 6:
            break
    return state.model_copy(update={"glossary_context": context})

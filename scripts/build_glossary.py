"""Извлечь справочные записи только из подготовленных учебных статей."""

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from scripts.corpus import GLOSSARY_SIZE, DataError, News, Term, TermSource, export_records, load_prepared, normalized, stable_id

_NAME = r"(?:[A-Z][a-z]+(?:[-'’][A-Z]?[a-z]+)*|[A-Z]{2,12})"
_NAMES = rf"{_NAME}(?:(?:\s+(?:of|the|and|for)\s+|\s+){_NAME}){{0,5}}"
_ENTITY = re.compile(rf"(?<![\w'’-]){_NAMES}(?![\w'’-])")
_ORG = re.compile(r"\b(?:Association|Authority|Bank|Board|Centre|Commission|Committee|Company|Confederation|"
                  r"Corporation|Council|Department|Federation|Fund|Group|Hospital|Institute|League|Ministry|Museum|"
                  r"Party|Service|Trust|Union|University|Ltd|PLC|Foundation|Society)\b")
_PLACE = re.compile(r"\b(?:Island|Islands|County|Valley|River|Province|Mount|Sea|Bay|District|Borough)\b")
_EVENT = re.compile(r"\b(?:Cup|Championships?|Elections?|Trophy)$")
_PERSON_CUE = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Professor|Sir|Dame|President|Minister|Senator|Mayor|"
                         r"Captain|Prince|Princess|King|Queen|Judge)\.?$")
_GEO_CUE = re.compile(r"\b(?:city of|town of|village of|county of|capital of|born in|based in|north of|south of)$", re.I)
_PLACE_PREP = re.compile(r"\b(?:in|from|across|near|around)$", re.I)
_REPORTING = re.compile(r"^\s+(?:said|told|added|wrote|won|played|died|announced|scored)\b", re.I)
_ACRONYM = re.compile(r"^\s*\(([A-Z]{2,12})\)")
_TITLE = re.compile(r"^(?:Mr|Mrs|Ms|Dr|Prof|Professor|Sir|Dame|President|Minister|Senator|Mayor|Captain|"
                    r"Prince|Princess|King|Queen|Judge)\s+")
_STOP = set("a an as at bbc but by he her his how i if in it its mr mrs ms my new no not our she so that the "
            "their them then there these they this those to we what when where which who why will with you "
            "your monday tuesday wednesday thursday friday saturday sunday january february march april may "
            "june july august september october november december".split())
# Словарь задаёт только шаблоны поиска. Контекст каждой записи берётся из статьи.
_TOPIC = re.compile(r"\b(?:air pollution|asylum seekers?|bankruptcy|benefit cuts|biodiversity|budget deficit|"
                    r"carbon emissions|ceasefire|child protection|climate change|coalition government|corruption|"
                    r"cyber attack|data protection|dementia|domestic violence|economic growth|election campaign|"
                    r"emergency services|energy prices|evacuation|flood warning|foreign policy|fuel prices|"
                    r"general election|genetic testing|greenhouse gases|hate crime|health care|heart disease|"
                    r"human rights|income tax|industrial action|inflation|infectious diseases?|interest rates|"
                    r"judicial review|legal aid|life expectancy|measles|mental health|minimum wage|monetary policy|"
                    r"mortgage rates|national debt|natural gas|nuclear power|oil prices|organ donation|"
                    r"peace talks|pension reform|planning permission|police investigation|public health|"
                    r"public spending|public transport|recession|recycling|redundancy|renewable energy|"
                    r"road safety|school funding|social care|social housing|social media|space exploration|"
                    r"state pension|tax evasion|trade unions?|unemployment|vaccination|voter turnout|"
                    r"welfare reform|wildlife conservation|world heritage|zero hour contracts?)\b", re.I)


@dataclass
class _Candidate:
    term: str
    category: str
    evidence: str
    news_id: UUID
    method: str
    review: bool
    documents: set[UUID] = field(default_factory=set)
    aliases: set[str] = field(default_factory=set)


def _evidence(text: str, start: int, end: int) -> str:
    left = text.rfind(" ", max(0, start - 85), max(0, start - 65)) + 1 if start > 85 else 0
    if start > 85 and left == 0:
        left = start
    right = text.find(" ", min(len(text), end + 110), min(len(text), end + 130))
    if right < 0:
        right = min(len(text), end + 110)
    return text[left:right].strip()


def _entities(text: str):
    for match in _ENTITY.finditer(text):
        term = match.group()
        start, end = match.span()
        title = _TITLE.match(term)
        if title:
            start += title.end()
            term = text[start:end]
        leading_stop = re.match(r"[A-Z][a-z]+\s+(?:the\s+)?", term)
        if leading_stop and leading_stop[0].split()[0].casefold() in _STOP:
            start += leading_stop.end()
            term = text[start:end]
        if normalized(term) in _STOP or len(term) < 3:
            continue
        before, after = text[max(0, start - 48):start].rstrip(), text[end:end + 40]
        acronym = _ACRONYM.match(after)
        initials = [word[0].upper() for word in term.split()]
        meaningful = [word[0].upper() for word in term.split() if word.casefold() not in {"of", "the", "and", "for"}]
        alias = (acronym[1] if acronym and len(initials) > 1
                 and acronym[1] in {"".join(initials), "".join(meaningful)} else None)
        organization = _ORG.search(term)
        location = _PLACE.search(term)
        if alias or (organization and not _ORG.fullmatch(term) and not _EVENT.search(term)):
            review = bool(not alias and organization and " and " in term and organization.end() < len(term))
            category, method = "organization", "expansion" if alias else "organization_marker"
        elif location and not _PLACE.fullmatch(term):
            if location.start() and location.end() < len(term):
                end = start + location.end()
                term = text[start:end]
            category, method, review = "place", "geographic_marker", False
        elif _GEO_CUE.search(before):
            category, method, review = "place", "geographic_marker", " and " in term or len(term.split()) > 3
        elif _PERSON_CUE.search(before) and any(word.isalpha() for word in term.split()):
            possessive = re.search(r"['’]s\b", term)
            if possessive:
                end = start + possessive.start()
                term = text[start:end]
            category, method, review = "person", "personal_title", len(term.split()) != 2
        elif _REPORTING.match(after) and 2 <= len(term.split()) <= 3:
            category, method, review = "person", "reporting_context", True
        elif _PLACE_PREP.search(before) and not term.isupper() and len(term.split()) <= 3:
            category, method, review = "place", "location_context", True
        elif term.isupper() and re.search(r"\bthe$", before, re.I):
            category, method, review = "organization", "acronym_context", True
        else:
            continue
        yield term, category, method, review, alias, start, end


def extract_terms(train: list[News], limit: int = GLOSSARY_SIZE) -> list[Term]:
    if limit < 1 or any(row.split != "train" for row in train):
        raise DataError("Справочник требует положительный лимит и только записи train.")
    candidates: dict[tuple[str, str, str], _Candidate] = {}
    for row in train:
        text = row.source_text
        matches = list(_entities(text))
        matches.extend((match[0], "topic", "news_term_pattern", False, None, *match.span())
                       for match in _TOPIC.finditer(text))
        for term, category, method, review, alias, start, end in matches:
            key = row.language, normalized(term), category
            prior = candidates.get(key)
            if prior is None or (prior.review and not review):
                current = _Candidate(term, category, _evidence(text, start, end), row.id, method, review)
                if prior:
                    current.documents = prior.documents
                    current.aliases = prior.aliases
                candidates[key] = current
            else:
                current = prior
            current.documents.add(row.id)
            if alias:
                current.aliases.add(alias)

    def rank(item):
        key, candidate = item
        return candidate.review, -len(candidate.documents), key

    ranked = {}
    for key, candidate in sorted(candidates.items(), key=rank):
        ranked.setdefault(key[:2], candidate)

    groups = defaultdict(list)
    for key, candidate in ranked.items():
        groups[candidate.category].append(key)
    quota = min(25, limit // 4)
    reserve = {key for group in groups.values() for key in group[:quota]}
    reserve.update([key for key in ranked if key not in reserve][:limit - len(reserve)])

    return [Term(
        id=stable_id("term", language, term_key), term=candidate.term, category=candidate.category,
        language=language, definition=candidate.evidence, aliases=sorted(candidate.aliases),
        source=TermSource(news_id=candidate.news_id, method=candidate.method, document_count=len(candidate.documents)),
        review_required=candidate.review,
    ) for (language, term_key), candidate in ranked.items() if (language, term_key) in reserve]


def glossary_stats(terms: list[Term]) -> dict:
    return {"glossary": len(terms), "categories": dict(Counter(term.category for term in terms)),
            "with_aliases": sum(bool(term.aliases) for term in terms),
            "review_required": sum(term.review_required for term in terms)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--count", type=int, default=GLOSSARY_SIZE)
    args = parser.parse_args()
    try:
        terms = extract_terms(load_prepared(args.data_dir / "import/news_train.json", News), args.count)
        if len(terms) < args.count:
            raise DataError(f"Найдено {len(terms)} из {args.count} записей.")
        export_records(args.data_dir / "import/glossary", terms)
    except (DataError, OSError) as exc:
        parser.exit(1, f"Ошибка сборки справочника: {exc}\n")
    print(json.dumps(glossary_stats(terms), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

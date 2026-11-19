"""Эвристики сопоставления имён, организаций, дат и чисел с новостью."""

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Iterator

from backend.agents.state import FactualFinding, ValidationCode
from backend.utils.text_processing import collapse_whitespace

_WORD = re.compile(r"[^\W_]+")
_CAPITAL_WORD = r"[A-ZА-ЯЁ][a-zа-яё]+(?:[-’'][A-ZА-ЯЁa-zа-яё]+)*"
_LEGAL_PREFIX = r"ООО|АО|ПАО|ОАО|ЗАО|НКО|АНО|ФГУП|ФГБУ"
_LEGAL_SUFFIX = r"Inc|Corp|Ltd|LLC|LLP|PLC|GmbH|Corporation|Limited"
_LEGAL_FORMS = set((_LEGAL_PREFIX + "|" + _LEGAL_SUFFIX).casefold().split("|"))
_QUOTED_NAME = r'«[^»]+»|"[^"]+"'
_ORGANIZATION = re.compile(
    rf"\b(?i:{_LEGAL_PREFIX}) +(?:{_QUOTED_NAME}|{_CAPITAL_WORD}(?: +{_CAPITAL_WORD}){{0,4}})"
    rf"|\b{_CAPITAL_WORD}(?: +{_CAPITAL_WORD}){{0,4}},? +(?i:{_LEGAL_SUFFIX})\b\.?"
    rf"|\b(?i:компания|корпорация|группа|банк|фонд) +(?P<quoted_name>{_QUOTED_NAME})"
)
_NAME = re.compile(
    rf"\b(?:(?:[A-ZА-ЯЁ]\. *){{1,3}}{_CAPITAL_WORD}"
    rf"|{_CAPITAL_WORD}(?: +{_CAPITAL_WORD}){{1,3}}"
    rf"|[A-Z][a-z]+[A-Z][A-Za-z]*"
    rf"|[A-ZА-ЯЁ]{{2,}})\b"
)
_MONTHS = {
    word: number
    for number, forms in enumerate((
        "января january jan", "февраля february feb", "марта march mar", "апреля april apr",
        "мая may", "июня june jun", "июля july jul", "августа august aug",
        "сентября september sep sept", "октября october oct", "ноября november nov",
        "декабря december dec",
    ), start=1)
    for word in forms.split()
}
_MONTH_PATTERN = "|".join(_MONTHS)
_TEXT_DATE = re.compile(
    rf"\b(?:(?P<day_first>\d{{1,2}}) +(?P<month_after>{_MONTH_PATTERN})\.?"
    rf"|(?P<month_first>{_MONTH_PATTERN})\.? +(?P<day_after>\d{{1,2}})(?:st|nd|rd|th)?)"
    r"(?:,? +(?P<year>\d{4}))?\b",
    re.IGNORECASE,
)
_NUMERIC_DATE = re.compile(r"(?<!\d)(\d{1,4})([./-])(\d{1,2})\2(\d{1,4})(?!\d)")
_NUMBER = re.compile(r"(?<!\w)[+−-]?(?:\d{1,3}(?: \d{3})+|\d+)(?:[.,]\d+)?(?: *%)?(?!\w)")
_PLACE = re.compile(rf"(?i:\b(?:in|near|from|across|в|из|возле|около)) +({_CAPITAL_WORD}(?: +{_CAPITAL_WORD}){{0,2}})")
_TITLE_NAME = re.compile(rf"(?i:\b(?:Mr|Mrs|Ms|Dr|Professor|господин|госпожа|президент))\.? +({_CAPITAL_WORD})")
_NAME_SUBJECT = re.compile(
    rf"\b({_CAPITAL_WORD}) +(?i:won|beat|defeated|said|claimed|testified|was acquitted|"
    r"заявил[аи]?|победил[аи]?|выиграл[аи]?)\b"
)
_ORG_WORD = re.compile(
    r"\b(?:university|organization|organisation|agency|company|police|court|association|council|committee|"
    r"corporation|организация|ассоциация|совет|комитет|университет|ооо|ао|пао|нко)\b", re.I,
)
_TITLE_PHRASE = re.compile(rf"\b{_CAPITAL_WORD}(?: +(?:(?:of|and|the|и) +)?{_CAPITAL_WORD}){{1,6}}\b")
_NAME_PARTS_IGNORED = {"the", "of", "and", "и"}
_PRONOUNS = {"he", "she", "it", "they", "we", "who", "он", "она", "они", "кто"}
_NON_PLACES = set(_MONTHS) | {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "понедельник", "вторник", "среду", "четверг", "пятницу", "субботу", "воскресенье", "the",
    "march", "may",
}
_RU_ENDINGS = ("ами", "ями", "ого", "ему", "ому", "ой", "ей", "ом", "ем", "ам", "ям", "ах", "ях",
               "ов", "ев", "а", "я", "у", "ю", "е", "и", "ы", "й", "ь")
_PLACE_ALIASES = (
    frozenset("казань|казани|казанью|kazan".split("|")),
    frozenset("москва|москвы|москве|москву|москвой|москвою|moscow|moskva".split("|")),
    frozenset("санкт петербург|санкт петербурга|санкт петербурге|санкт петербургу|санкт петербургом|"
              "saint petersburg|st petersburg|sankt peterburg".split("|")),
)
_TRANSLITERATION = str.maketrans(dict(zip(
    "абвгдежзийклмнопрстуфхцчшщъыьэюя",
    "a b v g d e zh z i y k l m n o p r s t u f kh ts ch sh shch _ y _ e yu ya".split(), strict=True,
)) | {"ъ": "", "ь": ""})
_NUMBER_WORDS = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve".split(), range(13), strict=True,
)) | {
    "ноль": 0, "один": 1, "одна": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5,
    "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10, "одиннадцать": 11, "двенадцать": 12,
}
_QUANTITY = re.compile(
    r"(?<!\w)(?P<number>[+−-]?(?:\d{1,3}(?:[ ,]\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)|"
    + "|".join(_NUMBER_WORDS) + r")(?!\w)(?:\s*(?P<unit>%|[^\W\d_]+))?", re.I,
)
_UNIT_STOP = {"the", "a", "an", "and", "of", "in", "on", "was", "were", "is", "to", "at", "и", "в", "на"}


@dataclass(frozen=True)
class Name:
    label: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class DateMention:
    start: int
    end: int
    key: tuple[int, int, int | None]


def glossary_names(entries: list[str]) -> list[Name]:
    names = []
    for entry in entries:
        forms = [form.strip() for form in entry.split(" — ", 1)[0].split("|") if form.strip()]
        aliases = tuple(dict.fromkeys(key for form in forms if (key := _name_key(form))))
        if aliases:
            names.append(Name(forms[0], aliases))
    return names


def is_name_only(text: str, glossary: list[Name]) -> bool:
    key = _name_key(text)
    if any(key in name.aliases for name in glossary):
        return True
    candidate = text.rstrip(" .!?…")
    return bool(_ORGANIZATION.fullmatch(candidate) or _NAME.fullmatch(candidate.strip(" «»“”\"")))


def possible_omissions(
    lead: str, summary: str, glossary: list[Name]
) -> dict[ValidationCode, list[str]]:
    missing: dict[ValidationCode, list[str]] = defaultdict(list)
    lead_key, summary_key = _name_key(lead), _name_key(summary)
    known = [item for item in glossary if _name_is_present(item, lead_key)]
    names = [*known]
    for candidate in _extract_names(lead):
        if not any(_name_is_present(candidate, form) for item in known for form in item.aliases):
            names.append(candidate)
    for name in names:
        if not _name_is_present(name, summary_key) and not any(
            _same_known_place(alias, summary_key) for alias in name.aliases
        ):
            missing[ValidationCode.POTENTIAL_MISSING_NAME].append(name.label)

    lead_dates = list(_dates(lead))
    summary_dates = list(_dates(summary))
    for value in lead_dates:
        if not any(_same_date(value.key, other.key) for other in summary_dates):
            missing[ValidationCode.POTENTIAL_MISSING_DATE].append(lead[value.start:value.end])

    summary_numbers = set(_numbers(summary, summary_dates))
    for value, label in _numbers(lead, lead_dates).items():
        if value not in summary_numbers:
            missing[ValidationCode.POTENTIAL_MISSING_NUMBER].append(label)
    return missing


def _family(word: str) -> set[str]:
    if not re.fullmatch(r"[а-я]{4,}", word):
        return {word}
    return {word} | {word[:-len(end)] for end in _RU_ENDINGS if word.endswith(end) and len(word) - len(end) >= 4}


def _same_word(first: str, second: str) -> bool:
    return bool(_family(first) & _family(second))


def _same_known_place(candidate: str, original_key: str) -> bool:
    return any(candidate in group and any(f" {name} " in f" {original_key} " for name in group)
               for group in _PLACE_ALIASES)


def _latin_forms(word: str) -> set[str]:
    return {form.translate(_TRANSLITERATION) for form in _family(word)}


def _acronyms(text: str) -> set[str]:
    return {
        "".join(word[0] for word in _name_key(match[0]).split() if word not in _NAME_PARTS_IGNORED)
        for match in _TITLE_PHRASE.finditer(text) if _ORG_WORD.search(match[0])
    }


def _confirmed_expansions(summary: str, source: str) -> set[str]:
    """Распознать составное название с сокращённым префиксом, например DR Congo."""
    confirmed = set()
    for match in _TITLE_PHRASE.finditer(summary):
        name = _name_key(match[0])
        parts = [word for word in name.split() if word not in _NAME_PARTS_IGNORED]
        if len(parts) >= 3:
            prefix = "".join(word[0] for word in parts[:-1])
            if f" {prefix} {parts[-1]} " in f" {source} ":
                confirmed.add(name)
    return confirmed


def new_entity_findings(original: str, summary: str, glossary: list[Name]) -> list[FactualFinding]:
    source = _name_key(original)
    source_words = set(source.split())
    source_forms = set().union(*map(_family, source_words))
    capitalized = {word for match in re.finditer(_CAPITAL_WORD, original) for word in _name_key(match[0]).split()}
    source_latin = {word for word in capitalized if re.fullmatch(r"[a-z]{4,}", word)}
    transliterated = {form for word in capitalized if re.fullmatch(r"[а-я]{4,}", word)
                      for form in _latin_forms(word)}
    aliases = {alias for item in glossary if _name_is_present(item, source) for alias in item.aliases}
    acronyms = _acronyms(original)
    expansions = _confirmed_expansions(summary, source)
    candidates = [(match[1], "place") for match in _PLACE.finditer(summary)
                  if (key := _name_key(match[1])) and key.split()[0] not in _NON_PLACES]
    candidates += [(match[1], "person") for match in _TITLE_NAME.finditer(summary)]
    candidates += [
        (name.label, "organization" if _ORG_WORD.search(name.label) or name.label.isupper() else "name")
        for name in _extract_names(summary)
    ]
    compounds = [f" {_name_key(label)} " for label, _ in candidates if " " in label]
    candidates += [(match[1], "person") for match in _NAME_SUBJECT.finditer(summary)
                   if (key := _name_key(match[1])) not in _PRONOUNS
                   and not any(f" {key} " in compound for compound in compounds)]
    findings = {}
    for label, category in candidates:
        key = _name_key(label)
        if not key or key in aliases or f" {key} " in f" {source} " or _same_known_place(key, source):
            continue
        if any(f" {key} " in f" {name} " for name in expansions):
            continue
        if label.isupper() and key in acronyms:
            continue
        parts = [word for word in key.split() if word not in _NAME_PARTS_IGNORED]
        unknown = [word for word in parts if not (
            len(word) == 1 or _family(word) & source_forms or word in transliterated
            or (re.fullmatch(r"[а-я]{4,}", word) and _latin_forms(word) & source_latin)
        )]
        if not unknown:
            continue
        hard_error = len(set(unknown)) >= 2
        if category == "name" and len(parts) == 2 and len(unknown) == 1:
            category = "person"
        findings.setdefault(key, FactualFinding(
            code=ValidationCode.UNSUPPORTED_NAME if hard_error else ValidationCode.UNCONFIRMED_ENTITY,
            severity="error" if hard_error else "review", category=category, subject=label,
        ))
    return list(findings.values())


@dataclass(frozen=True)
class _Quantity:
    label: str
    values: frozenset[Decimal]
    unit: str


def _quantities(text: str, dates: list[DateMention]) -> list[_Quantity]:
    masked = list(text)
    for date in dates:
        masked[date.start:date.end] = " " * (date.end - date.start)
    normalized = re.sub(r"(?<=\d)[-–—](?=\d)", ":", "".join(masked))
    quantities = []
    for match in _QUANTITY.finditer(normalized):
        raw = match["number"].casefold()
        unit = (match["unit"] or "").casefold()
        if unit in _UNIT_STOP:
            unit = ""
        if unit.startswith("percent") or unit.startswith("процент"):
            unit = "%"
        if raw in _NUMBER_WORDS:
            values = {Decimal(_NUMBER_WORDS[raw])}
        else:
            label = raw.replace(" ", "").replace("−", "-")
            if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", label):
                values = {Decimal(label.replace(",", ""))}
                if label.count(",") == 1 and "." not in label:
                    values.add(Decimal(label.replace(",", ".")))
            else:
                values = {Decimal(label.replace(",", "."))}
        quantities.append(_Quantity(" ".join(match[0].split()), frozenset(values), unit.rstrip("s")))
    return quantities


def new_number_findings(original: str, summary: str) -> list[FactualFinding]:
    source_dates, summary_dates = list(_dates(original)), list(_dates(summary))
    source_numbers = _quantities(original, source_dates)
    findings = []
    for mention in summary_dates:
        if not any(_same_date(item.key, mention.key) for item in source_dates):
            findings.append(FactualFinding(
                code=ValidationCode.UNCONFIRMED_NUMBER, severity="review", category="date",
                subject=summary[mention.start:mention.end],
            ))
    for value in _quantities(summary, summary_dates):
        if any(item.values & value.values and (value.unit != "%" or item.unit == "%") for item in source_numbers):
            continue
        if not value.unit and any(date.key[2] in value.values for date in source_dates):
            continue
        # Смена значения сопоставимой величины надёжнее, чем число без совпавшей единицы измерения.
        paired = bool(value.unit) and any(_same_word(item.unit, value.unit) for item in source_numbers)
        findings.append(FactualFinding(
            code=ValidationCode.UNSUPPORTED_NUMBER if paired else ValidationCode.UNCONFIRMED_NUMBER,
            severity="error" if paired else "review", category="number", subject=value.label,
        ))
    return findings


def _name_key(text: str) -> str:
    words = _WORD.findall(collapse_whitespace(text).casefold().replace("ё", "е"))
    if words and words[0] in _LEGAL_FORMS:
        words = words[1:]
    if words and words[-1] in _LEGAL_FORMS:
        words = words[:-1]
    return " ".join(words)


def _name_is_present(name: Name, text_key: str) -> bool:
    return any(f" {alias} " in f" {text_key} " for alias in name.aliases)


def _extract_names(text: str) -> Iterator[Name]:
    organizations = [match["quoted_name"] or match[0] for match in _ORGANIZATION.finditer(text)]
    other_names = (match[0] for match in _NAME.finditer(_ORGANIZATION.sub(" ", text)))
    for label in (*organizations, *other_names):
        if key := _name_key(label):
            yield Name(label, (key,))


def _dates(text: str) -> Iterator[DateMention]:
    for match in _NUMERIC_DATE.finditer(text):
        left, separator, middle, right = match.groups()
        if len(left) == 4 and len(right) <= 2:
            year, month, day = int(left), int(middle), int(right)
        elif len(right) == 4 and len(left) <= 2:
            year, month, day = int(right), int(middle), int(left)
            if separator == "/" and month > 12:
                month, day = day, month
        else:
            continue
        if _valid_date(year, month, day):
            yield DateMention(match.start(), match.end(), (month, day, year))
    for match in _TEXT_DATE.finditer(text):
        month = _MONTHS[(match["month_after"] or match["month_first"]).casefold()]
        day = int(match["day_first"] or match["day_after"])
        year = int(match["year"]) if match["year"] else None
        if _valid_date(year, month, day):
            yield DateMention(match.start(), match.end(), (month, day, year))


def _valid_date(year: int | None, month: int, day: int) -> bool:
    try:
        date(year if year is not None else 2000, month, day)
    except ValueError:
        return False
    return True


def _same_date(first: tuple[int, int, int | None], second: tuple[int, int, int | None]) -> bool:
    return first[:2] == second[:2] and (first[2] == second[2] or None in (first[2], second[2]))


def _numbers(text: str, dates: list[DateMention]) -> dict[tuple[Decimal, bool], str]:
    masked = list(text)
    for mention in dates:
        masked[mention.start:mention.end] = [" "] * (mention.end - mention.start)
    numbers: dict[tuple[Decimal, bool], str] = {}
    for match in _NUMBER.finditer("".join(masked)):
        label = match[0]
        value = label.replace(" ", "").replace(",", ".").replace("−", "-").rstrip("%")
        numbers.setdefault((Decimal(value), label.endswith("%")), label)
    return numbers

"""Детерминированная сегментация русских и английских предложений."""

import re
from typing import Literal
from unicodedata import normalize

Language = Literal["ru", "en"]
SummaryLanguage = Literal["auto", "ru", "en"]

LANGUAGE_INSTRUCTIONS: dict[Language, str] = {
    "ru": "Напиши резюме полностью на русском языке. Используй русские предложения и кириллицу, сохраняя допустимое написание имён.",
    "en": "Write the entire summary in English. Use English sentences and retain valid spellings of proper names.",
}


def detect_language(text: str) -> Language | None:
    """Определить преобладающую письменность для русского и английского текста."""
    cyrillic = len(re.findall("[а-яё]", text, re.I))
    latin = len(re.findall("[a-z]", text, re.I))
    return ("ru" if cyrillic > latin else "en") if cyrillic or latin else None

_BOUNDARY = re.compile(r"""(?P<mark>[.!?…]+)[»”’"')\]}]*(?= |$)""")
_PERIOD_WORD = re.compile(r"\b([a-zа-яё]+)$", re.IGNORECASE)
_INITIAL = re.compile(r"\b[A-ZА-ЯЁ]$")
_RUSSIAN_CHAIN = re.compile(r"\bт\.\s*([депкн])$", re.IGNORECASE)
_DOTTED_WORD = re.compile(r"\b(?:[a-z]\.)+[a-z]$", re.IGNORECASE)
_TITLES = {"dr", "mr", "mrs", "ms", "prof", "st", "ул", "им", "акад", "проф", "рис", "см"}
_ABBREVIATIONS = {
    "гг", "в", "вв", "руб", "млн", "млрд", "тыс", "коп", "шт",
    "etc", "inc", "ltd", "corp", "jr", "sr", "vs", "no",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
_EDGE_PUNCTUATION = " \t.!?…«»“”‘’\"'()[]"


def collapse_whitespace(text: str) -> str:
    return " ".join(normalize("NFKC", text).split())


def sentence_key(sentence: str) -> str:
    key = collapse_whitespace(sentence).strip(_EDGE_PUNCTUATION).casefold()
    return re.sub(r" *([,;:.!?]) *", r"\1", key)


def split_sentences(text: str) -> list[str]:
    text = collapse_whitespace(text)
    start = 0
    sentences: list[str] = []
    for boundary in _BOUNDARY.finditer(text):
        next_text = text[boundary.end():boundary.end() + 24].lstrip()
        prefix = text[max(start, boundary.start() - 48):boundary.start()]
        if next_text and boundary["mark"] == "." and _is_internal_period(prefix, next_text):
            continue
        sentence = text[start:boundary.end()].strip()
        if any(char.isalnum() for char in sentence):
            sentences.append(sentence)
        start = boundary.end()
    tail = text[start:].strip()
    if any(char.isalnum() for char in tail):
        sentences.append(tail)
    return sentences


def _is_internal_period(prefix: str, following: str) -> bool:
    if following[0] in "—–,;":
        return True

    chain = _RUSSIAN_CHAIN.search(prefix)
    if chain:
        return chain[1].casefold() in {"е", "к", "н"} or following[0].islower()

    match = _PERIOD_WORD.search(prefix)
    if not match:
        return False
    word = match[1].casefold()
    if word in _TITLES:
        return True
    if word == "т" and re.match(r"[депкн]\.", following, re.IGNORECASE):
        return True
    if word == "г":
        previous = prefix[:match.start()].rstrip()
        after_year = bool(previous and previous[-1].isdigit())
        return following[0].islower() or (following[0].isupper() and not after_year)
    if _INITIAL.search(prefix) and following[0].isupper():
        return True
    if _DOTTED_WORD.search(prefix):
        return prefix[-3:].casefold() in {"e.g", "i.e"} or (
            following[0].islower() or following[0].isdigit()
        )
    return word in _ABBREVIATIONS and (following[0].islower() or following[0].isdigit())

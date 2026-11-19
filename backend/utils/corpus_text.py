"""Общая нормализация для дедупликации корпуса и защиты тестового набора."""

import hashlib
import unicodedata

_CONTROL_CATEGORIES = {"Cc", "Cf", "Cs"}


def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value)
    value = "".join(" " if char.isspace() else char for char in value
                    if char.isspace() or unicodedata.category(char) not in _CONTROL_CATEGORIES)
    value = " ".join(value.split())
    return value if any(char.isalnum() for char in value) else ""


def normalized(value: str) -> str:
    return clean_text(value).casefold()


def content_hash(value: str) -> str:
    return hashlib.sha256(normalized(value).encode("utf-8")).hexdigest()

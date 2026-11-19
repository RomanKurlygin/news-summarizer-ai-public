"""Ограничение примеров, передаваемых в промпт Summarizer."""

import json
from collections.abc import Iterable
from enum import StrEnum

from backend.utils.text_processing import split_sentences

MAX_EXAMPLES = 1
MAX_CONTEXT_CHARS = 160


class StylePattern(StrEnum):
    SINGLE_BRIEF = "Lead with the main event in one concise sentence."
    SINGLE_DETAIL = "Use one sentence that leads with the main event and adds a brief supporting detail."
    MULTI_SENTENCE = "State the main event first, then add supporting detail in a separate sentence."


def reference_style(summary: str) -> StylePattern:
    """Вывести форму пересказа из эталона без передачи его слов или фактов."""
    if len(split_sentences(summary)) > 1:
        return StylePattern.MULTI_SENTENCE
    if len(summary.split()) > 30:
        return StylePattern.SINGLE_DETAIL
    return StylePattern.SINGLE_BRIEF


def limit_style_examples(examples: Iterable[str], *, max_examples: int = MAX_EXAMPLES,
                         max_chars: int = MAX_CONTEXT_CHARS) -> list[str]:
    allowed = set(StylePattern)
    return limit_examples(
        (example for example in examples if example in allowed),
        max_examples=min(max_examples, MAX_EXAMPLES), max_chars=min(max_chars, MAX_CONTEXT_CHARS),
    )


def limit_examples(examples: Iterable[str], *, max_examples: int = MAX_EXAMPLES,
                   max_chars: int = MAX_CONTEXT_CHARS) -> list[str]:
    if max_examples < 0 or max_chars < 0:
        raise ValueError("Лимиты контекста должны быть неотрицательными.")
    selected: list[str] = []
    for example in examples:
        if len(selected) >= max_examples:
            break
        if example.strip() and len(json.dumps([*selected, example], ensure_ascii=False)) <= max_chars:
            selected.append(example)
    return selected

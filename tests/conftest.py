from typing import Any

import pytest

from backend.data.llm_client import BaseLLMClient


class FakeLLMClient(BaseLLMClient):
    """Возвращает заданный ответ или ошибку. Используется только в тестах."""

    def __init__(self, answer: str = "", *, error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append((prompt, kwargs))
        if self.error is not None:
            raise self.error
        return self.answer


@pytest.fixture
def fake_llm():
    return FakeLLMClient

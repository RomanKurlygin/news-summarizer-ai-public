"""Асинхронный контракт генерации для адаптеров языковых моделей."""

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Сгенерировать текст по промпту с параметрами вызывающего агента."""

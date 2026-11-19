"""Общий контракт попытки генерации для сетевых и локальных моделей."""

from dataclasses import dataclass
from typing import Protocol

import httpx

from backend.utils.text_processing import Language


class LLMError(RuntimeError):
    def __init__(self, code: str, *, retryable: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class GenerationOptions:
    temperature: float
    max_tokens: int
    schema: dict | None = None
    task: str | None = None
    seed: int | None = None
    target_language: Language | None = None
    language_retry: bool = False


@dataclass(frozen=True)
class LLMReply:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class Provider(Protocol):
    async def complete(self, prompt: str, model: str, options: GenerationOptions) -> LLMReply: ...


def count_tokens(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def reply(text: object, actual_model: object, requested: str, input_tokens: object = None,
          output_tokens: object = None) -> LLMReply:
    if not isinstance(text, str) or not text.strip():
        raise LLMError("invalid_response")
    return LLMReply(text.strip(), actual_model if isinstance(actual_model, str) and actual_model else requested,
                    count_tokens(input_tokens), count_tokens(output_tokens))


async def json_request(client: httpx.AsyncClient, method: str, url: str, **kwargs) -> dict:
    try:
        response = await client.request(method, url, **kwargs)
        response.raise_for_status()
    except httpx.TimeoutException:
        raise LLMError("timeout", retryable=True) from None
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        label = {401: "unauthorized", 403: "forbidden", 404: "model_unavailable", 429: "rate_limit"}.get(
            code, f"http_{code}"
        )
        raise LLMError(label, retryable=code in {408, 429, 500, 502, 503, 504, 529}) from None
    except httpx.RequestError:
        raise LLMError("provider_unavailable", retryable=True) from None
    try:
        data = response.json()
    except ValueError:
        raise LLMError("invalid_response") from None
    if not isinstance(data, dict):
        raise LLMError("invalid_response")
    return data

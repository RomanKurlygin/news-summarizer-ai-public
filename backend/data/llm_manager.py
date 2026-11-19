"""Маршрутизация генерации и аудит каждой попытки LLM."""

import asyncio
import hashlib
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from uuid import UUID

import httpx

from backend.core.config import Settings
from backend.data.api_llm import AnthropicProvider, GeminiProvider, external_key
from backend.data.llm_client import BaseLLMClient
from backend.data.llm_types import GenerationOptions, LLMError, LLMReply, Provider
from backend.data.ollama import OllamaProvider
from backend.data.transformers_llm import TransformersProvider
from backend.utils.text_processing import LANGUAGE_INSTRUCTIONS, Language, detect_language

_EXTERNAL = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CallAttempt:
    request_id: UUID
    agent_name: str
    provider: str
    model: str
    created_at: datetime
    latency_ms: int
    status: str
    prompt_hash: str
    response_hash: str | None
    input_tokens: int | None
    output_tokens: int | None
    parameters: dict
    error_type: str | None


CallRecorder = Callable[[CallAttempt], Awaitable[None]]
ResponseValidator = Callable[[str], object]


class LLMManager:
    def __init__(self, settings: Settings, record: CallRecorder, *, allow_external: bool = False,
                 providers: Mapping[str, Provider]) -> None:
        self.settings = settings
        self.providers = providers
        self.record = record
        self.allow_external = allow_external
        self.language_retries = 0

    @staticmethod
    def providers_for(settings: Settings, client: httpx.AsyncClient) -> dict[str, Provider]:
        return {
            "ollama": OllamaProvider(client, str(settings.ollama_base_url), settings.ollama_keep_alive),
            "transformers": TransformersProvider(settings),
            "anthropic": AnthropicProvider(client),
            "gemini": GeminiProvider(client),
        }

    def model_for(self, provider: str) -> str | None:
        return {
            "ollama": self.settings.ollama_model, "transformers": self.settings.hf_model_id,
            "anthropic": self.settings.anthropic_model, "gemini": self.settings.gemini_model,
        }.get(provider)

    async def generate(self, prompt: str, *, request_id: UUID, agent_name: str, provider: str | None = None,
                       model: str | None = None, temperature: float = 0.0, max_tokens: int = 180,
                       schema: dict | None = None, validate_response: ResponseValidator | None = None,
                       seed: int | None = None, target_language: Language | None = None) -> str:
        if not prompt.strip() or not math.isfinite(temperature) or not 0 <= temperature <= 1 or not 0 < max_tokens <= 4096:
            raise ValueError("Некорректные параметры генерации.")
        chosen = provider or self.settings.llm_provider
        chosen_model = model or self.model_for(chosen)
        if chosen not in self.providers:
            raise LLMError("unknown_provider")
        if chosen in _EXTERNAL:
            if not self.allow_external or provider is None:
                raise LLMError("payment_not_authorized")
            external_key(_EXTERNAL[chosen])
        if not chosen_model or len(chosen_model) > 255 or not chosen_model.strip():
            raise LLMError("invalid_model")
        routes = [(chosen, chosen_model)]
        if (provider is None and model is None and chosen == "ollama" and self.settings.llm_allow_local_fallback
                and "transformers" in self.providers):
            routes.append(("transformers", self.settings.hf_model_id))
        last_error = LLMError("provider_unavailable")
        for name, model_name in routes:
            language_retry = False
            for attempt in range(1, self.settings.llm_max_attempts + 2):
                attempt_prompt = f"{LANGUAGE_INSTRUCTIONS[target_language]}\n{prompt}" if language_retry else prompt
                options = GenerationOptions(temperature, max_tokens, schema, agent_name, seed,
                                            target_language, language_retry)
                started_at, clock = datetime.now(UTC), perf_counter()
                response: LLMReply | None = None
                error: LLMError | None = None
                actual_language = None
                cancelled = False
                try:
                    async with asyncio.timeout(self.settings.llm_timeout_seconds):
                        response = await self.providers[name].complete(attempt_prompt, model_name, options)
                    if validate_response is not None:
                        try:
                            validate_response(response.text)
                        except (ValueError, RecursionError):
                            raise LLMError("invalid_response") from None
                    if target_language:
                        actual_language = detect_language(response.text)
                        if actual_language != target_language:
                            raise LLMError("language_mismatch")
                except asyncio.CancelledError:
                    cancelled = True
                    error = LLMError("cancelled")
                except TimeoutError:
                    error = LLMError("timeout", retryable=True)
                except LLMError as exc:
                    error = exc
                except Exception:
                    error = LLMError("provider_failure")

                await self._record(CallAttempt(
                    request_id, agent_name, name, response.model if response else model_name,
                    started_at, round((perf_counter() - clock) * 1000),
                    "cancelled" if cancelled else "failed" if error else "ok",
                    _hash(attempt_prompt), _hash(response.text) if response else None,
                    response.input_tokens if response else None, response.output_tokens if response else None,
                    {"temperature": temperature, "max_tokens": max_tokens, "structured": schema is not None,
                     "attempt": attempt, "fallback": name != chosen, "requested_model": model_name,
                     **({"seed": seed} if seed is not None else {}),
                     **({"target_language": target_language, "actual_language": actual_language,
                         "language_retry": language_retry} if target_language else {})},
                    error.code if error else None,
                ))
                if cancelled:
                    raise asyncio.CancelledError
                if error is None and response is not None:
                    return response.text
                last_error = error or LLMError("invalid_response")
                if target_language and last_error.code == "language_mismatch" and not self.language_retries:
                    self.language_retries += 1
                    language_retry = True
                    continue
                if language_retry or not last_error.retryable or attempt >= self.settings.llm_max_attempts:
                    break
                await asyncio.sleep(self.settings.llm_retry_delay_seconds * 2 ** (attempt - 1))
            if language_retry or (not last_error.retryable and last_error.code != "model_unavailable"):
                break
        raise last_error

    async def _record(self, attempt: CallAttempt) -> None:
        task = asyncio.create_task(self.record(attempt))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise


class ManagedAgentClient(BaseLLMClient):
    def __init__(self, manager: LLMManager, request_id: UUID, agent_name: str, *, provider: str | None = None,
                 model: str | None = None, schema: dict | None = None,
                 validate_response: ResponseValidator | None = None, seed: int | None = None) -> None:
        self.manager = manager
        self.request_id = request_id
        self.agent_name = agent_name
        self.provider = provider
        self.model = model
        self.schema = schema
        self.validate_response = validate_response
        self.seed = seed

    async def generate(self, prompt: str, **kwargs) -> str:
        try:
            return await self.manager.generate(
                prompt, request_id=self.request_id, agent_name=self.agent_name, provider=self.provider,
                model=self.model, schema=self.schema, validate_response=self.validate_response, seed=self.seed, **kwargs,
            )
        except LLMError as exc:
            if exc.code == "timeout":
                raise TimeoutError from None
            raise

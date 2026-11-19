"""Ресурсы и ограничения синхронных маршрутов генерации."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from hashlib import sha256
import json
from time import perf_counter
from typing import TypeVar
from uuid import UUID

import httpx
from fastapi import HTTPException, Request

from backend.api.schemas.summarization import CandidateError, CandidateResult, Configuration, SummaryResult
from backend.agents.state import AgentState, PipelineMode
from backend.core.config import Settings
from backend.core.exceptions import ServiceUnavailableError
from backend.data.llm_manager import LLMManager
from backend.data.ollama import OllamaProvider
from backend.db.cache import SummaryCache
from backend.db.reasoning import ReasoningLogStore
from backend.db.session import Database
from backend.evaluation.metrics_calculator import MetricsCalculator
from backend.retrieval.glossary import GlossaryTerm, glossary_context, glossary_snapshot
from backend.services.summarization import SummarizationService
from backend.utils.text_processing import SummaryLanguage

_FAILURES = {
    "model_unavailable": (503, "Модель недоступна у выбранного провайдера."),
    "provider_unavailable": (503, "Провайдер генерации недоступен."),
    "rate_limit": (503, "Провайдер временно ограничил запросы."),
    "rag_failed": (503, "Индекс поиска недоступен."),
    "glossary_failed": (503, "Справочник недоступен."),
    "timeout": (504, "Истекло время ожидания генерации."),
    "invalid_response": (502, "Модель вернула некорректный ответ."),
    "language_mismatch": (502, "Модель не соблюла выбранный язык резюме."),
    "sentence_limit": (502, "Модель превысила выбранный предел предложений."),
}
T = TypeVar("T")


def summary_result(text: str, report: dict) -> SummaryResult:
    resolved = next(iter(report.get("models", [])), {})
    provider = resolved.get("provider")
    return SummaryResult.model_validate({
        **report, "text": text, "provider": provider if provider in {"ollama", "transformers"} else None,
        "model": resolved.get("model"),
    })


class SummaryRuntime:
    def __init__(self, settings: Settings, database: Database, http: httpx.AsyncClient) -> None:
        self.settings = settings
        self.database = database
        self.metrics = MetricsCalculator(settings.embedding_cache_dir, settings.metric_nltk_dir)
        # HTTP принимает только уже установленные локальные модели.
        self.providers = LLMManager.providers_for(settings.model_copy(update={"hf_llm_local_files_only": True}), http)
        self._capacity = asyncio.Semaphore(2)
        self._index_lock = asyncio.Lock()
        self._embeddings = None

    @property
    def cache(self) -> SummaryCache:
        return SummaryCache(self.database.sessions)

    async def enrich(self, state, *, category: str | None, language: str | None, manifest=None, holdout=None):
        from backend.data.embeddings import Embeddings
        from backend.retrieval.index_bundle import open_indexes
        from scripts.build_indexes import load_holdout

        # Qdrant local нельзя открывать одновременно в двух клиентах.
        async with self._index_lock:
            if self._embeddings is None:
                self._embeddings = Embeddings(self.settings)
            guard = holdout or await asyncio.to_thread(load_holdout, self.settings.holdout_path)
            async with open_indexes(self.settings, guard, self._embeddings, manifest=manifest) as (retriever, _):
                return await retriever.enrich_state(state, category=category, language=language,
                                                    cross_language=state.target_language is not None and manifest is None)

    async def publish_reviewed_corpus(self) -> dict:
        from backend.data.embeddings import Embeddings
        from backend.retrieval.index_bundle import build_indexes
        from scripts.build_indexes import load_holdout, read_database

        async with self._index_lock:
            guard = await asyncio.to_thread(load_holdout, self.settings.holdout_path)
            documents = await read_database(self.settings)
            if self._embeddings is None:
                self._embeddings = Embeddings(self.settings)
            return await build_indexes(documents, guard, self.settings, self._embeddings)

    async def run_pipeline(
        self, text: str, *, mode: PipelineMode, provider: str, model: str | None, user_id: UUID | None,
        request_id: UUID, category: str | None = None, language: str | None = None, max_sentences: int = 3,
        temperature: float | None = None, timeout: float, seed: int | None = None,
        judge: tuple[str, str] | None = None, terms: tuple[GlossaryTerm, ...] | None = None,
        holdout=None, manifest=None, queue_review: bool = False, summary_language: SummaryLanguage | None = None,
    ) -> dict:
        async def enrich(state):
            index = {} if manifest is None else {"manifest": manifest, "holdout": holdout}
            return await self.enrich(state, category=category, language=language, **index)

        async def glossary(state):
            vocabulary = terms if terms is not None else await glossary_snapshot(self.database.sessions)
            return glossary_context(state, vocabulary, language=language,
                                    excluded_hashes=holdout.hashes if holdout else frozenset())

        service = SummarizationService(self.settings, self.database.sessions, self.providers,
                                       rag=enrich, glossary=glossary)
        async with asyncio.timeout(timeout):
            async with self._capacity:
                return await service.run(
                    text, provider=provider, model=model, mode=mode, max_sentences=max_sentences,
                    temperature=temperature, user_id=user_id, request_id=request_id, seed=seed, judge=judge,
                    queue_review=queue_review, summary_language=summary_language, rag_language=language,
                )

    async def generate(self, text: str, config: Configuration, user_id: UUID, request_id: UUID,
                       *, category: str | None = None, language: str | None = None,
                       queue_review: bool = False, summary_language: SummaryLanguage | None = None) -> CandidateResult:
        settings = self.settings
        key = None
        if queue_review and config.mode == "baseline" and summary_language and settings.baseline_cache_ttl_seconds:
            key = await self._baseline_key(text, config, user_id, category)
            if key and (cached := await self.cache.take(key)):
                clock, started = perf_counter(), datetime.now(UTC)
                state = AgentState(request_id=request_id, original_text=text, current_summary=cached["final_summary"],
                                   target_language=cached["target_language"], execution_status="draft", selected_iteration=0)
                state.record_step(name="Cache", started_at=started, started_clock=clock,
                                  input_summary=state.observation(), status="ok", model_name=cached["models"][0]["model"])
                cached.update(request_id=str(request_id), duration_ms=round((perf_counter() - clock) * 1000, 3),
                              llm_attempts=0, language_retries=0, cache_hit=True)
                await ReasoningLogStore(self.database.sessions).save_run(state, "baseline", user_id, parameters=cached)
                return CandidateResult(configuration=config, result=summary_result(text, cached))
        try:
            report = await self.run_pipeline(
                text, provider=config.provider, model=config.model, mode=config.mode,
                max_sentences=config.max_sentences, temperature=config.temperature,
                user_id=user_id, request_id=request_id, category=category, language=language,
                timeout=self.settings.api_run_timeout_seconds, queue_review=queue_review, summary_language=summary_language,
            )
        except TimeoutError:
            return CandidateResult(configuration=config, error=self.error("timeout", request_id))
        result = summary_result(text, report)
        if result.status in {"failed", "incomplete_configuration"}:
            code = next((code for code in report["error_codes"] if code in {
                "rag_failed", "glossary_failed", "language_mismatch", "sentence_limit",
            }),
                        report.get("failure_code"))
            return CandidateResult(configuration=config, result=result, error=self.error(code, request_id))
        if key and result.final_summary and result.status == "draft" and report["models"]:
            await self.cache.store(key, report, settings.baseline_cache_ttl_seconds)
        return CandidateResult(configuration=config, result=result)

    async def _baseline_key(self, text: str, config: Configuration, user_id: UUID, category: str | None) -> str | None:
        provider = self.providers.get(config.provider)
        if not isinstance(provider, OllamaProvider):
            return None
        model = config.model or self.settings.ollama_model
        try:
            response = await provider.client.get(provider.base_url + "/api/tags", timeout=2)
            tags = response.raise_for_status().json()["models"]
            digest = next((tag["digest"] for tag in tags if tag["name"] in {model, f"{model}:latest"}), None)
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return None
        if not isinstance(digest, str) or not digest:
            return None
        payload = ["baseline:v1", str(user_id), text, category, model, digest, config.model_dump(),
                   self.settings.summarizer_temperature, self.settings.summarizer_max_tokens]
        return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    @staticmethod
    def error(code: str | None, request_id: UUID) -> CandidateError:
        status, message = _FAILURES.get(code, (502, "Не удалось завершить суммаризацию."))
        return CandidateError(code=code if code in _FAILURES else "generation_failed",
                              http_status=status, message=message, request_id=request_id)


def get_runtime(request: Request) -> SummaryRuntime:
    runtime = request.app.state.summary_runtime
    if runtime is None:
        raise ServiceUnavailableError("Суммаризация недоступна без базы данных.")
    return runtime


async def during_request(request: Request, work: Callable[[], Awaitable[T]], timeout: float) -> T:
    async def disconnect():
        while (await request.receive())["type"] != "http.disconnect":
            pass

    worker = asyncio.create_task(work())
    listener = asyncio.create_task(disconnect())
    try:
        async with asyncio.timeout(timeout):
            done, _ = await asyncio.wait((worker, listener), return_when=asyncio.FIRST_COMPLETED)
            if worker in done:
                return worker.result()
            raise HTTPException(499, "Клиент отключился.")
    except TimeoutError:
        raise HTTPException(504, "Истекло время выполнения запроса.") from None
    finally:
        for task in (listener, worker):
            if not task.done():
                task.cancel()
        await asyncio.gather(listener, worker, return_exceptions=True)

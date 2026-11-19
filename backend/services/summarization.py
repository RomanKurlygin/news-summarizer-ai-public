"""Реальный запуск агентов, RAG и двух независимых журналов."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from time import perf_counter
from typing import get_args
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.agents.critic import Critic, CriticResult, parse_critic_response
from backend.agents.editor import Editor
from backend.agents.state import RAG_MODES, REVIEW_MODES, AgentState, ErrorCode, PipelineMode
from backend.agents.summarizer import Summarizer
from backend.agents.supervisor import Supervisor
from backend.agents.validator import Validator
from backend.core.config import Settings
from backend.data.llm_manager import CallAttempt, LLMManager, ManagedAgentClient
from backend.data.llm_types import Provider
from backend.db.llm_calls import LLMCallStore
from backend.db.reasoning import ReasoningLogStore
from backend.utils.text_processing import SummaryLanguage, detect_language, split_sentences

RagSearch = Callable[[AgentState], Awaitable[AgentState]]


def _validate_plain_summary(response: str) -> None:
    if response.lstrip().startswith(("{", "[", "```", "~~~")):
        raise ValueError("Резюме должно быть обычным текстом.")


class SummarizationService:
    def __init__(self, settings: Settings, sessions: async_sessionmaker[AsyncSession],
                 providers: Mapping[str, Provider], *, rag: RagSearch | None = None,
                 glossary: RagSearch | None = None) -> None:
        self.settings = settings
        self.calls = LLMCallStore(sessions)
        self.reasoning = ReasoningLogStore(sessions)
        self.providers = providers
        self.rag = rag
        self.glossary = glossary

    async def run(self, news: str, *, mode: PipelineMode = "full", provider: str | None = None,
                  model: str | None = None, allow_external: bool = False, request_id: UUID | None = None,
                  user_id: UUID | None = None, max_sentences: int | None = None,
                  temperature: float | None = None, judge: tuple[str, str] | None = None,
                  seed: int | None = None, queue_review: bool = False,
                  summary_language: SummaryLanguage | None = None, rag_language: str | None = None) -> dict:
        if not news.strip() or mode not in get_args(PipelineMode):
            raise ValueError("Нужна непустая новость и поддерживаемый режим.")
        started_at, clock = datetime.now(UTC), perf_counter()
        sentence_limit = 3 if max_sentences is None else max_sentences
        if not 1 <= sentence_limit <= 3:
            raise ValueError("Допустимая длина резюме — от одного до трёх предложений.")
        source_language = detect_language(news)
        target_language = source_language if summary_language == "auto" else summary_language
        state = AgentState(request_id=request_id or uuid4(), original_text=news, target_language=target_language,
                           execution_status="running")
        attempts: list[CallAttempt] = []
        versions: list[dict] = []
        last_step_calls = 0
        finalized = False
        report: dict = {}

        async def record(attempt: CallAttempt) -> None:
            await self.calls.record(attempt, len(attempts) + 1)
            attempts.append(attempt)

        def observe(step_state: AgentState) -> None:
            nonlocal state, last_step_calls
            state = step_state
            step = state.execution_history[-1]
            this_step = attempts[last_step_calls:]
            last_step_calls = len(attempts)
            if this_step:
                step.model_name = this_step[-1].model
            if state.current_summary and step.status == "ok" and (
                step.agent_name == "Summarizer" or (step.agent_name == "Editor" and state.editor_changed)
            ):
                versions.append({"iteration": state.iteration, "summary": state.current_summary})

        async def save(result: AgentState, selected_mode: PipelineMode, owner: UUID | None) -> None:
            nonlocal finalized
            model_calls = list(dict.fromkeys((item.provider, item.model) for item in attempts if item.status == "ok"))
            report.update(
                request_id=str(result.request_id), mode=selected_mode, status=result.execution_status,
                initial_summary=versions[0]["summary"] if versions else None, revisions=versions[1:],
                final_summary=result.current_summary, iterations=result.iteration,
                selected_iteration=result.selected_iteration,
                critic_score=result.critic_score, critic_fallback_used=result.critic_fallback_used,
                critic_passed=result.critic_passed, validation_passed=result.validation_passed,
                critical_factual_error=result.critical_factual_error, missing_key_fact=result.missing_key_fact,
                validation_errors=result.validation_errors, validation_warnings=result.validation_warnings,
                validation_error_codes=[code.value for code in result.validation_error_codes],
                validation_warning_codes=[code.value for code in result.validation_warning_codes],
                factual_findings=[finding.model_dump(mode="json") for finding in result.factual_findings],
                factual_review_required=result.factual_review_required,
                rag_ids=[str(value) for value in result.rag_document_ids], rag_examples=len(result.rag_context),
                rag_search_mode=result.rag_search_mode,
                models=[{"provider": name, "model": resolved} for name, resolved in model_calls],
                attempted_models=[{"provider": name, "model": resolved}
                                  for name, resolved in dict.fromkeys((item.provider, item.model) for item in attempts)],
                duration_ms=round((perf_counter() - clock) * 1000, 3),
                error_codes=[step.error_code.value for step in result.execution_history if step.error_code],
                failure_code=next(
                    (item.error_type for item in reversed(attempts) if item.error_type), None,
                ) if result.execution_status == "failed" else None,
                max_sentences=sentence_limit,
                summary_sentence_count=(len(split_sentences(result.current_summary)) if result.current_summary else None),
                llm_attempts=len(attempts),
            )
            if summary_language is not None:
                report.update(
                    source_language=source_language, rag_language=rag_language if selected_mode in RAG_MODES else None,
                    requested_summary_language=summary_language, target_language=target_language,
                    actual_summary_language=detect_language(result.current_summary or ""),
                    language_retries=manager.language_retries,
                )
            task = asyncio.create_task(
                self.reasoning.save_run(result, selected_mode, owner, existing_request=True, parameters=report,
                                        queue_review=queue_review)
            )
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                finalized = True
                raise
            finalized = True

        def step_failure(name: str, status: str, error_code: ErrorCode) -> None:
            state.record_step(name=name, started_at=started_at, started_clock=clock,
                              input_summary=state.observation(), status=status, error_code=error_code)

        config = self.settings
        manager = LLMManager(config, record, allow_external=allow_external, providers=self.providers)

        def client(name: str) -> ManagedAgentClient:
            chosen_provider, chosen_model = judge if name == "Critic" and judge else (provider, model)
            return ManagedAgentClient(
                manager, state.request_id, name, provider=chosen_provider, model=chosen_model, seed=seed,
                schema=CriticResult.model_json_schema() if name == "Critic" else None,
                validate_response=parse_critic_response if name == "Critic" else _validate_plain_summary,
            )

        # Внутренний таймаут применяется к каждой попытке. Внешний лимит агента учитывает повторы и резерв.
        timeout = 2 * config.llm_max_attempts * (config.llm_timeout_seconds + 2 * config.llm_retry_delay_seconds) + 10
        supervisor = Supervisor(
            lambda: client("Summarizer"), save, observe_step=observe,
            validator_factory=lambda: Validator(check_unsupported_names=True, check_source_claims=True,
                                                max_sentences=sentence_limit),
            summarizer_factory=lambda: Summarizer(
                client("Summarizer"), temperature=config.summarizer_temperature if temperature is None else temperature,
                max_tokens=config.summarizer_max_tokens, max_sentences=sentence_limit,
                enforce_sentence_limit=mode not in REVIEW_MODES and (max_sentences is not None or summary_language is not None),
            ),
            critic_factory=lambda: Critic(client("Critic"), timeout_seconds=timeout,
                                          temperature=config.critic_temperature, max_tokens=config.critic_max_tokens),
            editor_factory=lambda: Editor(client("Editor"), timeout_seconds=timeout,
                                          temperature=config.editor_temperature, max_tokens=config.editor_max_tokens,
                                          max_sentences=sentence_limit),
        )
        start = asyncio.create_task(self.calls.start_request(state.request_id, news, mode, user_id))
        try:
            await asyncio.shield(start)
            for name, enrichment, code in (
                ("Glossary", self.glossary if mode in {"full", "+dict"} else None, ErrorCode.GLOSSARY_FAILED),
                ("RAG", self.rag if mode in RAG_MODES else None, ErrorCode.RAG_FAILED),
            ):
                required = mode == "+dict" if name == "Glossary" else mode in RAG_MODES
                if not required and enrichment is None:
                    continue
                if enrichment is None:
                    state.execution_status = "incomplete_configuration"
                    step_failure(name, "failed", code)
                    await save(state, mode, user_id)
                    return report
                try:
                    rag_start, rag_clock, inputs = datetime.now(UTC), perf_counter(), state.observation()
                    enriched = await enrichment(state)
                    if enriched.request_id != state.request_id or enriched.original_text != state.original_text:
                        raise ValueError("RAG изменил входной запрос.")
                    state = enriched
                    if required or state.glossary_context:
                        state.record_step(name=name, started_at=rag_start, started_clock=rag_clock,
                                          input_summary=inputs, status="ok")
                except Exception:
                    state.execution_status = "failed"
                    step_failure(name, "failed", code)
                    await save(state, mode, user_id)
                    return report
            await supervisor.run(state, mode=mode, user_id=user_id)
        except asyncio.CancelledError:
            await start
            if not finalized:
                state.execution_status = "cancelled"
                step_failure("Supervisor", "cancelled", ErrorCode.CANCELLED)
                await save(state, mode, user_id)
            raise
        return report

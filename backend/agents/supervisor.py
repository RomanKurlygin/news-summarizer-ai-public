"""Оркестрация агентов с отдельными экземплярами на каждый запуск."""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from time import perf_counter
from uuid import UUID

from backend.agents.base import Agent
from backend.agents.critic import Critic
from backend.agents.editor import Editor
from backend.agents.state import CRITIC_THRESHOLD, REVIEW_MODES, AgentState, ErrorCode, PipelineMode
from backend.agents.summarizer import Summarizer
from backend.agents.validator import Validator
from backend.data.llm_client import BaseLLMClient
from backend.utils.text_processing import collapse_whitespace

AgentFactory = Callable[[], Agent]
RunRecorder = Callable[[AgentState, PipelineMode, UUID | None], Awaitable[None]]
StepObserver = Callable[[AgentState], None]


def _candidate_rank(state: AgentState) -> tuple[bool, bool, bool, float]:
    return (
        not state.critical_factual_error,
        state.validation_passed is True and not state.validation_errors,
        not state.factual_review_required,
        state.critic_score if state.critic_score is not None else -1.0,
    )


def _fail_review_step(state: AgentState, code: ErrorCode) -> None:
    state.execution_status = "failed"
    state.execution_history[-1].status = "failed"
    state.execution_history[-1].error_code = code


class Supervisor:
    def __init__(
        self,
        llm_client_factory: Callable[[], BaseLLMClient],
        save_run: RunRecorder,
        *,
        summarizer_factory: AgentFactory | None = None,
        observe_step: StepObserver | None = None,
        validator_factory: AgentFactory | None = Validator,
        critic_factory: AgentFactory | None = None,
        critic_llm_client_factory: Callable[[], BaseLLMClient] | None = None,
        editor_factory: AgentFactory | None = None,
        editor_llm_client_factory: Callable[[], BaseLLMClient] | None = None,
    ) -> None:
        self._llm_client_factory = llm_client_factory
        self._summarizer_factory = summarizer_factory
        self._observe_step = observe_step
        self._save_run = save_run
        self._validator_factory = validator_factory
        self._critic_factory = critic_factory
        if critic_factory is None and critic_llm_client_factory is not None:
            self._critic_factory = lambda: Critic(critic_llm_client_factory())
        self._editor_factory = editor_factory
        if editor_factory is None and editor_llm_client_factory is not None:
            self._editor_factory = lambda: Editor(editor_llm_client_factory())

    async def run(
        self, state: AgentState, *, mode: PipelineMode = "full", user_id: UUID | None = None
    ) -> AgentState:
        started_at = datetime.now(UTC)
        started_clock = perf_counter()
        work = AgentState(
            request_id=state.request_id,
            original_text=state.original_text,
            target_language=state.target_language,
            rag_context=state.rag_context,
            rag_document_ids=state.rag_document_ids,
            rag_search_mode=state.rag_search_mode,
            glossary_context=state.glossary_context,
            max_iterations=state.max_iterations,
            execution_status="running",
            execution_history=state.execution_history,
        )
        inputs = work.observation()
        agents: dict[str, Agent] = {}
        work = await self._run_step(
            work, agents, "Summarizer", self._summarizer_factory or (lambda: Summarizer(self._llm_client_factory()))
        )
        if work.execution_status != "failed":
            if mode in REVIEW_MODES:
                work = await self._review(work, agents)
            else:
                work.execution_status = "draft"

        if work.selected_iteration is None and work.current_summary:
            work.selected_iteration = work.iteration
        work.record_step(
            name="Supervisor", started_at=started_at, started_clock=started_clock,
            input_summary=inputs, status=work.execution_status,
        )
        await self._save_run(work, mode, user_id)
        return work

    async def _review(self, state: AgentState, agents: dict[str, Agent]) -> AgentState:
        if self._validator_factory is None:
            state.execution_status = "incomplete_configuration"
            return state

        best = None
        while True:
            state.validation_passed = None
            state.validation_errors = []
            state.validation_warnings = []
            state.validation_error_codes = []
            state.validation_warning_codes = []
            state.factual_findings = []
            state.reset_critique()

            state = await self._run_step(state, agents, "Validator", self._validator_factory)
            if state.execution_status == "failed":
                break
            if state.validation_passed is None:
                _fail_review_step(state, ErrorCode.INVALID_VALIDATION_RESULT)
                break
            if self._critic_factory is None:
                state.execution_status = "incomplete_configuration"
                break
            state = await self._run_step(state, agents, "Critic", self._critic_factory)
            if state.execution_status == "failed":
                break
            if state.critic_fallback_used:
                _fail_review_step(state, ErrorCode.LLM_FAILED)
                break
            if state.critic_score is None or state.critic_passed is None:
                _fail_review_step(state, ErrorCode.INVALID_CRITIC_RESULT)
                break

            if (
                state.current_summary and state.current_summary.strip()
                and state.validation_passed is True and not state.validation_errors
                and not state.factual_review_required
                and state.critic_passed is True
                and state.critic_score >= CRITIC_THRESHOLD
                and not state.critical_factual_error and not state.missing_key_fact
            ):
                state.execution_status = "verified"
                return state
            if best is None or _candidate_rank(state) > _candidate_rank(best):
                best = state.model_copy(deep=True, update={"execution_history": []})
            if self._editor_factory is None:
                state.execution_status = "incomplete_configuration"
                break
            if state.iteration >= state.max_iterations:
                state.execution_status = "iteration_limit"
                break

            previous_summary = collapse_whitespace(state.current_summary or "")
            state.iteration += 1
            state.editor_changed = None
            state = await self._run_step(state, agents, "Editor", self._editor_factory)
            if state.execution_status == "failed":
                break
            if (
                state.editor_changed is not True
                or collapse_whitespace(state.current_summary or "") == previous_summary
            ):
                state.editor_changed = False
                state.execution_status = "unchanged"
                break

        if best is None:
            return state
        result = best.model_copy(update={
            "iteration": state.iteration,
            "selected_iteration": best.iteration,
            "editor_changed": state.editor_changed,
            "execution_status": state.execution_status,
            "execution_history": state.execution_history,
        })
        if (result.execution_status in {"iteration_limit", "unchanged"}
                and result.validation_passed is True and not result.critical_factual_error
                and result.factual_review_required):
            result.execution_status = "needs_review"
        return result

    async def _run_step(
        self, state: AgentState, agents: dict[str, Agent], name: str, factory: AgentFactory
    ) -> AgentState:
        if name not in agents:
            started_at = datetime.now(UTC)
            started_clock = perf_counter()
            try:
                agents[name] = factory()
            except Exception:
                state.execution_status = "failed"
                state.record_step(
                    name=name, started_at=started_at, started_clock=started_clock,
                    input_summary=state.observation(), status="failed",
                    error_code=ErrorCode.AGENT_INITIALIZATION_FAILED,
                )
        if name in agents:
            state = await agents[name].process(state)
        if self._observe_step is not None:
            self._observe_step(state)
        return state

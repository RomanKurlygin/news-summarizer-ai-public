"""Запись запроса и шагов Supervisor одной транзакцией."""

import json
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.agents.state import CRITIC_THRESHOLD, AgentState, CriticCriterion, ExecutionStep, PipelineMode
from backend.db.models import ReasoningLog, SummarizationRequest
from backend.hitl.review import pending_review

_CRITIC_ACTIONS = {
    CriticCriterion.FACTUAL_ACCURACY: "Сверить факты резюме с исходной новостью.",
    CriticCriterion.COMPLETENESS: "Вернуть пропущенные ключевые факты.",
    CriticCriterion.COHERENCE: "Уточнить логическую последовательность событий.",
    CriticCriterion.READABILITY: "Упростить формулировки.",
    CriticCriterion.CONCISENESS: "Сократить повторы и второстепенные детали.",
}


def _step_output(step: ExecutionStep) -> str:
    if step.agent_name != "Critic":
        return json.dumps(step.output_summary)

    output = step.output_summary
    criteria = step.critic_criteria_scores
    actions = {criterion for criterion, score in criteria.items() if score < CRITIC_THRESHOLD}
    if output["critical_factual_error"]:
        actions.add(CriticCriterion.FACTUAL_ACCURACY)
    if output["missing_key_fact"]:
        actions.add(CriticCriterion.COMPLETENESS)

    suggestions = [message for criterion, message in _CRITIC_ACTIONS.items() if criterion in actions]
    if output["critic_fallback_used"]:
        suggestions = ["Повторить оценку после сбоя LLM."]
    elif output["critic_suggestions_count"] and not suggestions:
        suggestions = ["Уточнить формулировку по замечаниям Critic."]

    return json.dumps(
        {**output, "critic_criteria_scores": criteria, "critic_suggestions": suggestions},
        ensure_ascii=False,
    )


class ReasoningLogStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def save_run(
        self, state: AgentState, mode: PipelineMode, user_id: UUID | None,
        *, existing_request: bool = False, parameters: dict | None = None, queue_review: bool = False,
    ) -> None:
        request = SummarizationRequest(
            id=state.request_id,
            user_id=user_id,
            input_text=state.original_text,
            summary=state.current_summary,
            mode=mode,
            status=state.execution_status,
            parameters={
                "max_iterations": state.max_iterations,
                "selected_iteration": state.selected_iteration,
                **(parameters or {}),
            },
        )
        steps = [
            ReasoningLog(
                request_id=state.request_id,
                agent_name=step.agent_name,
                input_summary=json.dumps(step.input_summary),
                output_summary=_step_output(step),
                model_name=step.model_name,
                started_at=step.started_at,
                finished_at=step.finished_at,
                status=step.status,
                errors=(
                    ([step.error_code.value] if step.error_code is not None else [])
                    + [code.value for code in step.validation_error_codes]
                ),
                warnings=[code.value for code in step.validation_warning_codes] or (
                    ["validation_warnings_present"]
                    if step.output_summary["validation_warnings"] else []
                ),
                parameters={
                    **step.parameters,
                    "step": position,
                    "iteration": step.iteration,
                    "duration_ms": step.duration_ms,
                },
            )
            for position, step in enumerate(state.execution_history, start=1)
        ]
        async with self._sessions.begin() as session:
            if existing_request:
                updated = await session.scalar(
                    update(SummarizationRequest)
                    .where(SummarizationRequest.id == state.request_id, SummarizationRequest.status == "running")
                    .values(summary=request.summary, status=request.status, parameters=request.parameters)
                    .returning(SummarizationRequest.id)
                )
                if updated is None:
                    raise RuntimeError("Запрос для завершения отсутствует или уже завершён.")
                session.add_all(steps)
            else:
                request.reasoning_logs = steps
                session.add(request)
            if queue_review and user_id is not None and (item := pending_review(state, mode)):
                session.add(item)

    async def list_for_request(self, request_id: UUID) -> list[ReasoningLog]:
        query = (
            select(ReasoningLog)
            .where(ReasoningLog.request_id == request_id)
            .order_by(
                ReasoningLog.parameters["step"].as_integer(),
                ReasoningLog.started_at,
                ReasoningLog.id,
            )
        )
        async with self._sessions() as session:
            return list(await session.scalars(query))

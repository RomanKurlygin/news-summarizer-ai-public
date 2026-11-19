"""Очередь ручной проверки и сохранение решений в одной транзакции."""

from backend.agents.state import REVIEW_MODES, AgentState, PipelineMode
from backend.db.models import UncertainExample


def pending_review(state: AgentState, mode: PipelineMode) -> UncertainExample | None:
    if (mode not in REVIEW_MODES or not state.current_summary
            or state.execution_status in {"verified", "failed", "cancelled", "incomplete_configuration"}):
        return None
    reasons = [code.value for code in state.validation_error_codes]
    reasons.extend(finding.code.value for finding in state.factual_findings)
    if state.critic_passed is False:
        reasons.append("critic_rejected")
    if state.missing_key_fact:
        reasons.append("missing_key_fact")
    if state.critical_factual_error:
        reasons.append("critical_factual_error")
    if not reasons:
        return None
    return UncertainExample(
        request_id=state.request_id, input_text=state.original_text, model_output=state.current_summary,
        critic_score=state.critic_score, status="pending", reasons=list(dict.fromkeys(reasons)),
    )

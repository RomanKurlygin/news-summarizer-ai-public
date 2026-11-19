"""Состояние одного запуска и наблюдаемые события агентов."""

from datetime import UTC, datetime
from enum import StrEnum
from time import perf_counter
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.utils.text_processing import Language

MAX_EDITOR_ITERATIONS = 2
CRITIC_THRESHOLD = 7
RunMode = Literal["baseline", "full"]
PipelineMode = Literal["baseline", "+dict", "+rag", "+agents", "full"]
REVIEW_MODES = {"full", "+agents"}
RAG_MODES = {"full", "+rag"}
RagSearchMode = Literal["same_language", "cross_language", "mixed", "unfiltered", "no_results"]
ExecutionStatus = Literal[
    "pending", "running", "draft", "verified", "failed",
    "incomplete_configuration", "iteration_limit", "unchanged", "cancelled", "needs_review",
]
Observation = dict[str, int | float | bool | None]
Score = Annotated[float, Field(strict=True, ge=0, le=10, allow_inf_nan=False)]


class ErrorCode(StrEnum):
    EMPTY_NEWS = "empty_news"
    EMPTY_SUMMARY = "empty_summary"
    LLM_TIMEOUT = "llm_timeout"
    LLM_FAILED = "llm_generation_failed"
    LLM_INVALID_RESPONSE = "llm_invalid_response"
    LLM_LANGUAGE_MISMATCH = "language_mismatch"
    SENTENCE_LIMIT = "sentence_limit"
    AGENT_FAILED = "agent_failed"
    AGENT_INITIALIZATION_FAILED = "agent_initialization_failed"
    INVALID_VALIDATION_RESULT = "invalid_validation_result"
    INVALID_CRITIC_RESULT = "invalid_critic_result"
    RAG_FAILED = "rag_failed"
    GLOSSARY_FAILED = "glossary_failed"
    CANCELLED = "cancelled"


class CriticCriterion(StrEnum):
    FACTUAL_ACCURACY = "factual_accuracy"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    READABILITY = "readability"
    CONCISENESS = "conciseness"


class ValidationCode(StrEnum):
    UNSUPPORTED_ATTRIBUTION = "unsupported_attribution"
    MISSING_ORIGINAL_TEXT = "missing_original_text"
    MISSING_SUMMARY = "missing_summary"
    SENTENCE_COUNT = "sentence_count"
    SUMMARY_TOO_LONG = "summary_too_long"
    DUPLICATE_SENTENCE = "duplicate_sentence"
    VERBATIM_SENTENCE = "verbatim_sentence"
    UNSUPPORTED_NAME = "unsupported_name"
    UNATTRIBUTED_ALLEGATION = "unattributed_allegation"
    UNSUPPORTED_VEHICLE_IMPACT = "unsupported_vehicle_impact"
    UNSUPPORTED_CAUSAL_LINK = "unsupported_causal_link"
    UNSUPPORTED_NUMBER = "unsupported_number"
    UNCONFIRMED_ENTITY = "unconfirmed_entity"
    UNCONFIRMED_NUMBER = "unconfirmed_number"
    LOST_QUALIFICATION = "lost_qualification"
    UNCERTAIN_QUALIFICATION = "uncertain_qualification"
    POTENTIAL_MISSING_NAME = "potential_missing_name"
    POTENTIAL_MISSING_DATE = "potential_missing_date"
    POTENTIAL_MISSING_NUMBER = "potential_missing_number"


class FactualFinding(BaseModel):
    code: ValidationCode
    severity: Literal["error", "review"]
    category: Literal["person", "place", "organization", "name", "number", "date", "claim"]
    subject: str


class ExecutionStep(BaseModel):
    agent_name: str
    iteration: int
    started_at: datetime
    finished_at: datetime
    duration_ms: float
    status: str
    input_summary: Observation
    output_summary: Observation
    parameters: dict[str, int | float] = Field(default_factory=dict)
    model_name: str | None = None
    error_code: ErrorCode | None = None
    validation_error_codes: list[ValidationCode] = Field(default_factory=list)
    validation_warning_codes: list[ValidationCode] = Field(default_factory=list)
    critic_criteria_scores: dict[CriticCriterion, Score] = Field(default_factory=dict)


class AgentState(BaseModel):
    model_config = ConfigDict(validate_assignment=True, hide_input_in_errors=True)

    request_id: UUID = Field(default_factory=uuid4)
    original_text: str = Field(default="", repr=False)
    target_language: Language | None = None
    current_summary: str | None = Field(default=None, repr=False)
    rag_context: list[str] = Field(default_factory=list, repr=False)
    rag_document_ids: list[UUID] = Field(default_factory=list, repr=False)
    rag_search_mode: RagSearchMode | None = None
    glossary_context: list[str] = Field(default_factory=list, repr=False)
    iteration: int = Field(default=0, ge=0, le=MAX_EDITOR_ITERATIONS)
    selected_iteration: int | None = Field(default=None, ge=0, le=MAX_EDITOR_ITERATIONS)
    max_iterations: int = Field(default=MAX_EDITOR_ITERATIONS, ge=0, le=MAX_EDITOR_ITERATIONS)
    validation_passed: bool | None = None
    validation_errors: list[str] = Field(default_factory=list, repr=False)
    validation_warnings: list[str] = Field(default_factory=list, repr=False)
    validation_error_codes: list[ValidationCode] = Field(default_factory=list)
    validation_warning_codes: list[ValidationCode] = Field(default_factory=list)
    factual_findings: list[FactualFinding] = Field(default_factory=list, repr=False)
    critic_score: float | None = Field(default=None, ge=0, le=10)
    critic_reasoning: str | None = Field(default=None, repr=False)
    critic_suggestions: list[str] = Field(default_factory=list, repr=False)
    critic_passed: bool | None = None
    critic_criteria_scores: dict[CriticCriterion, Score] = Field(default_factory=dict)
    critical_factual_error: bool | None = None
    missing_key_fact: bool | None = None
    critic_fallback_used: bool = False
    editor_changed: bool | None = None
    execution_status: ExecutionStatus = "pending"
    execution_history: list[ExecutionStep] = Field(default_factory=list)

    @field_validator("original_text", mode="before")
    @classmethod
    def allow_missing_original(cls, value: object) -> object:
        return "" if value is None else value

    def reset_critique(self) -> None:
        """Подготовить поля оценки к новой проверке."""
        self.critic_score = None
        self.critic_reasoning = None
        self.critic_suggestions = []
        self.critic_passed = None
        self.critic_criteria_scores = {}
        self.critical_factual_error = None
        self.missing_key_fact = None
        self.critic_fallback_used = False

    @property
    def factual_review_required(self) -> bool:
        return any(finding.severity == "review" for finding in self.factual_findings)

    def observation(self) -> Observation:
        """Сводка для журнала из чисел и флагов."""
        return {
            "original_chars": len(self.original_text),
            "summary_chars": len(self.current_summary or ""),
            "rag_items": len(self.rag_context),
            "glossary_items": len(self.glossary_context),
            "validation_passed": self.validation_passed,
            "validation_errors": len(self.validation_errors),
            "validation_warnings": len(self.validation_warnings),
            "factual_review_required": self.factual_review_required,
            "critic_score": self.critic_score,
            "critic_passed": self.critic_passed,
            "critical_factual_error": self.critical_factual_error,
            "missing_key_fact": self.missing_key_fact,
            "critic_fallback_used": self.critic_fallback_used,
            "critic_suggestions_count": len(self.critic_suggestions),
            "editor_changed": self.editor_changed,
            "selected_iteration": self.selected_iteration,
        }

    def record_step(
        self,
        *,
        name: str,
        started_at: datetime,
        started_clock: float,
        input_summary: Observation,
        status: str,
        parameters: dict[str, int | float] | None = None,
        model_name: str | None = None,
        error_code: ErrorCode | None = None,
    ) -> None:
        is_validation = name == "Validator" and error_code is None
        self.execution_history.append(
            ExecutionStep(
                agent_name=name,
                iteration=self.iteration,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                duration_ms=round((perf_counter() - started_clock) * 1000, 3),
                status=status,
                input_summary=input_summary,
                output_summary=self.observation(),
                parameters=parameters or {},
                model_name=model_name,
                error_code=error_code,
                validation_error_codes=self.validation_error_codes.copy() if is_validation else [],
                validation_warning_codes=self.validation_warning_codes.copy() if is_validation else [],
                critic_criteria_scores=self.critic_criteria_scores.copy() if name == "Critic" else {},
            )
        )

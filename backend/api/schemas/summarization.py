"""Публичные контракты суммаризации и аудита."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, StringConstraints, model_validator

from backend.agents.state import ExecutionStatus, FactualFinding, PipelineMode, RagSearchMode, RunMode
from backend.utils.text_processing import Language, SummaryLanguage

MAX_NEWS_CHARS = 8000
MAX_GROUP_SIZE = 4
NewsText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_NEWS_CHARS)]
Category = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
LocalProvider = Literal["ollama", "transformers"]
RagLanguage = Annotated[Language | None, Field(validation_alias=AliasChoices("rag_language", "language"))]


class ModelChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: LocalProvider = "ollama"
    model: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255,
                                            pattern=r"^[\w.][\w./:@+-]*$")] | None = None
    temperature: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_model_id(self):
        if self.model is not None:
            if any(part in {"", ".", ".."} for part in self.model.split("/")):
                raise ValueError("Укажите идентификатор модели.")
            if self.provider == "transformers" and (":" in self.model or self.model.count("/") > 1):
                raise ValueError("Укажите идентификатор модели Hugging Face.")
            if self.provider == "ollama" and self.model.lower().endswith(("-cloud", ":cloud")):
                raise ValueError("HTTP API принимает локальные модели.")
        return self


class Configuration(ModelChoice):
    mode: RunMode = "full"
    max_sentences: int = Field(default=3, strict=True, ge=1, le=3)
    summary_language: SummaryLanguage = "auto"


class SummarizeInput(Configuration):
    text: NewsText
    category: Category | None = None
    rag_language: RagLanguage = None


class BatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[SummarizeInput] = Field(min_length=1, max_length=MAX_GROUP_SIZE)


class VariantsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: NewsText
    category: Category | None = None
    rag_language: RagLanguage = None
    variants: list[Configuration] = Field(min_length=2, max_length=MAX_GROUP_SIZE)


class EnsembleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: NewsText
    category: Category | None = None
    rag_language: RagLanguage = None
    summary_language: SummaryLanguage = "auto"
    max_sentences: int = Field(default=3, strict=True, ge=1, le=3)
    candidates: list[ModelChoice] = Field(min_length=2, max_length=MAX_GROUP_SIZE)


class Revision(BaseModel):
    iteration: int
    summary: str


class SummaryResult(BaseModel):
    request_id: UUID
    text: str
    status: ExecutionStatus
    mode: PipelineMode
    initial_summary: str | None = None
    final_summary: str | None = None
    provider: LocalProvider | None = None
    model: str | None = None
    max_sentences: int = 3
    validation_passed: bool | None = None
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    factual_findings: list[FactualFinding] = Field(default_factory=list)
    factual_review_required: bool = False
    critic_score: float | None = None
    critic_passed: bool | None = None
    iterations: int = 0
    selected_iteration: int | None = None
    revisions: list[Revision] = Field(default_factory=list)
    duration_ms: float | None = None
    rag_ids: list[UUID] = Field(default_factory=list)
    rag_search_mode: RagSearchMode | None = None
    summary_sentence_count: int | None = None
    source_language: Language | None = None
    rag_language: str | None = None
    requested_summary_language: SummaryLanguage | None = None
    target_language: Language | None = None
    actual_summary_language: Language | None = None
    language_retries: int = 0
    cache_hit: bool = False


class RequestDetails(SummaryResult):
    user_id: UUID | None
    created_at: datetime


class RequestPreview(BaseModel):
    request_id: UUID
    mode: PipelineMode
    status: ExecutionStatus
    preview: str
    created_at: datetime


class CallTrace(BaseModel):
    agent: str
    provider: str
    model: str
    status: str
    duration_ms: int | None
    tokens: int | None
    created_at: datetime
    language_retry: bool = False
    target_language: Language | None = None
    actual_language: Language | None = None
    language_mismatch: bool = False


class CandidateError(BaseModel):
    code: str
    message: str
    http_status: int
    request_id: UUID


class CandidateResult(BaseModel):
    configuration: Configuration
    result: SummaryResult | None = None
    error: CandidateError | None = None


class GroupResult(BaseModel):
    results: list[CandidateResult]


class EnsembleResult(GroupResult):
    criterion: str
    winner: SummaryResult | None


class ReasoningStep(BaseModel):
    id: UUID
    step: int | None
    iteration: int | None
    agent: str
    status: str
    model: str | None
    started_at: datetime
    finished_at: datetime | None
    duration_ms: float | None
    validation_passed: bool | None = None
    factual_review_required: bool | None = None
    critic_score: float | None = None
    critic_passed: bool | None = None
    critic_criteria_scores: dict[str, float] = Field(default_factory=dict)
    critical_factual_error: bool | None = None
    missing_key_fact: bool | None = None
    critic_fallback_used: bool | None = None
    editor_changed: bool | None = None
    errors: list[str]
    warnings: list[str]


class RequestReasoning(BaseModel):
    request_id: UUID
    user_id: UUID | None
    status: ExecutionStatus
    steps: list[ReasoningStep]
    revisions: list[Revision]

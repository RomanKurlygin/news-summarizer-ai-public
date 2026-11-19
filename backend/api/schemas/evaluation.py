"""Контракты оценки одной модели на отложенных новостях."""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints

from backend.agents.state import ExecutionStatus
from backend.api.schemas.summarization import MAX_GROUP_SIZE, CandidateError, Configuration
from backend.evaluation.metrics_calculator import SummaryMetrics

TestID = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]


class EvaluateInput(Configuration):
    test_ids: list[TestID] | None = Field(default=None, min_length=1, max_length=MAX_GROUP_SIZE)


class EvaluationItem(BaseModel):
    test_id: UUID
    source_document_id: str | None
    request_id: UUID
    status: ExecutionStatus
    metrics: SummaryMetrics | None = None
    error: CandidateError | None = None


class EvaluationOutput(BaseModel):
    id: UUID
    user_id: UUID
    created_at: datetime
    model_name: str
    configuration: Configuration
    sample_size: int
    scored_count: int
    evaluation_time: float
    metrics: SummaryMetrics
    metric_metadata: dict
    items: list[EvaluationItem]


class ExperimentHistoryItem(BaseModel):
    id: UUID
    experiment_id: UUID
    user_id: UUID
    model_name: str
    provider: str
    mode: str
    sample_size: int
    created_at: datetime
    evaluation_time: float
    aggregate: dict

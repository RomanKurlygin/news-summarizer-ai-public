"""Модели общей схемы PostgreSQL и SQLite."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    false,
    func,
    true,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_name)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


class Record:
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class User(Record, Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'editor', 'admin')", name="valid_role"),
    )

    email: Mapped[str] = mapped_column(String(320), unique=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), server_default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=true())


class KnowledgeBase(Record, Base):
    __tablename__ = "knowledge_base"
    __table_args__ = (
        Index("ix_knowledge_base_category_language", "category", "language"),
        Index("ix_knowledge_base_content_hash", "content_hash", unique=True),
    )

    source_text: Mapped[str] = mapped_column(Text)
    target_text: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(120), server_default="general")
    language: Mapped[str] = mapped_column(String(16), server_default="und")
    source_url: Mapped[str | None] = mapped_column(Text)
    source_dataset: Mapped[str | None] = mapped_column(String(160))
    source_document_id: Mapped[str | None] = mapped_column(String(255))
    source_split: Mapped[str | None] = mapped_column(String(16))
    content_hash: Mapped[str | None] = mapped_column(String(64))


class Glossary(Record, Base):
    __tablename__ = "glossary"
    __table_args__ = (
        UniqueConstraint("term", "category", "language", name="uq_glossary_term_category_language"),
        Index("ix_glossary_category_language", "category", "language"),
    )

    term: Mapped[str] = mapped_column(String(255))
    definition: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(120), server_default="general")
    language: Mapped[str] = mapped_column(String(16), server_default="und")
    aliases: Mapped[list[str] | None] = mapped_column(JSON)
    source: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    review_required: Mapped[bool] = mapped_column(Boolean, server_default=false())


class SummarizationRequest(Record, Base):
    __tablename__ = "summarization_requests"
    __table_args__ = (Index("ix_summarization_requests_user_created", "user_id", "created_at"),)

    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    input_text: Mapped[str] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String(32), server_default="full")
    status: Mapped[str] = mapped_column(String(32), server_default="pending")
    parameters: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    reasoning_logs: Mapped[list["ReasoningLog"]] = relationship(
        back_populates="request", lazy="raise", passive_deletes="all"
    )
    llm_calls: Mapped[list["LLMCall"]] = relationship(
        back_populates="request", lazy="raise", passive_deletes="all"
    )


class UncertainExample(Record, Base):
    __tablename__ = "uncertain_examples"
    __table_args__ = (
        CheckConstraint("critic_score BETWEEN 0 AND 10", name="critic_score_range"),
        Index("ix_uncertain_examples_status_created", "status", "created_at"),
    )

    request_id: Mapped[UUID] = mapped_column(
        ForeignKey("summarization_requests.id", ondelete="CASCADE"), unique=True
    )
    input_text: Mapped[str] = mapped_column(Text)
    model_output: Mapped[str] = mapped_column(Text)
    critic_score: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), server_default="pending")
    reasons: Mapped[list[str] | None] = mapped_column(JSON)


class CacheEntry(Record, Base):
    __tablename__ = "cache"
    __table_args__ = (CheckConstraint("hits >= 0", name="nonnegative_hits"),)

    input_key: Mapped[str] = mapped_column(String(255), unique=True)
    output: Mapped[dict[str, Any]] = mapped_column(JSON)
    hits: Mapped[int] = mapped_column(server_default="0")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)


class ReasoningLog(Record, Base):
    __tablename__ = "reasoning_logs"
    __table_args__ = (Index("ix_reasoning_logs_request_started", "request_id", "started_at"),)

    request_id: Mapped[UUID] = mapped_column(
        ForeignKey("summarization_requests.id", ondelete="CASCADE")
    )
    agent_name: Mapped[str] = mapped_column(String(100))
    input_summary: Mapped[str | None] = mapped_column(Text)
    output_summary: Mapped[str | None] = mapped_column(Text)
    model_name: Mapped[str | None] = mapped_column(String(255))
    parameters: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    errors: Mapped[list[str] | None] = mapped_column(JSON)
    warnings: Mapped[list[str] | None] = mapped_column(JSON)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32))

    request: Mapped[SummarizationRequest] = relationship(
        back_populates="reasoning_logs", lazy="raise"
    )


class LLMCall(Record, Base):
    __tablename__ = "llm_calls"
    __table_args__ = (
        CheckConstraint("tokens_used >= 0", name="nonnegative_tokens"),
        CheckConstraint("latency_ms >= 0", name="nonnegative_latency"),
        Index("ix_llm_calls_request_created", "request_id", "created_at"),
    )

    request_id: Mapped[UUID] = mapped_column(
        ForeignKey("summarization_requests.id", ondelete="CASCADE")
    )
    agent_name: Mapped[str] = mapped_column(String(100))
    model_name: Mapped[str] = mapped_column(String(255))
    provider: Mapped[str] = mapped_column(String(64))
    parameters: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    prompt_hash: Mapped[str] = mapped_column(String(64))
    response_hash: Mapped[str | None] = mapped_column(String(64))
    tokens_used: Mapped[int | None]
    latency_ms: Mapped[int | None]
    status: Mapped[str] = mapped_column(String(32))

    request: Mapped[SummarizationRequest] = relationship(
        back_populates="llm_calls", lazy="raise"
    )


class ExperimentJob(Record, Base):
    __tablename__ = "experiment_jobs"

    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    specification: Mapped[dict[str, Any]] = mapped_column(JSON)
    progress: Mapped[int] = mapped_column(server_default="0")
    total: Mapped[int]
    lease_token: Mapped[UUID | None]
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(String(100))


class EvaluationResult(Record, Base):
    __tablename__ = "evaluation_results"
    __table_args__ = (
        CheckConstraint("sample_size > 0", name="positive_sample_size"),
        CheckConstraint("evaluation_time >= 0", name="nonnegative_evaluation_time"),
        Index("ix_evaluation_results_model_mode_size", "model_name", "mode", "sample_size"),
        Index("ix_evaluation_results_user_created", "user_id", "created_at"),
    )

    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    experiment_id: Mapped[UUID | None] = mapped_column(ForeignKey("experiment_jobs.id", ondelete="CASCADE"), index=True)
    model_name: Mapped[str] = mapped_column(String(255))
    mode: Mapped[str] = mapped_column(String(32))
    sample_size: Mapped[int]
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    evaluation_time: Mapped[float]


class ModelComparison(Record, Base):
    __tablename__ = "model_comparisons"
    __table_args__ = (CheckConstraint("sample_size > 0", name="positive_sample_size"),)

    experiment_id: Mapped[UUID | None] = mapped_column(ForeignKey("experiment_jobs.id", ondelete="CASCADE"), index=True)
    models_compared: Mapped[list[str]] = mapped_column(JSON)
    sample_size: Mapped[int] = mapped_column(index=True)
    results_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    winner: Mapped[str | None] = mapped_column(String(255))


class Feedback(Record, Base):
    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint("decision IN ('approved', 'rejected')", name="valid_decision"),
        Index("ix_feedback_example_created", "uncertain_example_id", "created_at"),
    )

    uncertain_example_id: Mapped[UUID] = mapped_column(
        ForeignKey("uncertain_examples.id", ondelete="CASCADE")
    )
    editor_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    decision: Mapped[str] = mapped_column(String(16))
    corrected_summary: Mapped[str | None] = mapped_column(Text)
    comment: Mapped[str | None] = mapped_column(Text)

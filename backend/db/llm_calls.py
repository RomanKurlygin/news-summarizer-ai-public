"""Независимый журнал фактических попыток генерации."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.data.llm_manager import CallAttempt
from backend.db.models import LLMCall, SummarizationRequest


class LLMCallStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def start_request(self, request_id: UUID, news: str, mode: str, user_id: UUID | None) -> None:
        async with self.sessions.begin() as session:
            session.add(SummarizationRequest(id=request_id, user_id=user_id, input_text=news,
                                             mode=mode, status="running"))

    async def record(self, attempt: CallAttempt, order: int) -> None:
        total = (attempt.input_tokens + attempt.output_tokens
                 if attempt.input_tokens is not None and attempt.output_tokens is not None else None)
        async with self.sessions.begin() as session:
            session.add(LLMCall(
                request_id=attempt.request_id, agent_name=attempt.agent_name, provider=attempt.provider,
                model_name=attempt.model, created_at=attempt.created_at, latency_ms=attempt.latency_ms,
                status=attempt.status, prompt_hash=attempt.prompt_hash, response_hash=attempt.response_hash,
                tokens_used=total, parameters={
                    **attempt.parameters, "sequence": order, "error_type": attempt.error_type,
                    "input_tokens": attempt.input_tokens, "output_tokens": attempt.output_tokens,
                },
            ))

    async def list_for_request(self, request_id: UUID) -> list[LLMCall]:
        query = select(LLMCall).where(LLMCall.request_id == request_id).order_by(
            LLMCall.parameters["sequence"].as_integer(), LLMCall.created_at, LLMCall.id
        )
        async with self.sessions() as session:
            return list(await session.scalars(query))

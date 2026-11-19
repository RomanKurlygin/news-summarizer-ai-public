"""Оценка резюме внедрённым LLM-клиентом и проверка ответа судьи."""

import asyncio
import json
from pathlib import Path
from statistics import fmean
from string import Template
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from backend.agents.base import Agent, AgentError
from backend.agents.state import CRITIC_THRESHOLD, AgentState, CriticCriterion, ErrorCode, Score
from backend.data.llm_client import BaseLLMClient
from backend.data.llm_types import LLMError

_PROMPT = Template((Path(__file__).parent / "prompts" / "critic.txt").read_text(encoding="utf-8"))
_FALLBACK_SCORE = 5.0
_LLM_ERRORS = {ErrorCode.LLM_TIMEOUT, ErrorCode.LLM_FAILED, ErrorCode.LLM_INVALID_RESPONSE}
Explanation = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=400)]
Suggestion = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]


class CriticResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", hide_input_in_errors=True)

    factual_accuracy: Score
    completeness: Score
    coherence: Score
    readability: Score
    conciseness: Score
    overall_score: Score
    reasoning: Explanation
    suggestions: list[Suggestion] = Field(max_length=5)
    critical_factual_error: bool
    missing_key_fact: bool

    def criteria_scores(self) -> dict[CriticCriterion, float]:
        return {criterion: getattr(self, criterion.value) for criterion in CriticCriterion}

    @model_validator(mode="after")
    def calculate_overall_score(self) -> Self:
        self.overall_score = fmean(self.criteria_scores().values())
        return self


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise ValueError("В ответе Critic повторяются поля.")
    return fields


def parse_critic_response(response: str) -> CriticResult:
    return CriticResult.model_validate(json.loads(response, object_pairs_hook=_unique_fields))


class Critic(Agent):
    llm_client: BaseLLMClient

    def __init__(
        self,
        llm_client: BaseLLMClient,
        *,
        model_name: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 700,
        timeout_seconds: float = 20.0,
    ) -> None:
        if not 0 < timeout_seconds < float("inf"):
            raise ValueError("Таймаут Critic должен быть конечным положительным числом.")
        super().__init__(
            "Critic",
            llm_client=llm_client,
            model_name=model_name,
            parameters={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout_seconds": timeout_seconds,
            },
        )

    async def _process(self, state: AgentState) -> AgentState:
        if not state.original_text.strip():
            raise AgentError(ErrorCode.EMPTY_NEWS)
        if not state.current_summary or not state.current_summary.strip():
            raise AgentError(ErrorCode.EMPTY_SUMMARY)

        data = {
            "news": state.original_text,
            "summary": state.current_summary,
            "validator_warnings": state.validation_warnings,
        }
        if state.target_language:
            data["summary_language"] = state.target_language
        prompt = _PROMPT.substitute(data=json.dumps(data, ensure_ascii=False))
        if state.target_language:
            prompt = ("Резюме может быть на другом языке по выбору пользователя. Оцени факты по смыслу и не считай "
                      "перевод или эквивалентное написание имён ошибкой.\n" + prompt)
        try:
            async with asyncio.timeout(self.parameters["timeout_seconds"]):
                response = await self.llm_client.generate(
                    prompt,
                    temperature=self.parameters["temperature"],
                    max_tokens=self.parameters["max_tokens"],
                )
        except TimeoutError:
            raise AgentError(ErrorCode.LLM_TIMEOUT) from None
        except LLMError as exc:
            raise AgentError(ErrorCode.LLM_INVALID_RESPONSE if exc.code == "invalid_response" else ErrorCode.LLM_FAILED) from None
        except Exception:
            raise AgentError(ErrorCode.LLM_FAILED) from None

        if not isinstance(response, str) or not response.strip():
            raise AgentError(ErrorCode.LLM_INVALID_RESPONSE)
        try:
            assessment = parse_critic_response(response)
        except (ValueError, RecursionError):
            raise AgentError(ErrorCode.LLM_INVALID_RESPONSE) from None

        state.critic_criteria_scores = assessment.criteria_scores()
        state.critic_score = assessment.overall_score
        state.critic_reasoning = assessment.reasoning
        state.critic_suggestions = assessment.suggestions
        state.critical_factual_error = assessment.critical_factual_error
        state.missing_key_fact = assessment.missing_key_fact
        state.critic_fallback_used = False
        state.critic_passed = (
            assessment.overall_score >= CRITIC_THRESHOLD
            and not assessment.critical_factual_error
            and not assessment.missing_key_fact
        )
        return state

    def _on_failure(self, state: AgentState, code: ErrorCode) -> None:
        state.reset_critique()
        state.critic_passed = False
        if code in _LLM_ERRORS:
            state.critic_score = _FALLBACK_SCORE
            state.critic_fallback_used = True
            state.critic_reasoning = "Оценка недоступна из-за сбоя LLM. Использован резервный балл."

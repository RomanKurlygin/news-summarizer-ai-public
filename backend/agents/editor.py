"""Исправление резюме через внедрённый LLM-клиент."""

import asyncio
import json
from pathlib import Path
from string import Template

from backend.agents.base import Agent, AgentError
from backend.agents.state import AgentState, ErrorCode
from backend.data.llm_client import BaseLLMClient
from backend.data.llm_types import LLMError
from backend.utils.text_processing import LANGUAGE_INSTRUCTIONS, collapse_whitespace

_PROMPT = Template((Path(__file__).parent / "prompts" / "editor.txt").read_text(encoding="utf-8"))


class Editor(Agent):
    llm_client: BaseLLMClient

    def __init__(
        self,
        llm_client: BaseLLMClient,
        *,
        model_name: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 180,
        timeout_seconds: float = 20.0,
        max_sentences: int = 3,
    ) -> None:
        if not 0 < timeout_seconds < float("inf"):
            raise ValueError("Таймаут Editor должен быть конечным положительным числом.")
        super().__init__(
            "Editor",
            llm_client=llm_client,
            model_name=model_name,
            parameters={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout_seconds": timeout_seconds,
            },
        )
        self.max_sentences = max_sentences

    async def _process(self, state: AgentState) -> AgentState:
        if not state.original_text.strip():
            raise AgentError(ErrorCode.EMPTY_NEWS)
        if not state.current_summary or not state.current_summary.strip():
            raise AgentError(ErrorCode.EMPTY_SUMMARY)

        data = {
            "news": state.original_text,
            "summary": state.current_summary,
            "validator_errors": state.validation_errors,
            "validator_warnings": state.validation_warnings,
            "critic_criteria_scores": state.critic_criteria_scores,
            "critic_score": state.critic_score,
            "critic_suggestions": state.critic_suggestions,
            "critical_factual_error": state.critical_factual_error,
            "missing_key_fact": state.missing_key_fact,
            "iteration": state.iteration,
        }
        if self.max_sentences < 3:
            data["max_sentences"] = self.max_sentences
        target = state.target_language
        prompt = _PROMPT.substitute(language_rule=LANGUAGE_INSTRUCTIONS[target] if target else
                                    "Используй язык исходной новости.", data=json.dumps(data, ensure_ascii=False))
        language_option = {"target_language": target} if target else {}
        try:
            async with asyncio.timeout(self.parameters["timeout_seconds"]):
                response = await self.llm_client.generate(
                    prompt,
                    temperature=self.parameters["temperature"],
                    max_tokens=self.parameters["max_tokens"],
                    **language_option,
                )
        except TimeoutError:
            raise AgentError(ErrorCode.LLM_TIMEOUT) from None
        except LLMError as exc:
            raise AgentError(ErrorCode.LLM_LANGUAGE_MISMATCH if exc.code == "language_mismatch"
                             else ErrorCode.LLM_FAILED) from None
        except Exception:
            raise AgentError(ErrorCode.LLM_FAILED) from None

        updated = response.strip() if isinstance(response, str) else ""
        if (
            not any(char.isalnum() for char in updated)
            or updated.startswith(("```", "~~~", "{"))
            or (updated.startswith("[") and updated.endswith("]"))
        ):
            raise AgentError(ErrorCode.LLM_INVALID_RESPONSE)

        state.editor_changed = collapse_whitespace(updated) != collapse_whitespace(state.current_summary)
        if state.editor_changed:
            state.current_summary = updated
        return state

    def _on_failure(self, state: AgentState, code: ErrorCode) -> None:
        state.editor_changed = False

"""Первичное резюме новости через внедрённый LLM-клиент."""

import json
from pathlib import Path

from backend.agents.base import Agent, AgentError
from backend.agents.state import AgentState, ErrorCode
from backend.data.llm_client import BaseLLMClient
from backend.data.llm_types import LLMError
from backend.utils.rag_context import limit_style_examples
from backend.utils.text_processing import LANGUAGE_INSTRUCTIONS, split_sentences

_PROMPT = (Path(__file__).parent / "prompts" / "summarizer.txt").read_text(encoding="utf-8")


class Summarizer(Agent):
    llm_client: BaseLLMClient

    def __init__(
        self, llm_client: BaseLLMClient, *, temperature: float = 0.2, max_tokens: int = 180,
        max_sentences: int = 3, enforce_sentence_limit: bool = False,
    ) -> None:
        super().__init__(
            "Summarizer", llm_client=llm_client,
            parameters={"temperature": temperature, "max_tokens": max_tokens},
        )
        self.max_sentences = max_sentences
        self.enforce_sentence_limit = enforce_sentence_limit

    async def _process(self, state: AgentState) -> AgentState:
        news = state.original_text.strip()
        if not news:
            raise AgentError(ErrorCode.EMPTY_NEWS)

        data: dict[str, str | int | list[str]] = {"news": news}
        if self.max_sentences < 3:
            data["max_sentences"] = self.max_sentences
        style_examples = limit_style_examples(state.rag_context)
        if style_examples:
            data["style_examples"] = style_examples
        if state.glossary_context:
            data["glossary"] = state.glossary_context

        target = state.target_language
        language_rule = (f"Составь резюме новости из 1–3 предложений. {LANGUAGE_INSTRUCTIONS[target]}" if target else
                         "Составь резюме новости из 1–3 предложений на языке исходного текста.")
        prompt = _PROMPT.format(language_rule=language_rule, data=json.dumps(data, ensure_ascii=False, indent=2))
        language_option = {"target_language": target} if target else {}
        try:
            response = await self.llm_client.generate(prompt, **self.parameters, **language_option)
        except LLMError as exc:
            raise AgentError(ErrorCode.LLM_LANGUAGE_MISMATCH if exc.code == "language_mismatch"
                             else ErrorCode.LLM_FAILED) from None
        except Exception:
            raise AgentError(ErrorCode.LLM_FAILED) from None

        if not isinstance(response, str) or not response.strip():
            raise AgentError(ErrorCode.LLM_INVALID_RESPONSE)
        summary = response.strip()
        if self.enforce_sentence_limit:
            sentences = split_sentences(summary)
            if not sentences:
                raise AgentError(ErrorCode.LLM_INVALID_RESPONSE)
            if len(sentences) > self.max_sentences:
                raise AgentError(ErrorCode.SENTENCE_LIMIT)
        state.current_summary = summary
        return state

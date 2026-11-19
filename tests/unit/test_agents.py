import json
from uuid import UUID

import pytest
from pydantic import ValidationError

from backend.agents.base import Agent
from backend.agents.state import AgentState, ErrorCode
from backend.agents.summarizer import Summarizer
from backend.utils.rag_context import StylePattern

NEWS = "В Казани открыли три библиотеки 21 апреля 2026 года."
ANSWER = "21 апреля 2026 года в Казани открыли три библиотеки."


class SampleAgent(Agent):
    async def _process(self, state: AgentState) -> AgentState:
        state.current_summary = "Результат шага"
        return state


class FailingAgent(Agent):
    async def _process(self, state: AgentState) -> AgentState:
        state.current_summary = "Частичный результат"
        raise RuntimeError("internal-only-value")


def test_agent_state_defaults_are_independent():
    first = AgentState(original_text=NEWS)
    second = AgentState(original_text=NEWS)
    first.rag_context.append("Пример")
    first.validation_errors.append("Замечание")

    assert isinstance(first.request_id, UUID)
    assert first.request_id != second.request_id
    assert first.execution_status == "pending"
    assert first.current_summary is None
    assert first.execution_history == []
    assert first.max_iterations == 2 and first.iteration == 0
    assert first.validation_passed is None and first.critic_passed is None
    assert second.rag_context == second.validation_errors == []
    with pytest.raises(ValidationError):
        first.critic_score = 11
    with pytest.raises(ValidationError):
        AgentState(original_text=NEWS, max_iterations=3)


@pytest.mark.asyncio
async def test_agent_records_timing_and_last_result():
    with pytest.raises(TypeError):
        Agent("Abstract")
    agent = SampleAgent("Sample")
    state = AgentState(original_text=NEWS)

    result = await agent.process(state)
    step, = result.execution_history

    assert agent.last_result == result
    assert agent.name == step.agent_name == "Sample"
    assert step.started_at <= step.finished_at
    assert step.duration_ms >= 0
    assert step.status == "ok"
    assert state.current_summary is None and state.execution_history == []
    result.current_summary = "Последующая правка"
    assert agent.last_result.current_summary == "Результат шага"


@pytest.mark.asyncio
async def test_agent_keeps_previous_summary_on_exception():
    agent = FailingAgent("Failing")
    state = AgentState(original_text=NEWS, current_summary="Корректный черновик")

    result = await agent.process(state)
    step, = result.execution_history

    assert result.execution_status == step.status == "failed"
    assert result.current_summary == "Корректный черновик"
    assert step.error_code == ErrorCode.AGENT_FAILED
    assert "internal-only-value" not in step.model_dump_json()


@pytest.mark.asyncio
async def test_summarizer_uses_template_context_and_generation_parameters(fake_llm):
    client = fake_llm("  " + ANSWER + "\n")
    state = AgentState(
        original_text=NEWS, rag_context=[StylePattern.SINGLE_BRIEF], glossary_context=["Термин"],
        current_summary="Старый черновик",
    )
    result = await Summarizer(client, temperature=0.4, max_tokens=120).process(state)
    (prompt, options), = client.calls
    instructions, data = prompt.split("Входные данные в JSON:\n")
    assert json.loads(data) == {
        "news": NEWS,
        "style_examples": [StylePattern.SINGLE_BRIEF],
        "glossary": ["Термин"],
    }
    for text in (
        "1–3 предложений", "на языке исходного текста", "ключевые факты",
        "имена, даты и числа", "факты только из поля news",
        "не делай необоснованных выводов", "без заголовка и лишних пояснений",
        "только подсказку о форме",
    ):
        assert text in instructions

    assert options == {"temperature": 0.4, "max_tokens": 120}
    assert result.current_summary == ANSWER
    assert state.current_summary == "Старый черновик"
    assert result.execution_history[0].parameters == options


@pytest.mark.parametrize("foreign_fact", [
    "Sophia Carver", "Port Langley", "19 September 2014", "the suspension of all ferry services",
])
@pytest.mark.asyncio
async def test_summarizer_excludes_rag_entities_dates_and_events(fake_llm, foreign_fact):
    client = fake_llm(ANSWER)
    state = AgentState(original_text=NEWS, rag_context=[
        f"Example article: The report mentioned {foreign_fact}.",
        StylePattern.SINGLE_BRIEF, StylePattern.MULTI_SENTENCE,
    ])
    await Summarizer(client).process(state)
    prompt, _ = client.calls[0]
    assert foreign_fact not in prompt
    assert json.loads(prompt.split("Входные данные в JSON:\n")[1]) == {
        "news": NEWS, "style_examples": [StylePattern.SINGLE_BRIEF],
    }


@pytest.mark.asyncio
async def test_summarizer_skips_missing_optional_context(fake_llm):
    client = fake_llm(ANSWER)

    await Summarizer(client).process(AgentState(original_text=NEWS))

    prompt, _ = client.calls[0]
    assert json.loads(prompt.split("Входные данные в JSON:\n")[1]) == {"news": NEWS}


@pytest.mark.parametrize("text", ["", " \n\t "])
@pytest.mark.asyncio
async def test_summarizer_rejects_empty_news_before_generation(fake_llm, text):
    client = fake_llm(ANSWER)

    result = await Summarizer(client).process(AgentState(original_text=text))

    assert not client.calls
    assert result.current_summary is None
    assert result.execution_status == "failed"
    assert result.execution_history[0].error_code == ErrorCode.EMPTY_NEWS


@pytest.mark.parametrize(
    ("answer", "error", "expected"),
    [
        ("", TimeoutError("provider-only-value"), ErrorCode.LLM_FAILED),
        (" \t ", None, ErrorCode.LLM_INVALID_RESPONSE),
    ],
)
@pytest.mark.asyncio
async def test_summarizer_records_llm_failures(fake_llm, answer, error, expected):
    client = fake_llm(answer, error=error)

    result = await Summarizer(client).process(AgentState(original_text=NEWS))

    assert result.current_summary is None
    assert result.execution_status == "failed"
    assert result.execution_history[0].error_code == expected
    assert "provider-only-value" not in result.execution_history[0].model_dump_json()

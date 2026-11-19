import asyncio
from collections.abc import Callable
from unittest.mock import AsyncMock

import pytest

from backend.agents.base import Agent
from backend.agents.state import AgentState, ErrorCode
from backend.agents.supervisor import Supervisor

pytestmark = pytest.mark.asyncio


class TestStepAgent(Agent):
    __test__ = False

    def __init__(self, name: str, changes: Callable[[AgentState], dict]) -> None:
        super().__init__(name)
        self._changes = changes

    async def _process(self, state: AgentState) -> AgentState:
        for key, value in self._changes(state).items():
            setattr(state, key, value)
        return state


def factory(name, changes, created):
    def create():
        agent = TestStepAgent(name, changes)
        created.append(agent)
        return agent

    return create


def history(result):
    return [step.agent_name for step in result.execution_history]


async def test_supervisor_lazily_creates_agents_and_rechecks_edit(fake_llm):
    created = []
    recorder = AsyncMock()
    supervisor = Supervisor(
        lambda: fake_llm("Первый черновик"), recorder,
        validator_factory=factory(
            "Validator", lambda s: {"validation_passed": s.iteration == 1}, created
        ),
        critic_factory=factory(
            "Critic",
            lambda s: {"critic_score": 7 if s.iteration else 6, "critic_passed": True},
            created,
        ),
        editor_factory=factory(
            "Editor",
            lambda s: {"editor_changed": True, "current_summary": "Уточнённый черновик"},
            created,
        ),
    )
    state = AgentState(original_text="Текст для резюме")
    assert created == []

    result = await supervisor.run(state)

    assert [agent.name for agent in created] == ["Validator", "Critic", "Editor"]
    assert history(result) == [
        "Summarizer", "Validator", "Critic", "Editor", "Validator", "Critic", "Supervisor",
    ]
    assert result.execution_status == "verified" and result.iteration == 1
    assert result.critic_score == 7
    assert result.current_summary == "Уточнённый черновик"
    assert state.execution_history == [] and state.current_summary is None
    recorder.assert_awaited_once_with(result, "full", None)


async def test_baseline_runs_only_summarizer_and_emits_draft(fake_llm):
    created = []
    supervisor = Supervisor(
        lambda: fake_llm("Черновик"), AsyncMock(),
        validator_factory=factory("Validator", lambda s: {}, created),
        critic_factory=factory("Critic", lambda s: {}, created),
    )

    result = await supervisor.run(AgentState(original_text="Новость"), mode="baseline")

    assert result.execution_status == "draft"
    assert history(result) == ["Summarizer", "Supervisor"]
    assert not created


async def test_incomplete_full_mode_does_not_accept_previous_verdict(fake_llm):
    supervisor = Supervisor(lambda: fake_llm("Черновик"), AsyncMock(), validator_factory=None)
    state = AgentState(
        original_text="Новость", validation_passed=True, critic_passed=True, critic_score=10
    )

    result = await supervisor.run(state)

    assert result.execution_status == "incomplete_configuration"
    assert result.current_summary == "Черновик"
    assert result.validation_passed is result.critic_passed is result.critic_score is None
    assert history(result) == ["Summarizer", "Supervisor"]


async def test_editor_is_not_initialized_for_successful_initial_checks(fake_llm):
    created = []
    supervisor = Supervisor(
        lambda: fake_llm("Черновик"), AsyncMock(),
        validator_factory=factory("Validator", lambda s: {"validation_passed": True}, created),
        critic_factory=factory(
            "Critic", lambda s: {"critic_passed": True, "critic_score": 8}, created
        ),
        editor_factory=factory("Editor", lambda s: {}, created),
    )

    result = await supervisor.run(AgentState(original_text="Новость"))

    assert result.execution_status == "verified" and result.iteration == 0
    assert [agent.name for agent in created] == ["Validator", "Critic"]


async def test_supervisor_rechecks_both_edits_and_respects_limit(fake_llm):
    created = []
    supervisor = Supervisor(
        lambda: fake_llm("Черновик"), AsyncMock(),
        validator_factory=factory("Validator", lambda s: {"validation_passed": True}, created),
        critic_factory=factory(
            "Critic", lambda s: {"critic_passed": False, "critic_score": 9}, created
        ),
        editor_factory=factory(
            "Editor",
            lambda s: {"editor_changed": True, "current_summary": f"Версия {s.iteration}"},
            created,
        ),
    )

    result = await supervisor.run(AgentState(original_text="Новость"))

    assert result.execution_status == "iteration_limit" and result.iteration == 2
    assert history(result).count("Editor") == 2
    assert history(result).count("Validator") == history(result).count("Critic") == 3
    assert [agent.name for agent in created].count("Editor") == 1


@pytest.mark.parametrize("has_editor", [False, True], ids=["missing-editor", "same-text"])
async def test_supervisor_handles_missing_editor_or_no_change(fake_llm, has_editor):
    created = []
    supervisor = Supervisor(
        lambda: fake_llm("Черновик"), AsyncMock(),
        validator_factory=factory("Validator", lambda s: {"validation_passed": False}, created),
        critic_factory=factory(
            "Critic", lambda s: {"critic_passed": False, "critic_score": 5}, created
        ),
        editor_factory=(
            factory("Editor", lambda s: {"editor_changed": True}, created) if has_editor else None
        ),
    )

    result = await supervisor.run(AgentState(original_text="Новость"))

    assert result.execution_status == ("unchanged" if has_editor else "incomplete_configuration")


@pytest.mark.parametrize("during_factory", [False, True])
async def test_supervisor_registers_failures_and_skips_later_agents(fake_llm, during_factory):
    created = []

    def fail(*_args):
        raise RuntimeError("not-for-log")

    recorder = AsyncMock()
    supervisor = Supervisor(
        lambda: fake_llm("Черновик"), recorder,
        validator_factory=fail if during_factory else factory("Validator", fail, created),
        critic_factory=factory("Critic", lambda s: {}, created),
    )

    result = await supervisor.run(AgentState(original_text="Новость"))

    assert result.execution_status == "failed"
    assert history(result) == ["Summarizer", "Validator", "Supervisor"]
    assert result.execution_history[1].error_code == (
        ErrorCode.AGENT_INITIALIZATION_FAILED if during_factory else ErrorCode.AGENT_FAILED
    )
    recorder.assert_awaited_once_with(result, "full", None)


async def test_parallel_runs_use_distinct_agents_and_state(fake_llm):
    created = []
    clients = []
    barrier = asyncio.Barrier(2)

    class BlockingLLM(fake_llm):
        async def generate(self, prompt, **kwargs):
            await barrier.wait()
            return await super().generate(prompt, **kwargs)

    def create_llm():
        client = BlockingLLM(f"Отдельный черновик {len(clients) + 1}")
        clients.append(client)
        return client

    recorder = AsyncMock()
    supervisor = Supervisor(
        create_llm, recorder,
        validator_factory=factory("Validator", lambda s: {"validation_passed": True}, created),
        critic_factory=factory(
            "Critic", lambda s: {"critic_passed": True, "critic_score": 8}, created
        ),
    )

    first, second = await asyncio.wait_for(
        asyncio.gather(
            supervisor.run(AgentState(original_text="Первая новость")),
            supervisor.run(AgentState(original_text="Вторая новость")),
        ),
        timeout=2,
    )

    assert first.execution_status == second.execution_status == "verified"
    assert first.request_id != second.request_id
    assert first.current_summary == "Отдельный черновик 1"
    assert second.current_summary == "Отдельный черновик 2"
    assert "Первая новость" in clients[0].calls[0][0]
    assert "Вторая новость" in clients[1].calls[0][0]
    assert len(set(created)) == 4
    assert {a.last_result.request_id for a in created if a.name == "Validator"} == {
        first.request_id, second.request_id,
    }
    assert recorder.await_count == 2

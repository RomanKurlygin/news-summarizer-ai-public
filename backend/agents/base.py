"""Общий жизненный цикл агента."""

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from time import perf_counter

from backend.agents.state import AgentState, ErrorCode
from backend.core.exceptions import AppError
from backend.data.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class AgentError(AppError):
    """Ожидаемая ошибка шага с безопасным кодом для журнала."""

    def __init__(self, code: ErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


class Agent(ABC):
    def __init__(
        self,
        name: str,
        *,
        llm_client: BaseLLMClient | None = None,
        model_name: str | None = None,
        parameters: dict[str, int | float] | None = None,
    ) -> None:
        self.name = name
        self.llm_client = llm_client
        self.model_name = model_name
        self.parameters = dict(parameters or {})
        self.last_result: AgentState | None = None

    async def process(self, state: AgentState) -> AgentState:
        started_at = datetime.now(UTC)
        started_clock = perf_counter()
        inputs = state.observation()
        error_code = None
        try:
            result = await self._process(state.model_copy(deep=True))
            if result.request_id != state.request_id:
                raise AgentError(ErrorCode.AGENT_FAILED)
        except AgentError as exc:
            error_code = exc.code
        except Exception:
            error_code = ErrorCode.AGENT_FAILED

        if error_code is not None:
            result = state.model_copy(deep=True)
            result.execution_status = "failed"
            self._on_failure(result, error_code)

        result.execution_history = state.execution_history.copy()
        status = "failed" if error_code is not None else "ok"
        result.record_step(
            name=self.name,
            started_at=started_at,
            started_clock=started_clock,
            input_summary=inputs,
            status=status,
            parameters=self.parameters,
            model_name=self.model_name,
            error_code=error_code,
        )
        self.last_result = result.model_copy(deep=True)
        logger.info(
            "Agent %s finished (%s)", self.name, status,
            extra={"request_id": str(state.request_id)},
        )
        return result

    def _on_failure(self, state: AgentState, code: ErrorCode) -> None:
        """Дополнить копию исходного состояния данными о сбое."""

    @abstractmethod
    async def _process(self, state: AgentState) -> AgentState:
        """Изменить состояние шага и вернуть результат."""

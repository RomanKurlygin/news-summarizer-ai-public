import json
import os
from uuid import uuid4

import httpx
import pytest

from backend.core.config import Settings
from backend.data.llm_manager import LLMManager
from backend.data.ollama import OllamaProvider


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("RUN_LIVE_OLLAMA") != "1", reason="requires local Ollama with qwen2.5:1.5b")
async def test_live_ollama_generates_structured_response_through_manager():
    url = "http://127.0.0.1:11434"
    model = "qwen2.5:1.5b"
    settings = Settings(_env_file=None, ollama_base_url=url, ollama_model=model, llm_max_attempts=1)
    request_id = uuid4()
    calls = []

    async def record(call):
        calls.append(call)

    schema = {
        "type": "object", "properties": {"result": {"type": "string", "enum": ["ok"]}},
        "required": ["result"], "additionalProperties": False,
    }
    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds, trust_env=False) as http:
        await OllamaProvider(http, url).check_model(model)
        manager = LLMManager(settings, record, providers=LLMManager.providers_for(settings, http))
        response = await manager.generate(
            'Return a JSON object with result set to "ok".',
            request_id=request_id, agent_name="Probe", provider="ollama", model=model, max_tokens=32,
            schema=schema, validate_response=json.loads,
        )
    assert json.loads(response) == {"result": "ok"}
    (call,) = calls
    assert (call.request_id, call.provider, call.model, call.status) == (request_id, "ollama", model, "ok")
    assert call.input_tokens and call.output_tokens and call.prompt_hash and call.response_hash

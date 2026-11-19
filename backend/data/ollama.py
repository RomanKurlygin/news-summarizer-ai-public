"""Ollama HTTP API с длительным удержанием загруженной модели."""

import httpx

from backend.data.llm_types import GenerationOptions, LLMError, LLMReply, json_request, reply
from backend.data.local_prompts import local_chat, ollama_critic_schema


class OllamaProvider:
    def __init__(self, client: httpx.AsyncClient, base_url: str, keep_alive: str = "10m") -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive

    async def check_model(self, model: str) -> None:
        await json_request(self.client, "POST", f"{self.base_url}/api/show", json={"model": model})

    async def complete(self, prompt: str, model: str, options: GenerationOptions) -> LLMReply:
        messages = {message["role"]: message["content"] for message in local_chat(
            prompt, options.task, options.target_language, options.language_retry,
        )}
        payload = {
            "model": model, "prompt": messages["user"], "stream": False, "keep_alive": self.keep_alive,
            "options": {"temperature": options.temperature, "num_predict": options.max_tokens, "num_ctx": 8192},
        }
        if "system" in messages:
            payload["system"] = messages["system"]
        if options.seed is not None:
            payload["options"]["seed"] = options.seed
        if options.schema is not None:
            payload["format"] = ollama_critic_schema(options.schema) if options.task == "Critic" else options.schema
        data = await json_request(self.client, "POST", f"{self.base_url}/api/generate", json=payload)
        if data.get("done") is not True:
            raise LLMError("invalid_response")
        return reply(data.get("response"), data.get("model"), model,
                     data.get("prompt_eval_count"), data.get("eval_count"))

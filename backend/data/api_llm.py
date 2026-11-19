"""Внешние провайдеры через официальные HTTPS API."""

import os
import re

import httpx

from backend.data.llm_types import GenerationOptions, LLMError, LLMReply, json_request, reply

_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}")


def external_key(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise LLMError("missing_api_key")
    return value


def _schema(value: dict) -> dict:
    # Строгие числовые и строковые ограничения остаются в локальной Pydantic-схеме.
    stripped = {key: value[key] for key in ("type", "required", "additionalProperties") if key in value}
    if "properties" in value:
        stripped["properties"] = {key: _schema(part) for key, part in value["properties"].items()}
    if "items" in value:
        stripped["items"] = _schema(value["items"])
    return stripped


def _usage(data: dict, field: str) -> dict:
    value = data.get(field)
    return value if isinstance(value, dict) else {}


class AnthropicProvider:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def complete(self, prompt: str, model: str, options: GenerationOptions) -> LLMReply:
        key = external_key("ANTHROPIC_API_KEY")
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": options.temperature, "max_tokens": options.max_tokens}
        if options.schema is not None:
            payload["output_config"] = {"format": {"type": "json_schema", "schema": _schema(options.schema)}}
        data = await json_request(
            self.client, "POST", "https://api.anthropic.com/v1/messages", json=payload,
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        blocks = data.get("content")
        text = "\n".join(block["text"] for block in blocks if isinstance(block, dict)
                         and block.get("type") == "text" and isinstance(block.get("text"), str)) if isinstance(blocks, list) else ""
        usage = _usage(data, "usage")
        return reply(text, data.get("model"), model, usage.get("input_tokens"), usage.get("output_tokens"))


class GeminiProvider:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def complete(self, prompt: str, model: str, options: GenerationOptions) -> LLMReply:
        key = external_key("GEMINI_API_KEY")
        if not _MODEL_ID.fullmatch(model):
            raise LLMError("invalid_model")
        config = {"temperature": options.temperature, "maxOutputTokens": options.max_tokens}
        if options.schema is not None:
            config.update(responseMimeType="application/json", responseJsonSchema=_schema(options.schema))
        data = await json_request(
            self.client, "POST", f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": key},
            json={"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config},
        )
        candidates = data.get("candidates")
        first = candidates[0] if isinstance(candidates, list) and candidates else None
        content = first.get("content") if isinstance(first, dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        text = "\n".join(part["text"] for part in parts if isinstance(part, dict)
                         and not part.get("thought") and isinstance(part.get("text"), str)) if isinstance(parts, list) else ""
        usage = _usage(data, "usageMetadata")
        return reply(text, data.get("modelVersion"), model, usage.get("promptTokenCount"), usage.get("candidatesTokenCount"))

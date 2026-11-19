"""Локальная генерация на CPU с кэшированием Transformers."""

import asyncio
from contextlib import nullcontext
from functools import lru_cache
from threading import Event, Lock
from typing import Any

from backend.core.config import Settings
from backend.data.llm_types import GenerationOptions, LLMError, LLMReply, reply
from backend.data.local_prompts import local_chat

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
_cache: dict[tuple[str, str | None, str], tuple[Any, Any]] = {}
# Один слот для загрузки и генерации во всём процессе. Отмена запроса не освобождает
# слот раньше фактической остановки рабочего потока.
_cpu_slot = Lock()


def _load(model: str, revision: str | None, cache: str, offline: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    common = dict(revision=revision, cache_dir=cache, local_files_only=offline, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model, **common)
    engine = AutoModelForCausalLM.from_pretrained(
        model, **common, use_safetensors=True, dtype=torch.float32,
    ).eval()
    return tokenizer, engine


@lru_cache(maxsize=4)
def _grammar_tokenizer(tokenizer):
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data

    return build_token_enforcer_tokenizer_data(tokenizer)


def _json_decoder(tokenizer, schema):
    from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

    grammar = JsonSchemaParser(schema, config=CharacterLevelParserConfig(max_consecutive_whitespaces=2))
    return build_transformers_prefix_allowed_tokens_fn(_grammar_tokenizer(tokenizer), grammar)


def _critic_schema(schema: dict) -> dict:
    properties = {
        key: ({"type": "integer", "enum": list(range(11))}
              if details.get("type") == "number" and details.get("minimum") == 0 and details.get("maximum") == 10
              else details)
        for key, details in schema["properties"].items()
    }
    return {
        **schema,
        "properties": {
            **properties,
            "reasoning": {**properties["reasoning"], "maxLength": 150},
            "suggestions": {**properties["suggestions"], "maxItems": 2},
        },
    }


class TransformersProvider:
    def __init__(self, settings: Settings) -> None:
        self.cache_dir = str(settings.hf_llm_cache_dir.resolve())
        self.offline = settings.hf_llm_local_files_only
        self.model_revision = settings.hf_model_revision
        self.max_input = settings.hf_llm_max_input_tokens
        self.pinned_revisions: dict[str, str] = {}

    async def complete(self, prompt: str, model: str, options: GenerationOptions) -> LLMReply:
        cancelled = Event()
        try:
            return await asyncio.to_thread(self._complete, prompt, model, options, cancelled)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    def _complete(self, prompt: str, model: str, options: GenerationOptions, cancelled: Event) -> LLMReply:
        revision = self.pinned_revisions.get(model) or self.model_revision or (REVISION if model == MODEL else None)
        key = (model, revision, self.cache_dir)
        try:
            with _cpu_slot:
                if cancelled.is_set():
                    raise LLMError("cancelled")
                if key not in _cache:
                    _cache[key] = _load(*key, self.offline)
                tokenizer, engine = _cache[key]
                if cancelled.is_set():
                    raise LLMError("cancelled")
                return self._generate(tokenizer, engine, prompt, model, options, cancelled)
        except LLMError:
            raise
        except (MemoryError, RuntimeError) as exc:
            if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower() or "allocate memory" in str(exc).lower():
                raise LLMError("out_of_memory") from None
            raise LLMError("generation_failed") from None
        except OSError:
            raise LLMError("model_unavailable") from None
        except Exception:
            raise LLMError("generation_failed") from None

    def _generate(self, tokenizer, engine, prompt: str, model: str, options: GenerationOptions,
                  cancelled: Event) -> LLMReply:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class CancelGeneration(StoppingCriteria):
            def __call__(self, *args, **kwargs):
                return cancelled.is_set()

        inputs = tokenizer.apply_chat_template(
            local_chat(prompt, options.task, options.target_language, options.language_retry),
            add_generation_prompt=True, return_dict=True, return_tensors="pt",
        )
        input_tokens = inputs["input_ids"].shape[-1]
        context = getattr(engine.config, "max_position_embeddings", self.max_input + options.max_tokens)
        if input_tokens > self.max_input or input_tokens + options.max_tokens > context:
            raise LLMError("context_limit")
        sampling = {"temperature": options.temperature} if options.temperature > 0 else {
            "temperature": None, "top_p": None, "top_k": None,
        }
        decoder = {}
        if options.schema is not None:
            schema = _critic_schema(options.schema) if options.task == "Critic" else options.schema
            decoder["prefix_allowed_tokens_fn"] = _json_decoder(tokenizer, schema)
        rng = torch.random.fork_rng(devices=[]) if options.seed is not None else nullcontext()
        with torch.inference_mode(), rng:
            if options.seed is not None:
                torch.random.manual_seed(options.seed)
            output = engine.generate(
                **inputs, **sampling, **decoder, max_new_tokens=options.max_tokens, do_sample=options.temperature > 0,
                stopping_criteria=StoppingCriteriaList([CancelGeneration()]),
            )
        generated = output[0, input_tokens:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        if not text.strip():
            raise LLMError("invalid_response")
        return reply(text, model, model, input_tokens, len(generated))

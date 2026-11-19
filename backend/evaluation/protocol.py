"""Выборки, конфигурации и версии воспроизводимого эксперимента."""

import asyncio
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.api.schemas.summarization import ModelChoice
from backend.core.constants import PROJECT_ROOT
from backend.data.ollama import OllamaProvider
from backend.data.transformers_llm import MODEL, REVISION, TransformersProvider
from backend.retrieval.documents import Document, Holdout
from backend.retrieval.glossary import GlossaryTerm, glossary_snapshot
from backend.utils.corpus_text import normalized
from scripts.corpus import SAMPLE_SIZES, TEST_SIZE, News, evaluation_ids, load_prepared, read_json

Mode = Literal["baseline", "+dict", "+rag", "+agents", "+ensemble", "full"]
Size = Literal[20, 50, 100]
ENSEMBLE_TEMPERATURES = (0.0, 0.2)


class ExperimentError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class ExperimentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: list[ModelChoice] = Field(min_length=1, max_length=4)
    modes: list[Mode] = Field(min_length=1, max_length=6)
    sample_sizes: list[Size] = Field(min_length=1, max_length=3)
    seed: Literal[42] = 42
    parallelism: int = Field(default=1, ge=1, le=2)
    timeout: float = Field(default=120, ge=1, le=600)
    judge: ModelChoice = Field(default_factory=lambda: ModelChoice(model="qwen2.5:1.5b"))

    @model_validator(mode="after")
    def validate_unique(self):
        if (len(set(self.modes)) != len(self.modes) or len(set(self.sample_sizes)) != len(self.sample_sizes)
                or len({(item.provider, item.model) for item in self.models}) != len(self.models)):
            raise ValueError("Модели, режимы и размеры должны быть уникальны.")
        if any(item.model is None or item.temperature is not None for item in (*self.models, self.judge)):
            raise ValueError("Задайте явные модели. Температуры задаются единым протоколом эксперимента.")
        return self


def digest(obj) -> str:
    return sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def select_samples(path: Path, sizes: list[Size], diagnostic: int | None = None):
    rows = load_prepared(path, News)
    ids = [row.id for row in rows]
    if len(rows) < TEST_SIZE or len(set(ids)) != len(ids) or any(row.split != "test" for row in rows):
        raise ExperimentError("invalid_holdout")
    expected = evaluation_ids(rows)
    if read_json(path.with_name("evaluation_ids.json")) != expected or set(expected) != set(map(str, SAMPLE_SIZES)):
        raise ExperimentError("invalid_evaluation_ids")
    by_id = {str(row.id): row for row in rows}
    samples = {size: [by_id[key] for key in expected[str(size)][:diagnostic]] for size in sizes}
    holdout = Holdout.from_documents(
        Document.model_validate({**row.model_dump(exclude={"split"}), "source_split": row.split}) for row in rows
    )
    return samples, holdout, digest([row.model_dump(mode="json") for row in rows])


def model_key(model: ModelChoice) -> str:
    return f"{model.provider}:{model.model}"


def _hf_revision(provider: TransformersProvider, model: str) -> str:
    from huggingface_hub import snapshot_download

    revision = provider.model_revision or (REVISION if model == MODEL else None)
    path = snapshot_download(model, revision=revision, cache_dir=provider.cache_dir, local_files_only=True)
    return Path(path).name


async def model_versions(runtime, references: list[ModelChoice]) -> dict:
    versions = {}
    tags: dict[str, str] | None = None
    for ref in references:
        provider = runtime.providers[ref.provider]
        try:
            if isinstance(provider, OllamaProvider):
                if tags is None:
                    response = await provider.client.get(provider.base_url + "/api/tags", timeout=5)
                    response.raise_for_status()
                    tags = {item["name"]: item["digest"] for item in response.json()["models"]}
                name = ref.model
                revision = tags.get(name) or tags.get(name + ":latest")
            elif isinstance(provider, TransformersProvider):
                revision = await asyncio.to_thread(_hf_revision, provider, ref.model)
                provider.pinned_revisions[ref.model] = revision
            else:
                # Внедрённый адаптер может предоставлять версию напрямую.
                revision = await provider.model_version(ref.model)
        except (httpx.HTTPError, KeyError, OSError, ValueError, ImportError):
            raise ExperimentError("model_version_unavailable") from None
        if not isinstance(revision, str) or not revision:
            raise ExperimentError("model_version_unavailable")
        versions[model_key(ref)] = revision
    return versions


@dataclass(frozen=True)
class Protocol:
    metadata: dict
    samples: dict[int, list[News]]
    holdout: Holdout
    manifest: object | None
    terms: tuple[GlossaryTerm, ...]
    references: list[ModelChoice]


async def prepare_protocol(runtime, plan: ExperimentPlan, diagnostic: int | None) -> Protocol:
    samples, holdout, corpus_hash = await asyncio.to_thread(select_samples, runtime.settings.holdout_path,
                                                            plan.sample_sizes, diagnostic)
    manifest = None
    if set(plan.modes) & {"+rag", "full"}:
        from backend.retrieval.index_bundle import IndexManifest

        manifest = IndexManifest.model_validate_json((runtime.settings.rag_index_dir / "active.json").read_bytes())
        if not manifest.model_revision or not re.fullmatch(r"[a-f0-9]{40}", manifest.model_revision):
            raise ExperimentError("unpinned_embedding_revision")
    terms = (await glossary_snapshot(runtime.database.sessions)
             if set(plan.modes) & {"+dict", "full"} else ())
    if terms:
        test_rows = await asyncio.to_thread(load_prepared, runtime.settings.holdout_path, News)
        blocked = [normalized(row.target_text) for row in test_rows]
        rendered = [(term, normalized(" ".join((term.term, *term.aliases, term.definition)))) for term in terms]
        terms = tuple(term for term, value in rendered if not any(reference in value for reference in blocked))
    references = list(plan.models)
    if set(plan.modes) & {"+agents", "+ensemble"} and model_key(plan.judge) not in set(map(model_key, references)):
        references.append(plan.judge)
    versions = await model_versions(runtime, references)
    settings = runtime.settings
    implementation = (
        *sorted((PROJECT_ROOT / "backend/agents/prompts").glob("*.txt")),
        *sorted((PROJECT_ROOT / "backend/agents").glob("*.py")),
        *(PROJECT_ROOT / path for path in (
            "backend/data/local_prompts.py", "backend/data/ollama.py", "backend/data/transformers_llm.py",
            "backend/data/llm_manager.py", "backend/data/embeddings.py",
            "backend/services/summarization.py", "backend/retrieval/glossary.py",
            "backend/retrieval/index_bundle.py", "backend/retrieval/hybrid_retriever.py",
            "backend/retrieval/bm25_index.py", "backend/retrieval/vector_index.py",
            "backend/utils/rag_context.py", "backend/utils/corpus_text.py", "backend/utils/text_processing.py",
            "backend/api/runtime.py", "backend/evaluation/protocol.py",
            "backend/evaluation/metrics_calculator.py", "backend/evaluation/experiments.py",
        )),
    )
    metadata = {
        "plan": plan.model_dump(mode="json"), "corpus_sha256": corpus_hash,
        "sample_ids": {str(size): [str(row.id) for row in rows] for size, rows in samples.items()},
        "diagnostic": diagnostic is not None,
        "model_versions": versions, "index": manifest.model_dump(mode="json") if manifest else None,
        "retrieval": {"embedding_device": settings.embedding_device,
                      "embedding_batch_size": settings.embedding_batch_size} if manifest else None,
        "glossary_sha256": digest([asdict(term) for term in terms]) if set(plan.modes) & {"+dict", "full"} else None,
        "glossary_count": len(terms),
        "implementation_sha256": {path.relative_to(PROJECT_ROOT).as_posix():
                                  sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                  for path in implementation},
        "metrics": runtime.metrics.metadata(),
        "generation": {
            "temperature": 0.0, "ensemble_temperatures": list(ENSEMBLE_TEMPERATURES),
            "critic_temperature": settings.critic_temperature, "editor_temperature": settings.editor_temperature,
            "summarizer_max_tokens": settings.summarizer_max_tokens, "critic_max_tokens": settings.critic_max_tokens,
            "editor_max_tokens": settings.editor_max_tokens, "max_sentences": 3,
            "llm_timeout_seconds": settings.llm_timeout_seconds, "llm_max_attempts": settings.llm_max_attempts,
            "llm_retry_delay_seconds": settings.llm_retry_delay_seconds,
            "hf_max_input_tokens": settings.hf_llm_max_input_tokens, "fallback": False, "seed": plan.seed,
        },
    }
    return Protocol(metadata, samples, holdout, manifest, terms, references)

"""Ограниченные представления данных для интерактивного интерфейса."""

import asyncio
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException
import httpx
from pydantic import BaseModel, Field, ValidationError

from backend.api.deps import CurrentUser
from backend.api.routes.summarization import Runtime
from backend.api.schemas.summarization import ModelChoice
from backend.data.ollama import OllamaProvider
from backend.data.transformers_llm import MODEL, REVISION
from backend.evaluation.protocol import Mode

router = APIRouter(prefix="/api/v1", tags=["Данные интерфейса"])


def _cached_transformers(settings) -> list[str]:
    try:
        from huggingface_hub import snapshot_download

        revision = settings.hf_model_revision or (REVISION if settings.hf_model_id == MODEL else None)
        directory = Path(snapshot_download(settings.hf_model_id, revision=revision,
                                           cache_dir=settings.hf_llm_cache_dir, local_files_only=True))
        choice = ModelChoice(provider="transformers", model=settings.hf_model_id)
    except (ImportError, OSError, ValueError):
        return []
    if not (directory / "config.json").is_file() or not any(directory.glob("*.safetensors")):
        return []
    return [choice.model]


@router.get("/models")
async def available_models(_user: CurrentUser, runtime: Runtime) -> dict[str, list[str]]:
    ollama = runtime.providers.get("ollama")
    direct = []
    if isinstance(ollama, OllamaProvider):
        try:
            response = await ollama.client.get(ollama.base_url + "/api/tags", timeout=3)
            models = response.raise_for_status().json()["models"]
            for item in models:
                if not item.get("remote_host") and isinstance(item.get("name"), str):
                    try:
                        direct.append(ModelChoice(provider="ollama", model=item["name"]).model)
                    except ValidationError:
                        continue
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
            direct = []
    return {"ollama": sorted(set(direct)), "transformers": await asyncio.to_thread(_cached_transformers,
                                                                                 runtime.settings)}


class ArchivedRow(BaseModel):
    mode: Mode
    processed: int = Field(ge=1, le=100)
    successful: int = Field(ge=0, le=100)
    rouge_l: float | None = Field(default=None, ge=0, le=1)
    bertscore_f1: float | None = Field(default=None, ge=-1, le=1.01)
    bleu_1: float | None = Field(default=None, ge=0, le=1)
    bleu_2: float | None = Field(default=None, ge=0, le=1)
    meteor: float | None = Field(default=None, ge=0, le=1)
    mean_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    verified_rate: float | None = Field(default=None, ge=0, le=1)
    needs_review_rate: float | None = Field(default=None, ge=0, le=1)


class ComparisonArchive(BaseModel):
    baseline_id: UUID
    experiment_id: UUID
    rows: list[ArchivedRow] = Field(min_length=1, max_length=12)


def _load_archive(path: Path) -> ComparisonArchive:
    with path.open("rb") as stream:
        raw = stream.read(262_145)
    if len(raw) > 262_144:
        raise ValueError("Архив слишком большой.")
    result = ComparisonArchive.model_validate_json(raw)
    if (len({row.mode for row in result.rows}) != len(result.rows)
            or len({row.processed for row in result.rows}) != 1
            or any(row.successful > row.processed for row in result.rows)):
        raise ValueError("Несопоставимые результаты.")
    return result


@router.get("/evaluate/archive")
async def evaluation_archive(_user: CurrentUser, runtime: Runtime) -> ComparisonArchive:
    try:
        return await asyncio.to_thread(_load_archive, runtime.settings.comparison_archive_path)
    except FileNotFoundError:
        raise HTTPException(404, "Локальный архив сравнений не найден.") from None
    except (OSError, ValueError):
        raise HTTPException(422, "Архив сравнений не соответствует ожидаемому формату.") from None

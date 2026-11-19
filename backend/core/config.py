"""Конфигурация приложения из окружения и локального файла .env."""

from os import getenv
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.core.constants import APP_NAME, DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_SIZE, PROJECT_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        hide_input_in_errors=True,
    )

    app_name: str = APP_NAME
    app_env: Literal["development", "test", "staging", "production"] = "development"
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: list[str] = Field(default_factory=list)

    database_url: SecretStr | None = None
    test_database_url: SecretStr = SecretStr("sqlite+aiosqlite:///:memory:")
    db_pool_size: int = Field(default=5, ge=1)
    db_max_overflow: int = Field(default=5, ge=0)
    db_pool_timeout: float = Field(default=5, gt=0)
    db_health_timeout: float = Field(default=2, gt=0)
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")
    celery_broker_url: SecretStr = SecretStr("redis://localhost:6379/1")
    celery_result_backend: SecretStr = SecretStr("redis://localhost:6379/2")
    baseline_cache_ttl_seconds: int = Field(default=0, ge=0, le=86400)

    qdrant_url: AnyHttpUrl = Field(default="http://localhost:6333", repr=False)
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = Field(default="news_knowledge_base", pattern=r"^[A-Za-z0-9_-]+$", max_length=80)
    qdrant_mode: Literal["server", "local"] = "local"
    rag_index_dir: Path = PROJECT_ROOT / "data" / "indexes"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_revision: str | None = None
    embedding_size: int = Field(default=DEFAULT_EMBEDDING_SIZE, gt=0)
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=24, ge=1)
    embedding_cache_dir: Path = PROJECT_ROOT / ".cache" / "sentence-transformers"
    embedding_local_files_only: bool = False

    llm_provider: Literal["ollama", "transformers", "anthropic", "gemini"] = "ollama"
    ollama_base_url: AnyHttpUrl = Field(default="http://localhost:11434", repr=False)
    ollama_model: str = "qwen2.5:1.5b"
    ollama_keep_alive: str = "10m"
    hf_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    hf_model_revision: str | None = None
    hf_llm_cache_dir: Path = PROJECT_ROOT / ".cache" / "llm"
    hf_llm_local_files_only: bool = False
    hf_llm_max_input_tokens: int = Field(default=4096, gt=0)
    anthropic_model: str | None = None
    gemini_model: str | None = None
    llm_timeout_seconds: float = Field(default=90, gt=0, allow_inf_nan=False)
    llm_max_attempts: int = Field(default=2, ge=1, le=3)
    llm_retry_delay_seconds: float = Field(default=1, ge=0, le=10)
    llm_allow_local_fallback: bool = False
    summarizer_temperature: float = Field(default=0.15, ge=0, le=1)
    critic_temperature: float = Field(default=0.0, ge=0, le=1)
    editor_temperature: float = Field(default=0.1, ge=0, le=1)
    summarizer_max_tokens: int = Field(default=180, ge=1, le=4096)
    critic_max_tokens: int = Field(default=480, ge=1, le=4096)
    editor_max_tokens: int = Field(default=180, ge=1, le=4096)
    api_run_timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    api_group_timeout_seconds: float = Field(default=300, gt=0, allow_inf_nan=False)
    holdout_path: Path = PROJECT_ROOT / "data" / "test" / "news_test.json"
    comparison_archive_path: Path = PROJECT_ROOT / "data" / "indexes" / "experiments" / "stage13-1-comparison.json"
    metric_nltk_dir: Path = PROJECT_ROOT / ".cache" / "nltk"

    @property
    def anthropic_api_key(self) -> SecretStr | None:
        value = getenv("ANTHROPIC_API_KEY")
        return SecretStr(value) if value else None

    @property
    def gemini_api_key(self) -> SecretStr | None:
        value = getenv("GEMINI_API_KEY")
        return SecretStr(value) if value else None

    jwt_secret_key: SecretStr | None = None
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    jwt_access_token_expire_minutes: int = Field(default=30, gt=0)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_dir: Path = Path("logs")

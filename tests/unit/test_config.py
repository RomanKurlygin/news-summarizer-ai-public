import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.core.config import Settings


@pytest.fixture(autouse=True)
def clean_settings_environment(monkeypatch):
    for key in os.environ:
        if key.lower() in Settings.model_fields:
            monkeypatch.delenv(key)


def test_settings_start_with_local_defaults():
    settings = Settings(_env_file=None)

    assert settings.app_name == "NewsSummarizer AI"
    assert settings.app_env == "development"
    assert settings.app_host == "127.0.0.1"
    assert settings.app_port == 8000
    assert settings.cors_origins == []
    assert settings.log_level == "INFO"
    assert settings.log_dir == Path("logs")
    assert settings.database_url is None
    assert settings.jwt_secret_key is None
    assert settings.anthropic_api_key is None
    assert settings.gemini_api_key is None


def test_environment_overrides_dotenv(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        'APP_NAME="NewsSummarizer Test"\n'
        'APP_PORT=8080\n'
        'CORS_ORIGINS=["https://dotenv.example"]\n'
        'OLLAMA_MODEL=test-model\n'
        'JWT_SECRET_KEY=test-signing-key\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("APP_PORT", "8090")
    monkeypatch.setenv("CORS_ORIGINS", '["https://environment.example"]')

    settings = Settings(_env_file=dotenv)

    assert settings.app_name == "NewsSummarizer Test"
    assert settings.ollama_model == "test-model"
    assert settings.app_port == 8090
    assert settings.cors_origins == ["https://environment.example"]
    assert settings.jwt_secret_key.get_secret_value() == "test-signing-key"
    assert "test-signing-key" not in repr(settings)


def test_example_env_can_be_used_as_is():
    project_root = Path(__file__).resolve().parents[2]
    settings = Settings(_env_file=project_root / ".env.example")

    assert settings.app_host == "127.0.0.1"
    assert settings.app_port == 8000
    assert settings.database_url is None
    assert settings.jwt_secret_key is None
    assert settings.qdrant_api_key is None


def test_port_outside_tcp_range_is_rejected(monkeypatch):
    monkeypatch.setenv("APP_PORT", "65536")

    with pytest.raises(ValidationError, match="app_port"):
        Settings(_env_file=None)

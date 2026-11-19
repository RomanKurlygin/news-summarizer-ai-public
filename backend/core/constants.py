"""Общие константы приложения."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_NAME = "NewsSummarizer AI"
API_VERSION = "0.1.0"
REQUEST_ID_HEADER = "X-Request-ID"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
DEFAULT_EMBEDDING_SIZE = 384

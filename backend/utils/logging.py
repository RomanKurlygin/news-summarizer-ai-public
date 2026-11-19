"""Настройка журналов приложения."""

import logging
from logging.handlers import RotatingFileHandler

from backend.core.config import Settings
from backend.core.constants import PROJECT_ROOT

LOG_FILENAME = "news_summarizer.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3


def configure_logging(settings: Settings) -> None:
    """Настроить выделенный логгер backend для консоли и ротируемого файла."""
    logger = logging.getLogger("backend")
    logger.setLevel(settings.log_level)
    logger.propagate = False

    log_file = (PROJECT_ROOT / settings.log_dir / LOG_FILENAME).resolve()
    if any(
        isinstance(handler, RotatingFileHandler)
        and handler.baseFilename == str(log_file)
        for handler in logger.handlers
    ):
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s [request_id=%(request_id)s] %(message)s",
        defaults={"request_id": "-"},
    )
    for handler in (logging.StreamHandler(), file_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

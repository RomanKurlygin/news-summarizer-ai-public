import logging
from logging.handlers import RotatingFileHandler

import pytest

from backend.core.config import Settings
from backend.utils.logging import LOG_FILENAME, configure_logging


@pytest.fixture
def app_logger():
    parent = logging.getLogger("backend")
    previous_level, previous_propagate = parent.level, parent.propagate
    yield logging.getLogger("backend.test")
    for handler in parent.handlers[:]:
        parent.removeHandler(handler)
        handler.close()
    parent.setLevel(previous_level)
    parent.propagate = previous_propagate


def test_logs_to_console_and_file_once(app_logger, capsys, tmp_path):
    settings = Settings(_env_file=None, log_dir=tmp_path / "nested", log_level="DEBUG")
    configure_logging(settings)
    configure_logging(settings)

    app_logger.debug("test message", extra={"request_id": "unit-request"})
    app_logger.info("outside request")

    file_output = (settings.log_dir / LOG_FILENAME).read_text(encoding="utf-8")
    console_output = capsys.readouterr().err
    for output in (file_output, console_output):
        assert output.count("test message") == 1
        assert "DEBUG backend.test [request_id=unit-request]" in output
        assert "INFO backend.test [request_id=-]" in output


def test_log_level_filters_both_outputs(app_logger, capsys, tmp_path):
    configure_logging(Settings(_env_file=None, log_dir=tmp_path, log_level="WARNING"))

    app_logger.info("hidden info")
    app_logger.warning("visible warning")
    app_logger.error("visible error")

    file_output = (tmp_path / LOG_FILENAME).read_text(encoding="utf-8")
    for output in (file_output, capsys.readouterr().err):
        assert "hidden info" not in output
        assert "visible warning" in output
        assert "visible error" in output


def test_file_log_rotates(app_logger, tmp_path):
    configure_logging(Settings(_env_file=None, log_dir=tmp_path, log_level="INFO"))
    file_handler = next(
        handler
        for handler in logging.getLogger("backend").handlers
        if isinstance(handler, RotatingFileHandler)
    )
    file_handler.maxBytes = 256

    for number in range(6):
        app_logger.info("rotation test entry %d %s", number, "x" * 64)

    assert (tmp_path / f"{LOG_FILENAME}.1").exists()
    assert "rotation test entry 5" in (tmp_path / LOG_FILENAME).read_text(
        encoding="utf-8"
    )

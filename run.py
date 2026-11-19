"""Локальный запуск API через Uvicorn."""

import uvicorn

from backend.api.main import app


if __name__ == "__main__":
    settings = app.state.settings
    uvicorn.run(
        app,
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.log_level.lower(),
        access_log=False,
    )

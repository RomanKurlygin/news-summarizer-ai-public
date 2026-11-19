"""Контроллер локальных процессов Windows для start-local.ps1 и stop-local.ps1."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time

import httpx

from backend.core.config import Settings
from backend.core.constants import PROJECT_ROOT

STATE_DIR = PROJECT_ROOT / ".tools" / "local-stack"
API = "http://127.0.0.1:8000"
UI = "http://127.0.0.1:8501"


@contextmanager
def lock():
    import msvcrt

    with (STATE_DIR / "lock").open("a+b") as file:
        if file.tell() == 0:
            file.write(b"\0")
        file.flush()
        file.seek(0)
        try:
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit("Локальный запуск уже работает или запускается.") from None
        try:
            yield
        finally:
            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)


def free(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def serve(run_id: str, database: Path | None) -> None:
    settings = Settings()
    url = settings.database_url.get_secret_value() if settings.database_url else None
    if database is not None or url is None:
        file = (database or PROJECT_ROOT / "data" / "stage13-experiments.sqlite3").resolve()
        if not file.parent.is_dir():
            raise ValueError("Каталог локальной SQLite не найден.")
        url = f"sqlite+aiosqlite:///{file.as_posix()}"
    if not url.startswith("sqlite+aiosqlite:///"):
        raise ValueError("Для этого сценария укажите локальную SQLite через -Database.")

    env = os.environ | {"DATABASE_URL": url, "APP_HOST": "127.0.0.1", "APP_PORT": "8000",
                        "QDRANT_MODE": "local", "NEWS_API_URL": API, "PYTHONUTF8": "1"}
    env["JWT_SECRET_KEY"] = (settings.jwt_secret_key.get_secret_value() if settings.jwt_secret_key
                             else secrets.token_urlsafe(48))
    ollama = "http://127.0.0.1:11434"
    env["OLLAMA_BASE_URL"] = ollama
    children = []
    ready = STATE_DIR / "ready.json"
    stop = STATE_DIR / f"{run_id}.stop"

    def spawn(name, command, child_env=env):
        with (STATE_DIR / f"{name}.log").open("wb") as log:
            child = subprocess.Popen(command, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        children.append(child)
        return child

    def available(http, address):
        try:
            return http.get(address).is_success
        except httpx.HTTPError:
            return False

    def wait(http, address, child):
        for _ in range(100):
            if child.poll() is not None:
                break
            if available(http, address):
                return
            if stop.exists():
                break
            time.sleep(0.3)
        raise RuntimeError(f"Сервис не готов по адресу {address}.")

    if not all(free(port) for port in (8000, 8501)):
        raise RuntimeError("Порт 8000 или 8501 занят. Существующие процессы оставлены работать.")
    try:
        with httpx.Client(timeout=1.5, trust_env=False) as http:
            if not available(http, f"{ollama}/api/version"):
                executable = shutil.which("ollama")
                if executable is None or not free(11434):
                    raise RuntimeError("Локальная Ollama недоступна. Проверьте установку и порт 11434.")
                child = spawn("ollama", [executable, "serve"], env | {"OLLAMA_HOST": "127.0.0.1:11434"})
                wait(http, f"{ollama}/api/version", child)
            model = http.post(f"{ollama}/api/show", json={"model": settings.ollama_model})
            if not model.is_success:
                raise RuntimeError(f"Модель не найдена. Выполните ollama pull {settings.ollama_model}.")
            subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], env=env, check=True)
            wait(http, f"{API}/health", spawn("api", [sys.executable, "run.py"]))
            wait(http, f"{UI}/_stcore/health", spawn("streamlit", [
                sys.executable, "-m", "streamlit", "run", "streamlit_app/app.py",
                "--server.address=127.0.0.1", "--server.port=8501", "--server.headless=true",
            ]))
        staged = STATE_DIR / f"{run_id}.json.tmp"
        staged.write_text(json.dumps({"run_id": run_id, "pid": os.getpid()}), encoding="utf-8")
        staged.replace(ready)
        while not stop.exists():
            if any(child.poll() is not None for child in children):
                raise RuntimeError("Дочерний сервис завершился. Подробности в логах запуска.")
            time.sleep(0.4)
    finally:
        for child in reversed(children):
            if child.poll() is None:
                try:
                    child.send_signal(signal.CTRL_BREAK_EVENT)
                except OSError:
                    child.terminate()
        for child in reversed(children):
            try:
                child.wait(timeout=12)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=5)
        ready.unlink(missing_ok=True)
        stop.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--database", type=Path)
    args = parser.parse_args()
    if sys.platform != "win32" or len(args.run_id) != 32 or any(ch not in "0123456789abcdef" for ch in args.run_id):
        parser.error("Контроллер Windows ожидает идентификатор запуска из PowerShell.")
    os.chdir(PROJECT_ROOT)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with lock():
        try:
            serve(args.run_id, args.database)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, httpx.HTTPError) as exc:
            detail = str(exc) if isinstance(exc, (RuntimeError, ValueError)) else type(exc).__name__
            raise SystemExit(detail) from None


if __name__ == "__main__":
    main()

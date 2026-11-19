FROM python:3.11-slim-bookworm

ENV POETRY_VIRTUALENVS_CREATE=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN pip install --no-cache-dir poetry==2.5.1 \
    && useradd --create-home --user-group --uid 10001 summarizer \
    && install -d -o summarizer -g summarizer /app/.cache /rag /logs
COPY pyproject.toml poetry.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --only main,rag,evaluation,ui --no-root --no-interaction
COPY alembic.ini run.py ./
COPY migrations ./migrations
COPY backend ./backend
COPY scripts ./scripts
COPY streamlit_app ./streamlit_app
COPY .streamlit/config.toml ./.streamlit/config.toml
USER summarizer
CMD ["python", "run.py"]

"""Celery работает с JSON-сообщениями и хранит состояние эксперимента в БД."""

from celery import Celery

from backend.core.config import Settings

EVALUATE_TASK = "news_summarizer.evaluate_model_task"
COMPARE_TASK = "news_summarizer.compare_models_task"
PURGE_CACHE_TASK = "news_summarizer.purge_cache_task"


def celery_client(settings: Settings) -> Celery:
    client = Celery(
        "news_summarizer", broker=settings.celery_broker_url.get_secret_value(),
        include=["backend.worker.tasks"],
    )
    client.conf.update(
        task_serializer="json", accept_content=["json"], task_ignore_result=True,
        task_acks_late=True, task_reject_on_worker_lost=True, worker_prefetch_multiplier=1,
        broker_connection_timeout=3, broker_connection_retry_on_startup=True,
        broker_transport_options={"socket_connect_timeout": 3, "socket_timeout": 3,
                                  "visibility_timeout": 86400, "max_retries": 0},
        beat_schedule={"purge-expired-cache": {"task": PURGE_CACHE_TASK, "schedule": 300.0,
                                               "options": {"expires": 240}}},
    )
    return client


app = celery_client(Settings())

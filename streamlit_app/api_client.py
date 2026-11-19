"""Серверный HTTP-клиент Streamlit. Токен передаётся только FastAPI."""

from uuid import UUID

import httpx


class APIError(Exception):
    def __init__(self, message: str, status: int | None = None, request_id: str | None = None):
        super().__init__(message)
        self.status = status
        self.request_id = request_id


_MESSAGES = {
    401: "Сеанс истёк или учётные данные неверны.",
    403: "Недостаточно прав для доступа.",
    404: "Запись не найдена.",
    409: "Запись уже существует или изменилась. Обновите страницу.",
    422: "Проверьте поля формы и формат данных.",
    502: "Модель вернула ответ, который не удалось обработать.",
    503: "Сервис недоступен. Проверьте Ollama, базу данных и индекс поиска.",
    504: "Обработка превысила допустимое время.",
}
_DETAILS = {
    "model_unavailable": "Модель не установлена или недоступна. Обновите список моделей.",
    "provider_unavailable": "Провайдер недоступен. Проверьте запуск Ollama.",
    "rag_failed": "Поиск недоступен. Проверьте опубликованный индекс.",
    "broker_unavailable": "Брокер Redis недоступен.",
    "glossary_failed": "Не удалось открыть справочник.",
    "language_mismatch": "Модель не соблюла выбранный язык резюме. Подробности есть в журнале запроса.",
    "sentence_limit": "Модель превысила выбранный предел предложений. Попробуйте снова или выберите полный режим.",
    "timeout": _MESSAGES[504],
}


class APIClient:
    def __init__(self, base_url: str, token: str | None = None, *, transport=None):
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._transport = transport

    def request(self, method: str, path: str, *, body=None, params=None, csv_data: bytes | None = None):
        if not path.startswith("/") or path.startswith("//"):
            raise APIError("Маршрут должен относиться к настроенному FastAPI.")
        headers = self._headers | ({"Content-Type": "text/csv"} if csv_data is not None else {})
        try:
            with httpx.Client(base_url=self._base, timeout=httpx.Timeout(310, connect=4), trust_env=False,
                              follow_redirects=False, transport=self._transport) as client:
                response = client.request(method, path, json=body, params=params, content=csv_data, headers=headers)
        except httpx.TimeoutException:
            raise APIError(_MESSAGES[504]) from None
        except (httpx.RequestError, ValueError):
            raise APIError("Не удалось подключиться к FastAPI. Проверьте адрес и запуск сервера.") from None
        if not response.is_success:
            request_id = None
            message = _MESSAGES.get(response.status_code, "Не удалось выполнить запрос.")
            try:
                detail = response.json().get("detail")
                if isinstance(detail, dict):
                    message = _DETAILS.get(detail.get("code"), message)
                    if detail.get("request_id"):
                        request_id = str(UUID(detail["request_id"]))
            except (AttributeError, TypeError, ValueError):
                pass
            raise APIError(message, response.status_code, request_id)
        try:
            return response.json() if response.content else None
        except ValueError:
            raise APIError("Сервер вернул данные в неожиданном формате.") from None

    def get(self, path: str, params=None):
        return self.request("GET", path, params=params)

    def post(self, path: str, body=None):
        return self.request("POST", path, body=body)

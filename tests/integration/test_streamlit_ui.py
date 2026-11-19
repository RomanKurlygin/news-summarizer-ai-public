from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest

from streamlit_app.api_client import APIError

APP = Path(__file__).resolve().parents[2] / "streamlit_app" / "app.py"
OWN_REQUEST = "11111111-1111-4111-8111-111111111111"


class BrowserAPI:
    def __init__(self, token):
        self.token = token
        self.calls = []
        self.expired = False
        self.unavailable = False
        self.mode = "baseline"
        self.language = "auto"

    def result(self):
        return {
            "request_id": OWN_REQUEST, "text": "Тестовая новость.", "final_summary": "Тестовое резюме.",
            "initial_summary": "Тестовое резюме.", "mode": self.mode, "duration_ms": 100,
            "status": "draft" if self.mode == "baseline" else "needs_review",
            "critic_score": None if self.mode == "baseline" else 9,
            "validation_warnings": ["Нужно проверить имя."] if self.mode == "full" else [],
            "requested_summary_language": self.language, "target_language": "ru", "actual_summary_language": "ru",
        }

    def get(self, endpoint, params=None):
        self.calls.append(("GET", endpoint, params))
        if self.unavailable:
            raise APIError("Не удалось подключиться к FastAPI.")
        if self.expired or not self.token:
            raise APIError("Сеанс истёк", 401)
        if endpoint == "/api/v1/auth/me":
            return {"username": "fixture-reader", "role": "user"}
        if endpoint == "/api/v1/models":
            return {"ollama": ["local-test"], "transformers": []}
        if endpoint == "/api/v1/requests":
            return [{"request_id": OWN_REQUEST, "preview": "Личная новость",
                     "status": "draft", "created_at": "2026-11-18T12:00:00"}]
        if endpoint.endswith("/reasoning_log") and OWN_REQUEST in endpoint:
            return {"status": "draft", "revisions": [], "steps": [
                {"agent": "Summarizer", "status": "ok", "errors": [], "warnings": []},
            ]}
        if endpoint.endswith("/llm_calls") and OWN_REQUEST in endpoint:
            return []
        if endpoint == f"/api/v1/requests/{OWN_REQUEST}":
            return self.result()
        if endpoint == "/api/v1/evaluate/archive":
            return {"baseline_id": OWN_REQUEST, "experiment_id": OWN_REQUEST, "rows": [
                {"mode": "baseline", "processed": 20, "successful": 20, "rouge_l": 0.17, "bertscore_f1": 0.53,
                 "bleu_1": 0.2, "bleu_2": 0.1, "meteor": 0.2, "mean_seconds": 0.5},
            ]}
        raise APIError("Недостаточно прав для доступа.", 403)

    def post(self, endpoint, payload):
        self.calls.append(("POST", endpoint, {k: v for k, v in payload.items() if k != "password"}))
        if endpoint.endswith("/login"):
            self.token = "fixture-session"
            return {"access_token": self.token}
        self.mode = payload["mode"]
        self.language = payload["summary_language"]
        return self.result()


@pytest.fixture
def ui(monkeypatch):
    api = BrowserAPI(None)
    monkeypatch.setattr("streamlit_app.api_client.APIClient", lambda _base, _token=None: api)
    return AppTest.from_file(str(APP), default_timeout=12), api


def _button(at, label):
    return next(widget for widget in at.button if widget.label == label)


def _login(at):
    at.run()
    next(widget for widget in at.text_input if widget.label == "Имя пользователя").set_value("reader")
    next(widget for widget in at.text_input if widget.label == "Пароль").set_value("widget-fixture")
    _button(at, "Войти").click().run()
    assert not at.exception


def test_form_preserves_preferences_and_hides_absent_critic_score(ui):
    at, api = ui
    _login(at)
    at.text_area(key="news").set_value("В городе открылась библиотека.")
    at.selectbox(key="summary_sentences").set_value(1)
    at.selectbox(key="summary_language").set_value("ru")
    _button(at, "Суммаризировать").click().run()
    assert not at.exception and "Оценка Critic" not in [item.label for item in at.metric]
    assert at.session_state["preferences"]["max_sentences"] == 1
    assert any(call[1] == "/api/v1/summarize" and call[2]["summary_language"] == "ru" for call in api.calls)
    assert any("Получен: Русский" in item.value for item in at.caption)
    assert len(at.code) == 1 and at.code[0].value == "Тестовое резюме."
    at.sidebar.radio[0].set_value("История запросов").run()
    assert at.title[0].value == "История запросов"
    at.sidebar.radio[0].set_value("Суммаризация").run()
    assert at.selectbox(key="summary_sentences").value == 1
    at.selectbox(key="summary_mode").set_value("full")
    _button(at, "Суммаризировать").click().run()
    assert "Оценка Critic" in [item.label for item in at.metric]
    assert "Нужно проверить имя." in [item.value for item in at.warning]


def test_session_expiry_logout_and_role_gating(ui):
    at, api = ui
    _login(at)
    assert "Ручная проверка" not in at.sidebar.radio[0].options
    assert "База знаний и справочник" not in at.sidebar.radio[0].options
    api.expired = True
    at.run()
    assert any("Войдите повторно" in item.value for item in at.warning)
    assert "token" not in at.session_state and "preferences" not in at.session_state
    api.expired = False
    _login(at)
    _button(at, "Выйти").click().run()
    assert "token" not in at.session_state and not at.exception


def test_demo_switch_keeps_the_same_article_and_sets_full_and_two_sentences(ui):
    at, api = ui
    _login(at)
    _button(at, "Вставить новость").click().run()
    news = at.text_area(key="news").value
    assert at.selectbox(key="summary_mode").value == "baseline"
    at.selectbox(key="demo").set_value("B · Новость для RAG и агентов").run()
    _button(at, "Вставить новость").click().run()
    assert at.text_area(key="news").value == news
    assert at.selectbox(key="summary_mode").value == "full"
    assert at.selectbox(key="summary_sentences").value == 2
    _button(at, "Суммаризировать").click().run()
    assert api.calls[-1][2]["max_sentences"] == 2 and api.calls[-1][2]["mode"] == "full"


def test_history_and_archive_display_real_dtos_and_guard_unavailable_api(ui):
    at, api = ui
    _login(at)
    at.sidebar.radio[0].set_value("История запросов").run()
    assert "Личная новость" in at.selectbox[2].options[0]
    _button(at, "Открыть журнал этого запроса").click().run()
    assert at.title[0].value == "Работа агентов" and not at.exception
    assert any("Создание резюме" in item.label for item in at.expander)
    assert not any("Редактирование" in item.label for item in at.expander)
    at.sidebar.radio[0].set_value("Сравнение моделей и режимов").run()
    assert len(at.get("plotly_chart")) == 5 and at.dataframe and not at.exception
    api.unavailable = True
    at.run()
    assert "Не удалось подключиться к FastAPI." in [item.value for item in at.error]
    assert not at.exception and not at.get("plotly_chart")

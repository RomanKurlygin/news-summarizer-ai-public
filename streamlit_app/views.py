"""Страницы. Названия режимов и вердикты поступают из FastAPI."""

from datetime import date, timedelta
from uuid import UUID

import plotly.graph_objects as go
import streamlit as st

from streamlit_app.api_client import APIError
from streamlit_app.demo_news import SCENARIOS
from streamlit_app.presentation import AGENTS, MODES, STATUSES, archive_rows, history_rows, issue

SUMMARY_PAGE = "Суммаризация"
AGENTS_PAGE = "Работа агентов"
COMPARE_PAGE = "Сравнение моделей и режимов"
HISTORY_PAGE = "История запросов"
REVIEW_PAGE = "Ручная проверка"
DATA_PAGE = "База знаний и справочник"
MONITOR_PAGE = "Состояние сервисов"

_LANGUAGES = {"auto": "Автоматически", "ru": "Русский", "en": "Английский"}
_PROVIDERS = {"ollama": "Ollama", "transformers": "Transformers"}
_RAG_SEARCH = {"same_language": "На языке новости", "cross_language": "Межъязыковой поиск",
               "mixed": "На языке новости и межъязыковой", "unfiltered": "Без фильтра языка",
               "no_results": "Подходящих документов нет"}
_CAUTION = "«Автоматически проверено» означает, что проверки прошли. Даже высокая оценка Critic не гарантирует достоверность."
_SUMMARY_STYLE = """<style>
[class*="st-key-summary-prose-"] [data-testid="stCode"] pre,
[class*="st-key-summary-prose-"] [data-testid="stCode"] code,
[class*="st-key-summary-prose-"] [data-testid="stCode"] code span {
  font-family: 'Source Sans 3', 'Segoe UI', sans-serif !important;
  font-size: 1rem;
  line-height: 1.5;
}
</style>"""


def _status(code: str) -> str:
    return STATUSES.get(code, code)


def open_agents(request_id: str) -> None:
    st.session_state.active_request = request_id
    st.session_state.navigation = AGENTS_PAGE


def set_demo() -> None:
    text, mode, sentence_limit = SCENARIOS[st.session_state.demo]
    st.session_state.news = text
    st.session_state.draft = text
    st.session_state.summary_mode = mode
    st.session_state.preferences["mode"] = mode
    st.session_state.summary_language = "auto"
    st.session_state.preferences["language"] = "auto"
    st.session_state.summary_sentences = sentence_limit
    st.session_state.preferences["max_sentences"] = sentence_limit


def _available(client):
    available = client.get("/api/v1/models")
    return {key: models for key, models in available.items() if models}


def _generation_options(client):
    available = _available(client)
    if not available:
        st.warning("Доступных моделей нет. Запустите Ollama и проверьте, что модель установлена.")
        return None
    preferences = st.session_state.preferences
    providers = list(available)
    if st.session_state.get("llm_provider") not in providers:
        st.session_state.llm_provider = preferences.get("provider") if preferences.get("provider") in providers else providers[0]
    left, right = st.columns([2, 3])
    provider = left.selectbox("Провайдер", providers, format_func=_PROVIDERS.get, key="llm_provider")
    names = available[provider]
    if st.session_state.get("llm_model") not in names:
        st.session_state.llm_model = preferences.get("model") if preferences.get("model") in names else names[0]
    model = right.selectbox("Доступная модель", names, key="llm_model")
    preferences.update(provider=provider, model=model)
    return {"provider": provider, "model": model}


def summarize_page(client) -> None:
    st.caption("НОВЫЙ ЗАПРОС")
    st.title("Суммаризация")
    st.write("Сократите новость до 1–3 предложений и посмотрите, какие проверки прошёл результат.")
    options = _generation_options(client)
    if options is None:
        return
    single, batch = st.tabs(["Одна новость", "Пакет файлов"])
    prefs = st.session_state.preferences
    with single:
        demo, insert = st.columns([3, 1], vertical_alignment="bottom")
        demo.selectbox("Сценарии для защиты", list(SCENARIOS), key="demo")
        insert.button("Вставить новость", type="secondary", on_click=set_demo)
        st.session_state.setdefault("news", st.session_state.get("draft", ""))
        st.session_state.setdefault("summary_mode", prefs.get("mode", "baseline"))
        st.session_state.setdefault("summary_language", prefs.get("language") or "auto")
        st.session_state.setdefault("summary_sentences", prefs.get("max_sentences", 3))
        with st.form("summarization"):
            text = st.text_area("Текст новости", key="news", height=200, max_chars=8000,
                                placeholder="Вставьте текст или выберите демонстрационный сценарий.")
            left, mid, right = st.columns([2, 2, 1])
            mode = left.selectbox("Режим", ["baseline", "full"], format_func=MODES.get, key="summary_mode")
            language = mid.selectbox("Язык резюме", list(_LANGUAGES), format_func=_LANGUAGES.get, key="summary_language")
            maximum = right.selectbox("До предложений", [1, 2, 3], key="summary_sentences")
            submitted = st.form_submit_button("Суммаризировать", type="primary")
        if submitted:
            st.session_state.draft = text
            prefs.update(mode=mode, language=language, max_sentences=maximum)
            st.session_state.pop("last_summary", None)
            if not text.strip():
                st.warning("Введите текст новости.")
            else:
                try:
                    with st.spinner("Модель обрабатывает новость…"):
                        result = client.post("/api/v1/summarize", options | {
                            "text": text, "mode": mode, "summary_language": language,
                            "max_sentences": maximum,
                        })
                except APIError as error:
                    if error.request_id:
                        st.button("Посмотреть журнал ошибки", on_click=open_agents, args=[error.request_id])
                    raise
                st.session_state.last_summary = result
                st.session_state.active_request = result["request_id"]
        if result := st.session_state.get("last_summary"):
            show_summary(result)
            st.button("Открыть работу агентов", on_click=open_agents, args=[result["request_id"]])
    with batch:
        st.caption("До четырёх файлов .txt в UTF-8. Каждый файл содержит одну новость.")
        files = st.file_uploader("Файлы новостей", type=["txt"], accept_multiple_files=True)
        batch_mode = st.selectbox("Режим пакета", ["baseline", "full"], format_func=MODES.get)
        batch_language = st.selectbox("Язык резюме в пакете", list(_LANGUAGES), format_func=_LANGUAGES.get)
        if st.button("Обработать файлы", disabled=not files):
            if len(files) > 4:
                st.error("Выберите не более четырёх файлов.")
                return
            try:
                texts = [file.getvalue().decode("utf-8-sig").strip() for file in files]
            except UnicodeError:
                st.error("Файлы должны быть в UTF-8.")
                return
            if any(not 1 <= len(text) <= 8000 for text in texts):
                st.error("В каждом файле должно быть от 1 до 8000 символов текста.")
                return
            with st.spinner("Модель обрабатывает пакет…"):
                response = client.post("/api/v1/summarize/batch", {"items": [
                    options | {"text": text, "mode": batch_mode, "summary_language": batch_language,
                               "max_sentences": prefs.get("max_sentences", 3)}
                    for text in texts
                ]})
            for index, item in enumerate(response["results"], 1):
                with st.expander(f"Новость {index}", expanded=True):
                    if item.get("error"):
                        st.error(item["error"]["message"])
                    if item.get("result"):
                        show_summary(item["result"])


def show_summary(result: dict) -> None:
    st.subheader("Итоговое резюме")
    status = result["status"]
    if status == "verified":
        st.success(_status(status))
        st.caption(_CAUTION)
    elif status in {"needs_review", "unchanged", "iteration_limit"}:
        st.warning(_status(status))
        if status == "needs_review":
            st.caption("Нужно сверить замечание с исходной новостью вручную.")
    else:
        st.info(_status(status))
    if result.get("final_summary"):
        with st.container(key=f"summary-prose-{result['request_id']}"):
            st.html(_SUMMARY_STYLE)
            # Streamlit предоставляет штатную кнопку копирования для текстового блока.
            st.code(result["final_summary"], language=None, wrap_lines=True)
    if requested := result.get("requested_summary_language"):
        actual = _LANGUAGES.get(result.get("actual_summary_language"), "Не определён")
        selected = _LANGUAGES.get(result.get("target_language"), "Не определён")
        st.caption(f"Язык резюме · {_LANGUAGES[requested]} → {selected} · Получен: {actual}")
    if retries := result.get("language_retries"):
        st.caption(f"Повторные генерации из-за языка · {retries}")
    if (count := result.get("summary_sentence_count")) is not None:
        limit = result["max_sentences"]
        st.caption(f"Предложений · {count} из {limit}")
        if count > limit:
            st.warning("Черновик превышает выбранный предел предложений.")
    if search_mode := result.get("rag_search_mode"):
        st.caption(f"RAG · {_RAG_SEARCH[search_mode]} · Найдено: {len(result.get('rag_ids', []))}")
    columns = st.columns(3)
    if result.get("duration_ms") is not None:
        columns[0].metric("Время", f"{result['duration_ms'] / 1000:.2f} с")
    columns[1].metric("Попытки правки", result.get("iterations", 0))
    if result.get("critic_score") is not None and result["mode"] != "baseline":
        columns[2].metric("Оценка Critic", f"{result['critic_score']:g}/10")
    if result.get("model"):
        st.text(f"{result.get('provider', '')} · {result['model']}")
    if result.get("cache_hit"):
        st.caption("Повтор из кэша · модель не вызывалась.")
    for message in result.get("validation_errors", []):
        st.error(message)
    for message in result.get("validation_warnings", []):
        st.warning(message)
    with st.expander("Исходная новость и версии резюме"):
        st.text(result["text"])
        if result.get("initial_summary"):
            st.markdown("**Первоначальный вариант**")
            st.text(result["initial_summary"])
        for revision in result.get("revisions", []):
            st.markdown(f"**Правка {revision['iteration']}**")
            st.text(revision["summary"])


def show_agents(client, request_id: str, detail: dict | None = None) -> None:
    path = f"/api/v1/requests/{request_id}"
    detail = detail if detail is not None else client.get(path)
    log = client.get(path + "/reasoning_log")
    st.caption(_CAUTION)
    stages = [AGENTS.get(step["agent"], step["agent"]) for step in log["steps"]]
    if stages:
        st.text("Исходная новость → " + " → ".join(stages) + " → Результат")
    else:
        st.info("Журнал появится после завершения запроса.")
    st.subheader("Фактические шаги")
    for position, step in enumerate(log["steps"], 1):
        agent = AGENTS.get(step["agent"], step["agent"])
        duration = f" · {step['duration_ms']:.0f} мс" if step.get("duration_ms") is not None else ""
        with st.expander(f"{position}. {agent} · {_status(step['status'])}{duration}"):
            for error in step["errors"]:
                st.error(issue(error))
            for warning in step["warnings"]:
                st.warning(issue(warning))
            if step["agent"] == "RAG":
                st.caption(f"Найдено учебных записей: {len(detail.get('rag_ids', []))}.")
            if step["agent"] == "Critic" and step.get("critic_score") is not None:
                st.write(f"Оценка · {step['critic_score']:g}/10")
                for key, label in (
                    ("factual_accuracy", "Фактическая точность"), ("completeness", "Полнота"),
                    ("coherence", "Связность"), ("readability", "Удобочитаемость"),
                    ("conciseness", "Краткость"),
                ):
                    if key in step.get("critic_criteria_scores", {}):
                        st.caption(f"{label} · {step['critic_criteria_scores'][key]:g}/10")
                if step.get("critical_factual_error"):
                    st.warning(issue("critical_factual_error"))
                if step.get("missing_key_fact"):
                    st.warning(issue("missing_key_fact"))
            if step["agent"] == "Editor":
                revision = next((entry for entry in log["revisions"] if entry["iteration"] == step["iteration"]), None)
                st.text(revision["summary"] if revision else "На этом шаге текст не изменился.")
            if step["agent"] == "Supervisor":
                st.write(_status(log["status"]))
                if log["status"] == "verified":
                    st.caption("Правила Validator пройдены, оценка Critic достигла порога. Критические замечания не зарегистрированы.")
                elif log["status"] == "draft":
                    st.caption("В этом режиме была только генерация резюме.")
                else:
                    st.caption("Результат и замечания последней выбранной версии показаны ниже.")
    with st.expander("Вызовы языковых моделей"):
        calls = client.get(path + "/llm_calls")
        if calls:
            st.dataframe([{"Шаг": AGENTS.get(call["agent"], call["agent"]), "Модель": call["model"],
                           "Статус": issue("language_mismatch") if call.get("language_mismatch") else _status(call["status"]),
                           "Язык": _LANGUAGES.get(call.get("actual_language")), "Повтор": call.get("language_retry", False),
                           "Мс": call["duration_ms"], "Токены": call["tokens"]}
                          for call in calls], hide_index=True)
        else:
            st.caption("Вызовов нет.")


def agents_page(client) -> None:
    st.caption("ЖУРНАЛ ОБРАБОТКИ")
    st.title("Работа агентов")
    rows = client.get("/api/v1/requests", {"limit": 25})
    choices = {row["request_id"]: row["preview"] for row in rows}
    if active := st.session_state.get("active_request"):
        choices.setdefault(active, "Текущий запрос")
    if not choices:
        st.info("Сначала запустите суммаризацию.")
        return
    ids = list(choices)
    active = st.session_state.get("active_request")
    chosen = st.selectbox("Запрос для просмотра", ids, format_func=lambda key: f"{choices[key][:75]} · {key[:8]}",
                          index=ids.index(active) if active in ids else 0)
    detail = client.get(f"/api/v1/requests/{chosen}")
    show_agents(client, chosen, detail)
    show_summary(detail)


def history_page(client) -> None:
    st.caption("ЛИЧНЫЕ ЗАПРОСЫ")
    st.title("История запросов")
    left, middle, right = st.columns([2, 2, 3])
    mode = left.selectbox("Режим", ["all", *MODES], format_func=lambda key: MODES.get(key, "Все режимы"))
    status = middle.selectbox("Статус", ["all", *list(STATUSES)[:10]], format_func=lambda key: STATUSES.get(key, "Все статусы"))
    recent = right.checkbox("Фильтр по датам")
    filters = {key: value for key, value in {"mode": mode, "status": status}.items() if value != "all"}
    if recent:
        interval = right.date_input("Даты по UTC", (date.today() - timedelta(days=7), date.today()))
        if len(interval) != 2:
            st.info("Выберите обе даты.")
            return
        filters.update(date_from=interval[0].isoformat(), date_to=interval[1].isoformat())
    page = st.number_input("Страница", min_value=1, step=1, value=1)
    rows = client.get("/api/v1/requests", filters | {"limit": 20, "offset": (page - 1) * 20})
    if not rows:
        st.info("Пока нет запросов по этим условиям.")
        return
    by_id = {row["request_id"]: row for row in rows}
    selected = st.selectbox("Открыть запрос", list(by_id), format_func=lambda key: (
        f"{by_id[key]['created_at'][:10]} · {_status(by_id[key]['status'])} · {by_id[key]['preview'][:64]} · {key[:8]}"
    ))
    show_summary(client.get(f"/api/v1/requests/{selected}"))
    st.button("Открыть журнал этого запроса", on_click=open_agents, args=[selected])


def _chart(rows: list[dict], metrics: list[tuple[str, str]], title: str, y_label: str, *, percent=False) -> None:
    fig = go.Figure()
    for (key, name), color in zip(metrics, ["#007E76", "#5474A6", "#AE7540"], strict=False):
        measured = [row for row in rows if row.get(key) is not None]
        if measured:
            fig.add_bar(name=name, y=[row[key] for row in measured], x=[row["label"] for row in measured],
                        marker_color=color)
    if not fig.data:
        st.caption(f"{title} · данных пока нет.")
        return
    fig.update_layout(title=title, height=340, barmode="group", margin=dict(l=16, r=16, b=12, t=45),
                      xaxis_title="Конфигурация", yaxis_title=y_label, legend=dict(orientation="h", y=1.03))
    if percent:
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def comparison_page(client) -> None:
    st.caption("ОЦЕНКА КАЧЕСТВА")
    st.title("Сравнение моделей и режимов")
    source = st.radio("Источник результатов", ["Контрольный эксперимент", "Мои оценки", "Фоновые эксперименты"],
                      horizontal=True)
    if source == "Фоновые эксперименты":
        batch_evaluation(client)
        return
    if source == "Контрольный эксперимент":
        try:
            archive = client.get("/api/v1/evaluate/archive")
        except APIError as error:
            if error.status != 404:
                raise
            st.info("Локальный архив сравнения пока не настроен. Можно открыть собственные оценки.")
            return
        rows = archive_rows(archive)
        st.caption("Одинаковые 20 новостей XSum. Различия на этой выборке не доказывают статистического превосходства.")
        st.caption(f"Контроль · {archive['baseline_id']} · эксперимент · {archive['experiment_id']}")
    else:
        with st.expander("Оценить модель на одной контрольной новости"):
            options = _generation_options(client)
            if options:
                with st.form("evaluation"):
                    test_id = st.text_input("ID контрольной новости", max_chars=64,
                                            placeholder="Пустое поле — первый отложенный пример")
                    mode = st.selectbox("Режим оценки", ["baseline", "full"], format_func=MODES.get)
                    evaluate = st.form_submit_button("Запустить оценку")
                if evaluate:
                    payload = options | {"mode": mode}
                    if test_id.strip():
                        payload["test_ids"] = [test_id.strip()]
                    with st.spinner("Вычисляются метрики…"):
                        result = client.post("/api/v1/evaluate", payload)
                    st.success(f"Оценено примеров: {result['scored_count']}.")
        rows = history_rows(client.get("/api/v1/evaluate/history", {"limit": 20}))
        st.caption("Для строгого сравнения используйте одинаковые выборки и настройки.")
    if not rows:
        st.info("Сохранённых результатов пока нет.")
        return
    quality, performance = st.tabs(["Качество текста", "Время и проверки"])
    with quality:
        _chart(rows, [("rouge_l", "ROUGE-L"), ("bertscore_f1", "BERTScore F1")], "Сходство с эталоном", "Балл")
        _chart(rows, [("bleu_1", "BLEU-1"), ("bleu_2", "BLEU-2"), ("meteor", "METEOR")],
               "Лексическое сходство", "Балл")
        with st.expander("Профиль метрик и точные значения"):
            metrics = {"rouge_l": "ROUGE-L", "bleu_1": "BLEU-1", "bleu_2": "BLEU-2", "meteor": "METEOR"}
            comparable = [row for row in rows if all(row.get(key) is not None for key in metrics)]
            if comparable:
                fig = go.Figure()
                for row in comparable:
                    fig.add_scatterpolar(r=[row[key] for key in metrics], theta=list(metrics.values()),
                                         name=row["label"], mode="lines+markers")
                fig.update_layout(height=360, margin=dict(l=48, r=48, b=40, t=32),
                                  polar={"radialaxis": {"range": [0, 1]}}, legend={"orientation": "h"})
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.dataframe([
                {"Конфигурация": row["label"], "Примеров": row["processed"], "Успешно": row.get("successful")}
                | {name: row.get(key) for key, name in metrics.items()}
                | {"BERTScore F1": row.get("bertscore_f1"), "Среднее время, с": row.get("mean_seconds")}
                for row in rows
            ], hide_index=True)
    with performance:
        _chart(rows, [("mean_seconds", "Среднее время")], "Обработка одной новости", "Секунды")
        _chart(rows, [("success_rate", "Успешные генерации")], "Доля успешных генераций", "Доля", percent=True)
        _chart(rows, [("verified_rate", "Автоматически проверено"), ("needs_review_rate", "Нужна ручная проверка")],
               "Статусы автоматической проверки", "Доля", percent=True)


def batch_evaluation(client) -> None:
    models = {f"{provider} · {model}": {"provider": provider, "model": model}
              for provider, names in _available(client).items() for model in names}
    if models:
        with st.form("batch_experiment"):
            selected = st.multiselect("Модели", list(models), max_selections=4)
            modes = st.multiselect("Режимы эксперимента", list(MODES), default=["baseline"], format_func=MODES.get)
            sizes = st.multiselect("Размеры выборки", [20, 50, 100], default=[20])
            st.caption("Расчёт запускается только кнопкой ниже и может занять несколько минут или больше.")
            submit = st.form_submit_button("Поставить расчёт в очередь")
        if submit:
            if not selected or not modes or not sizes:
                st.warning("Выберите модели, режимы и размеры выборки.")
            else:
                result = client.post("/api/v1/evaluate/batch", {"models": [models[key] for key in selected],
                                                               "modes": modes, "sample_sizes": sizes})
                st.session_state.experiment = result["experiment_id"]
                st.success("Задача передана брокеру.")
    with st.form("experiment_status"):
        value = st.text_input("ID эксперимента", st.session_state.get("experiment", ""), max_chars=36)
        refresh = st.form_submit_button("Проверить состояние")
    if refresh and value.strip():
        try:
            identifier = str(UUID(value.strip()))
        except ValueError:
            st.error("Введите UUID эксперимента.")
            return
        job = client.get(f"/api/v1/evaluate/experiments/{identifier}")
        names = {"queued": "В очереди", "running": "В работе", "completed": "Завершён",
                 "failed": "Ошибка", "broker_unavailable": "Брокер недоступен"}
        st.write(names.get(job["status"], job["status"]))
        st.caption(f"Обработано · {job['progress']} из {job['total']}")


def monitor_page(client) -> None:
    st.title(MONITOR_PAGE)
    st.button("Обновить")
    result = client.get("/health/dependencies")
    names = {"api": "FastAPI", "database": "База данных", "ollama": "Ollama", "qdrant": "Qdrant",
             "rag_snapshot": "Индекс RAG", "redis": "Redis", "worker": "Celery worker"}
    statuses = {"ok": "Доступен", "unavailable": "Недоступен", "snapshot_available": "Снимок найден",
                "not_checked": "Проверка недоступна"}
    st.dataframe([{"Сервис": name, "Состояние": statuses[result[key]]} for key, name in names.items()], hide_index=True)
    if cache := result.get("cache"):
        st.subheader("Кэш baseline")
        st.caption("Включён" if cache["enabled"] else "Отключён")
        if "entries" in cache:
            left, right = st.columns(2)
            left.metric("Действующих записей", cache["entries"])
            right.metric("Попаданий", cache["hits"])


def review_page(client) -> None:
    st.caption("РЕДАКТОРСКАЯ ОЧЕРЕДЬ")
    st.title("Ручная проверка")
    st.write("Сверьте резюме с новостью. Решение редактора хранится отдельно от автоматического вердикта.")
    status = st.selectbox("Очередь", ["pending", "approved", "rejected"], format_func=_status)
    rows = client.get("/api/v1/hitl/queue", {"status": status})
    if notice := st.session_state.pop("review_notice", None):
        st.success(notice)
    if rows:
        choices = {row["id"]: row for row in rows}
        chosen = st.selectbox("Пример", list(choices), format_func=lambda key: f"{choices[key]['input_text'][:80]} · {key[:8]}")
        item = choices[chosen]
        endpoint = f"/api/v1/hitl/queue/{chosen}/decisions"
        with st.expander("Исходная новость", expanded=True):
            st.text(item["input_text"])
        st.markdown("**Резюме модели**")
        st.code(item["model_output"], language=None, wrap_lines=True)
        for code in item["reasons"]:
            st.warning(issue(code))
        if item["status"] == "pending":
            with st.form(f"decision_{chosen}"):
                corrected = st.text_area("Итоговый текст", item["model_output"][:600], max_chars=600)
                comment = st.text_area("Примечание", max_chars=1000)
                approve, reject = st.columns(2)
                yes = approve.form_submit_button("Подтвердить", type="primary")
                no = reject.form_submit_button("Отклонить")
            if yes or no:
                result = client.post(endpoint, {"decision": "approved" if yes else "rejected",
                                                "corrected_summary": corrected if yes else None,
                                                "comment": comment or None})
                note = {
                    "awaiting_index": "Решение сохранено. Пара добавлена в базу знаний и ждёт публикации индекса.",
                    "heldout": "Решение сохранено. Контрольная новость исключена из обучения.",
                    "already_in_corpus": "Решение сохранено. Ранее опубликованная обучающая пара сохранена.",
                    "holdout_unavailable": "Решение сохранено. Контрольный набор недоступен для проверки обучающей пары.",
                }
                st.session_state.review_notice = note.get(result["training_status"], "Решение сохранено.")
                st.rerun()
        else:
            for decision in client.get(endpoint):
                st.caption(f"{_status(decision['decision'])} · {decision['created_at'][:19]}")
                if decision.get("corrected_summary"):
                    st.text(decision["corrected_summary"])
                if decision.get("comment"):
                    st.text(decision["comment"])
    else:
        st.info("В этой очереди пока нет примеров.")
    st.divider()
    st.caption("Публикация пересобирает оба поисковых индекса. Пока работа идёт, активен прежний снимок.")
    if st.button("Опубликовать поисковые индексы"):
        with st.spinner("Строятся индексы…"):
            result = client.post("/api/v1/hitl/publish")
        st.success(f"Опубликовано документов: {result['documents']}. Снимок {result['version']}.")

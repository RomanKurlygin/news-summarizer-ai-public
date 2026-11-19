# REST API NewsSummarizer AI

Запустите Ollama с установленной моделью `qwen2.5:1.5b`, задайте `DATABASE_URL` и `JWT_SECRET_KEY` в окружении и выполните `poetry run alembic upgrade head`, затем `poetry run python run.py`. Для режима `full` нужны проверенные индексы BM25 и Qdrant. [Подготовка индексов](./rag.md)

В Swagger откройте `POST /api/v1/auth/register` или используйте существующую учётную запись. Получите `access_token` через `POST /api/v1/auth/login`. Нажмите **Authorize**, в поле **BearerAuth** вставьте значение токена и выполните `GET /api/v1/auth/me`. Заголовок `Authorization` Swagger добавляет автоматически. [Swagger UI](http://127.0.0.1:8000/docs)

## Генерация

`POST /api/v1/summarize` принимает следующую конфигурацию и текст.

```json
{
  "text": "The city reopened the public library on Monday after six months of renovations.",
  "provider": "ollama",
  "model": "qwen2.5:1.5b",
  "mode": "full",
  "max_sentences": 2
}
```

`text` содержит от 1 до 8000 символов после обрезки пробелов. `category` при наличии ограничивает выдачу RAG. Для генерации доступен `ollama` или уже загруженная локальная модель `transformers`. Если `model` опущено, выбирается модель выбранного провайдера из настроек. Варианты могут задавать температуру первой генерации.

- `baseline` выполняет только Summarizer. Статус результата `draft`. Если модель превысила `max_sentences`, возвращаются только первые разрешённые предложения.
- `full` использует справочник, гибридный RAG, Summarizer, Validator и Critic. При необходимости Supervisor вызывает Editor до двух раз, повторяя проверки после изменения текста. Validator считает превышение `max_sentences` ошибкой. Непрошедший черновик доступен с соответствующим статусом.

Ответ включает `request_id`, `text`, `initial_summary`, `final_summary`, фактические `provider` и `model`, результаты Validator и Critic, `iterations`, `status`, `duration_ms` и `rag_ids`. Поля непроведённых проверок равны `null`. `revisions` содержит изменившиеся версии, `selected_iteration` указывает на выбранную, начиная с 0 для черновика.

- `verified` выдаётся, когда Validator и Critic прошли проверку и отсутствуют критическая фактическая ошибка и замечание для ручной сверки.
- `needs_review` указывает на нерешённое существенное замечание. Результат доступен для дальнейшей проверки.
- `unchanged` означает, что Editor не изменил текст. `iteration_limit` означает исчерпание лимита правок.
- `failed`, `incomplete_configuration` и `cancelled` доступны в истории как результаты сбоев или отмены. Запись `running` доступна до завершения.

## Группы запросов

- `POST /api/v1/summarize/batch` принимает объект с массивом `items` из 1–4 полных запросов /summarize.
- `POST /api/v1/summarize/multi` принимает общие `text` и необязательное `category`, а также массив `variants` из 2–4 конфигураций без этих полей.
- `POST /api/v1/summarize/ensemble` принимает `text`, необязательное `category`, общий `max_sentences` и массив `candidates` из 2–4 конфигураций с `provider`, `model` и необязательной `temperature`. Все кандидаты выполняются в режиме `full`. Победитель выбирается из результатов `verified` по баллу Critic, при равенстве сохраняется первый. Если подходящих кандидатов нет, `winner` равен `null`.

В групповых ответах `results` сохраняет порядок входа. Для каждого элемента доступны `configuration`, `result` и отдельное `error` с кодом, описанием, HTTP-статусом и ID. У каждого кандидата собственный запрос и два журнала. Внутри группы обработка последовательная. На процесс Uvicorn допускается до двух активных генераций, доступ к Qdrant local сериализован. Лимиты времени по умолчанию составляют 120 секунд на генерацию и 300 секунд на весь запрос. Их можно изменить через `API_RUN_TIMEOUT_SECONDS` и `API_GROUP_TIMEOUT_SECONDS`.

## История и роли

- `GET /api/v1/requests` возвращает только запросы текущего пользователя с фильтрами `mode`, `status`, `date_from`, `date_to` по UTC, `limit`, `offset`. `GET /api/v1/requests/{request_id}/llm_calls` выдаёт безопасные метаданные вызовов владельцу или администратору.
- `GET /api/v1/models` возвращает установленные локальные модели. `GET /api/v1/evaluate/archive` выдаёт только разрешённые агрегаты настроенного локального архива. Оба маршрута требуют JWT.
- `GET /api/v1/hitl/queue`, чтение и запись `/api/v1/hitl/queue/{item_id}/decisions` и `POST /api/v1/hitl/publish` доступны редактору и администратору. Повторное решение даёт `409`. [Работа с очередью](./ui.md#страницы)

- `GET /api/v1/requests/{request_id}` возвращает результат, ID владельца и время создания.
- `GET /api/v1/requests/{request_id}/reasoning_log` возвращает владельца, финальный статус, упорядоченные шаги, оценки, коды замечаний и версии Editor. Записи `LLMCall` в базе связаны с тем же UUID. API журнала выдаёт отобранные наблюдаемые поля.
- `POST /api/v1/evaluate` вычисляет метрики. `GET /api/v1/evaluate/history` возвращает сохранённые оценки с пагинацией. [Метрики и формат оценки](./evaluation.md)
- `POST /api/v1/evaluate/batch` ставит в Celery пакет по фиксированным выборкам и возвращает `202` с `experiment_id`. При недоступном брокере возвращается `503` с кодом `broker_unavailable`.
- `GET /api/v1/evaluate/experiments/{experiment_id}` возвращает статус, `progress` и `total`, протокол и результаты. `GET /api/v1/evaluate/compare` отдаёт сохранённые сравнения. Допустимые фильтры — `experiment_id`, `sample_size`, `models` как список через запятую и `limit`.

Пользователь и редактор видят собственные запросы, журналы и оценки. Администратор видит запросы и оценки других пользователей. Для несуществующего UUID возвращается `404`, для чужого запроса без административной роли — `403`, для отсутствующего или просроченного токена и неактивной учётной записи — `401`.

Пример фонового пакета для `POST /api/v1/evaluate/batch` с заголовком Bearer.

```json
{
  "models": [
    {"provider": "ollama", "model": "qwen2.5:1.5b"},
    {"provider": "transformers", "model": "Qwen/Qwen2.5-0.5B-Instruct"}
  ],
  "modes": ["baseline", "+agents"],
  "sample_sizes": [20],
  "seed": 42,
  "parallelism": 1
}
```

Статус `queued` означает принятие сообщения брокером. `running` означает, что worker захватил задачу. `completed` означает, что получены итоговые вердикты всех примеров. Отдельные ошибки показаны в агрегате. `failed` и `cancelled` сохраняют последний прогресс. В истории этапа 13 запись содержит `experiment_id` и сводку `aggregate`. [Настройка Redis и worker](./experiments.md#redis-и-celery)

При неверных полях запроса API возвращает `422`. Ошибка ответа модели возвращается с `502`, недоступность модели, поиска или БД — с `503`, истечение таймаута — с `504`. При разрыве соединения незавершённая генерация отменяется. Начатый запрос сохраняет статус `cancelled`. Запущенное вычисление метрик на CPU может завершиться в фоновом потоке после отмены HTTP-ответа.

## Администрирование данных

Доступ имеют пользователи с ролью `admin`. Для каждого ресурса `knowledge-base` и `glossary` доступны `GET` и `POST /api/v1/admin/{resource}`, затем `GET`, `PUT` и `DELETE /api/v1/admin/{resource}/{row_id}`. Списки принимают `limit` и `offset`.

`POST /api/v1/admin/{resource}/import` принимает CSV UTF-8 в теле `text/csv`, до 100 строк и 256 КиБ. Для базы знаний обязательны `source_text,target_text`, для справочника — `term,definition`. Обоим можно задать `category,language`. В справочнике `aliases` принимает JSON-массив строк в CSV-поле. Ошибка любой записи откатывает весь импорт. Контрольный набор проверяется перед записью обучающей пары. После изменений базы знаний запустите `poetry run python -m scripts.build_indexes`, чтобы опубликовать новый снимок поиска.

Кэш результатов ещё не работает. Его ключ должен учитывать текст, модель, режим, параметры генерации, категорию и версию базы знаний.

## Проверка

```powershell
poetry run pytest tests/integration/test_stage12_api.py tests/unit/test_metrics_calculator.py
$env:RUN_LIVE_STAGE12 = '1'
poetry run pytest tests/integration/test_live_stage12_api.py
```

Сетевой тест выделяет отдельный порт loopback и временную SQLite, запускает Uvicorn, регистрирует временного пользователя, проверяет запросы с JWT и завершает сервер.

# NewsSummarizer AI

NewsSummarizer AI — приложение для сокращения русских и английских новостей до 1–3 предложений с автоматическими проверками и возможностью ручной проверки редактором.

В публичном репозитории находятся исходный код, тесты, миграции и конфигурации. Для первого запуска нужны настройка секретов, Docker Desktop и локальная модель Ollama. Учебный корпус, справочник, пользовательские базы и веса моделей готовятся отдельно.

## Архитектура

Streamlit отправляет авторизованные REST-запросы FastAPI. Сервис суммаризации использует общий LLMManager, сохраняет запросы и журналы в PostgreSQL и при необходимости обращается к поиску и Supervisor.

- **Summarizer** создаёт черновик.
- **Validator** проверяет длину, повторы и отдельные фактические расхождения правилами Python.
- **Critic** оценивает качество через LLM-as-a-Judge.
- **Editor** исправляет замечания, если потребуется.
- **Supervisor** управляет последовательностью и допускает до двух редакторских попыток.

`baseline` выполняет одиночную генерацию. `full` дополнительно использует справочник, RAG и цикл агентов. Сохранённые проверки и вызовы модели доступны владельцу запроса. Редактор и администратор могут обрабатывать очередь HITL.

Автоматические правила и небольшая LLM могут пропускать фактические и грамматические ошибки. Статус `verified` означает прохождение применённых проверок. Результаты с замечаниями требуют сверки с исходником.

## Технологии

Python 3.11, Poetry, FastAPI, Streamlit, Plotly, SQLAlchemy asyncio, Alembic, PostgreSQL 16, SQLite для разработки и тестов, JWT и bcrypt. Поиск использует SentenceTransformers MiniLM, BM25, Qdrant и Reciprocal Rank Fusion. Локальные LLM-провайдеры — Ollama и HuggingFace Transformers. Redis 7 служит брокером Celery, Nginx проксирует API и интерфейс.

Калькулятор качества использует SacreBLEU, rouge-score, NLTK и bert-score для BLEU-1/2, ROUGE-1/2/L, METEOR, chrF, chrF++, BERTScore и Exact Match. Оценка требует отдельно подготовленного контрольного корпуса.

## Запуск с Docker Desktop и Ollama на Windows

Установите Docker Desktop, включите WSL2 и Linux containers, дождитесь готовности Engine. Установите Ollama и загрузите модель. Ссылки на установщики — [Docker Desktop](https://www.docker.com/products/docker-desktop/) и [Ollama](https://ollama.com/download).

```powershell
ollama pull qwen2.5:1.5b
ollama list
git clone https://github.com/RomanKurlygin/news-summarizer-ai-public.git
cd news-summarizer-ai-public
```

Создайте локальный файл настроек с НОВЫМИ случайными секретами. Следующий пример требует PowerShell 5.1 или новее и записывает файл только при его отсутствии.

```powershell
if (Test-Path .env.docker) { throw 'Файл .env.docker уже существует. Сохраните действующие секреты.' }
$generator = [Security.Cryptography.RandomNumberGenerator]::Create()
$passwordBytes = New-Object byte[] 24
$jwtBytes = New-Object byte[] 48
try {
    $generator.GetBytes($passwordBytes)
    $generator.GetBytes($jwtBytes)
} finally {
    $generator.Dispose()
}
$dbPassword = -join ($passwordBytes | ForEach-Object { $_.ToString('x2') })
$jwtSecret = [Convert]::ToBase64String($jwtBytes)
$config = "POSTGRES_PASSWORD=$dbPassword" + [Environment]::NewLine +
          "JWT_SECRET_KEY=$jwtSecret" + [Environment]::NewLine
[IO.File]::WriteAllText((Join-Path (Get-Location) '.env.docker'), $config, (New-Object System.Text.UTF8Encoding($false)))
```

Файл `.env.docker` исключён из Git. Его секреты необходимо сохранять между запусками: PostgreSQL использует пароль, установленный при создании тома.

```powershell
$compose = @('compose', '--env-file', '.env.docker', '-f', 'compose.yaml', '-f', 'compose.windows.yaml')
docker @compose config --quiet
docker @compose up --build -d
docker @compose ps -a
```

Первая сборка скачивает несколько ГБ образов и CPU-зависимостей. Вариант `compose.windows.yaml` использует Windows Ollama по адресу `host.docker.internal:11434`; модель должна быть доступна из контейнеров. Если проверка `ollama-host-check` завершилась ошибкой, проверьте запуск Ollama, её сетевую привязку и доступ Docker Desktop.

После готовности сервисов откройте [Streamlit](http://127.0.0.1:8080) и [Swagger](http://127.0.0.1:8080/docs). Зарегистрируйте собственный аккаунт в интерфейсе и войдите. Готовых пользователей и паролей в репозитории нет.

```powershell
# Назначить редакторскую роль зарегистрированному пользователю
docker @compose exec app python -m scripts.set_role YOUR_USERNAME editor
# Проверить сообщения запуска
docker @compose logs --tail=60 init ollama-host-check app worker
# Остановить контейнеры проекта с сохранением томов
docker @compose stop
```

Для другой внешней точки входа задайте `NEWS_PROXY_PORT` в локальном `.env.docker`. Полная конфигурация также поддерживает Ollama внутри Compose. [Варианты развёртывания](docs/deployment.md)

## Возможности без учебных данных

После настройки модели и миграций доступны регистрация, вход, baseline на введённом пользователем тексте, история и журналы. Для новостей можно использовать собственные тексты. Данные приложения появляются только после выполнения запросов.

Для `full` и RAG нужен согласованный снимок BM25/Qdrant, учебные пары и отдельный holdout. Справочник требует самостоятельного импорта. Для оценки нужны эталоны, фиксированные ID и зависимости метрик, а для архивных графиков — собственные результаты сравнения. При отсутствующих данных приложение показывает соответствующее состояние или ошибку. Клонирование репозитория само по себе не создаёт эти данные.

Скрипты подготовки и проверки находятся в `scripts/`. Источник XSum доступен отдельно у его правообладателей. Перед загрузкой и использованием проверьте актуальные условия; право на распространение корпуса и извлечённых определений этим репозиторием не предоставляется. [Подготовка и формат данных](docs/dataset.md) · [RAG](docs/rag.md) · [Эксперименты](docs/experiments.md)

## Разработка и тесты

Для запуска из Python нужны Python 3.11 и Poetry 2.x.

```text
poetry install --with rag,evaluation,ui
poetry run pytest -q
```

Обычные тесты используют временные данные и контролируемые ответы провайдеров. Проверки реальных моделей включаются отдельно и требуют локальных весов. Инструкции для macOS подготовлены, проверка контейнерной сборки выполнялась на Windows x86-64.

[API](docs/api.md) · [Агенты](docs/agents.md) · [LLM](docs/llm.md) · [Метрики](docs/evaluation.md) · [Интерфейс](docs/ui.md)

## Права на материалы

Лицензия собственного кода пока не определена. Сторонние библиотеки, модели и данные используются на условиях их правообладателей. Оригинальные учебные материалы, полный корпус XSum, справочник и пользовательские данные в публичную версию не включены.

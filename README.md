# Intelligent-Assistant-for-Educational-Institutions

Практико-ориентированный проект интеллектуального ассистента для образовательных учреждений. Репозиторий включает:

- RAG API на FastAPI (`src/server`)
- веб-интерфейсы на встроенном `/web` и Streamlit (`src/client`)
- Telegram-бота с сохранением истории запросов в PostgreSQL (`src/bot`)
- индексатор документов, OCR/PDF-обработку и Chroma-векторное хранилище

![CI](https://github.com/krevetka-is-afk/Intelligent-Assistant-for-Educational-Institutions/actions/workflows/ci.yml/badge.svg)

## Что реализовано

- `POST /ask` защищён заголовком `X-API-Key`
- браузерный `/web` работает через bootstrap-admin, обычных web-пользователей и одноразовые invite-коды, а
  `POST /web/ask` использует HttpOnly-сессию без раскрытия backend `API_KEY` в JavaScript
- FastAPI, Streamlit и Telegram-бот используют единый env-контракт и структурированное логирование
- `docker-compose.yaml` поднимает `db`, `server`, `bot`, `client` с healthcheck и `restart: unless-stopped`
- при сбоях LLM RAG возвращает fallback-ответ и логирует причину на уровне `ERROR`

## Переменные окружения

Основной шаблон конфигурации: [`.env.example`](.env.example)

| Переменная                         | Где используется          | Назначение                                                                                                                                               |
|------------------------------------|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `APP_ENV`                          | `server`, `bot`, `client` | Имя окружения для логов                                                                                                                                  |
| `LOG_LEVEL`                        | `server`, `bot`, `client` | Уровень логирования                                                                                                                                      |
| `API_KEY`                          | `server`, `client`        | Shared secret для `X-API-Key`                                                                                                                            |
| `TELEGRAM_SERVICE_KEY`             | `server`, `bot`           | Отдельный служебный секрет для `X-Telegram-Service-Key`                                                                                                  |
| `SHOW_SOURCES`                     | `server`, `bot`, `client` | Показывать ли источники в `/web`, Streamlit и Telegram-боте                                                                                              |
| `WEB_BOOTSTRAP_ADMIN_TOKEN`        | `server`                  | Bootstrap token для создания первого web-admin                                                                                                           |
| `WEB_AUTH_DATABASE_URL`            | `server`                  | SQLAlchemy URL хранилища web users, invite-кодов и web-сессий. Если не задан, локально используется `./.web_auth.db`, а в контейнере `/data/web_auth.db` |
| `API_BASE_URL`                     | `bot`, `client`           | Базовый URL FastAPI                                                                                                                                      |
| `BOT_TOKEN`                        | `bot`                     | Telegram bot token                                                                                                                                       |
| `DATABASE_URL`                     | `bot`                     | SQLAlchemy URL для истории запросов                                                                                                                      |
| `POSTGRES_DB`                      | `compose`, `db`           | Имя базы PostgreSQL                                                                                                                                      |
| `POSTGRES_USER`                    | `compose`, `db`           | Пользователь PostgreSQL                                                                                                                                  |
| `POSTGRES_PASSWORD`                | `compose`, `db`           | Пароль PostgreSQL                                                                                                                                        |
| `OLLAMA_HOST`                      | `server`                  | URL локальной Ollama                                                                                                                                     |
| `LLM_MODEL`                        | `server`                  | Модель LLM                                                                                                                                               |
| `HF_EMBEDDING_MODEL`               | `server`, `indexer`       | Модель эмбеддингов                                                                                                                                       |
| `CHROMA_COLLECTION_NAME`           | `server`, `indexer`       | Имя коллекции Chroma                                                                                                                                     |
| `VECTOR_DB_DIR`                    | `server`, `indexer`       | Директория векторной БД                                                                                                                                  |
| `DOCUMENTS_DIR`                    | `server`, `indexer`       | Каталог корпуса документов                                                                                                                               |
| `RAG_TOP_K`                        | `server`                  | Сколько чанков доставать из Chroma                                                                                                                       |
| `RAG_MAX_CONTEXT_DOCUMENTS`        | `server`                  | Максимальное число найденных фрагментов, попадающих в prompt                                                                                             |
| `RAG_MAX_DOCUMENT_CHARS`           | `server`                  | Максимальная длина одного фрагмента в prompt                                                                                                             |
| `RAG_MAX_TOTAL_CONTEXT_CHARS`      | `server`                  | Общий лимит символов документного контекста в prompt                                                                                                     |
| `RAG_MAX_HISTORY_MESSAGES`         | `server`                  | Максимальное число сообщений истории, передаваемых модели как недоверенные данные                                                                        |
| `RAG_MAX_HISTORY_CHARS`            | `server`                  | Общий лимит символов истории в prompt                                                                                                                    |
| `RAG_SOURCE_SNIPPET_CHARS`         | `server`                  | Максимальная длина возвращаемой цитаты источника, включая многоточие при обрезке                                                                         |
| `RAG_TOTAL_TIMEOUT_SECONDS`        | `server`                  | Общий бюджет времени RAG                                                                                                                                 |
| `LLM_TIMEOUT_SECONDS`              | `server`                  | Таймаут вызова LLM                                                                                                                                       |
| `CONVERSATION_MEMORY_WINDOW`       | `server`                  | Размер окна памяти последних сообщений пользователя (по умолчанию `5`)                                                                                   |
| `CONVERSATION_MEMORY_TTL_SECONDS`  | `server`                  | TTL контекста диалога в секундах (по умолчанию `3600`)                                                                                                   |
| `CONVERSATION_MEMORY_MAX_SESSIONS` | `server`                  | Ограничение на число активных сессий контекста                                                                                                           |
| `PREPARE_RAG_ON_STARTUP`           | `server`                  | Подготавливать ли embeddings/vector store до ready-состояния сервиса                                                                                     |
| `AUTO_INDEX_ON_STARTUP`            | `server`                  | Автоматически индексировать `DOCUMENTS_DIR`, если vector store пуст на старте                                                                            |

`RAG_API_URL` оставлен только как legacy-алиас для Telegram-слоя и больше не является основной настройкой.

## RAG security barriers

RAG-запросы проходят через единую политику `rag-prompt-policy-v1`. Доверенные правила находятся
только в system role, а вопрос, история и найденные документы передаются модели одной user role
как JSON-данные. Строки из корпуса и истории не интерпретируются как разметка ролей, даже если
содержат `</document>`, `<system>` или похожие маркеры.

До вызова retrieval/LLM сервер отклоняет запросы на раскрытие system prompt, правил, API keys,
cookies, connection strings и служебной конфигурации. Такие ответы имеют `fallback_used=true` и `fallback_reason`,
например `policy_forbidden_control_or_secret_request` или
`policy_forbidden_instruction_override`; полный system text наружу не возвращается.

Контекст ограничивается детерминированными env-переменными `RAG_MAX_CONTEXT_DOCUMENTS`,
`RAG_MAX_DOCUMENT_CHARS`, `RAG_MAX_TOTAL_CONTEXT_CHARS`, `RAG_MAX_HISTORY_MESSAGES` и
`RAG_MAX_HISTORY_CHARS`. Источники в API-ответе возвращаются только из фактически найденных
документов, с allowlisted metadata и коротким snippet не длиннее `RAG_SOURCE_SNIPPET_CHARS`.
Если модель пытается сослаться на не найденный filename/title/URL или раскрыть служебные маркеры,
ответ заменяется безопасным отказом с `fallback_reason=policy_output_violation`.

Policy refusals не записываются в conversation memory и не отравляют следующий запрос. Обычный
успешный fallback при недоступной модели (`llm_timeout`/`llm_unavailable`) продолжает считаться
ответом по найденным документам и сохраняется по действующему контракту памяти.

В production автоиндексация отключена по умолчанию: `AUTO_INDEX_ON_STARTUP=0`. Корпус нужно
индексировать явной командой после ручного просмотра состава документов; закрытые документы не
следует добавлять в выпускной корпус без правил доступа.

## Локальный запуск

### 1. Установка зависимостей

```bash
git submodule update --init --recursive
uv venv .venv
source .venv/bin/activate
uv sync --group dev
export PYTHONPATH=.
```

### 2. Конфигурация

```bash
cp .env.example .env
```

Минимально для локальной разработки должны быть заданы:

```env
APP_ENV=development
LOG_LEVEL=INFO
API_KEY=change-me
SHOW_SOURCES=1
WEB_BOOTSTRAP_ADMIN_TOKEN=change-me-bootstrap-token
API_BASE_URL=http://localhost:8000
BOT_TOKEN=replace-with-real-token
DATABASE_URL=sqlite+aiosqlite:///./bot.db
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=qwen2.5:3b
```

`WEB_AUTH_DATABASE_URL` можно не задавать: сервер сам выберет подходящий путь для локального запуска и контейнера.
Если нужно временно скрыть источники во всех интерфейсах, установите `SHOW_SOURCES=0`.
Контекст последних сообщений хранится в БД `WEB_AUTH_DATABASE_URL`, поэтому при server deploy и рестартах не теряется.

Первый вход в `/web` делается через bootstrap token:

1. оператор сервера задаёт `WEB_BOOTSTRAP_ADMIN_TOKEN`
2. первый администратор открывает `/web` и создаёт admin-учётную запись
3. администратор выпускает одноразовые invite-коды для обычных web-пользователей
4. пользователь активирует invite-код и создаёт собственные login/password

### 3. Индексация документов

```bash
source .venv/bin/activate
export PYTHONPATH=.
uv run python -m src.server.app.index_documents \
  --input-dir "$(pwd)/data_and_documents" \
  --persist-dir "$(pwd)/src/server/chrome_langchain_db" \
  --rebuild
```

### 4. Запуск FastAPI

```bash
source .venv/bin/activate
export PYTHONPATH=.
uv run uvicorn src.server.app.main:app --reload
```

Основные endpoints:

- `GET /health` и `GET /live` — liveness HTTP-процесса (без проверки зависимостей)
- `GET /ready` — readiness RAG-сервиса: `initializing`/`failed` → HTTP 503,
  `ready`/`degraded` → HTTP 200; поле `checks` отдельно показывает состояния `rag` и `web_auth`
- `GET /metrics` c `X-API-Key`
- `GET /web`
- `POST /web/bootstrap`
- `POST /web/login`
- `POST /web/invite/accept`
- `POST /web/admin/invites`
- `POST /ask` c `X-API-Key`
- `POST /web/ask` c `X-API-Key` или серверной web-сессией после bootstrap/login/invite activation

Пример защищённого запроса:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"question":"Когда пересдача?"}'
```

Telegram-бот обращается к отдельному служебному маршруту;
сервер сам строит ключ памяти из проверенного `telegram_user_id`:

```bash
curl -X POST http://localhost:8000/telegram/ask \
  -H "Content-Type: application/json" \
  -H "X-Telegram-Service-Key: $TELEGRAM_SERVICE_KEY" \
  -d '{"telegram_user_id":123456, "question":"Когда пересдача?"}'
```

### 5. Запуск Streamlit

```bash
source .venv/bin/activate
export PYTHONPATH=.
uv run streamlit run src/client/app/streamlit_app.py
```

### 6. Запуск Telegram-бота

```bash
source .venv/bin/activate
export PYTHONPATH=.
uv run python -m src.bot.bot
```

## Docker Compose

Файл [`docker-compose.yaml`](docker-compose.yaml) поднимает:

- `db` на PostgreSQL 16
- `server` на `http://localhost:8000`
- `client` на `http://localhost:8501`
- `bot` как отдельный контейнер

Запуск:

```bash
cp .env.example .env
docker compose --profile dev up --build
```

Проверки состояния:

- `db`: `pg_isready`
- `server`: `GET /ready` для Docker readiness; `GET /health` и `GET /live` только для liveness
- `bot`: fail-fast старт + Docker restart policy

Временные файлы и `/tmp` для `server`, `bot`, `client` вынесены в `tmpfs`. Operational-логи пишутся только в
stdout/stderr контейнеров.

## Проверки

```bash
PYTHONPATH=. uv run pytest -q
PYTHONPATH=. uv run ruff check .
PYTHONPATH=. uv run black --check .
PYTHONPATH=. uv run isort --check-only .
```

Полный локальный прогон:

```bash
./uv-linters.sh
```

## Документация

- ТЗ: [
  `docs/technical-specification-for-IAfEI/ТЗ-общее/ТЗ-общее.pdf`](docs/technical-specification-for-IAfEI/ТЗ-общее/ТЗ-общее.pdf)

- Референс по структуре ТЗ: [
  `docs/technical-specification-for-IAfEI/README.md`](docs/technical-specification-for-IAfEI/README.md)

# Intelligent-Assistant-for-Educational-Institutions

Практико-ориентированный проект интеллектуального ассистента для образовательных учреждений. Репозиторий включает:

- RAG API на FastAPI (`src/server`)
- веб-интерфейсы на встроенном `/web` и Streamlit (`src/client`)
- Telegram-бота с сохранением истории запросов в PostgreSQL (`src/bot`)
- индексатор документов, OCR/PDF-обработку и Chroma-векторное хранилище

![CI](https://github.com/krevetka-is-afk/Intelligent-Assistant-for-Educational-Institutions/actions/workflows/ci.yml/badge.svg)

## Что реализовано

- `POST /ask` защищён заголовком `X-API-Key`
- браузерный `/web` использует серверный прокси `POST /web/ask` и не раскрывает API-ключ
- FastAPI, Streamlit и Telegram-бот используют единый env-контракт и структурированное логирование
- `docker-compose.yaml` поднимает `db`, `server`, `bot`, `client` с healthcheck и `restart: unless-stopped`
- при сбоях LLM RAG возвращает fallback-ответ и логирует причину на уровне `ERROR`

## Переменные окружения

Основной шаблон конфигурации: [`.env.example`](.env.example)

| Переменная | Где используется | Назначение |
| --- | --- | --- |
| `APP_ENV` | `server`, `bot`, `client` | Имя окружения для логов |
| `LOG_LEVEL` | `server`, `bot`, `client` | Уровень логирования |
| `API_KEY` | `server`, `bot`, `client` | Shared secret для `X-API-Key` |
| `API_BASE_URL` | `bot`, `client` | Базовый URL FastAPI |
| `BOT_TOKEN` | `bot` | Telegram bot token |
| `DATABASE_URL` | `bot` | SQLAlchemy URL для истории запросов |
| `POSTGRES_DB` | `compose`, `db` | Имя базы PostgreSQL |
| `POSTGRES_USER` | `compose`, `db` | Пользователь PostgreSQL |
| `POSTGRES_PASSWORD` | `compose`, `db` | Пароль PostgreSQL |
| `OLLAMA_HOST` | `server` | URL локальной Ollama |
| `LLM_MODEL` | `server` | Модель LLM |
| `HF_EMBEDDING_MODEL` | `server`, `indexer` | Модель эмбеддингов |
| `CHROMA_COLLECTION_NAME` | `server`, `indexer` | Имя коллекции Chroma |
| `VECTOR_DB_DIR` | `server`, `indexer` | Директория векторной БД |
| `DOCUMENTS_DIR` | `server`, `indexer` | Каталог корпуса документов |
| `RAG_TOP_K` | `server` | Сколько чанков доставать из Chroma |
| `RAG_TOTAL_TIMEOUT_SECONDS` | `server` | Общий бюджет времени RAG |
| `LLM_TIMEOUT_SECONDS` | `server` | Таймаут вызова LLM |

`RAG_API_URL` оставлен только как legacy-алиас для Telegram-слоя и больше не является основной настройкой.

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
API_BASE_URL=http://localhost:8000
BOT_TOKEN=replace-with-real-token
DATABASE_URL=sqlite+aiosqlite:///./bot.db
OLLAMA_HOST=http://localhost:11434
```

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

- `GET /health`
- `GET /metrics`
- `GET /web`
- `POST /ask` c `X-API-Key`
- `POST /web/ask` без клиентского секрета, только для встроенного веба

Пример защищённого запроса:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"question":"Когда пересдача?"}'
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
python src/bot/bot.py
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
- `server`: `GET /health`
- `bot`: fail-fast старт + Docker restart policy

Временные файлы и `/tmp` для `server`, `bot`, `client` вынесены в `tmpfs`. Operational-логи пишутся только в stdout/stderr контейнеров.

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

- ТЗ: [`docs/technical-specification-for-IAfEI/ТЗ-общее/ТЗ-общее.pdf`](docs/technical-specification-for-IAfEI/ТЗ-общее/ТЗ-общее.pdf)

- Референс по структуре ТЗ: [`docs/technical-specification-for-IAfEI/README.md`](docs/technical-specification-for-IAfEI/README.md)

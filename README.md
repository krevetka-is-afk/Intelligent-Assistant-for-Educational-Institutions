# Intelligent-Assistant-for-Educational-Institutions

Практико-ориентированный проект по созданию интеллектуального ассистента для образовательных учреждений. В текущем состоянии репозиторий содержит:

- RAG API на FastAPI (`src/server`)
- веб-интерфейс на Streamlit (`src/client`) и html
- сервисный слой Telegram-бота с историей запросов в БД (`src/bot`)
- OCR/PDF-обработку, индексатор документов и выдачу ответа вместе со списком источников

![CI](https://github.com/krevetka-is-afk/Intelligent-Assistant-for-Educational-Institutions/actions/workflows/ci.yml/badge.svg)

## Что уже реализовано

- команды и сервисный слой бота
- обработка текста, изображений через OCR и PDF
- отдельный CLI-индексатор документов для Chroma
- Chroma + HuggingFace ruBERT-tiny2 эмбеддинги по реальному корпусу документов
- сохранение истории запросов в БД
- fallback-ответ при недоступности LLM, confidence и метрики `/metrics`

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- локально установленный [Ollama](https://ollama.com/) с моделями:
  - `mistral:7b`
- Tesseract OCR с языками `rus` и `eng` для локального OCR/PDF-сценария

## Переменные окружения

| Переменная | Где используется | Значение по умолчанию | Назначение |
| --- | --- | --- | --- |
| `OLLAMA_HOST` | `server` | `http://localhost:11434` | URL локального Ollama |
| `LLM_MODEL` | `server` | `mistral:7b` | Модель Ollama для генерации ответа |
| `VECTOR_DB_DIR` | `server` | `src/server/chrome_langchain_db` | Путь к директории с Chroma DB |
| `DOCUMENTS_DIR` | `server`, `indexer` | `data_and_documents` | Каталог с исходным корпусом документов |
| `HF_EMBEDDING_MODEL` | `server`, `indexer` | `cointegrated/rubert-tiny2` | HuggingFace-модель эмбеддингов |
| `CHROMA_COLLECTION_NAME` | `server`, `indexer` | `edu_documents` | Имя коллекции Chroma |
| `RAG_TOP_K` | `server` | `4` | Сколько чанков извлекать из Chroma |
| `RAG_TOTAL_TIMEOUT_SECONDS` | `server` | `20` | Общий таймаут RAG |
| `LLM_TIMEOUT_SECONDS` | `server` | `18` | Таймаут вызова LLM внутри RAG |
| `API_BASE_URL` | `client`, `bot` | `http://localhost:8000` | Базовый URL FastAPI-сервера |
| `DATABASE_URL` | `bot` | нет | База истории запросов Telegram-слоя |

Пример локального `.env`:

```env
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=mistral:7b
DOCUMENTS_DIR=$(pwd)/data_and_documents
HF_EMBEDDING_MODEL=cointegrated/rubert-tiny2
CHROMA_COLLECTION_NAME=edu_documents
VECTOR_DB_DIR=/absolute/path/to/chroma_db
RAG_TOP_K=4
RAG_TOTAL_TIMEOUT_SECONDS=20
LLM_TIMEOUT_SECONDS=18
API_BASE_URL=http://localhost:8000
DATABASE_URL=sqlite+aiosqlite:///./bot.db
```

## Локальный запуск

### 1. Установка зависимостей

```bash
git submodule update --init --recursive
uv venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
uv sync --group dev
export PYTHONPATH=.
```

### 2. Индексация реального корпуса документов

```bash
source .venv/bin/activate
export PYTHONPATH=.
export DOCUMENTS_DIR="$(pwd)/data_and_documents"
export VECTOR_DB_DIR="$(pwd)/src/server/chrome_langchain_db"
uv run python -m src.server.app.index_documents --input-dir "$DOCUMENTS_DIR" --persist-dir "$VECTOR_DB_DIR" --rebuild
```

### 3. Запуск API

```bash
source .venv/bin/activate
export PYTHONPATH=.
export OLLAMA_HOST=http://localhost:11434
export VECTOR_DB_DIR="$(pwd)/src/server/chrome_langchain_db"
export LLM_MODEL=mistral:7b
uv run uvicorn src.server.app.main:app --reload
```

API будет доступно на `http://localhost:8000`, healthcheck: `http://localhost:8000/health`, метрики: `http://localhost:8000/metrics`.

### 4. Запуск веб-интерфейса

Доступен вариант через `http://localhost:8000/web`

Либо запуск Streamlit в отдельном терминале:

```bash
source .venv/bin/activate
export PYTHONPATH=.
export API_BASE_URL=http://localhost:8000
uv run streamlit run src/client/app/streamlit_app.py
```

Веб-интерфейс будет доступен на `http://localhost:8501`.

### 5. Telegram-слой локально

В этом репозитории Telegram-часть пока представлена сервисным слоем и обработчиками (`src/bot/service.py`, `src/bot/handlers/common.py`), которые:

- вызывают `/ask`
- сохраняют историю запросов в БД
- форматируют ответ и краткий список источников
- режут длинные сообщения под лимит Telegram

Для локальной проверки bot-а:

```bash
source .venv/bin/activate
export PYTHONPATH=.
export DATABASE_URL=sqlite+aiosqlite:///./bot.db
PYTHONPATH=. uv run pytest tests/test_bot_service.py tests/test_bot_handlers_common.py -q
```

Отдельный polling/webhook runner Telegram-бота и отдельный `bot`-сервис в `docker-compose.yaml` в текущем срезе репозитория отсутствуют.

## Запуск через Docker Compose

`docker-compose.yaml` поднимает:

- `server` на `http://localhost:8000`
- `client` на `http://localhost:8501`

Перед запуском создайте `.env` рядом с `docker-compose.yaml`, например:

```env
OLLAMA_HOST=http://host.docker.internal:11434
DOCUMENTS_DIR=/data_and_documents
LLM_MODEL=mistral:7b
API_BASE_URL=http://server:8000
DATABASE_URL=sqlite+aiosqlite:///./bot.db
```

Затем выполните:

```bash
docker compose --profile dev up --build
```

По умолчанию в compose:

- векторная БД хранится в volume `app-data`
- `OLLAMA_HOST` указывает на Ollama на хост-машине
- корпус документов доступен внутри server-контейнера в `/data_and_documents`
- tg_bot пока не выделен в отдельный контейнер

## Тесты и проверка

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

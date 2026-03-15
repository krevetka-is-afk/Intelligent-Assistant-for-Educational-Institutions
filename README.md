# Intelligent-Assistant-for-Educational-Institutions

Практико-ориентированный проект по созданию интеллектуального ассистента для образовательных учреждений. В текущем состоянии репозиторий содержит:

- RAG API на FastAPI (`src/server`)
- веб-интерфейс на Streamlit (`src/client`) и html
- сервисный слой Telegram-бота с историей запросов в БД (`src/bot`)
- OCR/PDF-обработку и выдачу ответа вместе со списком источников

![CI](https://github.com/krevetka-is-afk/Intelligent-Assistant-for-Educational-Institutions/actions/workflows/ci.yml/badge.svg)

## Что уже реализовано

- команды и сервисный слой бота
- обработка текста, изображений через OCR и PDF
- сохранение истории запросов в БД
- ответ с перечнем использованных источников

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- локально установленный [Ollama](https://ollama.com/) с моделями:
  - `gemma2:2b`
  - `mxbai-embed-large:latest`
- Tesseract OCR с языками `rus` и `eng` для локального OCR/PDF-сценария

## Переменные окружения

| Переменная | Где используется | Значение по умолчанию | Назначение |
| --- | --- | --- | --- |
| `OLLAMA_HOST` | `server` | `http://localhost:11434` | URL локального Ollama |
| `VECTOR_DB_DIR` | `server` | `src/server/chrome_langchain_db` | Путь к директории с Chroma DB |
| `API_BASE_URL` | `client`, `bot` | `http://localhost:8000` | Базовый URL FastAPI-сервера |
| `DATABASE_URL` | `bot` | нет | База истории запросов Telegram-слоя |

Пример локального `.env`:

```env
OLLAMA_HOST=http://localhost:11434
VECTOR_DB_DIR=/absolute/path/to/chroma_db
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

### 2. Запуск API

```bash
export OLLAMA_HOST=http://localhost:11434
export VECTOR_DB_DIR="$(pwd)/src/server/chrome_langchain_db"
uv run uvicorn src.server.app.main:app --reload
```

API будет доступно на `http://localhost:8000`, healthcheck: `http://localhost:8000/health`.

### 3. Запуск веб-интерфейса

Доступен вариант через `http://localhost:8000/web`

Либо запуск Streamlit в отдельном терминале:

```bash
source .venv/bin/activate
export PYTHONPATH=.
export API_BASE_URL=http://localhost:8000
uv run streamlit run src/client/app/streamlit_app.py
```

Веб-интерфейс будет доступен на `http://localhost:8501`.

### 4. Telegram-слой локально

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
uv run pytest tests/test_bot_service.py tests/test_bot_handlers_common.py -q
```

Отдельный polling/webhook runner Telegram-бота и отдельный `bot`-сервис в `docker-compose.yaml` в текущем срезе репозитория отсутствуют.

## Запуск через Docker Compose

`docker-compose.yaml` поднимает:

- `server` на `http://localhost:8000`
- `client` на `http://localhost:8501`

Перед запуском создайте `.env` рядом с `docker-compose.yaml`, например:

```env
OLLAMA_HOST=http://host.docker.internal:11434
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
- tg_bot пока не выделен в отдельный контейнер

## Тесты и проверка

```bash
uv run pytest -q
uv run ruff check .
uv run black --check .
uv run isort --check-only .
```

Полный локальный прогон:

```bash
./uv-linters.sh
```

## Документация

- ТЗ: [`docs/technical-specification-for-IAfEI/ТЗ-общее/ТЗ-общее.pdf`](docs/technical-specification-for-IAfEI/ТЗ-общее/ТЗ-общее.pdf)

- Референс по структуре ТЗ: [`docs/technical-specification-for-IAfEI/README.md`](docs/technical-specification-for-IAfEI/README.md)

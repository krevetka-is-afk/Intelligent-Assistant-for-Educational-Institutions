# Intelligent-Assistant-for-Educational-Institutions

Практико-ориентированный проект по созданию интеллектуального чат-бота для университета на основе RAG-архитектуры. Система будет использовать внутренние данные вуза (учебные планы, нормативные документы) для ответов на вопросы студентов и преподавателей.

![CI](https://github.com/krevetka-is-afk/Intelligent-Assistant-for-Educational-Institutions/actions/workflows/ci.yml/badge.svg)

## Быстрый старт

### Локальный запуск (рекомендуемый способ — через `uv` и `pyproject.toml`)

import docs if needed

```bash
git submodule update --init --recursive  # или клонируйте с флагом --recurse-submodules
```

```bash
uv venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
uv sync --group dev
export PYTHONPATH=.
```

start up server

```bash
uv run uvicorn src.server.app.main:app --reload
```

start up client

```bash
uv run streamlit run src/client/app/streamlit_app.py
```

### Запуск через `pip`

`pyproject.toml` источник зависимостей При желании можно установить проект напрямую:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install .
export PYTHONPATH=.
```

start up server

```bash
uvicorn src.server.app.main:app --reload
```

client

```bash
streamlit run src/client/app/streamlit_app.py
```

## Docker

Для сборки и запуска в Docker используется тот же `pyproject.toml`, зависимости устанавливаются через `uv`.

В контейнере путь к базе векторного индекса настраивается переменной окружения `VECTOR_DB_DIR` и по умолчанию настроен в `docker-compose.yaml` на `/app/chrome_langchain_db`, примонтированный как volume.

```bash
docker compose --profile dev up --build
```

## Contribute with uv

```bash
uv run pre-commit install
```

before PR

```bash
uv run ruff check --fix .
uv run black .
uv run isort .
uv run pytest -q
uv run pre-commit run --all-files
```

For a better contributing experience here is a ready script.

```bash
./uv-linters.sh
```

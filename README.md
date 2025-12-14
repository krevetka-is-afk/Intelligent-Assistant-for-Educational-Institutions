# Intelligent-Assistant-for-Educational-Institutions

Практико-ориентированный проект по созданию интеллектуального чат-бота для университета на основе RAG-архитектуры. Система будет использовать внутренние данные вуза (учебные планы, нормативные документы) для ответов на вопросы студентов и преподавателей.

![CI](https://github.com/krevetka-is-afk/Intelligent-Assistant-for-Educational-Institutions/actions/workflows/ci.yml/badge.svg)

## Быстрый старт

```bash
git submodule update --init --recursive  # или клонируйте с флагом --recurse-submodules
# python3 -m venv .venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
```

```bash
uvicorn app.main:app --reload
```

## Docker

```bash
docker compose --profile dev up --build
```

## Contribute

```bash
pre-commit install
```

before PR

```bash
ruff check --fix .
black .
isort .
pytest -q
pre-commit run --all-files
```

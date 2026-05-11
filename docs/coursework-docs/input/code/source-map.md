# Source map for generated documentation

Use these files as the main source map when writing PZ, PMI and RO.

## Product and architecture

- `README.md`
- `pyproject.toml`
- `app_runtime.py`

## Server — RAG core (Растворов)

- `src/server/app/main.py`
- `src/server/app/rag.py`
- `src/server/app/vector.py`
- `src/server/app/document_ingestion.py`
- `src/server/app/index_documents.py`
- `src/server/app/lexical.py`
- `src/server/app/conversation_memory.py`
- `src/server/app/metrics.py`
- `src/server/app/config.py`
- `src/server/app/auth_models.py`
- `src/server/app/auth_database.py`
- `src/server/app/auth_crud.py`

## Bot (Субботин)

- `src/bot/bot.py`
- `src/bot/api_client.py`
- `src/bot/service.py`
- `src/bot/core/config.py`
- `src/bot/core/models.py`
- `src/bot/core/crud.py`
- `src/bot/core/database.py`
- `src/bot/handlers/all_handlers.py`
- `src/bot/handlers/common.py`

## Web client (Субботин)

- `src/client/app/streamlit_app.py`

## Deployment and operations

- `docker-compose.yaml`
- `deployment/production/compose.yaml`
- `deployment/production/render-env.sh`

## Tests

- `tests/test_ask.py`
- `tests/test_rag.py`
- `tests/test_document_ingestion.py`
- `tests/test_health.py`
- `tests/test_server_config.py`
- `tests/test_bot_all_handles.py`
- `tests/test_bot_crud.py`
- `tests/test_bot_handlers_common.py`
- `tests/test_bot_service.py`
- `tests/test_bot_script_imports.py`
- `tests/conftest.py`

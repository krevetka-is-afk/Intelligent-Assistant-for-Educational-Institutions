# Release and quality context

## Canonical commands

```bash
uv sync --all-packages --group dev
./uv-linters.sh
pytest tests/
```

## Test coverage

- `tests/test_ask.py` — интеграционные тесты эндпоинта POST /ask
- `tests/test_rag.py` — юнит-тесты RAG-пайплайна (retrieval + generation)
- `tests/test_document_ingestion.py` — тесты загрузки и чанкинга документов
- `tests/test_health.py` — smoke-тест GET /health
- `tests/test_server_config.py` — тесты конфигурации сервера
- `tests/test_bot_*.py` — тесты Telegram-бота (хендлеры, CRUD, сервис, импорты)

## Main acceptance evidence

Для финальной сдачи необходимо предоставить:

- результат `pytest tests/` без падений;
- результат `./uv-linters.sh` (ruff check, ruff format --check);
- Docker Compose config validation (`docker compose config`);
- smoke-эндпоинты: `/health`, `/ask` с тестовым вопросом;
- ручные сценарии через Telegram-бот и Streamlit-клиент.

## Deployment

```bash
docker compose up -d          # локальный стенд
# production:
# deployment/production/compose.yaml
```

## Known caveats

- Для работы RAG-пайплайна требуется доступный экземпляр Ollama.
  `ALLOW_LLM_FALLBACK=1` переключает на lexical fallback — только для разработки,
  не является приёмочным свидетельством.
- Пересборка векторного индекса (POST /index) — фоновая операция, не прерывает сервис.
- ChromaDB хранит персистентный индекс на диске; при смене корпуса требуется
  повторная индексация.

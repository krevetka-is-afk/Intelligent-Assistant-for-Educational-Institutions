# Intelligent Assistant for Educational Institutions — code context

## Runtime

Основной runtime состоит из:

- `src/server` — FastAPI RAG-сервер (ядро системы);
- `src/bot` — Telegram-бот на aiogram 3.x;
- `src/client` — Streamlit веб-клиент;
- `deployment` — Docker Compose для production/staging;
- `tests` — pytest-тесты сервера, бота и RAG-пайплайна.

## Server modules

| Модуль | Ответственность |
| --- | --- |
| `src/server/app/main.py` | FastAPI app: маршруты /ask, /health, /index, /metrics |
| `src/server/app/rag.py` | RAG-пайплайн: retrieval (ChromaDB) + generation (LLM) |
| `src/server/app/vector.py` | ChromaDB client, эмбеддинги ruBERT-tiny2, поиск по cosine |
| `src/server/app/document_ingestion.py` | Загрузка PDF/HTML/TXT, чанкинг, OCR-изображений |
| `src/server/app/index_documents.py` | Фоновая индексация корпуса в ChromaDB |
| `src/server/app/lexical.py` | Лексический резервный поиск (fallback при недоступности LLM) |
| `src/server/app/conversation_memory.py` | История диалогов пользователя, хранение в PostgreSQL |
| `src/server/app/metrics.py` | Сбор метрик: время ответа, confidence, доля эскалаций |
| `src/server/app/auth_models.py` | Модели пользователей и ролей |
| `src/server/app/auth_database.py` | Подключение к PostgreSQL (asyncpg/SQLAlchemy) |
| `src/server/app/auth_crud.py` | CRUD операции: регистрация, токены, роли |
| `src/server/app/config.py` | Конфигурация через .env (BOT_TOKEN, DATABASE_URL, LLM_HOST) |

## Bot modules

| Модуль | Ответственность |
| --- | --- |
| `src/bot/bot.py` | Точка входа Telegram-бота, инициализация aiogram |
| `src/bot/api_client.py` | HTTP-клиент к FastAPI /ask |
| `src/bot/service.py` | Бизнес-логика бота: вопрос → API → ответ с источниками |
| `src/bot/core/models.py` | Модели данных бота (пользователи, запросы) |
| `src/bot/core/crud.py` | CRUD для истории запросов бота |
| `src/bot/core/database.py` | Подключение к БД из контекста бота |
| `src/bot/core/config.py` | Конфигурация бота |
| `src/bot/handlers/all_handlers.py` | Все хендлеры команд и сообщений |
| `src/bot/handlers/common.py` | Общие хендлеры (/start, /help, /admin) |

## Client

| Модуль | Ответственность |
| --- | --- |
| `src/client/app/streamlit_app.py` | Streamlit веб-интерфейс: диалог, история, источники |

## Recent important changes

- Добавлен lexical fallback при недоступности Ollama.
- Расширена схема метрик: confidence score, время ответа, флаги эскалации.
- Добавлена история диалогов в PostgreSQL (conversation_memory).
- Аутентификация по токену с ролевой моделью (пользователь/администратор).
- Поддержка OCR для изображений в document_ingestion.

## Documentation stance

Документы должны трактовать актуальное ТЗ (docs/technical-specification-for-IAfEI/),
код и тесты как взаимно проверяемые источники. Задокументированный scope:
RAG-пайплайн (ChromaDB + Ollama), FastAPI /ask API, Telegram-бот, Streamlit-клиент,
Docker Compose stack. Не следует заявлять как приёмочное требование функции,
не подтверждённые кодом и тестами.

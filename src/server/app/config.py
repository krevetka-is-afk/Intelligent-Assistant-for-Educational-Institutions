from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from app_runtime import getenv

SERVER_DIR = Path(__file__).resolve().parents[1]
DOCKER_VECTOR_DB_DIR = Path("/data")
DOCKER_DOCUMENTS_DIR = Path("/data_and_documents")
DOCKER_WEB_AUTH_DB_PATH = Path("/data/web_auth.db")


def _resolve_default_documents_dir() -> Path:
    project_root = os.getenv("PROJECT_ROOT")
    candidates = [
        Path(project_root).resolve() / "data_and_documents" if project_root else None,
        Path.cwd().resolve() / "data_and_documents",
        SERVER_DIR.parent / "data_and_documents",
        Path("/data_and_documents"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return (Path.cwd().resolve() / "data_and_documents").resolve()


def _is_running_in_container() -> bool:
    return Path("/.dockerenv").exists()


def _get_bool_env(name: str, default: bool) -> bool:
    raw = getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_vector_db_dir() -> Path:
    configured = getenv("VECTOR_DB_DIR")
    if configured is None:
        return DEFAULT_VECTOR_DB_DIR.resolve()

    candidate = Path(configured).expanduser()
    if candidate == DOCKER_VECTOR_DB_DIR and not _is_running_in_container():
        return DEFAULT_VECTOR_DB_DIR.resolve()
    return candidate.resolve()


def _resolve_documents_dir() -> Path:
    configured = getenv("DOCUMENTS_DIR")
    if configured is None:
        return DEFAULT_DOCUMENTS_DIR.resolve()

    candidate = Path(configured).expanduser()
    if candidate == DOCKER_DOCUMENTS_DIR and not _is_running_in_container():
        return DEFAULT_DOCUMENTS_DIR.resolve()
    return candidate.resolve()


def resolve_sqlite_path_from_url(database_url: str) -> Path | None:
    prefixes = ("sqlite+aiosqlite:///", "sqlite:///")
    for prefix in prefixes:
        if not database_url.startswith(prefix):
            continue

        raw_path = unquote(database_url[len(prefix) :])
        if raw_path in {"", ":memory:"}:
            return None

        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()

        base_dir = DOCKER_VECTOR_DB_DIR if _is_running_in_container() else Path.cwd().resolve()
        return (base_dir / candidate).resolve()
    return None


def _resolve_default_web_auth_db_url() -> str:
    if _is_running_in_container():
        return f"sqlite+aiosqlite:///{DOCKER_WEB_AUTH_DB_PATH.as_posix()}"

    default_path = (SERVER_DIR.parent.parent / ".web_auth.db").resolve()
    return f"sqlite+aiosqlite:///{default_path}"


def _resolve_web_auth_database_url() -> str:
    configured = getenv("WEB_AUTH_DATABASE_URL")
    if configured is None:
        return _resolve_default_web_auth_db_url()

    if _is_running_in_container():
        prefixes = ("sqlite+aiosqlite:///", "sqlite:///")
        for prefix in prefixes:
            if not configured.startswith(prefix):
                continue

            raw_path = unquote(configured[len(prefix) :])
            if raw_path in {"", ":memory:"}:
                return configured

            posix_path = PurePosixPath(raw_path)
            if not posix_path.is_absolute():
                posix_path = PurePosixPath(DOCKER_WEB_AUTH_DB_PATH.parent.as_posix()) / posix_path
            # create_async_engine требует sqlite+aiosqlite, а не sqlite:///
            path_str = posix_path.as_posix()
            return f"sqlite+aiosqlite:///{path_str}"

    resolved_path = resolve_sqlite_path_from_url(configured)
    if resolved_path is None:
        return configured
    return f"sqlite+aiosqlite:///{resolved_path}"


DEFAULT_VECTOR_DB_DIR = SERVER_DIR / "chrome_langchain_db"
DEFAULT_DOCUMENTS_DIR = _resolve_default_documents_dir()

API_KEY = getenv("API_KEY")
WEB_BOOTSTRAP_ADMIN_TOKEN = getenv("WEB_BOOTSTRAP_ADMIN_TOKEN")
WEB_AUTH_DATABASE_URL = _resolve_web_auth_database_url()
APP_ENV = getenv("APP_ENV", "development") or "development"
LOG_LEVEL = getenv("LOG_LEVEL", "INFO") or "INFO"
OLLAMA_HOST = (getenv("OLLAMA_HOST", "http://localhost:11434") or "http://localhost:11434").rstrip(
    "/"
)
VECTOR_DB_DIR = _resolve_vector_db_dir()
DOCUMENTS_DIR = _resolve_documents_dir()
CHROMA_COLLECTION_NAME = getenv("CHROMA_COLLECTION_NAME", "edu_documents") or "edu_documents"
HF_EMBEDDING_MODEL = (
    getenv("HF_EMBEDDING_MODEL", "cointegrated/rubert-tiny2") or "cointegrated/rubert-tiny2"
)
LLM_MODEL = getenv("LLM_MODEL", "mistral:7b") or "mistral:7b"
RAG_TOP_K = int(getenv("RAG_TOP_K", "4") or "4")
RAG_TOTAL_TIMEOUT_SECONDS = float(getenv("RAG_TOTAL_TIMEOUT_SECONDS", "420") or "420")
LLM_TIMEOUT_SECONDS = float(getenv("LLM_TIMEOUT_SECONDS", "360") or "360")
CONVERSATION_MEMORY_WINDOW = int(getenv("CONVERSATION_MEMORY_WINDOW", "5") or "5")
CONVERSATION_MEMORY_TTL_SECONDS = float(getenv("CONVERSATION_MEMORY_TTL_SECONDS", "3600") or "3600")
CONVERSATION_MEMORY_MAX_SESSIONS = int(
    getenv("CONVERSATION_MEMORY_MAX_SESSIONS", "10000") or "10000"
)
CHUNK_SIZE = int(getenv("RAG_CHUNK_SIZE", "500") or "500")
CHUNK_OVERLAP = int(getenv("RAG_CHUNK_OVERLAP", "100") or "100")
PREPARE_RAG_ON_STARTUP = _get_bool_env("PREPARE_RAG_ON_STARTUP", True)
AUTO_INDEX_ON_STARTUP = _get_bool_env("AUTO_INDEX_ON_STARTUP", True)
SHOW_SOURCES = _get_bool_env("SHOW_SOURCES", True)

LLM_PROMPT_TEMPLATE = """
Ты отвечаешь на вопросы студентов и сотрудников по документам учебного процесса.

Правила:
- Используй только факты из переданного контекста.
- Если данных недостаточно, прямо скажи об этом.
- Не выдумывай отсутствующие даты, правила или ссылки.
- Дай краткий ответ на русском языке.

История последних сообщений пользователя:
{conversation_history}

Контекст:
{information}

Вопрос:
{question}
"""


def validate_chunk_settings() -> None:
    if CHUNK_SIZE <= 0:
        raise ValueError("RAG_CHUNK_SIZE must be positive")
    if CHUNK_OVERLAP < 0:
        raise ValueError("RAG_CHUNK_OVERLAP must be non-negative")
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")


def validate_runtime_config() -> None:
    if API_KEY is None:
        raise RuntimeError("API_KEY is not set")
    if CONVERSATION_MEMORY_WINDOW <= 0:
        raise RuntimeError("CONVERSATION_MEMORY_WINDOW must be positive")
    if CONVERSATION_MEMORY_TTL_SECONDS <= 0:
        raise RuntimeError("CONVERSATION_MEMORY_TTL_SECONDS must be positive")
    if CONVERSATION_MEMORY_MAX_SESSIONS <= 0:
        raise RuntimeError("CONVERSATION_MEMORY_MAX_SESSIONS must be positive")
    validate_chunk_settings()

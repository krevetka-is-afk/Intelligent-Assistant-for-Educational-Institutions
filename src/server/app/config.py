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


def _get_nonempty_env(name: str) -> str | None:
    value = getenv(name)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _get_choice_env(name: str, default: str, allowed: set[str]) -> str:
    raw = _get_nonempty_env(name)
    normalized = (raw or default).strip().casefold()
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {choices}")
    return normalized


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

API_KEY = _get_nonempty_env("API_KEY")
TELEGRAM_SERVICE_KEY = _get_nonempty_env("TELEGRAM_SERVICE_KEY")
WEB_BOOTSTRAP_ADMIN_TOKEN = _get_nonempty_env("WEB_BOOTSTRAP_ADMIN_TOKEN")
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
HF_EMBEDDING_NORMALIZE = _get_bool_env("HF_EMBEDDING_NORMALIZE", False)
LLM_MODEL = getenv("LLM_MODEL", "qwen2.5:3b") or "qwen2.5:3b"
RAG_QUERY_REWRITE_MODEL = getenv("RAG_QUERY_REWRITE_MODEL", LLM_MODEL) or LLM_MODEL
RAG_TOP_K = int(getenv("RAG_TOP_K", "4") or "4")
RAG_MAX_CONTEXT_DOCUMENTS = int(
    getenv("RAG_MAX_CONTEXT_DOCUMENTS", str(RAG_TOP_K)) or str(RAG_TOP_K)
)
RAG_MAX_DOCUMENT_CHARS = int(getenv("RAG_MAX_DOCUMENT_CHARS", "1200") or "1200")
RAG_MAX_TOTAL_CONTEXT_CHARS = int(getenv("RAG_MAX_TOTAL_CONTEXT_CHARS", "3600") or "3600")
RAG_MAX_HISTORY_MESSAGES = int(getenv("RAG_MAX_HISTORY_MESSAGES", "5") or "5")
RAG_MAX_HISTORY_CHARS = int(getenv("RAG_MAX_HISTORY_CHARS", "1600") or "1600")
RAG_QUERY_REWRITE_ENABLED = _get_bool_env("RAG_QUERY_REWRITE_ENABLED", False)
RAG_CONTEXT_EXPANSION_ENABLED = _get_bool_env("RAG_CONTEXT_EXPANSION_ENABLED", True)
RAG_QUERY_REWRITE_TIMEOUT_SECONDS = float(getenv("RAG_QUERY_REWRITE_TIMEOUT_SECONDS", "8") or "8")
RAG_QUERY_REWRITE_MAX_HISTORY_MESSAGES = int(
    getenv("RAG_QUERY_REWRITE_MAX_HISTORY_MESSAGES", "3") or "3"
)
RAG_QUERY_REWRITE_MAX_HISTORY_CHARS = int(
    getenv("RAG_QUERY_REWRITE_MAX_HISTORY_CHARS", "600") or "600"
)
RAG_RETRIEVAL_MODE = _get_choice_env("RAG_RETRIEVAL_MODE", "hybrid", {"hybrid", "primary_dense"})
RAG_SOURCE_SNIPPET_CHARS = int(getenv("RAG_SOURCE_SNIPPET_CHARS", "320") or "320")
RAG_TOTAL_TIMEOUT_SECONDS = float(getenv("RAG_TOTAL_TIMEOUT_SECONDS", "420") or "420")
_lexical_index_override = _get_nonempty_env("LEXICAL_INDEX_PATH")
LEXICAL_INDEX_PATH = (
    Path(_lexical_index_override).expanduser().resolve()
    if _lexical_index_override is not None
    else (VECTOR_DB_DIR / "lexical_index.sqlite3").resolve()
)
RAG_CANDIDATE_POOL_SIZE = int(
    getenv("RAG_CANDIDATE_POOL_SIZE", str(max(RAG_TOP_K * 4, 10))) or str(max(RAG_TOP_K * 4, 10))
)
RAG_RRF_K = int(getenv("RAG_RRF_K", "60") or "60")
RAG_MAX_CHUNKS_PER_DOCUMENT = int(getenv("RAG_MAX_CHUNKS_PER_DOCUMENT", "2") or "2")
LLM_TIMEOUT_SECONDS = float(getenv("LLM_TIMEOUT_SECONDS", "360") or "360")
CONVERSATION_MEMORY_WINDOW = int(getenv("CONVERSATION_MEMORY_WINDOW", "5") or "5")
CONVERSATION_MEMORY_TTL_SECONDS = float(getenv("CONVERSATION_MEMORY_TTL_SECONDS", "3600") or "3600")
CONVERSATION_MEMORY_MAX_SESSIONS = int(
    getenv("CONVERSATION_MEMORY_MAX_SESSIONS", "10000") or "10000"
)
CHUNK_SIZE = int(getenv("RAG_CHUNK_SIZE", "500") or "500")
CHUNK_OVERLAP = int(getenv("RAG_CHUNK_OVERLAP", "100") or "100")
PREPARE_RAG_ON_STARTUP = _get_bool_env("PREPARE_RAG_ON_STARTUP", True)
AUTO_INDEX_ON_STARTUP = _get_bool_env("AUTO_INDEX_ON_STARTUP", APP_ENV != "production")
SHOW_SOURCES = _get_bool_env("SHOW_SOURCES", True)
DOCUMENT_OCR_ENABLED = _get_bool_env("DOCUMENT_OCR_ENABLED", False)
DOCUMENT_OCR_LANG = getenv("DOCUMENT_OCR_LANG", "rus+eng") or "rus+eng"
DOCUMENT_OCR_MAX_PAGES = int(getenv("DOCUMENT_OCR_MAX_PAGES", "5") or "5")
DOCUMENT_OCR_TIMEOUT_SECONDS = float(getenv("DOCUMENT_OCR_TIMEOUT_SECONDS", "30") or "30")


def validate_chunk_settings() -> None:
    if CHUNK_SIZE <= 0:
        raise ValueError("RAG_CHUNK_SIZE must be positive")
    if CHUNK_OVERLAP < 0:
        raise ValueError("RAG_CHUNK_OVERLAP must be non-negative")
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")


def validate_rag_policy_settings() -> None:
    positive_settings = {
        "RAG_MAX_CONTEXT_DOCUMENTS": RAG_MAX_CONTEXT_DOCUMENTS,
        "RAG_MAX_DOCUMENT_CHARS": RAG_MAX_DOCUMENT_CHARS,
        "RAG_MAX_TOTAL_CONTEXT_CHARS": RAG_MAX_TOTAL_CONTEXT_CHARS,
        "RAG_MAX_HISTORY_MESSAGES": RAG_MAX_HISTORY_MESSAGES,
        "RAG_MAX_HISTORY_CHARS": RAG_MAX_HISTORY_CHARS,
        "RAG_SOURCE_SNIPPET_CHARS": RAG_SOURCE_SNIPPET_CHARS,
        "RAG_CANDIDATE_POOL_SIZE": RAG_CANDIDATE_POOL_SIZE,
        "RAG_RRF_K": RAG_RRF_K,
        "RAG_MAX_CHUNKS_PER_DOCUMENT": RAG_MAX_CHUNKS_PER_DOCUMENT,
        "DOCUMENT_OCR_MAX_PAGES": DOCUMENT_OCR_MAX_PAGES,
    }
    for name, value in positive_settings.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if RAG_CANDIDATE_POOL_SIZE < RAG_TOP_K:
        raise ValueError("RAG_CANDIDATE_POOL_SIZE must be greater than or equal to RAG_TOP_K")
    if DOCUMENT_OCR_TIMEOUT_SECONDS <= 0:
        raise ValueError("DOCUMENT_OCR_TIMEOUT_SECONDS must be positive")


def validate_runtime_config() -> None:
    if API_KEY is None:
        raise RuntimeError("API_KEY is not set")
    if TELEGRAM_SERVICE_KEY is None:
        raise RuntimeError("TELEGRAM_SERVICE_KEY is not set")
    if CONVERSATION_MEMORY_WINDOW <= 0:
        raise RuntimeError("CONVERSATION_MEMORY_WINDOW must be positive")
    if CONVERSATION_MEMORY_TTL_SECONDS <= 0:
        raise RuntimeError("CONVERSATION_MEMORY_TTL_SECONDS must be positive")
    if CONVERSATION_MEMORY_MAX_SESSIONS <= 0:
        raise RuntimeError("CONVERSATION_MEMORY_MAX_SESSIONS must be positive")
    validate_chunk_settings()
    validate_rag_policy_settings()

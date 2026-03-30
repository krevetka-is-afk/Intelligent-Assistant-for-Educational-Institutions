from __future__ import annotations

import os
from pathlib import Path

from app_runtime import getenv

SERVER_DIR = Path(__file__).resolve().parents[1]
DOCKER_VECTOR_DB_DIR = Path("/data")
DOCKER_DOCUMENTS_DIR = Path("/data_and_documents")


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


DEFAULT_VECTOR_DB_DIR = SERVER_DIR / "chrome_langchain_db"
DEFAULT_DOCUMENTS_DIR = _resolve_default_documents_dir()

API_KEY = getenv("API_KEY")
WEB_UI_PASSWORD = getenv("WEB_UI_PASSWORD")
WEB_UI_PASSWORD = getenv("WEB_UI_PASSWORD")
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
RAG_TOTAL_TIMEOUT_SECONDS = float(getenv("RAG_TOTAL_TIMEOUT_SECONDS", "20") or "20")
LLM_TIMEOUT_SECONDS = float(getenv("LLM_TIMEOUT_SECONDS", "18") or "18")
CHUNK_SIZE = int(getenv("RAG_CHUNK_SIZE", "500") or "500")
CHUNK_OVERLAP = int(getenv("RAG_CHUNK_OVERLAP", "100") or "100")

LLM_PROMPT_TEMPLATE = """
Ты отвечаешь на вопросы студентов и сотрудников по документам учебного процесса.

Правила:
- Используй только факты из переданного контекста.
- Если данных недостаточно, прямо скажи об этом.
- Не выдумывай отсутствующие даты, правила или ссылки.
- Дай краткий ответ на русском языке.

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
    validate_chunk_settings()

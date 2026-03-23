from __future__ import annotations

import os
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]


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


DEFAULT_VECTOR_DB_DIR = SERVER_DIR / "chrome_langchain_db"
DEFAULT_DOCUMENTS_DIR = _resolve_default_documents_dir()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", str(DEFAULT_VECTOR_DB_DIR))).resolve()
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", str(DEFAULT_DOCUMENTS_DIR))).resolve()
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "edu_documents")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "cointegrated/rubert-tiny2")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral:7b")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_TOTAL_TIMEOUT_SECONDS = float(os.getenv("RAG_TOTAL_TIMEOUT_SECONDS", "20"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "18"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))

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

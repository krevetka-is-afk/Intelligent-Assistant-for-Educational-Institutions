from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document

from . import config
from .vector import RetrievedDocument

logger = logging.getLogger("server.lexical")

_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_]{2,}", re.UNICODE)


@dataclass(slots=True)
class _LexicalRow:
    chunk_id: str
    document_id: str
    source: str | None
    title: str | None
    page: int | None
    chunk_index: int | None
    char_start: int | None
    char_end: int | None
    indexed_at: str | None
    page_content: str


def _connect() -> sqlite3.Connection:
    db_path = Path(config.RAG_LEXICAL_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def _init_schema(connection: sqlite3.Connection) -> None:
    connection.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts
        USING fts5(
            chunk_id UNINDEXED,
            document_id UNINDEXED,
            source UNINDEXED,
            title UNINDEXED,
            page UNINDEXED,
            chunk_index UNINDEXED,
            char_start UNINDEXED,
            char_end UNINDEXED,
            indexed_at UNINDEXED,
            page_content,
            tokenize='unicode61 remove_diacritics 2'
        );
        """)


def initialize_lexical_index(*, rebuild: bool = False) -> None:
    with _connect() as connection:
        _init_schema(connection)
        if rebuild:
            connection.execute("DELETE FROM chunk_fts;")
        connection.commit()


def lexical_chunk_count() -> int:
    with _connect() as connection:
        _init_schema(connection)
        row = connection.execute("SELECT COUNT(*) AS count FROM chunk_fts;").fetchone()
        return int(row["count"]) if row is not None else 0


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_chunk_rows(chunks: Iterable[Any], *, fallback_document_id: str) -> list[_LexicalRow]:
    rows: list[_LexicalRow] = []
    for chunk in chunks:
        metadata = getattr(chunk, "metadata", {}) or {}
        chunk_id = str(metadata.get("chunk_id") or getattr(chunk, "id", "")).strip()
        page_content = str(getattr(chunk, "page_content", "")).strip()
        if not chunk_id or not page_content:
            continue

        document_id = str(metadata.get("document_id") or fallback_document_id).strip()
        if not document_id:
            continue

        rows.append(
            _LexicalRow(
                chunk_id=chunk_id,
                document_id=document_id,
                source=str(metadata.get("source") or "").strip() or None,
                title=str(metadata.get("title") or "").strip() or None,
                page=_coerce_int(metadata.get("page")),
                chunk_index=_coerce_int(metadata.get("chunk_index")),
                char_start=_coerce_int(metadata.get("char_start")),
                char_end=_coerce_int(metadata.get("char_end")),
                indexed_at=str(metadata.get("indexed_at") or "").strip() or None,
                page_content=page_content,
            )
        )
    return rows


def _insert_rows(connection: sqlite3.Connection, rows: list[_LexicalRow]) -> None:
    connection.executemany(
        """
        INSERT INTO chunk_fts (
            chunk_id,
            document_id,
            source,
            title,
            page,
            chunk_index,
            char_start,
            char_end,
            indexed_at,
            page_content
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        [
            (
                row.chunk_id,
                row.document_id,
                row.source,
                row.title,
                row.page,
                row.chunk_index,
                row.char_start,
                row.char_end,
                row.indexed_at,
                row.page_content,
            )
            for row in rows
        ],
    )


def upsert_document_chunks(document_id: str, chunks: list[Any]) -> None:
    rows = _normalize_chunk_rows(chunks, fallback_document_id=document_id)
    with _connect() as connection:
        _init_schema(connection)
        connection.execute("DELETE FROM chunk_fts WHERE document_id = ?;", (document_id,))
        if rows:
            _insert_rows(connection, rows)
        connection.commit()


def delete_document_chunks(document_id: str) -> None:
    with _connect() as connection:
        _init_schema(connection)
        connection.execute("DELETE FROM chunk_fts WHERE document_id = ?;", (document_id,))
        connection.commit()


def _build_match_query(raw_query: str) -> str | None:
    unique_tokens: list[str] = []
    seen_tokens: set[str] = set()
    for token in _TOKEN_RE.findall(raw_query.lower()):
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        unique_tokens.append(token)
        if len(unique_tokens) >= 24:
            break

    if not unique_tokens:
        return None

    escaped = ['"{}"'.format(token.replace('"', '""')) for token in unique_tokens]
    return " OR ".join(escaped)


def _rank_to_distance(rank: int, total: int) -> float:
    if total <= 1:
        return 0.25
    normalized_rank = (rank - 1) / (total - 1)
    return max(0.0, min(1.0, normalized_rank))


def lexical_search(question: str, *, k: int | None = None) -> list[RetrievedDocument]:
    top_k = k or config.RAG_HYBRID_LEXICAL_TOP_K
    match_query = _build_match_query(question)
    if top_k <= 0 or match_query is None:
        return []

    with _connect() as connection:
        _init_schema(connection)
        try:
            rows = connection.execute(
                """
                SELECT
                    chunk_id,
                    document_id,
                    source,
                    title,
                    page,
                    chunk_index,
                    char_start,
                    char_end,
                    indexed_at,
                    page_content,
                    bm25(chunk_fts) AS score
                FROM chunk_fts
                WHERE chunk_fts MATCH ?
                ORDER BY score ASC
                LIMIT ?;
                """,
                (match_query, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            logger.exception("FTS5 lexical search failed")
            return []

    retrieved: list[RetrievedDocument] = []
    total_rows = len(rows)
    for rank, row in enumerate(rows, start=1):
        metadata = {
            "document_id": row["document_id"],
            "chunk_id": row["chunk_id"],
            "source": row["source"],
            "title": row["title"],
            "page": _coerce_int(row["page"]),
            "chunk_index": _coerce_int(row["chunk_index"]),
            "char_start": _coerce_int(row["char_start"]),
            "char_end": _coerce_int(row["char_end"]),
            "indexed_at": row["indexed_at"],
        }
        document = Document(
            id=row["chunk_id"],
            page_content=str(row["page_content"] or ""),
            metadata={key: value for key, value in metadata.items() if value is not None},
        )
        retrieved.append(
            RetrievedDocument(
                document=document,
                distance=_rank_to_distance(rank, total_rows),
            )
        )

    return retrieved


def ensure_lexical_index_populated() -> int:
    initialize_lexical_index(rebuild=False)
    existing_count = lexical_chunk_count()
    if existing_count > 0:
        return existing_count

    try:
        from .vector import get_vector_store  # Local import to avoid module cycle.

        collection = get_vector_store()._collection
        payload = collection.get(include=["documents", "metadatas"])
    except Exception:
        logger.exception("Could not backfill lexical index from vector store")
        return 0

    ids = payload.get("ids", [])
    documents = payload.get("documents", [])
    metadatas = payload.get("metadatas", [])
    rows: list[_LexicalRow] = []
    for chunk_id, page_content, metadata in zip(ids, documents, metadatas):
        if not isinstance(metadata, dict):
            metadata = {}
        normalized = _LexicalRow(
            chunk_id=str(metadata.get("chunk_id") or chunk_id),
            document_id=str(metadata.get("document_id") or ""),
            source=str(metadata.get("source") or "").strip() or None,
            title=str(metadata.get("title") or "").strip() or None,
            page=_coerce_int(metadata.get("page")),
            chunk_index=_coerce_int(metadata.get("chunk_index")),
            char_start=_coerce_int(metadata.get("char_start")),
            char_end=_coerce_int(metadata.get("char_end")),
            indexed_at=str(metadata.get("indexed_at") or "").strip() or None,
            page_content=str(page_content or "").strip(),
        )
        if not normalized.chunk_id or not normalized.document_id or not normalized.page_content:
            continue
        rows.append(normalized)

    if not rows:
        return 0

    with _connect() as connection:
        _init_schema(connection)
        connection.execute("DELETE FROM chunk_fts;")
        _insert_rows(connection, rows)
        connection.commit()
    return len(rows)

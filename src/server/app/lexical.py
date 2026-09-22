from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from json import dumps, loads
from pathlib import Path
from typing import Iterable, Protocol

from langchain_core.documents import Document

from . import config

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class LexicalIndexUnavailableError(RuntimeError):
    """Raised when the lexical index cannot serve requests."""


class _ChunkLike(Protocol):
    id: str
    page_content: str
    metadata: dict[str, object]


@dataclass(slots=True)
class LexicalSearchResult:
    document: Document
    score: float


def resolve_lexical_index_path(
    persist_directory: Path | None = None,
    lexical_index_path: Path | None = None,
) -> Path:
    if lexical_index_path is not None:
        return lexical_index_path.expanduser().resolve()
    if persist_directory is not None:
        if persist_directory.expanduser().resolve() == Path(config.VECTOR_DB_DIR).resolve():
            return Path(config.LEXICAL_INDEX_PATH).expanduser().resolve()
        return (persist_directory / "lexical_index.sqlite3").resolve()
    return Path(config.LEXICAL_INDEX_PATH).expanduser().resolve()


def _connect(index_path: Path) -> sqlite3.Connection:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(index_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_lexical_index(index_path: Path, *, rebuild: bool = False) -> None:
    if rebuild and index_path.exists():
        index_path.unlink()

    with _connect(index_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS lexical_documents (
                document_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT NOT NULL
            )
            """)
        connection.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS lexical_chunks
            USING fts5(
                chunk_id UNINDEXED,
                document_id UNINDEXED,
                title,
                source,
                text,
                metadata UNINDEXED
            )
            """)


def reset_lexical_index(index_path: Path) -> None:
    initialize_lexical_index(index_path, rebuild=True)


def _metadata_value(metadata: dict[str, object], name: str) -> str:
    value = metadata.get(name)
    return value if isinstance(value, str) else ""


def replace_document_chunks(
    index_path: Path,
    document_id: str,
    chunks: Iterable[_ChunkLike],
) -> None:
    chunk_rows = list(chunks)
    if not chunk_rows:
        delete_document_chunks(index_path, document_id)
        return

    first_metadata = chunk_rows[0].metadata
    title = _metadata_value(first_metadata, "title") or document_id
    source = _metadata_value(first_metadata, "source")

    initialize_lexical_index(index_path)
    with _connect(index_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            connection.execute(
                "DELETE FROM lexical_chunks WHERE document_id = ?",
                (document_id,),
            )
            connection.execute(
                "DELETE FROM lexical_documents WHERE document_id = ?",
                (document_id,),
            )
            connection.execute(
                """
                INSERT INTO lexical_documents (document_id, title, source)
                VALUES (?, ?, ?)
                """,
                (document_id, title, source),
            )
            connection.executemany(
                """
                INSERT INTO lexical_chunks (chunk_id, document_id, title, source, text, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.id,
                        document_id,
                        _metadata_value(chunk.metadata, "title") or title,
                        _metadata_value(chunk.metadata, "source") or source,
                        chunk.page_content,
                        dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                    )
                    for chunk in chunk_rows
                ],
            )


def delete_document_chunks(index_path: Path, document_id: str) -> None:
    initialize_lexical_index(index_path)
    with _connect(index_path) as connection:
        with connection:
            connection.execute("DELETE FROM lexical_chunks WHERE document_id = ?", (document_id,))
            connection.execute(
                "DELETE FROM lexical_documents WHERE document_id = ?",
                (document_id,),
            )


def get_indexed_document_ids(index_path: Path) -> set[str]:
    initialize_lexical_index(index_path)
    with _connect(index_path) as connection:
        rows = connection.execute("SELECT document_id FROM lexical_documents").fetchall()
    return {str(row["document_id"]) for row in rows}


def delete_stale_documents(index_path: Path, *, active_document_ids: set[str]) -> list[str]:
    stale_document_ids = sorted(get_indexed_document_ids(index_path) - active_document_ids)
    for document_id in stale_document_ids:
        delete_document_chunks(index_path, document_id)
    return stale_document_ids


def _build_fts_query(query: str) -> str | None:
    tokens = _TOKEN_RE.findall(query.casefold())
    if not tokens:
        return None
    unique_tokens = list(dict.fromkeys(tokens))
    terms: list[str] = []
    for token in unique_tokens[:32]:
        terms.append(token)
        if len(token) > 4:
            terms.append(f"{token[:-1]}*")
    return " OR ".join(terms)


def search_lexical(
    query: str,
    *,
    limit: int,
    index_path: Path | None = None,
) -> list[LexicalSearchResult]:
    resolved_index_path = resolve_lexical_index_path(lexical_index_path=index_path)
    fts_query = _build_fts_query(query)
    if fts_query is None:
        return []
    if not resolved_index_path.exists():
        raise LexicalIndexUnavailableError("Lexical index does not exist")

    try:
        initialize_lexical_index(resolved_index_path)
        with _connect(resolved_index_path) as connection:
            rows = connection.execute(
                """
                SELECT
                    chunk_id,
                    document_id,
                    title,
                    source,
                    text,
                    metadata,
                    bm25(lexical_chunks) AS rank_score
                FROM lexical_chunks
                WHERE lexical_chunks MATCH ?
                ORDER BY rank_score ASC, chunk_id ASC
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
    except sqlite3.Error as exc:
        raise LexicalIndexUnavailableError("Lexical search failed") from exc

    results: list[LexicalSearchResult] = []
    for row in rows:
        score = float(row["rank_score"])
        try:
            metadata = loads(row["metadata"])
        except (TypeError, ValueError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.update(
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "title": row["title"],
                "source": row["source"],
                "retrieval_channel": "lexical",
            }
        )
        results.append(
            LexicalSearchResult(
                document=Document(
                    id=row["chunk_id"],
                    page_content=row["text"],
                    metadata=metadata,
                ),
                score=score,
            )
        )
    return results


def search(
    query: str,
    *,
    index_path: Path | None = None,
    k: int | None = None,
) -> list[LexicalSearchResult]:
    return search_lexical(query, limit=k or config.RAG_CANDIDATE_POOL_SIZE, index_path=index_path)

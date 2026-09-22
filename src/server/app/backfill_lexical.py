from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass
from json import dumps
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable
from uuid import uuid4

import chromadb

from app_runtime import log_extra, setup_logging

from . import config, lexical

logger = logging.getLogger("server.lexical_backfill")


@dataclass(frozen=True, slots=True)
class LexicalBackfillSummary:
    chroma_chunks: int
    lexical_chunks: int
    lexical_documents: int
    index_path: Path


@dataclass(frozen=True, slots=True)
class _ChromaChunk:
    chunk_id: str
    document_id: str
    title: str
    source: str
    text: str
    metadata: dict[str, Any]


def _metadata_value(metadata: dict[str, Any], name: str) -> str:
    value = metadata.get(name)
    return value if isinstance(value, str) else ""


def _document_id_for(chunk_id: str, metadata: dict[str, Any]) -> str:
    document_id = _metadata_value(metadata, "document_id")
    if document_id:
        return document_id
    return chunk_id.rsplit(":", maxsplit=1)[0] if ":" in chunk_id else chunk_id


def _normalize_chunk(
    *,
    chunk_id: str,
    text: str | None,
    metadata: dict[str, Any] | None,
) -> _ChromaChunk:
    normalized_metadata = dict(metadata or {})
    normalized_metadata.setdefault("chunk_id", chunk_id)
    document_id = _document_id_for(chunk_id, normalized_metadata)
    normalized_metadata.setdefault("document_id", document_id)
    return _ChromaChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title=_metadata_value(normalized_metadata, "title") or document_id,
        source=_metadata_value(normalized_metadata, "source"),
        text=text or "",
        metadata=normalized_metadata,
    )


def _iter_chroma_chunks(
    collection: Any,
    *,
    batch_size: int,
) -> Iterable[_ChromaChunk]:
    offset = 0
    while True:
        batch = collection.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        ids = [str(chunk_id) for chunk_id in batch.get("ids", [])]
        if not ids:
            break

        documents = batch.get("documents") or []
        metadatas = batch.get("metadatas") or []
        for index, chunk_id in enumerate(ids):
            text = documents[index] if index < len(documents) else None
            metadata = metadatas[index] if index < len(metadatas) else None
            yield _normalize_chunk(
                chunk_id=chunk_id,
                text=text if isinstance(text, str) else None,
                metadata=metadata if isinstance(metadata, dict) else None,
            )

        offset += len(ids)


def _create_staging_path(index_path: Path) -> Path:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        prefix=f".{index_path.name}.",
        suffix=f".{uuid4().hex}.tmp",
        dir=index_path.parent,
        delete=False,
    ) as handle:
        return Path(handle.name)


def _count_rows(connection: sqlite3.Connection, table_name: str) -> int:
    row = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    return int(row[0])


def _write_lexical_index(
    *,
    collection: Any,
    index_path: Path,
    batch_size: int,
) -> LexicalBackfillSummary:
    staging_path = _create_staging_path(index_path)
    chroma_count = int(collection.count())

    try:
        lexical.initialize_lexical_index(staging_path, rebuild=True)
        with sqlite3.connect(staging_path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                document_titles: dict[str, tuple[str, str]] = {}
                rows: list[tuple[str, str, str, str, str, str]] = []
                for chunk in _iter_chroma_chunks(collection, batch_size=batch_size):
                    document_titles.setdefault(chunk.document_id, (chunk.title, chunk.source))
                    rows.append(
                        (
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.title,
                            chunk.source,
                            chunk.text,
                            dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                        )
                    )
                    if len(rows) >= batch_size:
                        connection.executemany(
                            """
                            INSERT INTO lexical_chunks
                                (chunk_id, document_id, title, source, text, metadata)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            rows,
                        )
                        rows.clear()

                if rows:
                    connection.executemany(
                        """
                        INSERT INTO lexical_chunks
                            (chunk_id, document_id, title, source, text, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        rows,
                    )

                connection.executemany(
                    """
                    INSERT INTO lexical_documents (document_id, title, source)
                    VALUES (?, ?, ?)
                    """,
                    [
                        (document_id, title, source)
                        for document_id, (title, source) in sorted(document_titles.items())
                    ],
                )

            lexical_count = _count_rows(connection, "lexical_chunks")
            document_count = _count_rows(connection, "lexical_documents")

        if lexical_count != chroma_count:
            raise RuntimeError(
                "Lexical backfill count parity failed: "
                f"chroma={chroma_count} lexical={lexical_count}"
            )

        os.replace(staging_path, index_path)
        return LexicalBackfillSummary(
            chroma_chunks=chroma_count,
            lexical_chunks=lexical_count,
            lexical_documents=document_count,
            index_path=index_path,
        )
    except Exception:
        try:
            staging_path.unlink(missing_ok=True)
        finally:
            raise


def backfill_lexical_index(
    *,
    persist_directory: Path,
    collection_name: str,
    lexical_index_path: Path | None = None,
    batch_size: int = 1000,
) -> LexicalBackfillSummary:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    resolved_persist_directory = persist_directory.expanduser().resolve()
    resolved_index_path = lexical.resolve_lexical_index_path(
        persist_directory=resolved_persist_directory,
        lexical_index_path=lexical_index_path,
    )
    client = chromadb.PersistentClient(path=str(resolved_persist_directory))
    collection = client.get_collection(name=collection_name)
    return _write_lexical_index(
        collection=collection,
        index_path=resolved_index_path,
        batch_size=batch_size,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Restore the standalone SQLite FTS5 lexical index from an existing Chroma "
            "collection."
        )
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=config.VECTOR_DB_DIR,
        help="Directory where Chroma persists the collection.",
    )
    parser.add_argument(
        "--collection-name",
        default=config.CHROMA_COLLECTION_NAME,
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--lexical-index-path",
        type=Path,
        default=None,
        help="SQLite FTS5 index path. Defaults to <persist-dir>/lexical_index.sqlite3.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of Chroma rows to read and SQLite rows to write per batch.",
    )
    return parser


def main() -> int:
    setup_logging("server")
    parser = build_parser()
    args = parser.parse_args()

    try:
        summary = backfill_lexical_index(
            persist_directory=args.persist_dir,
            collection_name=args.collection_name,
            lexical_index_path=args.lexical_index_path,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        logger.exception(
            "Fatal lexical backfill error: %s",
            exc,
            extra=log_extra(stage="lexical_backfill", error_type=type(exc).__name__),
        )
        return 1

    logger.info(
        (
            "Lexical backfill finished: chroma_chunks=%s lexical_chunks=%s "
            "lexical_documents=%s index_path=%s"
        ),
        summary.chroma_chunks,
        summary.lexical_chunks,
        summary.lexical_documents,
        summary.index_path,
        extra=log_extra(stage="lexical_backfill"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

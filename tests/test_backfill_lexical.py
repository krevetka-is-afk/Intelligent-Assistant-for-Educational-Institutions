from __future__ import annotations

import sqlite3

import chromadb
import pytest

from src.server.app.backfill_lexical import backfill_lexical_index
from src.server.app.lexical import replace_document_chunks, search_lexical


def _create_collection(persist_dir, *, name: str = "edu_documents"):
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(name=name)


def _seed_collection(collection) -> None:
    collection.add(
        ids=["doc-a:00000", "doc-a:00001", "doc-b:00000"],
        documents=[
            "Порядок пересдачи экзамена опубликован в учебном регламенте.",
            "Апелляция подается через личный кабинет студента.",
            "Стипендия назначается приказом после сессии.",
        ],
        embeddings=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        metadatas=[
            {
                "chunk_id": "doc-a:00000",
                "document_id": "doc-a",
                "chunk_index": 0,
                "title": "Регламент пересдач",
                "source": "rules/retake.txt",
            },
            {
                "chunk_id": "doc-a:00001",
                "document_id": "doc-a",
                "chunk_index": 1,
                "title": "Регламент пересдач",
                "source": "rules/retake.txt",
            },
            {
                "chunk_id": "doc-b:00000",
                "document_id": "doc-b",
                "chunk_index": 0,
                "title": "Стипендии",
                "source": "finance/stipend.txt",
            },
        ],
    )


def test_backfill_lexical_index_restores_searchable_index_from_chroma(tmp_path):
    persist_dir = tmp_path / "chroma"
    index_path = tmp_path / "lexical.sqlite3"
    collection = _create_collection(persist_dir)
    _seed_collection(collection)

    summary = backfill_lexical_index(
        persist_directory=persist_dir,
        collection_name="edu_documents",
        lexical_index_path=index_path,
        batch_size=2,
    )

    assert summary.chroma_chunks == 3
    assert summary.lexical_chunks == 3
    assert summary.lexical_documents == 2
    assert summary.index_path == index_path.resolve()
    assert collection.count() == 3

    results = search_lexical("пересдача регламент", limit=5, index_path=index_path)
    assert results[0].document.id == "doc-a:00000"
    assert {result.document.id for result in results} == {"doc-a:00000", "doc-a:00001"}
    assert results[0].document.metadata["document_id"] == "doc-a"
    assert results[0].document.metadata["source"] == "rules/retake.txt"


def test_backfill_lexical_index_is_idempotent_and_replaces_stale_index(tmp_path):
    persist_dir = tmp_path / "chroma"
    index_path = tmp_path / "lexical.sqlite3"
    collection = _create_collection(persist_dir)
    _seed_collection(collection)
    replace_document_chunks(
        index_path,
        "old-doc",
        [
            type(
                "Chunk",
                (),
                {
                    "id": "old-doc:00000",
                    "page_content": "Старый индекс не должен остаться.",
                    "metadata": {
                        "chunk_id": "old-doc:00000",
                        "document_id": "old-doc",
                        "title": "Old",
                        "source": "old.txt",
                    },
                },
            )()
        ],
    )

    first = backfill_lexical_index(
        persist_directory=persist_dir,
        collection_name="edu_documents",
        lexical_index_path=index_path,
        batch_size=1,
    )
    second = backfill_lexical_index(
        persist_directory=persist_dir,
        collection_name="edu_documents",
        lexical_index_path=index_path,
        batch_size=1,
    )

    assert first == second
    assert search_lexical("старый", limit=5, index_path=index_path) == []
    assert search_lexical("стипендия", limit=5, index_path=index_path)

    with sqlite3.connect(index_path) as connection:
        lexical_chunks = connection.execute("SELECT COUNT(*) FROM lexical_chunks").fetchone()[0]
    assert lexical_chunks == collection.count()


def test_backfill_lexical_index_preserves_existing_index_when_parity_fails(
    tmp_path,
    monkeypatch,
):
    persist_dir = tmp_path / "chroma"
    index_path = tmp_path / "lexical.sqlite3"
    collection = _create_collection(persist_dir)
    _seed_collection(collection)
    replace_document_chunks(
        index_path,
        "safe-doc",
        [
            type(
                "Chunk",
                (),
                {
                    "id": "safe-doc:00000",
                    "page_content": "Существующий индекс остается на месте.",
                    "metadata": {
                        "chunk_id": "safe-doc:00000",
                        "document_id": "safe-doc",
                        "title": "Safe",
                        "source": "safe.txt",
                    },
                },
            )()
        ],
    )

    import src.server.app.backfill_lexical as backfill_module

    original_iter = backfill_module._iter_chroma_chunks

    def _drop_last_chunk(*args, **kwargs):
        chunks = list(original_iter(*args, **kwargs))
        yield from chunks[:-1]

    monkeypatch.setattr(backfill_module, "_iter_chroma_chunks", _drop_last_chunk)

    with pytest.raises(RuntimeError, match="count parity failed"):
        backfill_lexical_index(
            persist_directory=persist_dir,
            collection_name="edu_documents",
            lexical_index_path=index_path,
            batch_size=2,
        )

    assert search_lexical("остается", limit=5, index_path=index_path)
    assert search_lexical("пересдача", limit=5, index_path=index_path) == []
    assert not list(index_path.parent.glob(f".{index_path.name}.*.tmp"))

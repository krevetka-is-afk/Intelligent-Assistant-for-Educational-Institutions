from __future__ import annotations

import sqlite3

from langchain_core.documents import Document

from src.server.app import lexical, rag, vector


def _retrieved(chunk_id: str, *, distance: float = 0.1, text: str | None = None):
    return vector.RetrievedDocument(
        document=Document(
            page_content=text or chunk_id,
            metadata={"chunk_id": chunk_id, "source": f"{chunk_id}.txt"},
        ),
        distance=distance,
    )


def test_reciprocal_rank_fusion_prioritizes_overlap_between_rankings():
    dense = [
        _retrieved("dense-top"),
        _retrieved("shared"),
        _retrieved("dense-only"),
    ]
    lexical = [
        _retrieved("shared"),
        _retrieved("lexical-only"),
        _retrieved("dense-only"),
    ]

    fused = rag.reciprocal_rank_fusion(
        dense,
        lexical,
        top_k=4,
        rrf_k=60,
    )

    assert [item.document.metadata["chunk_id"] for item in fused] == [
        "shared",
        "dense-only",
        "dense-top",
        "lexical-only",
    ]


def test_reciprocal_rank_fusion_respects_rank_window_size():
    dense = [
        _retrieved("dense-top"),
        _retrieved("shared"),
        _retrieved("dense-only"),
    ]
    lexical = [
        _retrieved("shared"),
        _retrieved("lexical-only"),
        _retrieved("dense-only"),
    ]

    fused = rag.reciprocal_rank_fusion(
        dense,
        lexical,
        top_k=2,
        rrf_k=60,
    )

    assert [item.document.metadata["chunk_id"] for item in fused] == [
        "shared",
        "dense-only",
    ]


def test_lexical_search_returns_ranked_documents(tmp_path, monkeypatch):
    lexical_db = tmp_path / "rag_lexical.db"
    monkeypatch.setattr(lexical.config, "RAG_LEXICAL_DB_PATH", lexical_db)

    lexical.initialize_lexical_index(rebuild=True)
    chunks = [
        Document(
            id="doc-1:00000",
            page_content="Академический отпуск оформляется приказом деканата.",
            metadata={
                "document_id": "doc-1",
                "chunk_id": "doc-1:00000",
                "source": "rules.txt",
                "chunk_index": 0,
            },
        ),
        Document(
            id="doc-1:00001",
            page_content="График пересдач публикуется на сайте факультета.",
            metadata={
                "document_id": "doc-1",
                "chunk_id": "doc-1:00001",
                "source": "rules.txt",
                "chunk_index": 1,
            },
        ),
    ]
    lexical.upsert_document_chunks("doc-1", chunks)

    retrieved = lexical.lexical_search("академический отпуск", k=5)

    assert retrieved
    assert retrieved[0].document.metadata["chunk_id"] == "doc-1:00000"
    assert retrieved[0].document.metadata["source"] == "rules.txt"


def test_lexical_search_returns_empty_on_sqlite_operational_error(monkeypatch):
    class _Cursor:
        def fetchall(self):
            return []

    class _BrokenConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *args, **kwargs):
            statement = str(args[0]) if args else ""
            del kwargs
            if "SELECT" in statement.upper():
                raise sqlite3.OperationalError("fts is unavailable")
            return _Cursor()

    monkeypatch.setattr(lexical, "_connect", lambda: _BrokenConnection())

    assert lexical.lexical_search("пересдача математики", k=5) == []

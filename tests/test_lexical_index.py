from __future__ import annotations

import pytest

from src.server.app import config
from src.server.app.document_ingestion import ChunkRecord
from src.server.app.lexical import (
    LexicalIndexUnavailableError,
    delete_document_chunks,
    delete_stale_documents,
    get_indexed_document_ids,
    replace_document_chunks,
    reset_lexical_index,
    resolve_lexical_index_path,
    search_lexical,
)


def _chunk(
    chunk_id: str,
    document_id: str,
    text: str,
    *,
    title: str = "Учебный документ",
    source: str = "docs/source.txt",
) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        page_content=text,
        metadata={
            "chunk_id": chunk_id,
            "document_id": document_id,
            "chunk_index": int(chunk_id.rsplit(":", maxsplit=1)[-1]),
            "title": title,
            "source": source,
        },
    )


def test_search_lexical_returns_matching_chunks_with_metadata(tmp_path):
    index_path = tmp_path / "lexical.sqlite3"
    chunks = [
        _chunk(
            "retake:00000",
            "retake",
            "Первая пересдача проводится для студентов после экзаменационной сессии.",
            title="Положение о пересдачах",
            source="student_handbook/basic/retake.txt",
        ),
        _chunk(
            "discipline:00000",
            "discipline",
            "Дисциплинарное взыскание применяется за нарушение правил.",
            title="Дисциплина",
            source="discipline/rules.txt",
        ),
    ]

    replace_document_chunks(index_path, "retake", [chunks[0]])
    replace_document_chunks(index_path, "discipline", [chunks[1]])

    results = search_lexical("когда пересдача", limit=5, index_path=index_path)

    assert [result.document.metadata["document_id"] for result in results] == ["retake"]
    assert results[0].document.id == "retake:00000"
    assert results[0].document.metadata["source"] == "student_handbook/basic/retake.txt"
    assert results[0].document.metadata["retrieval_channel"] == "lexical"


def test_search_lexical_indexes_source_but_not_metadata_json(tmp_path):
    index_path = tmp_path / "lexical.sqlite3"
    replace_document_chunks(
        index_path,
        "retake",
        [
            _chunk(
                "retake:00000",
                "retake",
                "Обычный учебный текст",
                source="student_handbook/basic/retake-calendar.txt",
            )
        ],
    )

    assert search_lexical("retake-calendar", limit=5, index_path=index_path)
    assert search_lexical("00000", limit=5, index_path=index_path) == []


def test_search_lexical_distinguishes_missing_index_from_empty_query(tmp_path):
    missing_index_path = tmp_path / "missing.sqlite3"

    assert search_lexical("", limit=5, index_path=missing_index_path) == []
    with pytest.raises(LexicalIndexUnavailableError, match="does not exist"):
        search_lexical("пересдача", limit=5, index_path=missing_index_path)


def test_search_lexical_orders_equal_scores_by_chunk_id(tmp_path):
    index_path = tmp_path / "lexical.sqlite3"
    replace_document_chunks(
        index_path,
        "doc",
        [
            _chunk("doc:00002", "doc", "пересдача"),
            _chunk("doc:00001", "doc", "пересдача"),
        ],
    )

    results = search_lexical("пересдача", limit=5, index_path=index_path)

    assert [result.document.id for result in results] == ["doc:00001", "doc:00002"]


def test_replace_document_chunks_replaces_previous_rows(tmp_path):
    index_path = tmp_path / "lexical.sqlite3"
    replace_document_chunks(
        index_path,
        "doc",
        [_chunk("doc:00000", "doc", "Старый текст про расписание")],
    )

    replace_document_chunks(
        index_path,
        "doc",
        [_chunk("doc:00000", "doc", "Новый текст про пересдачу")],
    )

    assert search_lexical("старый", limit=5, index_path=index_path) == []
    results = search_lexical("пересдача", limit=5, index_path=index_path)
    assert len(results) == 1
    assert results[0].document.page_content == "Новый текст про пересдачу"


def test_delete_document_chunks_and_stale_cleanup(tmp_path):
    index_path = tmp_path / "lexical.sqlite3"
    replace_document_chunks(index_path, "active", [_chunk("active:00000", "active", "Активный")])
    replace_document_chunks(index_path, "stale", [_chunk("stale:00000", "stale", "Устаревший")])

    removed = delete_stale_documents(index_path, active_document_ids={"active"})

    assert removed == ["stale"]
    assert get_indexed_document_ids(index_path) == {"active"}

    delete_document_chunks(index_path, "active")
    assert get_indexed_document_ids(index_path) == set()


def test_reset_lexical_index_clears_existing_rows(tmp_path):
    index_path = tmp_path / "lexical.sqlite3"
    replace_document_chunks(index_path, "doc", [_chunk("doc:00000", "doc", "Текст")])

    reset_lexical_index(index_path)

    assert get_indexed_document_ids(index_path) == set()


def test_hybrid_retrieval_settings_are_validated(monkeypatch):
    monkeypatch.setattr(config, "RAG_TOP_K", 4)
    monkeypatch.setattr(config, "RAG_CANDIDATE_POOL_SIZE", 3)

    with pytest.raises(ValueError, match="RAG_CANDIDATE_POOL_SIZE"):
        config.validate_rag_policy_settings()


def test_resolve_lexical_index_path_honors_config_for_default_vector_dir(
    tmp_path,
    monkeypatch,
):
    vector_dir = tmp_path / "vector"
    configured_lexical = tmp_path / "configured" / "lexical.sqlite3"
    monkeypatch.setattr(config, "VECTOR_DB_DIR", vector_dir)
    monkeypatch.setattr(config, "LEXICAL_INDEX_PATH", configured_lexical)

    assert resolve_lexical_index_path(persist_directory=vector_dir) == configured_lexical.resolve()
    assert (
        resolve_lexical_index_path(persist_directory=tmp_path / "custom")
        == (tmp_path / "custom" / "lexical_index.sqlite3").resolve()
    )

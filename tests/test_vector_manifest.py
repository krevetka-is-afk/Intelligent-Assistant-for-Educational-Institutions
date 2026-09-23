from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.server.app import vector
from src.server.app.vector import VectorStoreUnavailableError


@pytest.fixture(autouse=True)
def clear_vector_cache():
    vector.clear_vector_cache()
    yield
    vector.clear_vector_cache()


@pytest.fixture
def configured_vector(monkeypatch, tmp_path):
    calls: list[dict[str, object]] = []

    class _FakeEmbeddings:
        def __init__(self, **kwargs):
            calls.append({"embeddings": kwargs})

    class _FakeChroma:
        def __init__(self, **kwargs):
            calls.append({"chroma": kwargs})

    monkeypatch.setattr(vector.config, "VECTOR_DB_DIR", tmp_path)
    monkeypatch.setattr(vector.config, "HF_EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setattr(vector.config, "HF_EMBEDDING_NORMALIZE", True)
    monkeypatch.setattr(vector.config, "CHROMA_COLLECTION_NAME", "edu_documents")
    monkeypatch.setattr(vector.config, "RAG_RETRIEVAL_MODE", "hybrid")
    monkeypatch.setattr(vector.config, "validate_chunk_settings", lambda: None)
    monkeypatch.setattr(vector, "HuggingFaceEmbeddings", _FakeEmbeddings)
    monkeypatch.setattr(vector, "Chroma", _FakeChroma)
    return tmp_path, calls


def _write_manifest(vector_db_dir: Path, **overrides: object) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "embedding_model": "BAAI/bge-m3",
        "normalize_embeddings": True,
        "collection_name": "edu_documents",
    }
    payload.update(overrides)
    (vector_db_dir / "reembed_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def test_get_vector_store_allows_absent_reembed_manifest(configured_vector):
    vector_db_dir, calls = configured_vector

    store = vector.get_vector_store()

    assert store is not None
    assert calls[-1]["chroma"]["persist_directory"] == str(vector_db_dir)


def test_get_vector_store_requires_reembed_manifest_for_bge_primary_dense(configured_vector):
    vector_db_dir, calls = configured_vector
    vector.config.RAG_RETRIEVAL_MODE = "primary_dense"

    with pytest.raises(VectorStoreUnavailableError) as exc_info:
        vector.get_vector_store()

    message = str(exc_info.value)
    assert "incompatible" in message
    assert str(vector_db_dir) not in message
    assert calls == []


def test_get_vector_store_allows_matching_reembed_manifest(configured_vector):
    vector_db_dir, calls = configured_vector
    _write_manifest(vector_db_dir, target="/copied/index/path")

    store = vector.get_vector_store()

    assert store is not None
    assert calls[-1]["chroma"]["collection_name"] == "edu_documents"


def test_get_vector_store_allows_matching_reembed_manifest_for_bge_primary_dense(
    configured_vector,
):
    vector_db_dir, calls = configured_vector
    vector.config.RAG_RETRIEVAL_MODE = "primary_dense"
    _write_manifest(vector_db_dir, target="/copied/index/path")

    store = vector.get_vector_store()

    assert store is not None
    assert calls[-1]["chroma"]["collection_name"] == "edu_documents"


@pytest.mark.parametrize(
    "overrides",
    [
        {"embedding_model": "cointegrated/rubert-tiny2"},
        {"normalize_embeddings": False},
        {"collection_name": "other_collection"},
    ],
)
def test_get_vector_store_rejects_mismatched_reembed_manifest(configured_vector, overrides):
    vector_db_dir, calls = configured_vector
    _write_manifest(vector_db_dir, **overrides)

    with pytest.raises(VectorStoreUnavailableError) as exc_info:
        vector.get_vector_store()

    message = str(exc_info.value)
    assert "incompatible" in message
    assert str(vector_db_dir) not in message
    assert calls == []


@pytest.mark.parametrize("raw_manifest", ["{bad json", "[]", "{}"])
def test_get_vector_store_rejects_malformed_reembed_manifest(configured_vector, raw_manifest):
    vector_db_dir, calls = configured_vector
    (vector_db_dir / "reembed_manifest.json").write_text(raw_manifest, encoding="utf-8")

    with pytest.raises(VectorStoreUnavailableError) as exc_info:
        vector.get_vector_store()

    message = str(exc_info.value)
    assert "incompatible" in message
    assert str(vector_db_dir) not in message
    assert calls == []

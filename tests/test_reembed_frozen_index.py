from __future__ import annotations

import json

import chromadb
import pytest

from src.server.app.reembed_frozen_index import reembed_frozen_index


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(index), float(len(text))] for index, text in enumerate(texts, start=1)]


def _collection(path, name: str):
    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection(name=name)


def test_reembed_frozen_index_preserves_ids_documents_and_metadata(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir(parents=True, exist_ok=True)
    (source / "lexical_index.sqlite3").write_bytes(b"frozen lexical db")
    source_collection = _collection(source, "edu_documents")
    source_collection.add(
        ids=["chunk-1", "chunk-2"],
        documents=["Первый фрагмент", "Второй фрагмент"],
        metadatas=[
            {"chunk_id": "chunk-1", "source": "a.txt", "page": 1},
            {"chunk_id": "chunk-2", "source": "b.txt", "page": 2},
        ],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
    )
    embedder = _FakeEmbedder()

    result = reembed_frozen_index(
        source=source,
        target=target,
        model="BAAI/bge-m3",
        collection_name="edu_documents",
        batch_size=1,
        normalize_embeddings=True,
        expected_count=2,
        embedding_function=embedder,
    )

    assert result.source_count == 2
    assert result.target_count == 2
    assert result.normalized is True
    assert result.lexical_source_path == source / "lexical_index.sqlite3"
    assert result.lexical_target_path == target / "lexical_index.sqlite3"
    assert (target / "lexical_index.sqlite3").read_bytes() == b"frozen lexical db"
    assert embedder.calls == [["Первый фрагмент"], ["Второй фрагмент"]]

    target_collection = _collection(target, "edu_documents")
    copied = target_collection.get(include=["documents", "metadatas", "embeddings"])
    assert copied["ids"] == ["chunk-1", "chunk-2"]
    assert copied["documents"] == ["Первый фрагмент", "Второй фрагмент"]
    assert copied["metadatas"] == [
        {"chunk_id": "chunk-1", "source": "a.txt", "page": 1},
        {"chunk_id": "chunk-2", "source": "b.txt", "page": 2},
    ]
    assert copied["embeddings"].tolist() == [[1.0, 15.0], [1.0, 15.0]]

    manifest = json.loads((target / "reembed_manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest.items()
        >= {
            "schema_version": 1,
            "collection_name": "edu_documents",
            "embedding_model": "BAAI/bge-m3",
            "normalize_embeddings": True,
            "source_count": 2,
            "target_count": 2,
            "lexical_index_source": str(source / "lexical_index.sqlite3"),
            "lexical_index_target": str(target / "lexical_index.sqlite3"),
            "lexical_index_size_bytes": len(b"frozen lexical db"),
            "lexical_index_copied": True,
            "ids_documents_metadatas_preserved": True,
            "source_files_rechunked": False,
        }.items()
    )


def test_reembed_frozen_index_rejects_non_empty_target(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _collection(source, "edu_documents").add(
        ids=["chunk-1"],
        documents=["Source"],
        metadatas=[{"chunk_id": "chunk-1"}],
        embeddings=[[0.1, 0.2]],
    )
    _collection(target, "edu_documents").add(
        ids=["existing"],
        documents=["Existing"],
        metadatas=[{"chunk_id": "existing"}],
        embeddings=[[0.0, 0.0]],
    )

    with pytest.raises(RuntimeError, match="not empty"):
        reembed_frozen_index(
            source=source,
            target=target,
            model="BAAI/bge-m3",
            collection_name="edu_documents",
            batch_size=64,
            embedding_function=_FakeEmbedder(),
        )


def test_vector_embedding_function_honors_normalization_flag(monkeypatch):
    from src.server.app import vector

    observed: dict[str, object] = {}

    class _FakeHuggingFaceEmbeddings:
        def __init__(self, **kwargs):
            observed.update(kwargs)

    monkeypatch.setattr(vector.config, "HF_EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setattr(vector.config, "HF_EMBEDDING_NORMALIZE", True)
    monkeypatch.setattr(vector, "HuggingFaceEmbeddings", _FakeHuggingFaceEmbeddings)
    vector.clear_vector_cache()

    try:
        vector.get_embedding_function()
    finally:
        vector.clear_vector_cache()

    assert observed == {
        "model": "BAAI/bge-m3",
        "encode_kwargs": {"normalize_embeddings": True},
        "query_encode_kwargs": {"normalize_embeddings": True},
    }


def test_reembed_frozen_index_rejects_target_nested_inside_source(tmp_path):
    source = tmp_path / "source"
    target = source / "nested-target"

    with pytest.raises(RuntimeError, match="must not overlap"):
        reembed_frozen_index(
            source=source,
            target=target,
            model="BAAI/bge-m3",
            collection_name="edu_documents",
            batch_size=64,
            embedding_function=_FakeEmbedder(),
        )


def test_reembed_frozen_index_rejects_target_parent_of_source_before_rebuild(tmp_path):
    target = tmp_path / "target"
    source = target / "source"
    source.mkdir(parents=True)
    sentinel = source / "do-not-delete.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(RuntimeError, match="must not overlap"):
        reembed_frozen_index(
            source=source,
            target=target,
            model="BAAI/bge-m3",
            collection_name="edu_documents",
            batch_size=64,
            rebuild_target=True,
            embedding_function=_FakeEmbedder(),
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_reembed_frozen_index_fails_when_batch_documents_are_missing(tmp_path, monkeypatch):
    from src.server.app import reembed_frozen_index as module

    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir(parents=True, exist_ok=True)
    (source / "lexical_index.sqlite3").write_bytes(b"frozen lexical db")

    class _SourceCollection:
        metadata = {"hnsw:space": "cosine"}

        def count(self) -> int:
            return 1

        def get(self, **kwargs):
            return {"ids": ["chunk-1"], "documents": None, "metadatas": [{"chunk_id": "chunk-1"}]}

    class _TargetCollection:
        def count(self) -> int:
            return 0

        def add(self, **kwargs):
            raise AssertionError("target add must not run with incomplete source batch")

    monkeypatch.setattr(
        module,
        "_get_or_fail_source_collection",
        lambda source, collection_name: _SourceCollection(),
    )

    def _prepare_target_collection(*args, **kwargs):
        return _TargetCollection()

    monkeypatch.setattr(module, "_prepare_target_collection", _prepare_target_collection)

    with pytest.raises(RuntimeError, match="did not include documents"):
        reembed_frozen_index(
            source=source,
            target=target,
            model="BAAI/bge-m3",
            collection_name="edu_documents",
            batch_size=64,
            embedding_function=_FakeEmbedder(),
        )


def test_reembed_frozen_index_fails_when_batch_metadata_length_mismatches(tmp_path, monkeypatch):
    from src.server.app import reembed_frozen_index as module

    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir(parents=True, exist_ok=True)
    (source / "lexical_index.sqlite3").write_bytes(b"frozen lexical db")

    class _SourceCollection:
        metadata = None

        def count(self) -> int:
            return 2

        def get(self, **kwargs):
            return {
                "ids": ["chunk-1", "chunk-2"],
                "documents": ["one", "two"],
                "metadatas": [{"chunk_id": "chunk-1"}],
            }

    class _TargetCollection:
        def count(self) -> int:
            return 0

        def add(self, **kwargs):
            raise AssertionError("target add must not run with mismatched metadata")

    monkeypatch.setattr(
        module,
        "_get_or_fail_source_collection",
        lambda source, collection_name: _SourceCollection(),
    )

    def _prepare_target_collection(*args, **kwargs):
        return _TargetCollection()

    monkeypatch.setattr(module, "_prepare_target_collection", _prepare_target_collection)

    with pytest.raises(RuntimeError, match="metadatas length mismatch"):
        reembed_frozen_index(
            source=source,
            target=target,
            model="BAAI/bge-m3",
            collection_name="edu_documents",
            batch_size=64,
            embedding_function=_FakeEmbedder(),
        )

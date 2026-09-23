from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from . import config


class VectorStoreUnavailableError(RuntimeError):
    """Raised when the vector store cannot serve requests."""


class EmptyVectorStoreError(VectorStoreUnavailableError):
    """Raised when the vector store exists but contains no indexed documents."""


@dataclass(slots=True)
class RetrievedDocument:
    document: Document
    distance: float
    _retrieval_diagnostics: dict[str, Any] = field(default_factory=dict, repr=False)


_embedding_function: HuggingFaceEmbeddings | None = None
_vector_store: Chroma | None = None
_read_collection_cache: tuple[str, str, Any] | None = None

_REEMBED_MANIFEST_NAME = "reembed_manifest.json"
_MANIFEST_ERROR_MESSAGE = "Vector index manifest is incompatible with current vector configuration"
_BGE_M3_MODEL = "BAAI/bge-m3"


def clear_vector_cache() -> None:
    global _embedding_function, _read_collection_cache, _vector_store
    _embedding_function = None
    _read_collection_cache = None
    _vector_store = None


def get_embedding_function() -> HuggingFaceEmbeddings:
    global _embedding_function
    if _embedding_function is None:
        embedding_kwargs = {"normalize_embeddings": config.HF_EMBEDDING_NORMALIZE}
        _embedding_function = HuggingFaceEmbeddings(
            model=config.HF_EMBEDDING_MODEL,
            encode_kwargs=embedding_kwargs,
            query_encode_kwargs=embedding_kwargs,
        )
    return _embedding_function


def _validate_reembed_manifest(vector_db_dir: Path) -> None:
    manifest_path = vector_db_dir / _REEMBED_MANIFEST_NAME
    if not manifest_path.exists():
        if (
            config.HF_EMBEDDING_MODEL == _BGE_M3_MODEL
            and config.RAG_RETRIEVAL_MODE == "primary_dense"
        ):
            raise VectorStoreUnavailableError(_MANIFEST_ERROR_MESSAGE)
        return

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise VectorStoreUnavailableError(_MANIFEST_ERROR_MESSAGE) from None

    expected = {
        "embedding_model": config.HF_EMBEDDING_MODEL,
        "normalize_embeddings": config.HF_EMBEDDING_NORMALIZE,
        "collection_name": config.CHROMA_COLLECTION_NAME,
    }
    if not isinstance(payload, dict):
        raise VectorStoreUnavailableError(_MANIFEST_ERROR_MESSAGE)

    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise VectorStoreUnavailableError(_MANIFEST_ERROR_MESSAGE)


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        config.validate_chunk_settings()
        vector_db_dir = Path(config.VECTOR_DB_DIR)
        _validate_reembed_manifest(vector_db_dir)
        _vector_store = Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            persist_directory=str(vector_db_dir),
            embedding_function=get_embedding_function(),
        )
    return _vector_store


def get_vector_store_document_count(vector_store: Chroma | None = None) -> int:
    resolved_store = vector_store or get_vector_store()
    try:
        count = resolved_store._collection.count()
    except Exception as exc:
        raise VectorStoreUnavailableError("Could not access vector store collection") from exc
    return int(count)


def _get_read_collection() -> Any:
    global _read_collection_cache
    vector_db_dir = str(config.VECTOR_DB_DIR)
    collection_name = config.CHROMA_COLLECTION_NAME
    if (
        _read_collection_cache is not None
        and _read_collection_cache[0] == vector_db_dir
        and _read_collection_cache[1] == collection_name
    ):
        return _read_collection_cache[2]
    try:
        import chromadb

        client = chromadb.PersistentClient(path=vector_db_dir)
        collection = client.get_collection(collection_name)
    except Exception as exc:
        raise VectorStoreUnavailableError("Could not open vector store collection") from exc
    _read_collection_cache = (vector_db_dir, collection_name, collection)
    return collection


def get_chunks_by_ids(ids: list[str]) -> dict[str, Document]:
    unique_ids = list(dict.fromkeys(item for item in ids if item))
    if not unique_ids:
        return {}

    try:
        raw = _get_read_collection().get(ids=unique_ids, include=["documents", "metadatas"])
    except Exception as exc:
        raise VectorStoreUnavailableError("Could not read vector store documents by id") from exc

    result: dict[str, Document] = {}
    raw_ids = raw.get("ids") or []
    raw_documents = raw.get("documents") or []
    raw_metadatas = raw.get("metadatas") or []
    for index, item_id in enumerate(raw_ids):
        content = raw_documents[index] if index < len(raw_documents) else ""
        metadata = raw_metadatas[index] if index < len(raw_metadatas) else {}
        result[str(item_id)] = Document(
            page_content=str(content or ""),
            metadata=dict(metadata or {}),
            id=str(item_id),
        )
    return result


def _ensure_index_ready(vector_store: Chroma) -> int:
    count = get_vector_store_document_count(vector_store)
    if count == 0:
        raise EmptyVectorStoreError(
            "Vector index is empty. Run `python -m src.server.app.index_documents --rebuild` first."
        )
    return count


def ensure_vector_store_ready() -> int:
    vector_store = get_vector_store()
    return _ensure_index_ready(vector_store)


def similarity_search(question: str, *, k: int | None = None) -> list[RetrievedDocument]:
    vector_store = get_vector_store()
    _ensure_index_ready(vector_store)
    top_k = k or config.RAG_TOP_K

    try:
        results = vector_store.similarity_search_with_score(question, k=top_k)
    except Exception as exc:
        raise VectorStoreUnavailableError("Similarity search failed") from exc

    return [
        RetrievedDocument(document=document, distance=float(score)) for document, score in results
    ]


def get_retriever() -> BaseRetriever:
    vector_store = get_vector_store()
    _ensure_index_ready(vector_store)
    return vector_store.as_retriever(search_kwargs={"k": config.RAG_TOP_K})

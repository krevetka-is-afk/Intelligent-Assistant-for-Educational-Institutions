from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


_embedding_function: HuggingFaceEmbeddings | None = None
_vector_store: Chroma | None = None


def clear_vector_cache() -> None:
    global _embedding_function, _vector_store
    _embedding_function = None
    _vector_store = None


def get_embedding_function() -> HuggingFaceEmbeddings:
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = HuggingFaceEmbeddings(model_name=config.HF_EMBEDDING_MODEL)
    return _embedding_function


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        config.validate_chunk_settings()
        _vector_store = Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            persist_directory=str(Path(config.VECTOR_DB_DIR)),
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

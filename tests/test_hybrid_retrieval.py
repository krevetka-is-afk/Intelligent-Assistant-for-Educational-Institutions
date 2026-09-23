from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from src.server.app import rag
from src.server.app.vector import RetrievedDocument


@dataclass(frozen=True, slots=True)
class _LexicalResult:
    document: Document
    score: float


def _doc(
    text: str,
    *,
    chunk_id: str,
    document_id: str,
    chunk_index: int,
    quality_status: str = "clean",
    quality_reasons: str = "",
) -> Document:
    metadata = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "source": f"{document_id}.txt",
        "chunk_index": chunk_index,
    }
    if quality_status != "clean" or quality_reasons:
        metadata["quality_status"] = quality_status
        metadata["quality_reasons"] = quality_reasons
    return Document(
        page_content=text,
        metadata=metadata,
    )


def _retrieved(
    text: str,
    *,
    chunk_id: str,
    document_id: str,
    chunk_index: int,
    distance: float,
    quality_status: str = "clean",
    quality_reasons: str = "",
) -> RetrievedDocument:
    return RetrievedDocument(
        document=_doc(
            text,
            chunk_id=chunk_id,
            document_id=document_id,
            chunk_index=chunk_index,
            quality_status=quality_status,
            quality_reasons=quality_reasons,
        ),
        distance=distance,
    )


def test_hybrid_retrieval_uses_unweighted_rrf_and_exact_chunk_dedupe(monkeypatch):
    dense_documents = [
        _retrieved("dense alpha", chunk_id="a:0", document_id="a", chunk_index=0, distance=0.1),
        _retrieved("shared beta", chunk_id="b:0", document_id="b", chunk_index=0, distance=0.2),
    ]
    lexical_documents = [
        _LexicalResult(
            document=_doc("shared beta lexical", chunk_id="b:0", document_id="b", chunk_index=0),
            score=0.95,
        ),
        _LexicalResult(
            document=_doc("lexical gamma", chunk_id="c:0", document_id="c", chunk_index=0),
            score=0.9,
        ),
    ]

    monkeypatch.setattr(rag, "similarity_search", lambda question, k: dense_documents)
    monkeypatch.setattr(
        rag,
        "search_lexical",
        lambda query, *, limit, index_path=None: lexical_documents,
    )
    monkeypatch.setattr(rag.config, "RAG_CANDIDATE_POOL_SIZE", 4, raising=False)
    monkeypatch.setattr(rag.config, "RAG_MAX_CHUNKS_PER_DOCUMENT", 1, raising=False)

    retrieved, metadata, diagnostics = rag.retrieve_documents("beta", k=3)

    assert [item.document.metadata["chunk_id"] for item in retrieved] == ["b:0", "a:0", "c:0"]
    assert (
        metadata.items()
        >= {
            "retrieval_strategy": "hybrid",
            "retrieval_candidate_pool_size": 4,
            "retrieval_dense_candidate_count": 2,
            "retrieval_lexical_candidate_count": 2,
            "retrieval_lexical_available": True,
            "retrieval_query_count": 1,
            "retrieval_rewrite_used": False,
            "retrieval_original_dense_candidate_count": 2,
            "retrieval_original_lexical_candidate_count": 2,
            "retrieval_expanded_dense_candidate_count": 0,
            "retrieval_expanded_lexical_candidate_count": 0,
        }.items()
    )
    assert diagnostics["candidate_count"] == 3
    assert retrieved[0]._retrieval_diagnostics["channel_ranks"] == {"dense": 2, "lexical": 1}
    assert retrieved[0]._retrieval_diagnostics["channels"] == ["dense", "lexical"]


def test_hybrid_retrieval_diversifies_documents_then_fills_without_adjacent_chunks(monkeypatch):
    dense_documents = [
        _retrieved("d1 c0", chunk_id="d1:0", document_id="d1", chunk_index=0, distance=0.1),
        _retrieved("d1 c1", chunk_id="d1:1", document_id="d1", chunk_index=1, distance=0.11),
        _retrieved("d2 c0", chunk_id="d2:0", document_id="d2", chunk_index=0, distance=0.12),
        _retrieved("d1 c2", chunk_id="d1:2", document_id="d1", chunk_index=2, distance=0.13),
    ]

    monkeypatch.setattr(rag, "similarity_search", lambda question, k: dense_documents)
    monkeypatch.setattr(rag, "search_lexical", lambda query, *, limit, index_path=None: [])
    monkeypatch.setattr(rag.config, "RAG_CANDIDATE_POOL_SIZE", 4, raising=False)
    monkeypatch.setattr(rag.config, "RAG_MAX_CHUNKS_PER_DOCUMENT", 2, raising=False)

    retrieved, metadata, _diagnostics = rag.retrieve_documents("query", k=3)

    assert [item.document.metadata["chunk_id"] for item in retrieved] == ["d1:0", "d2:0", "d1:2"]
    assert metadata["retrieval_strategy"] == "dense_only"
    assert metadata["retrieval_candidate_pool_size"] == 4
    assert metadata["retrieval_lexical_available"] is True


def test_hybrid_retrieval_degrades_to_dense_when_lexical_search_fails(monkeypatch):
    dense_documents = [
        _retrieved("dense only", chunk_id="d:0", document_id="d", chunk_index=0, distance=0.1)
    ]

    def _raise_lexical(query: str, *, limit: int, index_path=None):
        del query, limit, index_path
        raise RuntimeError("sqlite is locked")

    monkeypatch.setattr(rag, "similarity_search", lambda question, k: dense_documents)
    monkeypatch.setattr(rag, "search_lexical", _raise_lexical)

    retrieved, metadata, diagnostics = rag.retrieve_documents("query", k=1)

    assert retrieved == dense_documents
    assert metadata["retrieval_strategy"] == "dense_only"
    assert metadata["retrieval_lexical_available"] is False
    assert diagnostics["strategy"] == "dense_only"
    assert retrieved[0]._retrieval_diagnostics["channel_ranks"] == {"dense": 1}
    assert retrieved[0].distance == 0.1


def test_hybrid_retrieval_bounds_lexical_only_candidate_distances(monkeypatch):
    lexical_documents = [
        _LexicalResult(
            document=_doc("lexical high", chunk_id="l:0", document_id="l", chunk_index=0),
            score=7.0,
        ),
        _LexicalResult(
            document=_doc("lexical low", chunk_id="m:0", document_id="m", chunk_index=0),
            score=-3.0,
        ),
    ]

    monkeypatch.setattr(rag, "similarity_search", lambda question, k: [])
    monkeypatch.setattr(
        rag,
        "search_lexical",
        lambda query, *, limit, index_path=None: lexical_documents,
    )
    monkeypatch.setattr(rag.config, "RAG_CANDIDATE_POOL_SIZE", 2, raising=False)

    retrieved, metadata, diagnostics = rag.retrieve_documents("query", k=2)

    assert metadata["retrieval_strategy"] == "hybrid"
    assert [item.document.metadata["chunk_id"] for item in retrieved] == ["l:0", "m:0"]
    assert all(0.0 <= item.distance <= 1.0 for item in retrieved)
    assert retrieved[0].distance == 0.0
    assert retrieved[1].distance > retrieved[0].distance
    assert diagnostics["selected"][0]["channel_scores"]["lexical_score"] == 7.0


def test_hybrid_retrieval_penalty_lets_close_clean_candidate_beat_noisy_candidate(
    monkeypatch,
):
    dense_documents = [
        _retrieved(
            "1 2 3 4 5 назад далее",
            chunk_id="noisy:0",
            document_id="noisy",
            chunk_index=0,
            distance=0.05,
            quality_status="review",
            quality_reasons="high_digit_ratio,navigation_lines",
        ),
        _retrieved(
            "Правила пересдачи",
            chunk_id="clean:0",
            document_id="clean",
            chunk_index=0,
            distance=0.06,
        ),
    ]

    monkeypatch.setattr(rag, "similarity_search", lambda question, k: dense_documents)
    monkeypatch.setattr(rag, "search_lexical", lambda query, *, limit, index_path=None: [])
    monkeypatch.setattr(rag.config, "RAG_CANDIDATE_POOL_SIZE", 2, raising=False)
    monkeypatch.setattr(rag.config, "RAG_MAX_CHUNKS_PER_DOCUMENT", 1, raising=False)

    retrieved, _metadata, diagnostics = rag.retrieve_documents("пересдача", k=2)

    assert [item.document.metadata["document_id"] for item in retrieved] == ["clean", "noisy"]
    assert retrieved[1]._retrieval_diagnostics["quality_demoted"] is True
    assert diagnostics["selected"][1]["quality_demoted"] is True


def test_hybrid_retrieval_strong_noisy_dual_channel_match_beats_weak_clean_match(
    monkeypatch,
):
    dense_documents = [
        _retrieved(
            "Полный фрагмент про пересдачи из студенческого справочника",
            chunk_id="noisy:0",
            document_id="noisy",
            chunk_index=0,
            distance=0.02,
            quality_status="review",
            quality_reasons="many_hyperlinks,navigation_lines",
        ),
        _retrieved(
            "Общий учебный регламент без ответа на вопрос",
            chunk_id="clean:0",
            document_id="clean",
            chunk_index=0,
            distance=0.75,
        ),
    ]
    lexical_documents = [
        _LexicalResult(
            document=_doc(
                "Полный фрагмент про пересдачи из студенческого справочника",
                chunk_id="noisy:0",
                document_id="noisy",
                chunk_index=0,
                quality_status="review",
                quality_reasons="many_hyperlinks,navigation_lines",
            ),
            score=0.99,
        )
    ]

    monkeypatch.setattr(rag, "similarity_search", lambda question, k: dense_documents)
    monkeypatch.setattr(
        rag,
        "search_lexical",
        lambda query, *, limit, index_path=None: lexical_documents,
    )
    monkeypatch.setattr(rag.config, "RAG_CANDIDATE_POOL_SIZE", 2, raising=False)
    monkeypatch.setattr(rag.config, "RAG_MAX_CHUNKS_PER_DOCUMENT", 1, raising=False)

    retrieved, metadata, diagnostics = rag.retrieve_documents("когда пересдача", k=2)

    assert [item.document.metadata["document_id"] for item in retrieved] == ["noisy", "clean"]
    assert metadata["retrieval_strategy"] == "hybrid"
    assert retrieved[0]._retrieval_diagnostics["quality_demoted"] is True
    assert retrieved[0]._retrieval_diagnostics["channels"] == ["dense", "lexical"]
    assert diagnostics["selected"][0]["channel_ranks"] == {"dense": 1, "lexical": 1}


def test_hybrid_retrieval_keeps_noisy_document_as_fallback_when_no_clean_candidate(
    monkeypatch,
):
    dense_documents = [
        RetrievedDocument(
            document=_doc(
                "1 2 3 4 5 назад далее",
                chunk_id="noisy:0",
                document_id="noisy",
                chunk_index=0,
                quality_status="review",
                quality_reasons="high_digit_ratio,navigation_lines",
            ),
            distance=0.05,
        )
    ]

    monkeypatch.setattr(rag, "similarity_search", lambda question, k: dense_documents)
    monkeypatch.setattr(rag, "search_lexical", lambda query, *, limit, index_path=None: [])
    monkeypatch.setattr(rag.config, "RAG_CANDIDATE_POOL_SIZE", 1, raising=False)

    retrieved, metadata, diagnostics = rag.retrieve_documents("пересдача", k=1)

    assert [item.document.metadata["document_id"] for item in retrieved] == ["noisy"]
    assert metadata["retrieval_strategy"] == "dense_only"
    assert retrieved[0]._retrieval_diagnostics["quality_demoted"] is True
    assert diagnostics["selected"][0]["quality_demoted"] is True

from __future__ import annotations

import asyncio

from langchain_core.documents import Document

from src.server.app import rag
from src.server.app.vector import RetrievedDocument


def test_compute_confidence_applies_fallback_penalty():
    docs = [
        RetrievedDocument(document=Document(page_content="a", metadata={}), distance=0.1),
        RetrievedDocument(document=Document(page_content="b", metadata={}), distance=0.2),
        RetrievedDocument(document=Document(page_content="c", metadata={}), distance=0.3),
    ]

    regular = rag.compute_confidence(docs, fallback_used=False)
    fallback = rag.compute_confidence(docs, fallback_used=True)

    assert regular == 0.86
    assert fallback == 0.645


def test_deduplicate_sources_uses_source_page_and_chunk():
    docs = [
        RetrievedDocument(
            document=Document(
                page_content="Chunk 1",
                metadata={"source": "a.txt", "page": 1, "chunk_index": 0, "title": "A"},
            ),
            distance=0.1,
        ),
        RetrievedDocument(
            document=Document(
                page_content="Chunk 1 duplicate",
                metadata={"source": "a.txt", "page": 1, "chunk_index": 0, "title": "A"},
            ),
            distance=0.2,
        ),
    ]

    sources = rag.deduplicate_sources(docs)

    assert sources == [
        {
            "content": "Chunk 1",
            "metadata": {"source": "a.txt", "page": 1, "chunk_index": 0, "title": "A"},
        }
    ]


def test_ask_question_returns_fallback_when_llm_fails(monkeypatch):
    docs = [
        RetrievedDocument(
            document=Document(
                page_content="В приказе сказано, что пересдача проходит в июле.",
                metadata={"source": "rules.txt", "title": "Правила", "chunk_index": 0},
            ),
            distance=0.15,
        )
    ]

    monkeypatch.setattr(rag, "similarity_search", lambda question, k: docs)

    def _raise_llm(question: str, retrieved_documents):
        raise RuntimeError("llm down")

    monkeypatch.setattr(rag, "invoke_llm", _raise_llm)

    result = asyncio.run(rag.ask_question("Когда пересдача?"))

    assert result.metadata["fallback_used"] is True
    assert result.metadata["fallback_reason"] == "llm_unavailable"
    assert result.metadata["num_sources"] == 1
    assert result.answer.startswith("LLM временно недоступна")
    assert result.sources[0]["metadata"]["title"] == "Правила"


def test_ask_question_uses_conversation_history_in_retrieval_query(monkeypatch):
    dense_docs = [
        RetrievedDocument(
            document=Document(page_content="x", metadata={"source": "s"}),
            distance=0.1,
        )
    ]
    lexical_docs = [
        RetrievedDocument(
            document=Document(page_content="x lexical", metadata={"source": "lex"}),
            distance=0.2,
        )
    ]
    fused_docs = [
        RetrievedDocument(
            document=Document(page_content="x fused", metadata={"source": "fused"}),
            distance=0.05,
        )
    ]
    observed: dict[str, object] = {}

    def _similarity_search(question: str, k: int):
        observed["dense_query"] = question
        observed["dense_k"] = k
        return dense_docs

    def _lexical_search(question: str, k: int):
        observed["lexical_query"] = question
        observed["lexical_k"] = k
        return lexical_docs

    def _rrf(
        dense_documents: list[RetrievedDocument],
        lexical_documents: list[RetrievedDocument],
        *,
        top_k: int,
        rrf_k: int,
    ):
        observed["rrf_dense"] = dense_documents
        observed["rrf_lexical"] = lexical_documents
        observed["rrf_top_k"] = top_k
        observed["rrf_k"] = rrf_k
        return fused_docs

    monkeypatch.setattr(rag, "similarity_search", _similarity_search)
    monkeypatch.setattr(rag, "lexical_search", _lexical_search)
    monkeypatch.setattr(rag, "reciprocal_rank_fusion", _rrf)
    monkeypatch.setattr(rag, "invoke_llm", lambda question, retrieved_documents, _: "ok")

    result = asyncio.run(
        rag.ask_question(
            "А что по дедлайну?",
            conversation_history=[
                "Я учусь в кампусе Санкт-Петербург",
                "Я на 2 курсе",
                "У меня пересдача в июле",
                "Какие документы нужны?",
                "И куда нести?",
                "Можно ли подать онлайн?",
            ],
        )
    )

    assert result.answer == "ok"
    expected_query = (
        "Я на 2 курсе\n"
        "У меня пересдача в июле\n"
        "Какие документы нужны?\n"
        "И куда нести?\n"
        "Можно ли подать онлайн?\n"
        "А что по дедлайну?"
    )
    assert observed["dense_query"] == expected_query
    assert observed["lexical_query"] == expected_query
    assert observed["dense_k"] == max(rag.config.RAG_TOP_K, rag.config.RAG_HYBRID_DENSE_TOP_K)
    assert observed["lexical_k"] == max(rag.config.RAG_TOP_K, rag.config.RAG_HYBRID_LEXICAL_TOP_K)
    assert observed["rrf_dense"] == dense_docs
    assert observed["rrf_lexical"] == lexical_docs
    assert observed["rrf_top_k"] == rag.config.RAG_TOP_K
    assert observed["rrf_k"] == rag.config.RAG_HYBRID_RRF_K

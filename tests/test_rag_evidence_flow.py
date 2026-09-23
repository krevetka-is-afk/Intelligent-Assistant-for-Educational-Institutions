from __future__ import annotations

import asyncio

from langchain_core.documents import Document

from src.server.app import rag
from src.server.app.rag_evidence_judge import parse_answer_judge_result
from src.server.app.rag_evidence_models import EvidenceAnswerJudgeResult
from src.server.app.rag_evidence_windows import build_structural_windows
from src.server.app.vector import RetrievedDocument


def _retrieved(content: str, *, chunk_index: int, document_id: str = "doc") -> RetrievedDocument:
    return RetrievedDocument(
        document=Document(
            page_content=content,
            metadata={
                "chunk_id": f"{document_id}:{chunk_index:05d}",
                "document_id": document_id,
                "source_sha256": f"sha-{document_id}",
                "source": f"{document_id}.txt",
                "title": f"Doc {document_id}",
                "page": 1,
                "chunk_index": chunk_index,
                "char_start": chunk_index * 100,
                "char_end": chunk_index * 100 + len(content),
            },
        ),
        distance=0.1 + chunk_index / 100,
    )


def test_evidence_rerank_keeps_top16_and_selects_final_top4(monkeypatch):
    monkeypatch.setattr(rag.config, "RAG_EVIDENCE_RERANK_ENABLED", True)
    monkeypatch.setattr(rag.config, "RAG_EVIDENCE_OFFLINE_CAPTURE_ENABLED", True)
    monkeypatch.setattr(rag.config, "RAG_EVIDENCE_CANDIDATE_TOP_K", 16)
    monkeypatch.setattr(rag.config, "RAG_EVIDENCE_FINAL_TOP_K", 4)
    monkeypatch.setattr(rag.config, "RAG_CANDIDATE_POOL_SIZE", 16)
    docs = [
        _retrieved(
            "пересдача экзамена дата приказ" if index == 12 else f"общий текст {index}",
            chunk_index=0,
            document_id=f"doc-{index}",
        )
        for index in range(16)
    ]

    monkeypatch.setattr(rag, "dense_similarity_search", lambda question, k: docs[:k])
    monkeypatch.setattr(rag, "lexical_similarity_search", lambda question, k: ([], False))

    selected, metadata, diagnostics = rag.retrieve_documents("Какая дата пересдачи экзамена?")

    assert metadata["rag_evidence_rerank_enabled"] is True
    assert len(diagnostics["top16_documents"]) == 16
    assert len(selected) == 4
    assert selected[0].document.metadata["document_id"] == "doc-12"
    assert diagnostics["evidence_rerank"]["selected_count"] == 4


def test_structural_window_uses_trusted_neighbors_without_overlap(monkeypatch):
    anchor = _retrieved("anchor regulation", chunk_index=1)
    previous = Document(
        page_content="previous condition",
        metadata={
            **anchor.document.metadata,
            "chunk_id": "doc:00000",
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 18,
        },
    )
    next_document = Document(
        page_content="next exception",
        metadata={
            **anchor.document.metadata,
            "chunk_id": "doc:00002",
            "chunk_index": 2,
            "char_start": 250,
            "char_end": 264,
        },
    )

    monkeypatch.setattr(
        "src.server.app.rag_evidence_windows.get_chunks_by_ids",
        lambda ids: {
            "doc:00000": previous,
            "doc:00001": anchor.document,
            "doc:00002": next_document,
        },
    )

    result = build_structural_windows([anchor])

    assert result.diagnostics["documents"][0]["status"] == "expanded"
    assert "anchor regulation" in result.prompt_documents[0].document.page_content
    assert "previous condition" in result.prompt_documents[0].document.page_content
    assert "next exception" in result.prompt_documents[0].document.page_content


def test_answer_judge_rejects_unknown_evidence_ids():
    result = parse_answer_judge_result(
        '{"verdict":"complete","evidence_ids":["known","unknown"],"missing_aspects":[]}',
        {"known"},
    )

    assert result.verdict == "unknown"
    assert result.reason == "judge_returned_unknown_evidence_id"
    assert result.evidence_ids == ["known"]


def test_ask_question_records_actual_prompt_documents_for_evidence_flow(monkeypatch):
    monkeypatch.setattr(rag.config, "RAG_EVIDENCE_OFFLINE_CAPTURE_ENABLED", True)
    monkeypatch.setattr(rag.config, "RAG_EVIDENCE_STRUCTURAL_WINDOW_ENABLED", True)
    monkeypatch.setattr(rag.config, "RAG_EVIDENCE_SUFFICIENCY_ENABLED", True)
    monkeypatch.setattr(rag.config, "RAG_CONTEXT_EXPANSION_ENABLED", False)
    anchor = _retrieved("anchor regulation", chunk_index=1)
    previous = Document(
        page_content="previous condition",
        metadata={
            **anchor.document.metadata,
            "chunk_id": "doc:00000",
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 18,
        },
    )
    next_document = Document(
        page_content="next exception",
        metadata={
            **anchor.document.metadata,
            "chunk_id": "doc:00002",
            "chunk_index": 2,
            "char_start": 250,
            "char_end": 264,
        },
    )

    monkeypatch.setattr(rag, "retrieve_documents", lambda *args, **kwargs: ([anchor], {}, {}))
    monkeypatch.setattr(
        rag.query_rewrite,
        "rewrite_retrieval_query",
        lambda *args, **kwargs: rag.query_rewrite.QueryRewriteResult(
            query=args[0],
            used=False,
            fallback_reason=None,
            history_used=False,
            diagnostics={},
        ),
    )
    monkeypatch.setattr(
        "src.server.app.rag_evidence_windows.get_chunks_by_ids",
        lambda ids: {
            "doc:00000": previous,
            "doc:00001": anchor.document,
            "doc:00002": next_document,
        },
    )
    monkeypatch.setattr(rag, "invoke_llm", lambda *args, **kwargs: "anchor regulation [1].")
    monkeypatch.setattr(
        rag,
        "heuristic_answer_judge",
        lambda **kwargs: EvidenceAnswerJudgeResult(verdict="complete", evidence_ids=["doc:00001"]),
    )

    result = asyncio.run(rag.ask_question("previous condition regulation next exception"))

    prompt_text = result.retrieval_diagnostics["actual_prompt_documents"][0]["text"]
    assert "previous condition" in prompt_text
    assert result.retrieval_diagnostics["evidence_sufficiency"]["status"] == "sufficient"
    assert result.sources[0]["content"].startswith("previous condition")


def test_ask_question_sanitizes_known_control_marker_without_model_repair(monkeypatch):
    docs = [_retrieved("В расписании указана дата пересдачи.", chunk_index=0)]

    monkeypatch.setattr(rag.config, "RAG_EVIDENCE_MARKER_REPAIR_ENABLED", True)
    monkeypatch.setattr(rag, "retrieve_documents", lambda *args, **kwargs: (docs, {}, {}))
    monkeypatch.setattr(
        rag.query_rewrite,
        "rewrite_retrieval_query",
        lambda *args, **kwargs: rag.query_rewrite.QueryRewriteResult(
            query=args[0],
            used=False,
            fallback_reason=None,
            history_used=False,
            diagnostics={},
        ),
    )
    monkeypatch.setattr(
        rag,
        "invoke_llm",
        lambda *args, **kwargs: "В untrusted_documents нет сведений о другой дате.",
    )

    result = asyncio.run(rag.ask_question("Когда пересдача?"))

    assert result.answer == "В найденных источниках нет сведений о другой дате."
    assert result.metadata["fallback_used"] is False
    assert result.metadata["policy_output_violation_reason"] is None
    assert result.metadata["policy_output_marker_sanitization_attempted"] is True
    assert result.metadata["policy_output_marker_sanitization_changed"] is True
    assert result.metadata["policy_output_repair_attempted"] is False


def test_failed_evidence_retry_fallback_uses_retry_evidence_context(monkeypatch):
    initial = _retrieved("INITIALONLY пересдача дата", chunk_index=0, document_id="initial")
    alternate = _retrieved(
        "ALTERNATEONLY приказ итоговая выдержка",
        chunk_index=0,
        document_id="alternate",
    )
    answers = iter(["Начальный неподтвержденный ответ [1].", "Retry still unsupported [1]."])

    monkeypatch.setattr(rag.config, "RAG_EVIDENCE_OFFLINE_CAPTURE_ENABLED", True)
    monkeypatch.setattr(rag.config, "RAG_EVIDENCE_SUFFICIENCY_ENABLED", True)
    monkeypatch.setattr(rag.config, "RAG_EVIDENCE_RETRY_ENABLED", True)
    monkeypatch.setattr(rag.config, "RAG_EVIDENCE_FINAL_TOP_K", 1)
    monkeypatch.setattr(rag.config, "RAG_CONTEXT_EXPANSION_ENABLED", False)
    monkeypatch.setattr(
        rag,
        "retrieve_documents",
        lambda *args, **kwargs: (
            [initial],
            {},
            {"_candidate_top16_documents_internal": [initial, alternate]},
        ),
    )
    monkeypatch.setattr(
        rag.query_rewrite,
        "rewrite_retrieval_query",
        lambda *args, **kwargs: rag.query_rewrite.QueryRewriteResult(
            query=args[0],
            used=False,
            fallback_reason=None,
            history_used=False,
            diagnostics={},
        ),
    )
    judge_results = iter(
        [
            EvidenceAnswerJudgeResult(
                verdict="unsupported",
                missing_aspects=["ALTERNATEONLY приказ"],
                reason="initial_unsupported",
            ),
            EvidenceAnswerJudgeResult(verdict="unsupported", reason="retry_unsupported"),
        ]
    )
    monkeypatch.setattr(rag, "heuristic_answer_judge", lambda **kwargs: next(judge_results))
    monkeypatch.setattr(
        rag,
        "invoke_llm",
        lambda question, retrieved_documents, conversation_history=None: next(answers),
    )

    result = asyncio.run(rag.ask_question("пересдача дата приказ"))

    assert result.metadata["fallback_reason"] == "evidence_retry_incomplete"
    assert result.metadata["rag_evidence_retry_attempted"] is True
    assert result.metadata["rag_evidence_retry_succeeded"] is False
    assert "ALTERNATEONLY" in result.answer
    assert "INITIALONLY" not in result.answer
    assert result.sources[0]["content"].startswith("ALTERNATEONLY")
    assert result.retrieved_documents[0].document.metadata["document_id"] == "alternate"
    final_prompt_text = result.retrieval_diagnostics["actual_prompt_documents"][0]["text"]
    assert "ALTERNATEONLY" in final_prompt_text
    assert "INITIALONLY" not in final_prompt_text

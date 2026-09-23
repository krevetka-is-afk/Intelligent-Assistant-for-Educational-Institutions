from langchain_core.documents import Document

from src.server.app.rag_stage7_evaluation import (
    MODE_DENSE_RAW,
    Stage7CaseModeCapture,
    Stage7EvaluationMetrics,
    _disciplinary_conduct_answer,
    _rank_documents,
    _retake_date_uncertainty_answer,
    _summarize,
)
from src.server.app.vector import RetrievedDocument


def _doc(chunk_id: str, document_id: str, *, distance: float) -> RetrievedDocument:
    return RetrievedDocument(
        document=Document(
            page_content=f"content {chunk_id}",
            metadata={
                "chunk_id": chunk_id,
                "document_id": document_id,
                "source": f"{document_id}.txt",
                "chunk_index": 0,
            },
        ),
        distance=distance,
        _retrieval_diagnostics={"original": True},
    )


def test_stage7_rank_documents_clones_inputs_before_mutating_rank_diagnostics():
    dense = [_doc("dense-1", "doc-a", distance=0.2)]
    lexical = [_doc("lexical-1", "doc-b", distance=0.3)]

    ranked, diagnostics = _rank_documents(dense_documents=dense, lexical_documents=lexical, top_k=2)

    assert len(ranked) == 2
    assert diagnostics["candidate_count"] == 2
    assert dense[0].distance == 0.2
    assert lexical[0].distance == 0.3
    assert dense[0]._retrieval_diagnostics == {"original": True}
    assert lexical[0]._retrieval_diagnostics == {"original": True}
    assert ranked[0]._retrieval_diagnostics["rank"] == 1


def test_stage7_rank_documents_fuses_rewrite_as_independent_ranked_lane():
    original = [_doc("original-1", "doc-a", distance=0.2)]
    rewritten = [_doc("rewrite-1", "doc-b", distance=0.1)]

    ranked, diagnostics = _rank_documents(
        dense_documents=original,
        lexical_documents=[],
        dense_rewrite_documents=rewritten,
        top_k=2,
    )

    assert {item.document.metadata["document_id"] for item in ranked} == {"doc-a", "doc-b"}
    assert diagnostics["candidate_count"] == 2
    assert rewritten[0]._retrieval_diagnostics == {"original": True}
    assert any("dense_rewrite" in item._retrieval_diagnostics["channels"] for item in ranked)


def test_stage7_summarize_counts_retrieval_policy_and_fallback_metrics():
    capture = Stage7CaseModeCapture(
        case_id="case-1",
        case_variant="clean-history",
        mode=MODE_DENSE_RAW,
        question="question",
        conversation_history=[],
        retrieval_query="question",
        dense_top5=[],
        lexical_top5=[],
        candidate_top5=[],
        final_top4=[],
        bounded_prompt_documents=[],
        answer="answer",
        sources=[],
        retrieval_metadata={},
        retrieval_diagnostics={},
        policy_metadata={},
        metrics=Stage7EvaluationMetrics(
            expected_document_ranks={"doc": 1},
            expected_document_ranks_at_5={"doc": 1},
            expected_documents_found=["doc"],
            expected_documents_found_at_5=["doc"],
            forbidden_cluster_matches=[],
            retrieval_hit_at_4=True,
            retrieval_hit_at_5=True,
            retake_content_evidence_at_4=None,
            retake_content_evidence_at_5=None,
            forbidden_context_at_4=False,
            answer_policy_violated=True,
            answer_policy_reasons=["out_of_range_source_index"],
            disciplinary_conduct_answer=True,
            retake_date_uncertainty_answer=None,
            fallback_used=True,
            fallback_reason="policy_output_violation",
        ),
        retrieval_time_ms=1,
        generation_time_ms=2,
        total_time_ms=3,
    )

    summary = _summarize([capture])

    assert summary[MODE_DENSE_RAW]["cases"] == 1
    assert summary[MODE_DENSE_RAW]["retrieval_hit_at_4"] == 1
    assert summary[MODE_DENSE_RAW]["answer_policy_violations"] == 1
    assert summary[MODE_DENSE_RAW]["disciplinary_conduct_answers"] == 1
    assert summary[MODE_DENSE_RAW]["disciplinary_answer_cases"] == 1
    assert summary[MODE_DENSE_RAW]["retake_date_uncertainty_answers"] == 0
    assert summary[MODE_DENSE_RAW]["retake_answer_cases"] == 0
    assert summary[MODE_DENSE_RAW]["fallback_used"] == 1
    assert summary[MODE_DENSE_RAW]["avg_total_time_ms"] == 3


def test_stage7_scores_disciplinary_conduct_answer():
    assert _disciplinary_conduct_answer(
        case_id="disciplinary-actions-clean-core",
        answer=(
            "К действиям относятся: непосредственно совершившие дисциплинарный проступок, "
            "склонившие других, содействовавшие и использовавшие иных лиц."
        ),
    )
    assert (
        _disciplinary_conduct_answer(
            case_id="disciplinary-actions-clean-core",
            answer="В источниках нет детализированных правил о конкретных действиях.",
        )
        is False
    )
    assert _disciplinary_conduct_answer(case_id="retake-periods-clean-core", answer="x") is None


def test_stage7_scores_retake_date_uncertainty_answer():
    assert _retake_date_uncertainty_answer(
        case_id="retake-periods-clean-core",
        answer=(
            "Точных календарных дат в найденных фрагментах нет. "
            "Есть правило: промежуток между первой и второй пересдачей не менее 5 дней."
        ),
    )
    assert (
        _retake_date_uncertainty_answer(
            case_id="retake-periods-clean-core",
            answer="Пересдачи проходят в период пересдач.",
        )
        is False
    )
    assert (
        _retake_date_uncertainty_answer(case_id="disciplinary-actions-clean-core", answer="x")
        is None
    )

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

if TYPE_CHECKING:
    from .vector import RetrievedDocument

DEFAULT_STAGE7_REPORT_DIR = (
    Path(__file__).resolve().parents[3] / ".omx" / "reports" / "rag-stage7-2026-09-23"
)
DEFAULT_STAGE7_SNAPSHOT_DIR = (
    Path(__file__).resolve().parents[3] / ".omx" / "snapshots" / "rag-stage7-2026-09-23"
)

MODE_DENSE_RAW = "dense-only"
MODE_PRIMARY_DENSE_RANKER = "primary"
MODE_HYBRID = "hybrid"
MODE_POLICY_FALLBACK = "policy-fallback"
STAGE7_MODES = (
    MODE_DENSE_RAW,
    MODE_PRIMARY_DENSE_RANKER,
    MODE_HYBRID,
    MODE_POLICY_FALLBACK,
)

_CHROMA_COLLECTION: Any | None = None
_TINY_TOKENIZER: Any | None = None
_TINY_MODEL: Any | None = None
_VERIFIED_TINY_MODEL = "cointegrated/rubert-tiny2"


@dataclass(frozen=True, slots=True)
class Stage7EvaluationMetrics:
    expected_document_ranks: dict[str, int | None]
    expected_document_ranks_at_5: dict[str, int | None]
    expected_documents_found: list[str]
    expected_documents_found_at_5: list[str]
    forbidden_cluster_matches: list[str]
    retrieval_hit_at_4: bool
    retrieval_hit_at_5: bool
    retake_content_evidence_at_4: bool | None
    retake_content_evidence_at_5: bool | None
    forbidden_context_at_4: bool
    answer_policy_violated: bool
    answer_policy_reasons: list[str]
    disciplinary_conduct_answer: bool | None
    retake_date_uncertainty_answer: bool | None
    fallback_used: bool
    fallback_reason: str | None


@dataclass(frozen=True, slots=True)
class Stage7CaseModeCapture:
    case_id: str
    case_variant: str
    mode: str
    question: str
    conversation_history: list[str]
    retrieval_query: str
    dense_top5: list[dict[str, Any]]
    lexical_top5: list[dict[str, Any]]
    candidate_top5: list[dict[str, Any]]
    final_top4: list[dict[str, Any]]
    bounded_prompt_documents: list[dict[str, Any]]
    answer: str
    sources: list[dict[str, Any]]
    retrieval_metadata: dict[str, Any]
    retrieval_diagnostics: dict[str, Any]
    policy_metadata: dict[str, Any]
    metrics: Stage7EvaluationMetrics
    retrieval_time_ms: int
    generation_time_ms: int
    total_time_ms: int
    expanded_retrieval_query: str | None = None
    dense_rewrite_top5: list[dict[str, Any]] = field(default_factory=list)
    lexical_rewrite_top5: list[dict[str, Any]] = field(default_factory=list)


def _apply_frozen_index(index_dir: Path) -> None:
    resolved = index_dir.expanduser().resolve()
    os.environ["VECTOR_DB_DIR"] = str(resolved)
    os.environ["LEXICAL_INDEX_PATH"] = str((resolved / "lexical_index.sqlite3").resolve())


def _progress(message: str) -> None:
    print(f"[stage7] {message}", file=sys.stderr, flush=True)


def _load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _query_embedding_backend() -> str:
    from . import config

    if config.HF_EMBEDDING_MODEL == _VERIFIED_TINY_MODEL:
        return "verified_tiny2_cls_normalized"
    return "production_huggingface_embeddings"


def _embed_tiny_query(query: str) -> list[float]:
    global _TINY_TOKENIZER, _TINY_MODEL
    import torch
    import torch.nn.functional as functional
    from transformers.models.bert.modeling_bert import BertModel
    from transformers.models.bert.tokenization_bert import BertTokenizer

    if _TINY_TOKENIZER is None or _TINY_MODEL is None:
        _progress(f"loading verified direct embedding model {_VERIFIED_TINY_MODEL}")
        _TINY_TOKENIZER = BertTokenizer.from_pretrained(_VERIFIED_TINY_MODEL)
        _TINY_MODEL = BertModel.from_pretrained(_VERIFIED_TINY_MODEL)
        _TINY_MODEL.eval()
    assert _TINY_TOKENIZER is not None and _TINY_MODEL is not None
    encoded = _TINY_TOKENIZER(
        [query], padding=True, truncation=True, max_length=2048, return_tensors="pt"
    )
    with torch.no_grad():
        embedding = _TINY_MODEL(**encoded).last_hidden_state[:, 0]
        embedding = functional.normalize(embedding, p=2, dim=1)
    return embedding[0].cpu().tolist()


def _get_chroma_collection(index_dir: Path, *, collection_name: str) -> Any:
    global _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is None:
        import chromadb

        client = chromadb.PersistentClient(path=str(index_dir.expanduser().resolve()))
        _CHROMA_COLLECTION = client.get_collection(collection_name)
    return _CHROMA_COLLECTION


def _resolve_hf_cache_revision(model_name: str) -> str | None:
    model_dir_name = "models--" + model_name.replace("/", "--")
    refs_path = Path.home() / ".cache" / "huggingface" / "hub" / model_dir_name / "refs" / "main"
    if not refs_path.exists():
        return None
    return refs_path.read_text(encoding="utf-8").strip() or None


def _dense_similarity_search_direct(
    query: str, *, k: int, index_dir: Path
) -> list[RetrievedDocument]:
    from . import config
    from .vector import RetrievedDocument, get_embedding_function

    backend = _query_embedding_backend()
    embedding = (
        _embed_tiny_query(query)
        if backend == "verified_tiny2_cls_normalized"
        else get_embedding_function().embed_query(query)
    )
    collection = _get_chroma_collection(index_dir, collection_name=config.CHROMA_COLLECTION_NAME)
    raw = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    documents = raw.get("documents") or [[]]
    metadatas = raw.get("metadatas") or [[]]
    distances = raw.get("distances") or [[]]
    ids = raw.get("ids") or [[]]

    results: list[RetrievedDocument] = []
    for index, content in enumerate(documents[0]):
        metadata = metadatas[0][index] if index < len(metadatas[0]) else {}
        distance = distances[0][index] if index < len(distances[0]) else 1.0
        document_id = ids[0][index] if index < len(ids[0]) else None
        results.append(
            RetrievedDocument(
                document=Document(
                    page_content=str(content or ""),
                    metadata=dict(metadata or {}),
                    id=document_id,
                ),
                distance=float(distance),
                _retrieval_diagnostics={
                    "dense_distance": float(distance),
                    "dense_query_backend": backend,
                    "dense_query_normalized": (
                        backend == "verified_tiny2_cls_normalized" or config.HF_EMBEDDING_NORMALIZE
                    ),
                },
            )
        )
    return results


def _clone_retrieved_documents(items: Sequence[RetrievedDocument]) -> list[RetrievedDocument]:
    from .vector import RetrievedDocument

    cloned: list[RetrievedDocument] = []
    for item in items:
        metadata = item.document.metadata if isinstance(item.document.metadata, dict) else {}
        cloned.append(
            RetrievedDocument(
                document=Document(
                    page_content=str(item.document.page_content),
                    metadata=dict(metadata),
                    id=getattr(item.document, "id", None),
                ),
                distance=float(item.distance),
                _retrieval_diagnostics=dict(item._retrieval_diagnostics),
            )
        )
    return cloned


def _candidate_payload(item: RetrievedDocument, *, rank: int) -> dict[str, Any]:
    from . import rag

    return {
        "rank": rank,
        "distance": float(item.distance),
        "metadata": rag.normalize_source_metadata(item.document.metadata),
        "content_preview": " ".join(item.document.page_content.split())[:500],
        "retrieval_diagnostics": dict(item._retrieval_diagnostics),
    }


def _candidate_payloads(items: Sequence[RetrievedDocument], *, limit: int) -> list[dict[str, Any]]:
    return [
        _candidate_payload(item, rank=index) for index, item in enumerate(items[:limit], start=1)
    ]


def _bounded_prompt_payloads(items: Sequence[RetrievedDocument]) -> list[dict[str, Any]]:
    return [
        {
            **_candidate_payload(item, rank=index),
            "content": item.document.page_content,
        }
        for index, item in enumerate(items, start=1)
    ]


def _disciplinary_conduct_answer(*, case_id: str, answer: str) -> bool | None:
    if not case_id.startswith("disciplinary-"):
        return None
    answer_text = answer.casefold()
    unsupported_absence_claims = (
        "нет детализирован",
        "отсутствуют детализирован",
        "отсутствуют конкретн",
        "не указаны конкретн",
    )
    if any(marker in answer_text for marker in unsupported_absence_claims):
        return False
    conduct_markers = (
        "непосредственно соверш",
        "совершению дисциплинарного проступка",
        "склонив",
        "содейств",
        "использовав",
        "организовав",
        "руководив",
        "нарушение правил",
        "нарушения правил",
        "нарушение локальных",
        "нарушения локальных",
    )
    return any(marker in answer_text for marker in conduct_markers)


def _retake_date_uncertainty_answer(*, case_id: str, answer: str) -> bool | None:
    if not case_id.startswith("retake-"):
        return None
    answer_text = answer.casefold()
    date_uncertainty_markers = (
        "точных календарных дат",
        "конкретных календарных дат",
        "календарные даты не",
        "точные даты не",
        "конкретные даты не",
        "дат в найденных",
    )
    timing_rule_markers = (
        "не менее 5",
        "не менее пяти",
        "учебный офис",
        "учебным офисом",
        "текущий период",
        "календарный год",
        "период пересдач",
    )
    return any(marker in answer_text for marker in date_uncertainty_markers) and any(
        marker in answer_text for marker in timing_rule_markers
    )


def _expected_forbidden_metrics(
    *,
    case: Any,
    final_documents: Sequence[RetrievedDocument],
    candidate_documents: Sequence[RetrievedDocument],
    answer: str,
    sources: list[dict[str, Any]],
    fallback_used: bool,
    fallback_reason: str | None,
) -> Stage7EvaluationMetrics:
    from . import rag
    from .prompt_policy import build_source_allowlist, evaluate_answer_policy
    from .rag_evaluation import _matches_any

    expected_document_ranks: dict[str, int | None] = {
        expected_document: None for expected_document in case.expected_documents
    }
    expected_document_ranks_at_5: dict[str, int | None] = {
        expected_document: None for expected_document in case.expected_documents
    }
    expected_documents_found: list[str] = []
    expected_documents_found_at_5: list[str] = []
    forbidden_matches: list[str] = []

    def collect_expected(
        documents: Sequence[RetrievedDocument],
        ranks: dict[str, int | None],
        found: list[str],
    ) -> None:
        for rank, retrieved in enumerate(documents, start=1):
            metadata = rag.normalize_source_metadata(retrieved.document.metadata)
            expected_matches = _matches_any(metadata, case.expected_documents)
            for expected_document in expected_matches:
                if ranks.get(expected_document) is None:
                    ranks[expected_document] = rank
                if expected_document not in found:
                    found.append(expected_document)

    collect_expected(final_documents, expected_document_ranks, expected_documents_found)
    collect_expected(
        candidate_documents, expected_document_ranks_at_5, expected_documents_found_at_5
    )

    for retrieved in final_documents:
        metadata = rag.normalize_source_metadata(retrieved.document.metadata)
        for forbidden_cluster in _matches_any(metadata, case.forbidden_clusters):
            if forbidden_cluster not in forbidden_matches:
                forbidden_matches.append(forbidden_cluster)

    retake_content_evidence_at_4 = _retake_content_evidence(
        case=case,
        documents=final_documents,
    )
    retake_content_evidence_at_5 = _retake_content_evidence(
        case=case,
        documents=candidate_documents,
    )

    policy_result = evaluate_answer_policy(
        answer,
        source_count=len(sources),
        source_allowlist=build_source_allowlist(sources),
    )
    return Stage7EvaluationMetrics(
        expected_document_ranks=expected_document_ranks,
        expected_document_ranks_at_5=expected_document_ranks_at_5,
        expected_documents_found=expected_documents_found,
        expected_documents_found_at_5=expected_documents_found_at_5,
        forbidden_cluster_matches=forbidden_matches,
        retrieval_hit_at_4=bool(expected_documents_found),
        retrieval_hit_at_5=bool(expected_documents_found_at_5),
        retake_content_evidence_at_4=retake_content_evidence_at_4,
        retake_content_evidence_at_5=retake_content_evidence_at_5,
        forbidden_context_at_4=bool(forbidden_matches),
        answer_policy_violated=policy_result.violated,
        answer_policy_reasons=list(policy_result.reasons),
        disciplinary_conduct_answer=_disciplinary_conduct_answer(
            case_id=case.id,
            answer=answer,
        ),
        retake_date_uncertainty_answer=_retake_date_uncertainty_answer(
            case_id=case.id,
            answer=answer,
        ),
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )


def _retake_content_evidence(
    *,
    case: Any,
    documents: Sequence[RetrievedDocument],
) -> bool | None:
    if not str(case.id).startswith("retake-"):
        return None
    for retrieved in documents:
        metadata = (
            retrieved.document.metadata if isinstance(retrieved.document.metadata, dict) else {}
        )
        haystack = " ".join(
            str(value).casefold()
            for value in (
                metadata.get("document_id"),
                metadata.get("source"),
                metadata.get("title"),
                retrieved.document.page_content[:500],
            )
            if value is not None
        )
        if "пересда" in haystack or "retake" in haystack:
            return True
    return False


def _rank_documents(
    *,
    dense_documents: Sequence[RetrievedDocument],
    lexical_documents: Sequence[RetrievedDocument],
    top_k: int,
    dense_rewrite_documents: Sequence[RetrievedDocument] = (),
    lexical_rewrite_documents: Sequence[RetrievedDocument] = (),
) -> tuple[list[RetrievedDocument], dict[str, Any]]:
    from . import config, rag

    ranked, diagnostics = rag._rank_hybrid_documents(
        dense_documents=_clone_retrieved_documents(dense_documents),
        lexical_documents=_clone_retrieved_documents(lexical_documents),
        dense_rewrite_documents=_clone_retrieved_documents(dense_rewrite_documents),
        lexical_rewrite_documents=_clone_retrieved_documents(lexical_rewrite_documents),
        top_k=top_k,
        rrf_k=rag._get_positive_int_config("RAG_RRF_K", config.RAG_RRF_K),
        max_chunks_per_document=rag._get_positive_int_config(
            "RAG_MAX_CHUNKS_PER_DOCUMENT",
            config.RAG_MAX_CHUNKS_PER_DOCUMENT,
        ),
    )
    return _clone_retrieved_documents(ranked), diagnostics


def _invoke_answer(
    mode: str,
    question: str,
    documents: list[RetrievedDocument],
    history: list[str],
    *,
    llm_available: bool,
    replay_answer: str | None = None,
):
    from . import rag
    from .prompt_policy import build_source_allowlist, evaluate_answer_policy

    generation_started = perf_counter()
    fallback_used = False
    fallback_reason: str | None = None
    policy_metadata: dict[str, Any] = {}

    try:
        compiled_prompt = rag._prompt_compiler.compile(
            question=question,
            retrieved_documents=documents,
            conversation_history=history,
        )
        bounded_documents = rag._attach_retrieval_diagnostics(
            list(compiled_prompt.retrieved_documents),
            documents,
        )
        sources = rag.deduplicate_sources(bounded_documents)
        if replay_answer is not None:
            answer = replay_answer
        else:
            try:
                if not llm_available:
                    raise RuntimeError("stage7_llm_not_available")
                answer = rag.invoke_llm(question, bounded_documents, history)
            except Exception as exc:
                fallback_used = True
                fallback_reason = f"llm_unavailable:{type(exc).__name__}"
                answer = rag.build_fallback_answer(bounded_documents)

        source_allowlist = build_source_allowlist(sources)
        policy_result = evaluate_answer_policy(
            answer,
            source_count=len(sources),
            source_allowlist=source_allowlist,
        )
        policy_repair_attempted = False
        policy_repair_succeeded = False
        policy_repair_skipped_reason = None
        if mode == MODE_POLICY_FALLBACK and policy_result.violated:
            if rag._requires_safe_policy_refusal(policy_result):
                fallback_used = True
                fallback_reason = "policy_output_violation"
                policy_repair_skipped_reason = "unsafe_policy_reason"
                answer = rag.SAFE_POLICY_REFUSAL
            elif rag._allows_policy_repair(policy_result) and llm_available:
                policy_repair_attempted = True
                try:
                    repair_answer = rag.invoke_llm(
                        rag._build_repair_question(question),
                        bounded_documents,
                        history,
                    )
                except Exception:
                    repair_answer = ""
                repair_policy_result = evaluate_answer_policy(
                    repair_answer or rag.build_empty_answer(),
                    source_count=len(sources),
                    source_allowlist=source_allowlist,
                )
                policy_result = repair_policy_result
                if repair_policy_result.violated:
                    fallback_used = True
                    fallback_reason = "policy_output_violation"
                    policy_repair_skipped_reason = (
                        "unsafe_policy_reason"
                        if rag._requires_safe_policy_refusal(repair_policy_result)
                        else None
                    )
                    answer = (
                        rag.SAFE_POLICY_REFUSAL
                        if rag._requires_safe_policy_refusal(repair_policy_result)
                        else rag.build_policy_output_fallback_answer(bounded_documents)
                    )
                else:
                    policy_repair_succeeded = True
                    answer = repair_answer or rag.build_empty_answer()
            else:
                fallback_used = True
                fallback_reason = "policy_output_violation"
                policy_repair_skipped_reason = "llm_unavailable"
                answer = rag.build_policy_output_fallback_answer(bounded_documents)
        policy_metadata = {
            "policy_output_violation_reason": policy_result.primary_reason,
            "policy_output_violation_reasons": list(policy_result.reasons),
            "policy_output_violation_match_counts": policy_result.match_counts,
            "policy_output_repair_attempted": policy_repair_attempted,
            "policy_output_repair_succeeded": policy_repair_succeeded,
            "policy_output_repair_skipped_reason": policy_repair_skipped_reason,
            "policy_replayed_initial_answer": replay_answer is not None,
        }
        generation_elapsed = perf_counter() - generation_started
        return (
            answer,
            bounded_documents,
            sources,
            fallback_used,
            fallback_reason,
            policy_metadata,
            round(generation_elapsed * 1000),
        )
    except Exception as exc:
        generation_elapsed = perf_counter() - generation_started
        fallback_used = True
        fallback_reason = f"stage7_generation_failed:{type(exc).__name__}"
        answer = rag.build_fallback_answer(documents)
        return (
            answer,
            documents,
            rag.deduplicate_sources(documents),
            fallback_used,
            fallback_reason,
            {"stage7_generation_error": type(exc).__name__},
            round(generation_elapsed * 1000),
        )


def _capture_case_mode(
    *,
    case: Any,
    mode: str,
    dense_documents: Sequence[RetrievedDocument],
    lexical_documents: Sequence[RetrievedDocument],
    dense_rewrite_documents: Sequence[RetrievedDocument],
    lexical_rewrite_documents: Sequence[RetrievedDocument],
    expanded_retrieval_query: str | None,
    rewrite_diagnostics: dict[str, Any],
    llm_available: bool,
    replay_answer: str | None = None,
) -> Stage7CaseModeCapture:
    from . import config, rag

    total_started = perf_counter()
    retrieval_query = rag.build_retrieval_query(case.question, case.conversation_history)
    retrieval_started = perf_counter()

    retrieval_metadata: dict[str, Any] = {
        "retrieval_candidate_pool_size": max(
            config.RAG_TOP_K,
            config.RAG_CANDIDATE_POOL_SIZE,
        ),
        "retrieval_dense_candidate_count": len(dense_documents),
        "retrieval_lexical_candidate_count": len(lexical_documents),
        "retrieval_expanded_dense_candidate_count": len(dense_rewrite_documents),
        "retrieval_expanded_lexical_candidate_count": len(lexical_rewrite_documents),
        "query_rewrite": rewrite_diagnostics,
    }
    retrieval_diagnostics: dict[str, Any] = {}

    if mode == MODE_DENSE_RAW:
        candidate_top5_docs = _clone_retrieved_documents(dense_documents[:5])
        final_documents = _clone_retrieved_documents(dense_documents[: config.RAG_TOP_K])
        retrieval_metadata["retrieval_strategy"] = "dense_raw"
    elif mode == MODE_PRIMARY_DENSE_RANKER:
        candidate_top5_docs, candidate_diagnostics = _rank_documents(
            dense_documents=dense_documents,
            lexical_documents=[],
            top_k=5,
        )
        final_documents, retrieval_diagnostics = _rank_documents(
            dense_documents=dense_documents,
            lexical_documents=[],
            top_k=config.RAG_TOP_K,
        )
        retrieval_diagnostics["candidate_top5"] = candidate_diagnostics
        retrieval_metadata["retrieval_strategy"] = "primary_dense_ranker"
    elif mode in {MODE_HYBRID, MODE_POLICY_FALLBACK}:
        candidate_top5_docs, candidate_diagnostics = _rank_documents(
            dense_documents=dense_documents,
            lexical_documents=lexical_documents,
            dense_rewrite_documents=dense_rewrite_documents,
            lexical_rewrite_documents=lexical_rewrite_documents,
            top_k=5,
        )
        final_documents, retrieval_diagnostics = _rank_documents(
            dense_documents=dense_documents,
            lexical_documents=lexical_documents,
            dense_rewrite_documents=dense_rewrite_documents,
            lexical_rewrite_documents=lexical_rewrite_documents,
            top_k=config.RAG_TOP_K,
        )
        retrieval_diagnostics["candidate_top5"] = candidate_diagnostics
        retrieval_metadata["retrieval_strategy"] = (
            "hybrid_policy_replay" if mode == MODE_POLICY_FALLBACK else "hybrid"
        )
    else:
        raise ValueError(f"Unknown stage-7 mode: {mode}")

    retrieval_elapsed = perf_counter() - retrieval_started
    (
        answer,
        bounded_documents,
        sources,
        fallback_used,
        fallback_reason,
        policy_metadata,
        generation_time_ms,
    ) = _invoke_answer(
        mode,
        case.question,
        final_documents,
        list(case.conversation_history),
        llm_available=llm_available,
        replay_answer=replay_answer,
    )
    total_elapsed = perf_counter() - total_started

    metrics = _expected_forbidden_metrics(
        case=case,
        final_documents=bounded_documents,
        candidate_documents=candidate_top5_docs,
        answer=answer,
        sources=sources,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )

    return Stage7CaseModeCapture(
        case_id=case.id,
        case_variant="dirty-history" if case.conversation_history else "clean-history",
        mode=mode,
        question=case.question,
        conversation_history=list(case.conversation_history),
        retrieval_query=retrieval_query,
        expanded_retrieval_query=expanded_retrieval_query,
        dense_top5=_candidate_payloads(dense_documents, limit=5),
        lexical_top5=_candidate_payloads(lexical_documents, limit=5),
        dense_rewrite_top5=_candidate_payloads(dense_rewrite_documents, limit=5),
        lexical_rewrite_top5=_candidate_payloads(lexical_rewrite_documents, limit=5),
        candidate_top5=_candidate_payloads(candidate_top5_docs, limit=5),
        final_top4=_candidate_payloads(final_documents, limit=config.RAG_TOP_K),
        bounded_prompt_documents=_bounded_prompt_payloads(bounded_documents),
        answer=answer,
        sources=sources,
        retrieval_metadata=retrieval_metadata,
        retrieval_diagnostics=retrieval_diagnostics,
        policy_metadata=policy_metadata,
        metrics=metrics,
        retrieval_time_ms=round(retrieval_elapsed * 1000),
        generation_time_ms=generation_time_ms,
        total_time_ms=round(total_elapsed * 1000),
    )


def _capture_case(
    case: Any,
    *,
    modes: Sequence[str],
    llm_available: bool,
    index_dir: Path,
    query_rewrite_enabled: bool,
) -> list[Stage7CaseModeCapture]:
    from . import config, query_rewrite, rag

    retrieval_query = rag.build_retrieval_query(case.question, case.conversation_history)
    rewrite_started = perf_counter()
    rewrite_result = query_rewrite.rewrite_retrieval_query(
        retrieval_query,
        list(case.conversation_history),
        enabled=query_rewrite_enabled,
    )
    rewrite_time_ms = round((perf_counter() - rewrite_started) * 1000)
    expanded_query = rewrite_result.query if rewrite_result.used else None
    candidate_pool_size = max(config.RAG_TOP_K, config.RAG_CANDIDATE_POOL_SIZE, 5)
    _progress(f"{case.id}: dense retrieval start")
    dense_documents = _dense_similarity_search_direct(
        retrieval_query,
        k=candidate_pool_size,
        index_dir=index_dir,
    )
    _progress(f"{case.id}: lexical retrieval start")
    lexical_documents, _lexical_available = rag.lexical_similarity_search(
        retrieval_query,
        k=candidate_pool_size,
    )
    dense_rewrite_documents: list[RetrievedDocument] = []
    lexical_rewrite_documents: list[RetrievedDocument] = []
    if expanded_query is not None:
        _progress(f"{case.id}: rewritten dense retrieval start")
        dense_rewrite_documents = _dense_similarity_search_direct(
            expanded_query,
            k=candidate_pool_size,
            index_dir=index_dir,
        )
        _progress(f"{case.id}: rewritten lexical retrieval start")
        lexical_rewrite_documents, _expanded_lexical_available = rag.lexical_similarity_search(
            expanded_query,
            k=candidate_pool_size,
        )
    captures: list[Stage7CaseModeCapture] = []
    hybrid_initial_answer: str | None = None
    for mode in modes:
        _progress(f"{case.id}: mode {mode} start")
        capture = _capture_case_mode(
            case=case,
            mode=mode,
            dense_documents=_clone_retrieved_documents(dense_documents),
            lexical_documents=_clone_retrieved_documents(lexical_documents),
            dense_rewrite_documents=_clone_retrieved_documents(dense_rewrite_documents),
            lexical_rewrite_documents=_clone_retrieved_documents(lexical_rewrite_documents),
            expanded_retrieval_query=expanded_query,
            rewrite_diagnostics={
                **rewrite_result.diagnostics,
                "rewrite_time_ms": rewrite_time_ms,
                "applied_to_mode": mode in {MODE_HYBRID, MODE_POLICY_FALLBACK},
            },
            llm_available=llm_available,
            replay_answer=(hybrid_initial_answer if mode == MODE_POLICY_FALLBACK else None),
        )
        if mode == MODE_HYBRID:
            hybrid_initial_answer = capture.answer
        captures.append(capture)
    return captures


def _ollama_model_available(*, host: str, model: str, timeout_seconds: float) -> bool:
    url = f"{host.rstrip('/')}/api/tags"
    request = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, ValueError):
        return False
    models = payload.get("models")
    if not isinstance(models, list):
        return False
    for item in models:
        if not isinstance(item, dict):
            continue
        if item.get("name") == model or item.get("model") == model:
            return True
    return False


def _summarize(captures: Sequence[Stage7CaseModeCapture]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in STAGE7_MODES:
        mode_captures = [capture for capture in captures if capture.mode == mode]
        if not mode_captures:
            continue
        summary[mode] = {
            "cases": len(mode_captures),
            "retrieval_hit_at_4": sum(
                1 for capture in mode_captures if capture.metrics.retrieval_hit_at_4
            ),
            "retrieval_hit_at_5": sum(
                1 for capture in mode_captures if capture.metrics.retrieval_hit_at_5
            ),
            "retake_content_evidence_at_4": sum(
                1
                for capture in mode_captures
                if capture.metrics.retake_content_evidence_at_4 is True
            ),
            "retake_content_evidence_at_5": sum(
                1
                for capture in mode_captures
                if capture.metrics.retake_content_evidence_at_5 is True
            ),
            "forbidden_context_at_4": sum(
                1 for capture in mode_captures if capture.metrics.forbidden_context_at_4
            ),
            "answer_policy_violations": sum(
                1 for capture in mode_captures if capture.metrics.answer_policy_violated
            ),
            "disciplinary_conduct_answers": sum(
                1
                for capture in mode_captures
                if capture.metrics.disciplinary_conduct_answer is True
            ),
            "disciplinary_answer_cases": sum(
                1
                for capture in mode_captures
                if capture.metrics.disciplinary_conduct_answer is not None
            ),
            "retake_date_uncertainty_answers": sum(
                1
                for capture in mode_captures
                if capture.metrics.retake_date_uncertainty_answer is True
            ),
            "retake_answer_cases": sum(
                1
                for capture in mode_captures
                if capture.metrics.retake_date_uncertainty_answer is not None
            ),
            "fallback_used": sum(1 for capture in mode_captures if capture.metrics.fallback_used),
            "clean_cases": sum(
                1 for capture in mode_captures if capture.case_variant == "clean-history"
            ),
            "dirty_cases": sum(
                1 for capture in mode_captures if capture.case_variant == "dirty-history"
            ),
            "avg_total_time_ms": round(
                sum(capture.total_time_ms for capture in mode_captures) / len(mode_captures)
            ),
        }
    return summary


def _json_default(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_stage7_report(
    path: Path,
    *,
    captures: Sequence[Stage7CaseModeCapture],
    cases_path: Path,
    snapshot_dir: Path,
    index_dir: Path,
    snapshot_manifest: dict[str, Any] | None,
    query_rewrite_enabled: bool = False,
) -> None:
    from . import config
    from .prompt_policy import PROMPT_POLICY_VERSION

    payload = {
        "schema_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "cases_path": str(cases_path),
        "snapshot_dir": str(snapshot_dir),
        "snapshot_manifest": snapshot_manifest,
        "settings": {
            "rag_top_k": config.RAG_TOP_K,
            "rag_candidate_pool_size": config.RAG_CANDIDATE_POOL_SIZE,
            "rag_max_context_documents": config.RAG_MAX_CONTEXT_DOCUMENTS,
            "rag_max_document_chars": config.RAG_MAX_DOCUMENT_CHARS,
            "rag_max_total_context_chars": config.RAG_MAX_TOTAL_CONTEXT_CHARS,
            "rag_max_history_messages": config.RAG_MAX_HISTORY_MESSAGES,
            "rag_max_history_chars": config.RAG_MAX_HISTORY_CHARS,
            "embedding_model": config.HF_EMBEDDING_MODEL,
            "hf_embedding_normalize": config.HF_EMBEDDING_NORMALIZE,
            "llm_model": config.LLM_MODEL,
            "query_rewrite_enabled": query_rewrite_enabled,
            "query_rewrite_model": config.RAG_QUERY_REWRITE_MODEL,
            "query_rewrite_timeout_seconds": config.RAG_QUERY_REWRITE_TIMEOUT_SECONDS,
            "ollama_host": config.OLLAMA_HOST,
            "vector_db_dir": str(config.VECTOR_DB_DIR),
            "lexical_index_path": str(config.LEXICAL_INDEX_PATH),
            "prompt_policy_version": PROMPT_POLICY_VERSION,
            "dense_query_backend": _query_embedding_backend(),
            "dense_query_backend_reason": (
                "tiny2 CLS+Normalize was parity-checked against its production "
                "SentenceTransformer; other models use the production embedding function"
            ),
            "hf_embedding_cache_revision": _resolve_hf_cache_revision(config.HF_EMBEDDING_MODEL),
            "index_dir": str(index_dir),
            "index_seed_dir": str(snapshot_dir / "index.seed"),
        },
        "modes": list(STAGE7_MODES),
        "summary": _summarize(captures),
        "captures": captures,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def capture_stage7_suite(
    *,
    cases_path: Path,
    snapshot_dir: Path,
    index_dir: Path | None,
    modes: Sequence[str],
    query_rewrite_enabled: bool = False,
    ollama_probe_timeout_seconds: float = 2.0,
) -> list[Stage7CaseModeCapture]:
    resolved_index_dir = index_dir or (snapshot_dir / "index")
    _apply_frozen_index(resolved_index_dir)

    _progress("importing RAG modules")
    from . import config
    from .rag_evaluation import load_evaluation_cases
    from .vector import clear_vector_cache

    clear_vector_cache()
    llm_available = _ollama_model_available(
        host=config.OLLAMA_HOST,
        model=config.LLM_MODEL,
        timeout_seconds=ollama_probe_timeout_seconds,
    )
    _progress(f"LLM {config.LLM_MODEL} available via {config.OLLAMA_HOST}: {llm_available}")
    cases = load_evaluation_cases(cases_path)
    captures: list[Stage7CaseModeCapture] = []
    for case in cases:
        captures.extend(
            _capture_case(
                case,
                modes=modes,
                llm_available=llm_available,
                index_dir=resolved_index_dir,
                query_rewrite_enabled=query_rewrite_enabled,
            )
        )
    return captures


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture stage-7 RAG quality comparison matrix.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "tests"
        / "fixtures"
        / "rag_eval"
        / "cases.v1.json",
    )
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_STAGE7_SNAPSHOT_DIR)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Optional working index copy; defaults to SNAPSHOT_DIR/index.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_STAGE7_REPORT_DIR / "matrix.prefix.json",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=STAGE7_MODES,
        default=list(STAGE7_MODES),
    )
    parser.add_argument("--ollama-probe-timeout-seconds", type=float, default=2.0)
    parser.add_argument(
        "--query-rewrite",
        action="store_true",
        help="Apply generic LLM query rewrite to hybrid and policy/fallback modes only.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    snapshot_manifest_path = args.snapshot_dir / "manifest.json"
    snapshot_manifest = (
        _load_json_file(snapshot_manifest_path) if snapshot_manifest_path.exists() else None
    )
    captures = capture_stage7_suite(
        cases_path=args.cases,
        snapshot_dir=args.snapshot_dir,
        index_dir=args.index_dir,
        modes=args.modes,
        query_rewrite_enabled=args.query_rewrite,
        ollama_probe_timeout_seconds=args.ollama_probe_timeout_seconds,
    )
    write_stage7_report(
        args.output,
        captures=captures,
        cases_path=args.cases,
        snapshot_dir=args.snapshot_dir,
        index_dir=args.index_dir or (args.snapshot_dir / "index"),
        snapshot_manifest=snapshot_manifest,
        query_rewrite_enabled=args.query_rewrite,
    )


if __name__ == "__main__":
    main()

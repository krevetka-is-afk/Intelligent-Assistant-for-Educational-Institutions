from __future__ import annotations

import re
from math import isfinite
from time import perf_counter
from typing import Any

from . import config
from .rag_evidence_models import EvidenceRerankResult
from .vector import RetrievedDocument

_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{3,}")


def _tokens(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _TOKEN_RE.finditer(text)}


def _chunk_id(retrieved: RetrievedDocument) -> str | None:
    metadata = retrieved.document.metadata if isinstance(retrieved.document.metadata, dict) else {}
    value = metadata.get("chunk_id")
    return str(value) if value is not None else None


def _heuristic_evidence_score(question: str, retrieved: RetrievedDocument) -> float:
    question_terms = _tokens(question)
    if not question_terms:
        return 0.0

    text_terms = _tokens(str(retrieved.document.page_content or ""))
    if not text_terms:
        return 0.0

    overlap = len(question_terms & text_terms) / len(question_terms)
    density = len(question_terms & text_terms) / max(1, len(text_terms))
    relevance = max(0.0, 1.0 - float(retrieved.distance))
    return round((0.68 * overlap) + (0.22 * relevance) + (0.10 * density), 8)


def _candidate_payload(retrieved: RetrievedDocument, rank: int) -> dict[str, Any]:
    metadata = retrieved.document.metadata if isinstance(retrieved.document.metadata, dict) else {}
    return {
        "rank": rank,
        "text": str(retrieved.document.page_content or ""),
        "metadata": dict(metadata),
    }


def _cross_encoder_scores(
    *,
    question: str,
    candidates: list[RetrievedDocument],
) -> tuple[list[float], str | None]:
    try:
        from .rag_context_reranker_pilot import _load_cross_encoder, score_with_cross_encoder
    except Exception as exc:
        return [], f"import_failed:{type(exc).__name__}"
    cache_info = getattr(_load_cross_encoder, "cache_info", lambda: None)()
    if cache_info is not None and getattr(cache_info, "currsize", 0) <= 0:
        return [], "cross_encoder_not_prewarmed"

    payloads = [_candidate_payload(item, rank) for rank, item in enumerate(candidates, start=1)]
    try:
        scores = list(score_with_cross_encoder(question, payloads))
    except Exception as exc:
        return [], f"score_failed:{type(exc).__name__}"
    if len(scores) != len(candidates):
        return [], "invalid_score_count"
    if not all(isinstance(score, int | float) and isfinite(float(score)) for score in scores):
        return [], "invalid_score_type"
    return [float(score) for score in scores], None


def _score_candidates(
    *,
    question: str,
    candidates: list[RetrievedDocument],
) -> tuple[list[float], str, str | None]:
    method = getattr(config, "RAG_EVIDENCE_RERANK_METHOD", "heuristic")
    if method == "cross_encoder":
        scores, fallback_reason = _cross_encoder_scores(question=question, candidates=candidates)
        if fallback_reason is None:
            return scores, "cross_encoder", None
        return [], "rrf_fallback", fallback_reason
    return (
        [_heuristic_evidence_score(question, item) for item in candidates],
        ("heuristic_overlap_v1"),
        None,
    )


def rerank_evidence_candidates(
    *,
    question: str,
    candidates: list[RetrievedDocument],
    final_k: int,
) -> EvidenceRerankResult:
    started = perf_counter()
    scores, method, fallback_reason = _score_candidates(question=question, candidates=candidates)
    if method == "rrf_fallback":
        selected = candidates[:final_k]
        diagnostics: dict[str, Any] = {
            "enabled": True,
            "method": method,
            "fallback_reason": fallback_reason,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "latency_ms": round((perf_counter() - started) * 1000),
            "candidate_ids": [_chunk_id(item) for item in candidates],
            "selected_ids": [_chunk_id(item) for item in selected],
        }
        return EvidenceRerankResult(
            selected_documents=selected,
            candidate_documents=candidates,
            diagnostics=diagnostics,
        )
    scored: list[tuple[float, int, RetrievedDocument]] = []
    for index, retrieved in enumerate(candidates, start=1):
        score = scores[index - 1]
        diagnostics = dict(retrieved._retrieval_diagnostics)
        diagnostics.update(
            {
                "evidence_candidate_rank": index,
                "evidence_rerank_score": score,
                "evidence_rerank_method": method,
            }
        )
        retrieved._retrieval_diagnostics = diagnostics
        scored.append((score, index, retrieved))

    ordered = sorted(scored, key=lambda item: (-item[0], item[1]))
    selected = [item[2] for item in ordered[:final_k]]
    for rank, retrieved in enumerate(selected, start=1):
        retrieved._retrieval_diagnostics = {
            **retrieved._retrieval_diagnostics,
            "evidence_final_rank": rank,
        }

    diagnostics: dict[str, Any] = {
        "enabled": True,
        "method": method,
        "fallback_reason": fallback_reason,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "latency_ms": round((perf_counter() - started) * 1000),
        "candidate_ids": [_chunk_id(item) for item in candidates],
        "selected_ids": [_chunk_id(item) for item in selected],
    }
    return EvidenceRerankResult(
        selected_documents=selected,
        candidate_documents=candidates,
        diagnostics=diagnostics,
    )

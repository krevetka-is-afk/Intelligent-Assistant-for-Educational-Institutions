from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from app_runtime import log_extra

from . import config, query_rewrite
from .prompt_policy import (
    PROMPT_POLICY_VERSION,
    SAFE_POLICY_REFUSAL,
    AnswerPolicyAudit,
    AnswerPolicyReason,
    AnswerPolicyResult,
    CompiledPrompt,
    PromptCompiler,
    PromptPolicyViolation,
    build_source_allowlist,
    evaluate_answer_policy,
    sanitize_known_control_marker_artifacts,
)
from .rag_evidence_judge import (
    assess_evidence_sufficiency,
    invoke_answer_judge,
    unknown_answer_judge,
)
from .rag_evidence_models import EvidenceAnswerJudgeResult
from .rag_evidence_rerank import rerank_evidence_candidates
from .rag_evidence_windows import build_structural_windows
from .vector import RetrievedDocument, get_chunks_by_ids, similarity_search

_search_lexical: Callable[..., Any] | None
try:
    from .lexical import search_lexical as _imported_search_lexical
except ImportError:  # pragma: no cover - lexical lane may be absent during partial builds
    _search_lexical = None
else:
    _search_lexical = _imported_search_lexical
search_lexical = _search_lexical

_ALLOWED_METADATA_KEYS = {
    "source",
    "title",
    "url",
    "page",
    "mime_type",
    "chunk_index",
    "char_start",
    "char_end",
    "document_id",
    "chunk_id",
    "indexed_at",
    "source_type",
    "source_size",
    "source_sha256",
    "quality_status",
    "quality_score",
    "quality_reasons",
    "quality_flags",
    "chunk_strategy",
    "chunk_fallback_reason",
    "docx_tables",
    "docx_has_tables",
    "docx_table_paragraphs",
    "docx_read_order_ambiguous",
}

_llm_chain = None
_prompt_compiler = PromptCompiler()
logger = logging.getLogger("server.rag")
_REPAIRABLE_ANSWER_POLICY_REASONS: frozenset[AnswerPolicyReason] = frozenset(
    {
        "out_of_range_source_index",
        "unverified_labeled_source",
        "unverified_url",
        "unverified_filename",
        "known_control_marker_artifact",
    }
)


def heuristic_answer_judge(**kwargs: Any) -> tuple[EvidenceAnswerJudgeResult, dict[str, object]]:
    return invoke_answer_judge(**kwargs)


def _coerce_answer_judge_output(
    output: Any,
) -> tuple[EvidenceAnswerJudgeResult, dict[str, object]]:
    if isinstance(output, tuple) and len(output) == 2:
        result, diagnostics = output
        if isinstance(result, EvidenceAnswerJudgeResult) and isinstance(diagnostics, dict):
            return result, diagnostics
    if isinstance(output, EvidenceAnswerJudgeResult):
        return output, {
            "enabled": True,
            "verdict": output.verdict,
            "evidence_ids": output.evidence_ids,
            "reason_code": output.reason,
            "missing_aspect_count": len(output.missing_aspects),
        }
    return unknown_answer_judge("invalid_judge_output")


@dataclass(slots=True)
class RAGResponse:
    answer: str
    sources: list[dict[str, Any]]
    metadata: dict[str, Any]
    retrieved_documents: list[RetrievedDocument]
    policy_audit: AnswerPolicyAudit | None = None
    retrieval_diagnostics: dict[str, Any] = field(default_factory=dict, repr=False)


def _get_llm_chain():
    global _llm_chain
    if _llm_chain is None:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_message}"),
                ("user", "{user_message}"),
            ]
        )
        model_kwargs: dict[str, Any] = {
            "model": config.LLM_MODEL,
            "base_url": config.OLLAMA_HOST,
        }
        if config.RAG_OFFLINE_GENERATION_SEED is not None:
            model_kwargs["seed"] = config.RAG_OFFLINE_GENERATION_SEED
        if config.RAG_OFFLINE_GENERATION_TEMPERATURE is not None:
            model_kwargs["temperature"] = config.RAG_OFFLINE_GENERATION_TEMPERATURE
        model = OllamaLLM(**model_kwargs)
        _llm_chain = prompt | model
    return _llm_chain


def _normalize_metadata_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def normalize_source_metadata(raw_metadata: Any) -> dict[str, Any]:
    if not isinstance(raw_metadata, dict):
        return {}

    normalized = {
        key: _normalize_metadata_value(value)
        for key, value in raw_metadata.items()
        if key in _ALLOWED_METADATA_KEYS and _normalize_metadata_value(value) is not None
    }

    source = normalized.get("source")
    title = normalized.get("title")
    if title is None and source is not None:
        normalized["title"] = source

    return normalized


def _bounded_source_content(page_content: str) -> str:
    compact = " ".join(page_content.split())
    if len(compact) <= config.RAG_SOURCE_SNIPPET_CHARS:
        return compact
    if config.RAG_SOURCE_SNIPPET_CHARS <= 3:
        return compact[: config.RAG_SOURCE_SNIPPET_CHARS]
    return compact[: config.RAG_SOURCE_SNIPPET_CHARS - 3].rstrip() + "..."


def deduplicate_sources(retrieved_documents: list[RetrievedDocument]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any, Any]] = set()
    sources: list[dict[str, Any]] = []

    for retrieved in retrieved_documents:
        metadata = normalize_source_metadata(retrieved.document.metadata)
        source_key = (
            metadata.get("source"),
            metadata.get("page"),
            metadata.get("chunk_index"),
        )
        if source_key in seen:
            continue
        seen.add(source_key)
        sources.append(
            {
                "content": _bounded_source_content(retrieved.document.page_content),
                "metadata": metadata,
            }
        )

    return sources


def compute_confidence(
    retrieved_documents: list[RetrievedDocument], *, fallback_used: bool
) -> float:
    if not retrieved_documents:
        return 0.0

    relevances = [max(0.0, 1.0 - float(item.distance)) for item in retrieved_documents]
    top1 = relevances[0]
    top3_avg = sum(relevances[:3]) / min(3, len(relevances))
    confidence = max(0.0, min(1.0, 0.6 * top1 + 0.4 * top3_avg))
    if fallback_used:
        confidence *= 0.75
    return round(max(0.0, min(1.0, confidence)), 4)


def build_context(retrieved_documents: list[RetrievedDocument]) -> str:
    context_parts: list[str] = []
    for index, retrieved in enumerate(retrieved_documents, start=1):
        metadata = normalize_source_metadata(retrieved.document.metadata)
        title = metadata.get("title") or metadata.get("source") or f"Документ {index}"
        page = metadata.get("page")
        location = f", стр. {page}" if page is not None else ""
        context_parts.append(
            f"[{index}] {title}{location}\n{retrieved.document.page_content.strip()}"
        )
    return "\n\n".join(context_parts)


def _metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if not value.isdecimal():
            return None
        return int(value)
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return integer if integer == value else None


def _trusted_anchor_metadata(
    metadata: dict[str, Any],
) -> tuple[str, str, Any, int, int, int] | None:
    document_id = metadata.get("document_id")
    source_sha256 = metadata.get("source_sha256")
    page = metadata.get("page")
    chunk_index = _metadata_int(metadata, "chunk_index")
    char_start = _metadata_int(metadata, "char_start")
    char_end = _metadata_int(metadata, "char_end")
    if (
        not isinstance(document_id, str)
        or not document_id
        or not isinstance(source_sha256, str)
        or not source_sha256
        or chunk_index is None
        or chunk_index < 0
        or char_start is None
        or char_end is None
        or char_start < 0
        or char_end <= char_start
    ):
        return None
    return document_id, source_sha256, page, chunk_index, char_start, char_end


def _chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}:{chunk_index:05d}"


def _same_chunk_version(
    metadata: dict[str, Any],
    *,
    document_id: str,
    source_sha256: str,
    page: Any,
    chunk_index: int,
) -> bool:
    return (
        metadata.get("document_id") == document_id
        and metadata.get("source_sha256") == source_sha256
        and metadata.get("page") == page
        and _metadata_int(metadata, "chunk_index") == chunk_index
    )


def _trusted_neighbor_order(
    previous_metadata: dict[str, Any],
    anchor_metadata: dict[str, Any],
    next_metadata: dict[str, Any],
) -> bool:
    previous_start = _metadata_int(previous_metadata, "char_start")
    previous_end = _metadata_int(previous_metadata, "char_end")
    anchor_start = _metadata_int(anchor_metadata, "char_start")
    anchor_end = _metadata_int(anchor_metadata, "char_end")
    next_start = _metadata_int(next_metadata, "char_start")
    next_end = _metadata_int(next_metadata, "char_end")
    if None in {
        previous_start,
        previous_end,
        anchor_start,
        anchor_end,
        next_start,
        next_end,
    }:
        return False
    assert previous_start is not None
    assert previous_end is not None
    assert anchor_start is not None
    assert anchor_end is not None
    assert next_start is not None
    assert next_end is not None
    return (
        previous_start < previous_end
        and anchor_start < anchor_end
        and next_start < next_end
        and previous_start < anchor_start
        and previous_end >= anchor_start
        and previous_end <= anchor_end
        and next_start >= anchor_start
        and anchor_end >= next_start
        and anchor_end < next_end
    )


def _prefix_suffix_overlap(left: str, right: str) -> int:
    max_length = min(len(left), len(right))
    for length in range(max_length, 0, -1):
        if left[-length:] == right[:length]:
            return length
    return 0


def _merge_three_chunks(
    previous: str,
    anchor: str,
    next_text: str,
    *,
    previous_overlap: int,
    next_overlap: int,
) -> tuple[str, int, int]:
    merged = previous + anchor[previous_overlap:]
    anchor_start = len(previous) - previous_overlap
    anchor_end = anchor_start + len(anchor)
    merged += next_text[next_overlap:]
    return merged, anchor_start, anchor_end


def _clip_preserving_anchor(
    text: str, *, anchor_start: int, anchor_end: int, limit: int
) -> str | None:
    if limit <= 0 or len(text) <= limit:
        return text
    anchor_length = anchor_end - anchor_start
    if anchor_length > limit:
        return None
    if anchor_length == limit:
        return text[anchor_start:anchor_end]

    side_budget = limit - anchor_length
    left_keep = min(anchor_start, side_budget // 2)
    right_keep = min(len(text) - anchor_end, side_budget - left_keep)
    left_keep = min(anchor_start, side_budget - right_keep)
    start = anchor_start - left_keep
    end = anchor_end + right_keep
    return text[start:end]


def _expanded_context_budget(document_count: int) -> int:
    if document_count <= 0:
        return 0
    fair_total = max(1, config.RAG_MAX_TOTAL_CONTEXT_CHARS // document_count)
    return min(config.RAG_MAX_DOCUMENT_CHARS, fair_total)


def expand_context_documents_for_generation(
    retrieved_documents: list[RetrievedDocument],
) -> list[RetrievedDocument]:
    if not config.RAG_CONTEXT_EXPANSION_ENABLED:
        return retrieved_documents

    prompt_documents = retrieved_documents[: config.RAG_MAX_CONTEXT_DOCUMENTS]
    if not prompt_documents:
        return retrieved_documents

    budget = _expanded_context_budget(len(prompt_documents))
    expanded: list[RetrievedDocument] = []
    for retrieved in prompt_documents:
        expanded.append(_expand_single_context_document(retrieved, content_limit=budget))
    if len(retrieved_documents) > len(prompt_documents):
        expanded.extend(retrieved_documents[len(prompt_documents) :])
    return expanded


def _expand_single_context_document(
    retrieved: RetrievedDocument,
    *,
    content_limit: int,
) -> RetrievedDocument:
    metadata = retrieved.document.metadata if isinstance(retrieved.document.metadata, dict) else {}
    trusted = _trusted_anchor_metadata(metadata)
    if trusted is None:
        return retrieved

    document_id, source_sha256, page, chunk_index, _char_start, _char_end = trusted
    if chunk_index == 0:
        return retrieved

    previous_id = _chunk_id(document_id, chunk_index - 1)
    expected_anchor_id = _chunk_id(document_id, chunk_index)
    anchor_id = metadata.get("chunk_id")
    if anchor_id is not None and anchor_id != expected_anchor_id:
        return retrieved
    anchor_id = expected_anchor_id
    next_id = _chunk_id(document_id, chunk_index + 1)
    try:
        chunks_by_id = get_chunks_by_ids([previous_id, anchor_id, next_id])
    except Exception:
        return retrieved

    previous = chunks_by_id.get(previous_id)
    anchor = chunks_by_id.get(anchor_id)
    next_document = chunks_by_id.get(next_id)
    if previous is None or anchor is None or next_document is None:
        return retrieved
    if str(anchor.page_content or "") != str(retrieved.document.page_content or ""):
        return retrieved

    previous_metadata = previous.metadata if isinstance(previous.metadata, dict) else {}
    anchor_metadata = anchor.metadata if isinstance(anchor.metadata, dict) else {}
    next_metadata = next_document.metadata if isinstance(next_document.metadata, dict) else {}
    if not (
        _same_chunk_version(
            previous_metadata,
            document_id=document_id,
            source_sha256=source_sha256,
            page=page,
            chunk_index=chunk_index - 1,
        )
        and _same_chunk_version(
            anchor_metadata,
            document_id=document_id,
            source_sha256=source_sha256,
            page=page,
            chunk_index=chunk_index,
        )
        and _same_chunk_version(
            next_metadata,
            document_id=document_id,
            source_sha256=source_sha256,
            page=page,
            chunk_index=chunk_index + 1,
        )
        and _trusted_neighbor_order(previous_metadata, anchor_metadata, next_metadata)
    ):
        return retrieved

    previous_text = str(previous.page_content or "")
    anchor_text = str(anchor.page_content or "")
    next_text = str(next_document.page_content or "")
    previous_overlap = _prefix_suffix_overlap(previous_text, anchor_text)
    next_overlap = _prefix_suffix_overlap(anchor_text, next_text)
    if previous_overlap == 0 or next_overlap == 0:
        return retrieved

    merged, anchor_start, anchor_end = _merge_three_chunks(
        previous_text,
        anchor_text,
        next_text,
        previous_overlap=previous_overlap,
        next_overlap=next_overlap,
    )
    clipped = _clip_preserving_anchor(
        merged,
        anchor_start=anchor_start,
        anchor_end=anchor_end,
        limit=content_limit,
    )
    if not clipped:
        return retrieved

    return RetrievedDocument(
        document=Document(
            page_content=clipped,
            metadata=dict(metadata),
            id=getattr(retrieved.document, "id", None),
        ),
        distance=retrieved.distance,
        _retrieval_diagnostics=dict(retrieved._retrieval_diagnostics),
    )


def build_conversation_history(conversation_history: list[str] | None) -> str:
    if not conversation_history:
        return "Нет."
    return "\n".join(
        f"{index}. {message}" for index, message in enumerate(conversation_history, start=1)
    )


def build_retrieval_query(question: str, conversation_history: list[str] | None) -> str:
    del conversation_history
    return question.strip()


def _get_positive_int_config(name: str, default: int) -> int:
    value = getattr(config, name, default)
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return default
    return resolved if resolved > 0 else default


def _bounded_distance(value: Any, *, fallback: float) -> float:
    try:
        distance = float(value)
    except (TypeError, ValueError):
        distance = fallback
    return max(0.0, min(1.0, distance))


def dense_similarity_search(question: str, *, k: int) -> list[RetrievedDocument]:
    return similarity_search(question, k=k)


def _chunk_key(retrieved: RetrievedDocument) -> tuple[Any, ...]:
    metadata = retrieved.document.metadata or {}
    chunk_id = metadata.get("chunk_id")
    if chunk_id is not None:
        return ("chunk_id", chunk_id)
    return (
        "chunk",
        metadata.get("document_id"),
        metadata.get("source"),
        metadata.get("page"),
        metadata.get("chunk_index"),
        retrieved.document.page_content,
    )


def _document_key(retrieved: RetrievedDocument) -> tuple[Any, ...]:
    metadata = retrieved.document.metadata or {}
    source_sha256 = metadata.get("source_sha256")
    if source_sha256 is not None:
        return ("source_sha256", source_sha256)
    document_id = metadata.get("document_id")
    if document_id is not None:
        return ("document_id", document_id)
    return ("document", metadata.get("source"), metadata.get("title"))


def _chunk_index(retrieved: RetrievedDocument) -> int | None:
    value = (retrieved.document.metadata or {}).get("chunk_index")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


QUALITY_REVIEW_RRF_MULTIPLIER = 0.75


def _quality_demoted(retrieved: RetrievedDocument) -> bool:
    metadata = retrieved.document.metadata or {}
    flags = metadata.get("quality_flags")
    has_flags = bool(flags)
    return metadata.get("quality_status") == "review" or has_flags


def _quality_score_multiplier(retrieved: RetrievedDocument) -> float:
    return QUALITY_REVIEW_RRF_MULTIPLIER if _quality_demoted(retrieved) else 1.0


def _lexical_result_to_retrieved(result: Any) -> RetrievedDocument | None:
    document = getattr(result, "document", None)
    score = getattr(result, "score", None)
    if isinstance(result, dict):
        document = result.get("document", document)
        score = result.get("score", score)
        if document is None and "content" in result:
            document = Document(
                page_content=str(result.get("content") or ""),
                metadata=dict(result.get("metadata") or {}),
            )
    if document is None:
        return None
    if not hasattr(document, "page_content") or not hasattr(document, "metadata"):
        return None
    if score is None:
        lexical_score = 0.0
    else:
        try:
            lexical_score = float(score)
        except (TypeError, ValueError):
            lexical_score = 0.0
    return RetrievedDocument(
        document=document,
        distance=_bounded_distance(1.0 - lexical_score, fallback=1.0),
        _retrieval_diagnostics={"lexical_score": lexical_score},
    )


def lexical_similarity_search(question: str, *, k: int) -> tuple[list[RetrievedDocument], bool]:
    if search_lexical is None:
        return [], False

    kwargs: dict[str, Any] = {"limit": k}
    if hasattr(config, "LEXICAL_INDEX_PATH"):
        kwargs["index_path"] = getattr(config, "LEXICAL_INDEX_PATH")

    try:
        raw_results = search_lexical(question, **kwargs)
    except Exception as exc:
        logger.warning(
            "Lexical search failed, degrading to dense-only retrieval",
            extra=log_extra(stage="retrieval", error_type=type(exc).__name__),
        )
        return [], False

    retrieved: list[RetrievedDocument] = []
    for result in raw_results:
        item = _lexical_result_to_retrieved(result)
        if item is not None:
            retrieved.append(item)
    return retrieved, True


def _is_adjacent_to_selected(
    retrieved: RetrievedDocument,
    selected_by_document: dict[tuple[Any, ...], set[int]],
) -> bool:
    current_index = _chunk_index(retrieved)
    if current_index is None:
        return False
    selected_indices = selected_by_document.get(_document_key(retrieved), set())
    return any(abs(current_index - selected_index) == 1 for selected_index in selected_indices)


def _rank_hybrid_documents(
    *,
    dense_documents: list[RetrievedDocument],
    lexical_documents: list[RetrievedDocument],
    top_k: int,
    rrf_k: int,
    max_chunks_per_document: int,
    dense_rewrite_documents: list[RetrievedDocument] | None = None,
    lexical_rewrite_documents: list[RetrievedDocument] | None = None,
) -> tuple[list[RetrievedDocument], dict[str, Any]]:
    candidates: dict[tuple[Any, ...], dict[str, Any]] = {}

    def add_candidate(channel: str, rank: int, retrieved: RetrievedDocument) -> None:
        key = _chunk_key(retrieved)
        candidate = candidates.setdefault(
            key,
            {
                "key": key,
                "document_key": _document_key(retrieved),
                "retrieved": retrieved,
                "rrf_score": 0.0,
                "channel_ranks": {},
                "channel_scores": {},
                "best_rank": rank,
            },
        )
        dense_channel = channel.startswith("dense")
        has_dense_channel = any(
            existing_channel.startswith("dense") for existing_channel in candidate["channel_ranks"]
        )
        if dense_channel:
            candidate["retrieved"] = retrieved
            score_key = "dense_distance" if channel == "dense" else f"{channel}_distance"
            candidate["channel_scores"][score_key] = float(retrieved.distance)
        elif not has_dense_channel:
            candidate["retrieved"] = retrieved
        candidate["rrf_score"] += 1.0 / (rrf_k + rank)
        candidate["channel_ranks"][channel] = rank
        candidate["best_rank"] = min(candidate["best_rank"], rank)
        if channel.startswith("lexical"):
            score = retrieved._retrieval_diagnostics.get("lexical_score")
            if score is not None:
                score_key = "lexical_score" if channel == "lexical" else f"{channel}_score"
                candidate["channel_scores"][score_key] = score

    for rank, retrieved in enumerate(dense_documents, start=1):
        add_candidate("dense", rank, retrieved)
    for rank, retrieved in enumerate(lexical_documents, start=1):
        add_candidate("lexical", rank, retrieved)
    for rank, retrieved in enumerate(dense_rewrite_documents or [], start=1):
        add_candidate("dense_rewrite", rank, retrieved)
    for rank, retrieved in enumerate(lexical_rewrite_documents or [], start=1):
        add_candidate("lexical_rewrite", rank, retrieved)

    for candidate in candidates.values():
        quality_multiplier = _quality_score_multiplier(candidate["retrieved"])
        candidate["quality_multiplier"] = quality_multiplier
        candidate["ranking_score"] = float(candidate["rrf_score"]) * quality_multiplier

    ordered_candidates = sorted(
        candidates.values(),
        key=lambda item: (
            -item["ranking_score"],
            item["best_rank"],
            repr(item["key"]),
        ),
    )

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[Any, ...]] = set()
    selected_by_document: dict[tuple[Any, ...], set[int]] = {}
    document_counts: dict[tuple[Any, ...], int] = {}

    def select(candidate: dict[str, Any]) -> None:
        retrieved = candidate["retrieved"]
        key = candidate["key"]
        document_key = candidate["document_key"]
        selected.append(candidate)
        selected_keys.add(key)
        document_counts[document_key] = document_counts.get(document_key, 0) + 1
        current_index = _chunk_index(retrieved)
        if current_index is not None:
            selected_by_document.setdefault(document_key, set()).add(current_index)

    def eligible(
        candidate: dict[str, Any],
        *,
        require_new_document: bool,
        suppress_adjacent: bool,
    ) -> bool:
        if len(selected) >= top_k or candidate["key"] in selected_keys:
            return False
        retrieved = candidate["retrieved"]
        document_key = candidate["document_key"]
        if require_new_document and document_counts.get(document_key, 0) > 0:
            return False
        if document_counts.get(document_key, 0) >= max_chunks_per_document:
            return False
        return not (suppress_adjacent and _is_adjacent_to_selected(retrieved, selected_by_document))

    for candidate in ordered_candidates:
        if eligible(candidate, require_new_document=True, suppress_adjacent=True):
            select(candidate)
    for candidate in ordered_candidates:
        if eligible(candidate, require_new_document=False, suppress_adjacent=True):
            select(candidate)
    for candidate in ordered_candidates:
        if eligible(candidate, require_new_document=False, suppress_adjacent=False):
            select(candidate)

    selected_documents: list[RetrievedDocument] = []
    selected_diagnostics: list[dict[str, Any]] = []
    max_ranking_score = max(
        (float(candidate["ranking_score"]) for candidate in selected),
        default=0.0,
    )
    for rank, candidate in enumerate(selected, start=1):
        retrieved = candidate["retrieved"]
        if any(channel.startswith("dense") for channel in candidate["channel_ranks"]):
            retrieved.distance = _bounded_distance(retrieved.distance, fallback=1.0)
        else:
            normalized_fusion_score = (
                float(candidate["ranking_score"]) / max_ranking_score
                if max_ranking_score > 0
                else 0.0
            )
            retrieved.distance = _bounded_distance(1.0 - normalized_fusion_score, fallback=1.0)
        quality_demoted = _quality_demoted(retrieved)
        diagnostics = {
            "rank": rank,
            "chunk_id": (retrieved.document.metadata or {}).get("chunk_id"),
            "document_id": (retrieved.document.metadata or {}).get("document_id"),
            "quality_flags": (retrieved.document.metadata or {}).get("quality_flags") or [],
            "quality_demoted": quality_demoted,
            "quality_score_multiplier": candidate["quality_multiplier"],
            "channels": sorted(candidate["channel_ranks"]),
            "channel_ranks": dict(candidate["channel_ranks"]),
            "channel_scores": dict(candidate["channel_scores"]),
            "rrf_score": round(float(candidate["rrf_score"]), 8),
            "ranking_score": round(float(candidate["ranking_score"]), 8),
        }
        retrieved._retrieval_diagnostics = diagnostics
        selected_documents.append(retrieved)
        selected_diagnostics.append(diagnostics)

    return selected_documents, {
        "selected": selected_diagnostics,
        "candidate_count": len(ordered_candidates),
    }


def _attach_retrieval_diagnostics(
    bounded_documents: list[RetrievedDocument],
    source_documents: list[RetrievedDocument],
) -> list[RetrievedDocument]:
    diagnostics_by_key = {
        _chunk_key(retrieved): retrieved._retrieval_diagnostics for retrieved in source_documents
    }
    for retrieved in bounded_documents:
        retrieved._retrieval_diagnostics = dict(diagnostics_by_key.get(_chunk_key(retrieved), {}))
    return bounded_documents


def retrieve_documents(
    question: str,
    *,
    k: int | None = None,
    expanded_query: str | None = None,
) -> tuple[
    list[RetrievedDocument],
    dict[str, Any],
    dict[str, Any],
]:
    top_k = k or config.RAG_TOP_K
    evidence_rerank_enabled = bool(getattr(config, "RAG_EVIDENCE_RERANK_ENABLED", False))
    evidence_candidate_top_k = _get_positive_int_config("RAG_EVIDENCE_CANDIDATE_TOP_K", 16)
    evidence_final_top_k = _get_positive_int_config("RAG_EVIDENCE_FINAL_TOP_K", top_k)
    ranking_top_k = max(top_k, evidence_candidate_top_k) if evidence_rerank_enabled else top_k
    candidate_pool_size = max(
        ranking_top_k,
        _get_positive_int_config("RAG_CANDIDATE_POOL_SIZE", max(top_k * 4, top_k)),
    )
    rrf_k = _get_positive_int_config("RAG_RRF_K", 60)
    max_chunks_per_document = _get_positive_int_config("RAG_MAX_CHUNKS_PER_DOCUMENT", 1)
    normalized_expanded_query = (expanded_query or "").strip()
    use_expanded_query = bool(
        normalized_expanded_query
        and normalized_expanded_query.casefold() != question.strip().casefold()
    )
    retrieval_mode = getattr(config, "RAG_RETRIEVAL_MODE", "hybrid")
    primary_dense_mode = retrieval_mode == "primary_dense"

    dense_documents = dense_similarity_search(question, k=candidate_pool_size)
    if primary_dense_mode:
        lexical_documents: list[RetrievedDocument] = []
        lexical_available = False
    else:
        lexical_documents, lexical_available = lexical_similarity_search(
            question,
            k=candidate_pool_size,
        )
    expanded_dense_documents: list[RetrievedDocument] = []
    expanded_lexical_documents: list[RetrievedDocument] = []
    expanded_lexical_available = False
    expanded_search_failed = False
    expanded_search_error_type: str | None = None
    expanded_search_error_stage: str | None = None
    if use_expanded_query:
        try:
            expanded_dense_documents = dense_similarity_search(
                normalized_expanded_query,
                k=candidate_pool_size,
            )
            if not primary_dense_mode:
                expanded_lexical_documents, expanded_lexical_available = lexical_similarity_search(
                    normalized_expanded_query,
                    k=candidate_pool_size,
                )
        except Exception as exc:
            expanded_search_failed = True
            expanded_search_error_type = type(exc).__name__
            expanded_search_error_stage = "dense" if not expanded_dense_documents else "lexical"
            expanded_dense_documents = []
            expanded_lexical_documents = []
            expanded_lexical_available = False
            logger.warning(
                "Expanded retrieval failed, degrading to original-query candidates",
                extra=log_extra(
                    stage="retrieval",
                    error_type=expanded_search_error_type,
                    retrieval_lane="expanded",
                    retrieval_stage=expanded_search_error_stage,
                ),
            )

    all_dense_documents = dense_documents + expanded_dense_documents
    all_lexical_documents = lexical_documents + expanded_lexical_documents
    retrieved_documents, diagnostics = _rank_hybrid_documents(
        dense_documents=dense_documents,
        lexical_documents=lexical_documents,
        dense_rewrite_documents=expanded_dense_documents,
        lexical_rewrite_documents=expanded_lexical_documents,
        top_k=ranking_top_k,
        rrf_k=rrf_k,
        max_chunks_per_document=max_chunks_per_document,
    )
    candidate_top_documents = retrieved_documents[:evidence_candidate_top_k]
    diagnostics["top16"] = [dict(item._retrieval_diagnostics) for item in candidate_top_documents]
    if evidence_rerank_enabled:
        diagnostics["_candidate_top16_documents_internal"] = list(candidate_top_documents)
        if getattr(config, "RAG_EVIDENCE_OFFLINE_CAPTURE_ENABLED", False):
            top16_payload = _documents_payload(candidate_top_documents)
            diagnostics["top16_documents"] = top16_payload
            diagnostics["candidate_top16_documents"] = top16_payload
            diagnostics["rrf_top16_documents"] = top16_payload
        try:
            rerank_result = rerank_evidence_candidates(
                question=question,
                candidates=retrieved_documents[:evidence_candidate_top_k],
                final_k=min(top_k, evidence_final_top_k),
            )
            retrieved_documents = rerank_result.selected_documents
            diagnostics["evidence_rerank"] = rerank_result.diagnostics
        except Exception as exc:
            diagnostics["evidence_rerank"] = {
                "enabled": True,
                "fallback": "rrf_top_k",
                "error_type": type(exc).__name__,
            }
            retrieved_documents = retrieved_documents[:top_k]
    if primary_dense_mode:
        strategy = "primary_dense_ranker"
    else:
        strategy = "hybrid" if all_lexical_documents else "dense_only"
    query_count = 2 if use_expanded_query else 1

    retrieval_metadata = {
        "retrieval_strategy": strategy,
        "retrieval_candidate_pool_size": candidate_pool_size,
        "retrieval_dense_candidate_count": len(all_dense_documents),
        "retrieval_lexical_candidate_count": len(all_lexical_documents),
        "retrieval_lexical_available": lexical_available or expanded_lexical_available,
        "retrieval_query_count": query_count,
        "retrieval_rewrite_used": use_expanded_query,
        "retrieval_original_dense_candidate_count": len(dense_documents),
        "retrieval_original_lexical_candidate_count": len(lexical_documents),
        "retrieval_expanded_dense_candidate_count": len(expanded_dense_documents),
        "retrieval_expanded_lexical_candidate_count": len(expanded_lexical_documents),
        "retrieval_expanded_search_failed": expanded_search_failed,
        "retrieval_expanded_search_error_type": expanded_search_error_type,
        "retrieval_expanded_search_error_stage": expanded_search_error_stage,
        "rag_evidence_rerank_enabled": evidence_rerank_enabled,
        "rag_evidence_candidate_top_k": evidence_candidate_top_k,
        "rag_evidence_final_top_k": evidence_final_top_k,
    }
    diagnostics.update(
        {
            "strategy": strategy,
            "candidate_pool_size": candidate_pool_size,
            "rrf_k": rrf_k,
            "max_chunks_per_document": max_chunks_per_document,
            "lexical_available": lexical_available or expanded_lexical_available,
            "query_count": query_count,
            "rewrite_used": use_expanded_query,
            "original_dense_candidate_count": len(dense_documents),
            "original_lexical_candidate_count": len(lexical_documents),
            "expanded_dense_candidate_count": len(expanded_dense_documents),
            "expanded_lexical_candidate_count": len(expanded_lexical_documents),
            "expanded_search_failed": expanded_search_failed,
            "expanded_search_error_type": expanded_search_error_type,
            "expanded_search_error_stage": expanded_search_error_stage,
        }
    )
    return retrieved_documents, retrieval_metadata, diagnostics


def invoke_llm(
    question: str,
    retrieved_documents: list[RetrievedDocument],
    conversation_history: list[str] | None = None,
) -> str:
    compiled_prompt = _prompt_compiler.compile(
        question=question,
        retrieved_documents=retrieved_documents,
        conversation_history=conversation_history,
    )
    return invoke_llm_with_prompt(compiled_prompt)


def compile_generation_prompt(
    *,
    question: str,
    retrieved_documents: list[RetrievedDocument],
    conversation_history: list[str] | None = None,
) -> CompiledPrompt:
    prompt_documents = expand_context_documents_for_generation(retrieved_documents)
    return _prompt_compiler.compile(
        question=question,
        retrieved_documents=prompt_documents,
        conversation_history=conversation_history,
    )


def invoke_llm_with_prompt(compiled_prompt: CompiledPrompt) -> str:
    chain = _get_llm_chain()
    system_message = compiled_prompt.messages[0][1]
    user_message = compiled_prompt.messages[1][1]
    response = chain.invoke(
        {
            "system_message": system_message,
            "user_message": user_message,
        }
    )
    return str(response).strip()


def build_empty_answer() -> str:
    return "Не удалось найти релевантные документы по этому вопросу."


def _build_document_fallback_answer(
    retrieved_documents: list[RetrievedDocument], *, prefix: str
) -> str:
    snippets: list[str] = []
    for index, retrieved in enumerate(retrieved_documents[:4], start=1):
        compact = " ".join(retrieved.document.page_content.split())
        snippet = compact[:260].rstrip()
        if len(compact) > 260:
            snippet += "..."
        snippets.append(f"{index}. {snippet}")

    if not snippets:
        return build_empty_answer()

    return prefix + "\n\n" + "\n\n".join(snippets)


def build_fallback_answer(retrieved_documents: list[RetrievedDocument]) -> str:
    return _build_document_fallback_answer(
        retrieved_documents,
        prefix=(
            "LLM временно недоступна, поэтому показываю наиболее релевантные фрагменты "
            "из найденных документов."
        ),
    )


def build_policy_output_fallback_answer(retrieved_documents: list[RetrievedDocument]) -> str:
    return _build_document_fallback_answer(
        retrieved_documents,
        prefix=(
            "Не удалось подтвердить ссылку в сгенерированном ответе, поэтому показываю "
            "нейтральную выдержку из найденных документов."
        ),
    )


def _requires_safe_policy_refusal(policy_result: AnswerPolicyResult) -> bool:
    return "control_marker_leak" in policy_result.reasons


def _allows_policy_repair(policy_result: AnswerPolicyResult) -> bool:
    reasons = set(policy_result.reasons)
    return bool(reasons) and reasons <= _REPAIRABLE_ANSWER_POLICY_REASONS


def _try_sanitize_known_control_marker(
    answer: str,
    *,
    source_count: int,
    source_allowlist: set[str],
) -> tuple[str, AnswerPolicyResult, dict[str, Any]]:
    sanitization = sanitize_known_control_marker_artifacts(answer)
    diagnostics = {
        "attempted": True,
        "changed": sanitization.changed,
        "skipped_reason": sanitization.skipped_reason,
    }
    if not sanitization.changed:
        policy_result = evaluate_answer_policy(
            answer,
            source_count=source_count,
            source_allowlist=source_allowlist,
        )
        return answer, policy_result, diagnostics

    policy_result = evaluate_answer_policy(
        sanitization.sanitized_answer,
        source_count=source_count,
        source_allowlist=source_allowlist,
    )
    diagnostics["succeeded"] = not policy_result.violated
    return sanitization.sanitized_answer, policy_result, diagnostics


def _build_repair_question(question: str) -> str:
    return (
        "Повтори ответ на вопрос пользователя, используя только переданные найденные "
        "документы. Удали неподтвержденные ссылки, URL, имена файлов и источники. "
        "Если нужны ссылки, используй только номера существующих источников в формате "
        "[1], [2] и так далее.\n\n"
        f"Вопрос пользователя: {question}"
    )


def _build_evidence_retry_question(question: str, missing_aspects: list[str]) -> str:
    missing = ", ".join(missing_aspects[:6]) if missing_aspects else "неполные аспекты ответа"
    return (
        "Повтори ответ на вопрос пользователя. Используй только переданные найденные "
        "документы и явно не добавляй сведения, которых нет в контексте. Уточни или "
        f"исправь неполные части: {missing}.\n\n"
        f"Вопрос пользователя: {question}"
    )


_EVIDENCE_TERM_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{3,}")
_RAW_DIAGNOSTIC_KEYS = {
    "actual_prompt_documents",
    "final_top4",
    "top16_documents",
    "candidate_top16_documents",
    "rrf_top16_documents",
    "initial_actual_prompt_documents",
    "initial_final_top4",
    "retry_actual_prompt_documents",
    "retry_final_top4",
    "evidence_answer_judge_private",
}


def _evidence_terms(values: list[str] | tuple[str, ...] | set[str] | str | None) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        text = values
    else:
        text = " ".join(str(value) for value in values)
    return {match.group(0).casefold() for match in _EVIDENCE_TERM_RE.finditer(text)}


def _chunk_identifier(retrieved: RetrievedDocument) -> str:
    metadata = retrieved.document.metadata if isinstance(retrieved.document.metadata, dict) else {}
    chunk_id = metadata.get("chunk_id") or metadata.get("evidence_window_anchor_chunk_id")
    if chunk_id is not None:
        return str(chunk_id)
    return "|".join(
        str(metadata.get(key) or "")
        for key in ("document_id", "page", "chunk_index", "char_start", "char_end")
    )


def _candidate_top16_documents(
    retrieval_diagnostics: dict[str, Any],
    fallback_documents: list[RetrievedDocument],
) -> list[RetrievedDocument]:
    value = retrieval_diagnostics.get("_candidate_top16_documents_internal")
    if isinstance(value, list) and all(isinstance(item, RetrievedDocument) for item in value):
        return list(value)
    return list(fallback_documents)


def _compile_evidence_documents(
    *,
    question: str,
    selected_documents: list[RetrievedDocument],
    conversation_history: list[str] | None,
    structural_windows_enabled: bool,
) -> tuple[list[RetrievedDocument], list[RetrievedDocument], dict[str, Any] | None]:
    if structural_windows_enabled:
        window_result = build_structural_windows(selected_documents, question=question)
        compiled_prompt = _prompt_compiler.compile(
            question=question,
            retrieved_documents=window_result.prompt_documents,
            conversation_history=conversation_history,
        )
        bounded = _attach_retrieval_diagnostics(
            list(compiled_prompt.retrieved_documents),
            window_result.prompt_documents,
        )
        return bounded, bounded, window_result.diagnostics

    compiled_prompt = _prompt_compiler.compile(
        question=question,
        retrieved_documents=selected_documents,
        conversation_history=conversation_history,
    )
    bounded = _attach_retrieval_diagnostics(
        list(compiled_prompt.retrieved_documents),
        selected_documents,
    )
    return bounded, bounded, None


def _select_alternate_evidence_documents(
    *,
    question: str,
    missing_aspects: list[str],
    current_documents: list[RetrievedDocument],
    retrieval_diagnostics: dict[str, Any],
    conversation_history: list[str] | None,
    structural_windows_enabled: bool,
) -> tuple[list[RetrievedDocument], list[RetrievedDocument], dict[str, Any]]:
    candidates = _candidate_top16_documents(retrieval_diagnostics, current_documents)
    current_ids = {_chunk_identifier(item) for item in current_documents}
    aspect_terms = _evidence_terms(missing_aspects)
    question_terms = _evidence_terms(question)
    scoring_terms = aspect_terms or question_terms
    if not candidates or not scoring_terms:
        return (
            current_documents,
            current_documents,
            {
                "status": "unchanged",
                "reason": "no_alternate_terms",
                "candidate_count": len(candidates),
            },
        )

    scored: list[tuple[int, int, int, RetrievedDocument]] = []
    for index, candidate in enumerate(candidates):
        text_terms = _evidence_terms(str(candidate.document.page_content or ""))
        missing_overlap = len(aspect_terms & text_terms) if aspect_terms else 0
        query_overlap = len(question_terms & text_terms) if question_terms else 0
        is_current = int(_chunk_identifier(candidate) in current_ids)
        scored.append((missing_overlap, query_overlap, -is_current, candidate))

    ordered = [item[3] for item in sorted(scored, key=lambda item: item[:3], reverse=True)]
    selected: list[RetrievedDocument] = []
    seen: set[str] = set()
    max_documents = min(
        _get_positive_int_config("RAG_EVIDENCE_FINAL_TOP_K", config.RAG_TOP_K),
        max(1, config.RAG_TOP_K),
    )
    for candidate in ordered:
        identifier = _chunk_identifier(candidate)
        if identifier in seen:
            continue
        selected.append(candidate)
        seen.add(identifier)
        if len(selected) >= max_documents:
            break

    selected_ids = {_chunk_identifier(item) for item in selected}
    changed = selected_ids != current_ids and bool(selected_ids - current_ids)
    if not changed:
        return (
            current_documents,
            current_documents,
            {
                "status": "unchanged",
                "reason": "no_different_evidence",
                "candidate_count": len(candidates),
                "selected_ids": sorted(selected_ids),
            },
        )

    augmented_question = question
    if missing_aspects:
        augmented_question = question + "\n" + "\n".join(missing_aspects[:6])
    bounded, generation, window_diagnostics = _compile_evidence_documents(
        question=augmented_question,
        selected_documents=selected,
        conversation_history=conversation_history,
        structural_windows_enabled=structural_windows_enabled,
    )
    diagnostics: dict[str, Any] = {
        "status": "reselected",
        "candidate_count": len(candidates),
        "selected_ids": [_chunk_identifier(item) for item in selected],
        "previous_ids": sorted(current_ids),
        "missing_aspect_count": len(missing_aspects),
    }
    if window_diagnostics is not None:
        diagnostics["structural_windows"] = window_diagnostics
    return bounded, generation, diagnostics


def _maybe_store_raw_diagnostics(
    retrieval_diagnostics: dict[str, Any],
    key: str,
    value: Any,
) -> None:
    if getattr(config, "RAG_EVIDENCE_OFFLINE_CAPTURE_ENABLED", False):
        retrieval_diagnostics[key] = value


def _store_final_context_diagnostics(
    retrieval_diagnostics: dict[str, Any],
    *,
    generation_documents: list[RetrievedDocument],
    bounded_documents: list[RetrievedDocument],
) -> None:
    _maybe_store_raw_diagnostics(
        retrieval_diagnostics,
        "actual_prompt_documents",
        _documents_payload(generation_documents),
    )
    _maybe_store_raw_diagnostics(
        retrieval_diagnostics,
        "final_top4",
        _documents_payload(bounded_documents),
    )


def _finalize_retrieval_diagnostics(
    retrieval_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    finalized = {
        key: value for key, value in retrieval_diagnostics.items() if not str(key).startswith("_")
    }
    if not getattr(config, "RAG_EVIDENCE_OFFLINE_CAPTURE_ENABLED", False):
        for key in _RAW_DIAGNOSTIC_KEYS:
            finalized.pop(key, None)
    return finalized


def _documents_payload(retrieved_documents: list[RetrievedDocument]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for index, retrieved in enumerate(retrieved_documents, start=1):
        metadata = (
            retrieved.document.metadata if isinstance(retrieved.document.metadata, dict) else {}
        )
        payloads.append(
            {
                "rank": index,
                "text": str(retrieved.document.page_content or ""),
                "metadata": dict(metadata),
                "distance": float(retrieved.distance),
                "diagnostics": dict(retrieved._retrieval_diagnostics),
            }
        )
    return payloads


def _policy_metadata(
    *,
    sources: list[dict[str, Any]],
    retrieved_documents: list[RetrievedDocument],
    fallback_used: bool,
    fallback_reason: str | None,
    retrieval_elapsed: float,
    generation_elapsed: float,
    total_elapsed: float,
    policy_result: AnswerPolicyResult | None = None,
    policy_repair_attempted: bool = False,
    policy_repair_succeeded: bool = False,
    policy_repair_skipped_reason: str | None = None,
    policy_marker_sanitization: dict[str, Any] | None = None,
    retrieval_metadata: dict[str, Any] | None = None,
    evidence_sufficiency: dict[str, Any] | None = None,
    evidence_answer_judge: dict[str, Any] | None = None,
    evidence_retry_attempted: bool = False,
    evidence_retry_succeeded: bool = False,
    evidence_retry_skipped_reason: str | None = None,
) -> dict[str, Any]:
    metadata = {
        "model": config.LLM_MODEL,
        "embedding_model": config.HF_EMBEDDING_MODEL,
        "policy_version": PROMPT_POLICY_VERSION,
        "num_sources": len(sources),
        "confidence": compute_confidence(retrieved_documents, fallback_used=fallback_used),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "retrieval_time_ms": round(retrieval_elapsed * 1000),
        "generation_time_ms": round(generation_elapsed * 1000),
        "total_time_ms": round(total_elapsed * 1000),
        "policy_output_violation_reason": (
            policy_result.primary_reason if policy_result is not None else None
        ),
        "policy_output_violation_reasons": (
            list(policy_result.reasons) if policy_result is not None else []
        ),
        "policy_output_violation_match_counts": (
            policy_result.match_counts if policy_result is not None else {}
        ),
        "policy_output_repair_attempted": policy_repair_attempted,
        "policy_output_repair_succeeded": policy_repair_succeeded,
        "policy_output_repair_skipped_reason": policy_repair_skipped_reason,
        "policy_output_marker_sanitization_attempted": (
            bool(policy_marker_sanitization) if policy_marker_sanitization is not None else False
        ),
        "policy_output_marker_sanitization_changed": (
            policy_marker_sanitization.get("changed") if policy_marker_sanitization else False
        ),
        "policy_output_marker_sanitization_skipped_reason": (
            policy_marker_sanitization.get("skipped_reason") if policy_marker_sanitization else None
        ),
        "rag_evidence_sufficiency_status": (
            evidence_sufficiency.get("status") if evidence_sufficiency else None
        ),
        "rag_evidence_answer_verdict": (
            evidence_answer_judge.get("verdict") if evidence_answer_judge else None
        ),
        "rag_evidence_retry_attempted": evidence_retry_attempted,
        "rag_evidence_retry_succeeded": evidence_retry_succeeded,
        "rag_evidence_retry_skipped_reason": evidence_retry_skipped_reason,
    }
    if retrieval_metadata is not None:
        metadata.update(retrieval_metadata)
    return metadata


def build_policy_refusal(
    *,
    reason: str,
    retrieval_elapsed: float,
    total_elapsed: float,
) -> RAGResponse:
    return RAGResponse(
        answer=SAFE_POLICY_REFUSAL,
        sources=[],
        metadata=_policy_metadata(
            sources=[],
            retrieved_documents=[],
            fallback_used=True,
            fallback_reason=reason,
            retrieval_elapsed=retrieval_elapsed,
            generation_elapsed=0,
            total_elapsed=total_elapsed,
        ),
        retrieved_documents=[],
        retrieval_diagnostics={},
    )


async def ask_question(question: str, conversation_history: list[str] | None = None) -> RAGResponse:
    total_started = perf_counter()

    try:
        _prompt_compiler.compile(
            question=question,
            retrieved_documents=[],
            conversation_history=conversation_history,
        )
    except PromptPolicyViolation as exc:
        total_elapsed = perf_counter() - total_started
        logger.info(
            "RAG request rejected by prompt policy",
            extra=log_extra(stage="policy", error_type=exc.reason),
        )
        return build_policy_refusal(
            reason=exc.reason,
            retrieval_elapsed=0,
            total_elapsed=total_elapsed,
        )

    retrieval_query = build_retrieval_query(question, conversation_history)
    retrieval_started = perf_counter()
    rewrite_result = await asyncio.to_thread(
        query_rewrite.rewrite_retrieval_query,
        retrieval_query,
        conversation_history,
    )
    retrieved_documents, retrieval_metadata, retrieval_diagnostics = await asyncio.to_thread(
        retrieve_documents,
        retrieval_query,
        k=config.RAG_TOP_K,
        expanded_query=rewrite_result.query if rewrite_result.used else None,
    )
    retrieval_elapsed = perf_counter() - retrieval_started
    rewrite_metadata = {
        "query_rewrite_used": rewrite_result.used,
        "query_rewrite_fallback_reason": rewrite_result.fallback_reason,
        "query_rewrite_history_used": rewrite_result.history_used,
    }
    retrieval_metadata.update(rewrite_metadata)
    retrieval_diagnostics["query_rewrite"] = dict(rewrite_result.diagnostics)

    if not retrieved_documents:
        total_elapsed = perf_counter() - total_started
        return RAGResponse(
            answer=build_empty_answer(),
            sources=[],
            metadata={
                "model": config.LLM_MODEL,
                "embedding_model": config.HF_EMBEDDING_MODEL,
                "policy_version": PROMPT_POLICY_VERSION,
                "num_sources": 0,
                "confidence": 0.0,
                "fallback_used": False,
                "fallback_reason": None,
                "retrieval_time_ms": round(retrieval_elapsed * 1000),
                "generation_time_ms": 0,
                "total_time_ms": round(total_elapsed * 1000),
                **retrieval_metadata,
            },
            retrieved_documents=retrieved_documents,
            retrieval_diagnostics=_finalize_retrieval_diagnostics(retrieval_diagnostics),
        )

    structural_windows_enabled = bool(
        getattr(config, "RAG_EVIDENCE_STRUCTURAL_WINDOW_ENABLED", False)
    )
    bounded_documents, generation_documents, window_diagnostics = _compile_evidence_documents(
        question=question,
        selected_documents=retrieved_documents,
        conversation_history=conversation_history,
        structural_windows_enabled=structural_windows_enabled,
    )
    if window_diagnostics is not None:
        retrieval_diagnostics["evidence_structural_windows"] = window_diagnostics

    if not structural_windows_enabled and config.RAG_CONTEXT_EXPANSION_ENABLED:
        expanded_documents = expand_context_documents_for_generation(retrieved_documents)
        if any(
            expanded is not original
            for expanded, original in zip(expanded_documents, retrieved_documents)
        ):
            generation_prompt = _prompt_compiler.compile(
                question=question,
                retrieved_documents=expanded_documents,
                conversation_history=conversation_history,
            )
            generation_documents = list(generation_prompt.retrieved_documents)
    sources = deduplicate_sources(bounded_documents)
    source_allowlist = build_source_allowlist(sources)
    _store_final_context_diagnostics(
        retrieval_diagnostics,
        generation_documents=generation_documents,
        bounded_documents=bounded_documents,
    )
    evidence_sufficiency: dict[str, Any] | None = None
    evidence_answer_judge: dict[str, Any] | None = None
    evidence_retry_attempted = False
    evidence_retry_succeeded = False
    evidence_retry_skipped_reason: str | None = None
    pre_generation_abstained = False
    if getattr(config, "RAG_EVIDENCE_SUFFICIENCY_ENABLED", False):
        sufficiency_result = assess_evidence_sufficiency(
            question=question,
            prompt_documents=generation_documents,
        )
        evidence_sufficiency = asdict(sufficiency_result)
        retrieval_diagnostics["evidence_sufficiency"] = evidence_sufficiency
        if sufficiency_result.status in {"partial", "none"}:
            alternate_bounded, alternate_generation, alternate_diagnostics = (
                _select_alternate_evidence_documents(
                    question=question,
                    missing_aspects=sufficiency_result.missing_aspects,
                    current_documents=bounded_documents,
                    retrieval_diagnostics=retrieval_diagnostics,
                    conversation_history=conversation_history,
                    structural_windows_enabled=structural_windows_enabled,
                )
            )
            retrieval_diagnostics["evidence_sufficiency_action"] = alternate_diagnostics
            if alternate_diagnostics.get("status") == "reselected":
                alternate_sufficiency = assess_evidence_sufficiency(
                    question=question,
                    prompt_documents=alternate_generation,
                )
                retrieval_diagnostics["evidence_sufficiency_after_action"] = asdict(
                    alternate_sufficiency
                )
                if alternate_sufficiency.status == "sufficient":
                    bounded_documents = alternate_bounded
                    generation_documents = alternate_generation
                    sources = deduplicate_sources(bounded_documents)
                    source_allowlist = build_source_allowlist(sources)
                    evidence_sufficiency = asdict(alternate_sufficiency)
                    retrieval_diagnostics["evidence_sufficiency"] = evidence_sufficiency
                    _store_final_context_diagnostics(
                        retrieval_diagnostics,
                        generation_documents=generation_documents,
                        bounded_documents=bounded_documents,
                    )
                else:
                    pre_generation_abstained = True
            else:
                pre_generation_abstained = True
            if pre_generation_abstained:
                retrieval_diagnostics["evidence_retrieval_miss"] = {
                    "status": "abstained",
                    "reason": "evidence_insufficient",
                    "sufficiency_status": sufficiency_result.status,
                }
    generation_elapsed = 0.0
    fallback_used = False
    fallback_reason: str | None = None
    policy_result: AnswerPolicyResult | None = None
    policy_repair_attempted = False
    policy_repair_succeeded = False
    policy_repair_skipped_reason: str | None = None
    policy_marker_sanitization: dict[str, Any] | None = None

    remaining_budget = max(0.0, config.RAG_TOTAL_TIMEOUT_SECONDS - retrieval_elapsed)
    llm_timeout = min(config.LLM_TIMEOUT_SECONDS, remaining_budget)

    if pre_generation_abstained:
        fallback_used = True
        fallback_reason = "evidence_insufficient"
        answer = build_empty_answer()
    elif llm_timeout <= 0:
        fallback_used = True
        fallback_reason = "rag_timeout_budget_exhausted"
        answer = build_fallback_answer(bounded_documents)
    else:
        generation_started = perf_counter()
        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(
                    invoke_llm,
                    question,
                    generation_documents,
                    conversation_history,
                ),
                timeout=llm_timeout,
            )
            if not answer:
                answer = build_empty_answer()
            else:
                policy_result = evaluate_answer_policy(
                    answer,
                    source_count=len(sources),
                    source_allowlist=source_allowlist,
                )
                if policy_result.violated:
                    marker_repair_enabled = bool(
                        getattr(config, "RAG_EVIDENCE_MARKER_REPAIR_ENABLED", False)
                    )
                    if marker_repair_enabled and set(policy_result.reasons) == {
                        "known_control_marker_artifact"
                    }:
                        answer, policy_result, policy_marker_sanitization = (
                            _try_sanitize_known_control_marker(
                                answer,
                                source_count=len(sources),
                                source_allowlist=source_allowlist,
                            )
                        )
                    if not policy_result.violated:
                        pass
                    elif (
                        not marker_repair_enabled
                        and "known_control_marker_artifact" in policy_result.reasons
                    ):
                        fallback_used = True
                        fallback_reason = "policy_output_violation"
                        policy_repair_skipped_reason = "marker_repair_disabled"
                        answer = SAFE_POLICY_REFUSAL
                    elif _requires_safe_policy_refusal(policy_result):
                        fallback_used = True
                        fallback_reason = "policy_output_violation"
                        policy_repair_skipped_reason = "unsafe_policy_reason"
                        answer = SAFE_POLICY_REFUSAL
                    elif _allows_policy_repair(policy_result):
                        elapsed_after_first_generation = perf_counter() - generation_started
                        repair_remaining_budget = max(
                            0.0,
                            config.RAG_TOTAL_TIMEOUT_SECONDS
                            - retrieval_elapsed
                            - elapsed_after_first_generation,
                        )
                        repair_timeout = min(config.LLM_TIMEOUT_SECONDS, repair_remaining_budget)

                        if repair_timeout <= 0:
                            fallback_used = True
                            fallback_reason = "policy_output_violation"
                            policy_repair_skipped_reason = "total_budget_exhausted"
                            answer = build_policy_output_fallback_answer(bounded_documents)
                        else:
                            policy_repair_attempted = True
                            repair_answer = await asyncio.wait_for(
                                asyncio.to_thread(
                                    invoke_llm,
                                    _build_repair_question(question),
                                    generation_documents,
                                    conversation_history,
                                ),
                                timeout=repair_timeout,
                            )
                            repair_answer = repair_answer or build_empty_answer()
                            repair_policy_result = evaluate_answer_policy(
                                repair_answer,
                                source_count=len(sources),
                                source_allowlist=source_allowlist,
                            )
                            policy_result = repair_policy_result
                            if repair_policy_result.violated:
                                fallback_used = True
                                fallback_reason = "policy_output_violation"
                                policy_repair_skipped_reason = (
                                    "unsafe_policy_reason"
                                    if _requires_safe_policy_refusal(repair_policy_result)
                                    else None
                                )
                                answer = (
                                    SAFE_POLICY_REFUSAL
                                    if _requires_safe_policy_refusal(repair_policy_result)
                                    else build_policy_output_fallback_answer(bounded_documents)
                                )
                            else:
                                policy_repair_succeeded = True
                                answer = repair_answer
                    else:
                        fallback_used = True
                        fallback_reason = "policy_output_violation"
                        policy_repair_skipped_reason = "unsupported_policy_reason"
                        answer = build_policy_output_fallback_answer(bounded_documents)
                if not fallback_used and getattr(config, "RAG_EVIDENCE_SUFFICIENCY_ENABLED", False):
                    judge_timeout = min(
                        config.RAG_EVIDENCE_JUDGE_TIMEOUT_SECONDS,
                        max(
                            0.0,
                            config.RAG_TOTAL_TIMEOUT_SECONDS
                            - retrieval_elapsed
                            - (perf_counter() - generation_started),
                        ),
                    )
                    if judge_timeout <= 0:
                        judge_result, judge_diagnostics = unknown_answer_judge(
                            "total_budget_exhausted"
                        )
                    else:
                        try:
                            judge_result, judge_diagnostics = await asyncio.wait_for(
                                asyncio.to_thread(
                                    lambda: _coerce_answer_judge_output(
                                        heuristic_answer_judge(
                                            question=question,
                                            answer=answer,
                                            prompt_documents=generation_documents,
                                            timeout_seconds=judge_timeout,
                                        )
                                    )
                                ),
                                timeout=judge_timeout,
                            )
                        except asyncio.TimeoutError:
                            judge_result, judge_diagnostics = unknown_answer_judge("timeout")
                        except Exception as exc:
                            judge_result, judge_diagnostics = unknown_answer_judge(
                                f"error:{type(exc).__name__}"
                            )
                    evidence_answer_judge = dict(judge_diagnostics)
                    retrieval_diagnostics["evidence_answer_judge"] = evidence_answer_judge
                    _maybe_store_raw_diagnostics(
                        retrieval_diagnostics,
                        "evidence_answer_judge_private",
                        asdict(judge_result),
                    )
                    if judge_result.verdict in {"incomplete", "unsupported"} and getattr(
                        config, "RAG_EVIDENCE_RETRY_ENABLED", False
                    ):
                        retry_missing_aspects = list(judge_result.missing_aspects)
                        if not retry_missing_aspects and evidence_sufficiency:
                            raw_missing = evidence_sufficiency.get("missing_aspects")
                            if isinstance(raw_missing, list):
                                retry_missing_aspects = [str(item) for item in raw_missing]
                        (
                            retry_bounded_documents,
                            retry_generation_documents,
                            retry_evidence_diagnostics,
                        ) = _select_alternate_evidence_documents(
                            question=question,
                            missing_aspects=retry_missing_aspects,
                            current_documents=bounded_documents,
                            retrieval_diagnostics=retrieval_diagnostics,
                            conversation_history=conversation_history,
                            structural_windows_enabled=structural_windows_enabled,
                        )
                        retrieval_diagnostics["evidence_retry_evidence_selection"] = (
                            retry_evidence_diagnostics
                        )
                        if retry_evidence_diagnostics.get("status") != "reselected":
                            evidence_retry_skipped_reason = str(
                                retry_evidence_diagnostics.get("reason") or "no_alternate_evidence"
                            )
                            fallback_used = True
                            fallback_reason = "evidence_retry_incomplete"
                            answer = build_policy_output_fallback_answer(bounded_documents)
                        else:
                            elapsed_before_retry = perf_counter() - generation_started
                            retry_remaining_budget = max(
                                0.0,
                                config.RAG_TOTAL_TIMEOUT_SECONDS
                                - retrieval_elapsed
                                - elapsed_before_retry,
                            )
                            retry_timeout = min(config.LLM_TIMEOUT_SECONDS, retry_remaining_budget)
                            retry_sources = deduplicate_sources(retry_bounded_documents)
                            retry_source_allowlist = build_source_allowlist(retry_sources)
                            if retry_timeout <= 0:
                                evidence_retry_skipped_reason = "total_budget_exhausted"
                                fallback_used = True
                                fallback_reason = "evidence_retry_incomplete"
                                bounded_documents = retry_bounded_documents
                                generation_documents = retry_generation_documents
                                sources = retry_sources
                                source_allowlist = retry_source_allowlist
                                _store_final_context_diagnostics(
                                    retrieval_diagnostics,
                                    generation_documents=generation_documents,
                                    bounded_documents=bounded_documents,
                                )
                                answer = build_policy_output_fallback_answer(bounded_documents)
                            else:
                                evidence_retry_attempted = True
                                _maybe_store_raw_diagnostics(
                                    retrieval_diagnostics,
                                    "initial_actual_prompt_documents",
                                    _documents_payload(generation_documents),
                                )
                                _maybe_store_raw_diagnostics(
                                    retrieval_diagnostics,
                                    "initial_final_top4",
                                    _documents_payload(bounded_documents),
                                )
                                _maybe_store_raw_diagnostics(
                                    retrieval_diagnostics,
                                    "retry_actual_prompt_documents",
                                    _documents_payload(retry_generation_documents),
                                )
                                _maybe_store_raw_diagnostics(
                                    retrieval_diagnostics,
                                    "retry_final_top4",
                                    _documents_payload(retry_bounded_documents),
                                )
                                retry_answer = await asyncio.wait_for(
                                    asyncio.to_thread(
                                        invoke_llm,
                                        _build_evidence_retry_question(
                                            question,
                                            retry_missing_aspects,
                                        ),
                                        retry_generation_documents,
                                        conversation_history,
                                    ),
                                    timeout=retry_timeout,
                                )
                                retry_answer = retry_answer or build_empty_answer()
                                retry_policy_result = evaluate_answer_policy(
                                    retry_answer,
                                    source_count=len(retry_sources),
                                    source_allowlist=retry_source_allowlist,
                                )
                                policy_result = retry_policy_result
                                if retry_policy_result.violated:
                                    fallback_used = True
                                    fallback_reason = "policy_output_violation"
                                    if policy_repair_attempted:
                                        policy_repair_skipped_reason = "policy_repair_slot_used"
                                    elif _requires_safe_policy_refusal(retry_policy_result):
                                        policy_repair_skipped_reason = "unsafe_policy_reason"
                                        answer = SAFE_POLICY_REFUSAL
                                    elif _allows_policy_repair(retry_policy_result):
                                        policy_repair_attempted = True
                                        repair_remaining_budget = max(
                                            0.0,
                                            config.RAG_TOTAL_TIMEOUT_SECONDS
                                            - retrieval_elapsed
                                            - (perf_counter() - generation_started),
                                        )
                                        repair_timeout = min(
                                            config.LLM_TIMEOUT_SECONDS,
                                            repair_remaining_budget,
                                        )
                                        if repair_timeout <= 0:
                                            policy_repair_skipped_reason = "total_budget_exhausted"
                                        else:
                                            repair_answer = await asyncio.wait_for(
                                                asyncio.to_thread(
                                                    invoke_llm,
                                                    _build_repair_question(question),
                                                    retry_generation_documents,
                                                    conversation_history,
                                                ),
                                                timeout=repair_timeout,
                                            )
                                            repair_answer = repair_answer or build_empty_answer()
                                            retry_policy_result = evaluate_answer_policy(
                                                repair_answer,
                                                source_count=len(retry_sources),
                                                source_allowlist=retry_source_allowlist,
                                            )
                                            policy_result = retry_policy_result
                                            if retry_policy_result.violated:
                                                policy_repair_skipped_reason = (
                                                    "unsafe_policy_reason"
                                                    if _requires_safe_policy_refusal(
                                                        retry_policy_result
                                                    )
                                                    else None
                                                )
                                            else:
                                                policy_repair_succeeded = True
                                                fallback_used = False
                                                fallback_reason = None
                                                retry_answer = repair_answer
                                    else:
                                        policy_repair_skipped_reason = "unsupported_policy_reason"
                                    if fallback_used:
                                        bounded_documents = retry_bounded_documents
                                        generation_documents = retry_generation_documents
                                        sources = retry_sources
                                        source_allowlist = retry_source_allowlist
                                        _store_final_context_diagnostics(
                                            retrieval_diagnostics,
                                            generation_documents=generation_documents,
                                            bounded_documents=bounded_documents,
                                        )
                                        answer = (
                                            SAFE_POLICY_REFUSAL
                                            if _requires_safe_policy_refusal(retry_policy_result)
                                            else build_policy_output_fallback_answer(
                                                bounded_documents
                                            )
                                        )
                                if not fallback_used:
                                    retry_judge_timeout = min(
                                        config.RAG_EVIDENCE_JUDGE_TIMEOUT_SECONDS,
                                        max(
                                            0.0,
                                            config.RAG_TOTAL_TIMEOUT_SECONDS
                                            - retrieval_elapsed
                                            - (perf_counter() - generation_started),
                                        ),
                                    )
                                    if retry_judge_timeout <= 0:
                                        retry_judge_result, retry_judge_diagnostics = (
                                            unknown_answer_judge("total_budget_exhausted")
                                        )
                                    else:
                                        try:
                                            retry_judge_result, retry_judge_diagnostics = (
                                                await asyncio.wait_for(
                                                    asyncio.to_thread(
                                                        lambda: _coerce_answer_judge_output(
                                                            heuristic_answer_judge(
                                                                question=question,
                                                                answer=retry_answer,
                                                                prompt_documents=(
                                                                    retry_generation_documents
                                                                ),
                                                                timeout_seconds=(
                                                                    retry_judge_timeout
                                                                ),
                                                            )
                                                        )
                                                    ),
                                                    timeout=retry_judge_timeout,
                                                )
                                            )
                                        except asyncio.TimeoutError:
                                            retry_judge_result, retry_judge_diagnostics = (
                                                unknown_answer_judge("timeout")
                                            )
                                        except Exception as exc:
                                            retry_judge_result, retry_judge_diagnostics = (
                                                unknown_answer_judge(f"error:{type(exc).__name__}")
                                            )
                                    evidence_answer_judge = dict(retry_judge_diagnostics)
                                    retrieval_diagnostics["evidence_answer_judge"] = (
                                        evidence_answer_judge
                                    )
                                    _maybe_store_raw_diagnostics(
                                        retrieval_diagnostics,
                                        "evidence_answer_judge_private",
                                        asdict(retry_judge_result),
                                    )
                                    evidence_retry_succeeded = (
                                        retry_judge_result.verdict == "complete"
                                        and bool(retry_judge_result.evidence_ids)
                                    )
                                    if evidence_retry_succeeded:
                                        answer = retry_answer
                                        bounded_documents = retry_bounded_documents
                                        generation_documents = retry_generation_documents
                                        sources = retry_sources
                                        source_allowlist = retry_source_allowlist
                                        _store_final_context_diagnostics(
                                            retrieval_diagnostics,
                                            generation_documents=generation_documents,
                                            bounded_documents=bounded_documents,
                                        )
                                    else:
                                        fallback_used = True
                                        fallback_reason = "evidence_retry_incomplete"
                                        evidence_retry_skipped_reason = (
                                            retry_judge_result.reason or retry_judge_result.verdict
                                        )
                                        bounded_documents = retry_bounded_documents
                                        generation_documents = retry_generation_documents
                                        sources = retry_sources
                                        source_allowlist = retry_source_allowlist
                                        _store_final_context_diagnostics(
                                            retrieval_diagnostics,
                                            generation_documents=generation_documents,
                                            bounded_documents=bounded_documents,
                                        )
                                        answer = build_policy_output_fallback_answer(
                                            bounded_documents
                                        )
        except asyncio.TimeoutError:
            fallback_used = True
            fallback_reason = (
                "policy_output_violation"
                if policy_result is not None and policy_result.violated
                else "llm_timeout"
            )
            logger.error(
                "LLM call timed out, switching to fallback",
                extra=log_extra(stage="llm", error_type="TimeoutError"),
            )
            answer = (
                build_policy_output_fallback_answer(bounded_documents)
                if fallback_reason == "policy_output_violation"
                else build_fallback_answer(bounded_documents)
            )
        except Exception as exc:
            fallback_used = True
            fallback_reason = (
                "policy_output_violation"
                if policy_result is not None and policy_result.violated
                else "llm_unavailable"
            )
            logger.error(
                "LLM call failed, switching to fallback",
                extra=log_extra(stage="llm", error_type=type(exc).__name__),
            )
            answer = (
                build_policy_output_fallback_answer(bounded_documents)
                if fallback_reason == "policy_output_violation"
                else build_fallback_answer(bounded_documents)
            )
        finally:
            generation_elapsed = perf_counter() - generation_started

    total_elapsed = perf_counter() - total_started
    metadata = _policy_metadata(
        sources=sources,
        retrieved_documents=bounded_documents,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        retrieval_elapsed=retrieval_elapsed,
        generation_elapsed=generation_elapsed,
        total_elapsed=total_elapsed,
        policy_result=policy_result,
        policy_repair_attempted=policy_repair_attempted,
        policy_repair_succeeded=policy_repair_succeeded,
        policy_repair_skipped_reason=policy_repair_skipped_reason,
        policy_marker_sanitization=policy_marker_sanitization,
        retrieval_metadata=retrieval_metadata,
        evidence_sufficiency=evidence_sufficiency,
        evidence_answer_judge=evidence_answer_judge,
        evidence_retry_attempted=evidence_retry_attempted,
        evidence_retry_succeeded=evidence_retry_succeeded,
        evidence_retry_skipped_reason=evidence_retry_skipped_reason,
    )

    return RAGResponse(
        answer=answer,
        sources=sources,
        metadata=metadata,
        retrieved_documents=bounded_documents,
        policy_audit=policy_result.audit if policy_result is not None else None,
        retrieval_diagnostics=_finalize_retrieval_diagnostics(retrieval_diagnostics),
    )

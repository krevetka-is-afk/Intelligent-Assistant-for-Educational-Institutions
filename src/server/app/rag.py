from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
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
)
from .vector import RetrievedDocument, similarity_search

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
    }
)


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
        model = OllamaLLM(model=config.LLM_MODEL, base_url=config.OLLAMA_HOST)
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
    candidate_pool_size = max(
        top_k,
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
        top_k=top_k,
        rrf_k=rrf_k,
        max_chunks_per_document=max_chunks_per_document,
    )
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


def _build_repair_question(question: str) -> str:
    return (
        "Повтори ответ на вопрос пользователя, используя только переданные найденные "
        "документы. Удали неподтвержденные ссылки, URL, имена файлов и источники. "
        "Если нужны ссылки, используй только номера существующих источников в формате "
        "[1], [2] и так далее.\n\n"
        f"Вопрос пользователя: {question}"
    )


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
    retrieval_metadata: dict[str, Any] | None = None,
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
            retrieval_diagnostics=retrieval_diagnostics,
        )

    compiled_prompt = _prompt_compiler.compile(
        question=question,
        retrieved_documents=retrieved_documents,
        conversation_history=conversation_history,
    )
    bounded_documents = _attach_retrieval_diagnostics(
        list(compiled_prompt.retrieved_documents),
        retrieved_documents,
    )
    sources = deduplicate_sources(bounded_documents)
    source_allowlist = build_source_allowlist(sources)
    generation_elapsed = 0.0
    fallback_used = False
    fallback_reason: str | None = None
    policy_result: AnswerPolicyResult | None = None
    policy_repair_attempted = False
    policy_repair_succeeded = False
    policy_repair_skipped_reason: str | None = None

    remaining_budget = max(0.0, config.RAG_TOTAL_TIMEOUT_SECONDS - retrieval_elapsed)
    llm_timeout = min(config.LLM_TIMEOUT_SECONDS, remaining_budget)

    if llm_timeout <= 0:
        fallback_used = True
        fallback_reason = "rag_timeout_budget_exhausted"
        answer = build_fallback_answer(bounded_documents)
    else:
        generation_started = perf_counter()
        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(invoke_llm, question, bounded_documents, conversation_history),
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
                    if _requires_safe_policy_refusal(policy_result):
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
                                    bounded_documents,
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
        retrieval_metadata=retrieval_metadata,
    )

    return RAGResponse(
        answer=answer,
        sources=sources,
        metadata=metadata,
        retrieved_documents=bounded_documents,
        policy_audit=policy_result.audit if policy_result is not None else None,
        retrieval_diagnostics=retrieval_diagnostics,
    )

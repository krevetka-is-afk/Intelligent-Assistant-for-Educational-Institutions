from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from app_runtime import log_extra

from . import config
from .prompt_policy import (
    PROMPT_POLICY_VERSION,
    SAFE_POLICY_REFUSAL,
    CompiledPrompt,
    PromptCompiler,
    PromptPolicyViolation,
    answer_violates_policy,
    build_source_allowlist,
)
from .vector import RetrievedDocument, similarity_search

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
}

_llm_chain = None
_prompt_compiler = PromptCompiler()
logger = logging.getLogger("server.rag")


@dataclass(slots=True)
class RAGResponse:
    answer: str
    sources: list[dict[str, Any]]
    metadata: dict[str, Any]
    retrieved_documents: list[RetrievedDocument]


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
    if not conversation_history:
        return question

    # Keep retrieval query compact: only the latest few user turns plus current question.
    recent_messages = [message.strip() for message in conversation_history[-3:] if message.strip()]
    if not recent_messages:
        return question
    return "\n".join([*recent_messages, question])


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


def build_fallback_answer(retrieved_documents: list[RetrievedDocument]) -> str:
    snippets: list[str] = []
    for index, retrieved in enumerate(retrieved_documents[:4], start=1):
        compact = " ".join(retrieved.document.page_content.split())
        snippet = compact[:260].rstrip()
        if len(compact) > 260:
            snippet += "..."
        snippets.append(f"{index}. {snippet}")

    if not snippets:
        return build_empty_answer()

    return (
        "LLM временно недоступна, поэтому показываю наиболее релевантные фрагменты "
        "из найденных документов.\n\n" + "\n\n".join(snippets)
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
) -> dict[str, Any]:
    return {
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
    }


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
    retrieved_documents = await asyncio.to_thread(
        similarity_search,
        retrieval_query,
        k=config.RAG_TOP_K,
    )
    retrieval_elapsed = perf_counter() - retrieval_started

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
            },
            retrieved_documents=retrieved_documents,
        )

    compiled_prompt = _prompt_compiler.compile(
        question=question,
        retrieved_documents=retrieved_documents,
        conversation_history=conversation_history,
    )
    bounded_documents = list(compiled_prompt.retrieved_documents)
    sources = deduplicate_sources(bounded_documents)
    source_allowlist = build_source_allowlist(sources)
    generation_elapsed = 0.0
    fallback_used = False
    fallback_reason: str | None = None

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
            elif answer_violates_policy(
                answer,
                source_count=len(sources),
                source_allowlist=source_allowlist,
            ):
                fallback_used = True
                fallback_reason = "policy_output_violation"
                logger.warning(
                    "LLM response rejected by prompt policy",
                    extra=log_extra(stage="policy", error_type=fallback_reason),
                )
                answer = SAFE_POLICY_REFUSAL
        except asyncio.TimeoutError:
            fallback_used = True
            fallback_reason = "llm_timeout"
            logger.error(
                "LLM call timed out, switching to fallback",
                extra=log_extra(stage="llm", error_type="TimeoutError"),
            )
            answer = build_fallback_answer(bounded_documents)
        except Exception as exc:
            fallback_used = True
            fallback_reason = "llm_unavailable"
            logger.error(
                "LLM call failed, switching to fallback: %s",
                exc,
                extra=log_extra(stage="llm", error_type=type(exc).__name__),
            )
            answer = build_fallback_answer(bounded_documents)
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
    )

    return RAGResponse(
        answer=answer,
        sources=sources,
        metadata=metadata,
        retrieved_documents=bounded_documents,
    )

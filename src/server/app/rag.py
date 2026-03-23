from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from . import config
from .vector import RetrievedDocument, similarity_search

_ALLOWED_METADATA_KEYS = {
    "source",
    "title",
    "page",
    "mime_type",
    "chunk_index",
    "char_start",
    "char_end",
    "document_id",
    "chunk_id",
    "indexed_at",
}

_llm_chain = None


@dataclass(slots=True)
class RAGResponse:
    answer: str
    sources: list[dict[str, Any]]
    metadata: dict[str, Any]
    retrieved_documents: list[RetrievedDocument]


def _get_llm_chain():
    global _llm_chain
    if _llm_chain is None:
        prompt = ChatPromptTemplate.from_template(config.LLM_PROMPT_TEMPLATE)
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
        sources.append({"content": retrieved.document.page_content, "metadata": metadata})

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


def invoke_llm(question: str, retrieved_documents: list[RetrievedDocument]) -> str:
    chain = _get_llm_chain()
    response = chain.invoke(
        {
            "information": build_context(retrieved_documents),
            "question": question,
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


async def ask_question(question: str) -> RAGResponse:
    total_started = perf_counter()

    retrieval_started = perf_counter()
    retrieved_documents = await asyncio.to_thread(similarity_search, question, k=config.RAG_TOP_K)
    retrieval_elapsed = perf_counter() - retrieval_started

    if not retrieved_documents:
        total_elapsed = perf_counter() - total_started
        return RAGResponse(
            answer=build_empty_answer(),
            sources=[],
            metadata={
                "model": config.LLM_MODEL,
                "embedding_model": config.HF_EMBEDDING_MODEL,
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

    sources = deduplicate_sources(retrieved_documents)
    generation_elapsed = 0.0
    fallback_used = False
    fallback_reason: str | None = None

    remaining_budget = max(0.0, config.RAG_TOTAL_TIMEOUT_SECONDS - retrieval_elapsed)
    llm_timeout = min(config.LLM_TIMEOUT_SECONDS, remaining_budget)

    if llm_timeout <= 0:
        fallback_used = True
        fallback_reason = "rag_timeout_budget_exhausted"
        answer = build_fallback_answer(retrieved_documents)
    else:
        generation_started = perf_counter()
        try:
            answer = await asyncio.wait_for(
                asyncio.to_thread(invoke_llm, question, retrieved_documents),
                timeout=llm_timeout,
            )
            if not answer:
                answer = build_empty_answer()
        except asyncio.TimeoutError:
            fallback_used = True
            fallback_reason = "llm_timeout"
            answer = build_fallback_answer(retrieved_documents)
        except Exception:
            fallback_used = True
            fallback_reason = "llm_unavailable"
            answer = build_fallback_answer(retrieved_documents)
        finally:
            generation_elapsed = perf_counter() - generation_started

    total_elapsed = perf_counter() - total_started
    metadata = {
        "model": config.LLM_MODEL,
        "embedding_model": config.HF_EMBEDDING_MODEL,
        "num_sources": len(sources),
        "confidence": compute_confidence(retrieved_documents, fallback_used=fallback_used),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "retrieval_time_ms": round(retrieval_elapsed * 1000),
        "generation_time_ms": round(generation_elapsed * 1000),
        "total_time_ms": round(total_elapsed * 1000),
    }

    return RAGResponse(
        answer=answer,
        sources=sources,
        metadata=metadata,
        retrieved_documents=retrieved_documents,
    )

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from . import config, rag
from .prompt_policy import answer_violates_policy, build_source_allowlist
from .vector import RetrievedDocument, similarity_search

DEFAULT_EVALUATION_CASES_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "rag_eval" / "cases.v1.json"
)


@dataclass(frozen=True, slots=True)
class RAGEvaluationCase:
    id: str
    question: str
    conversation_history: list[str]
    expected_documents: list[str]
    forbidden_clusters: list[str]
    minimum_answer_points: list[str]
    allow_no_calendar_dates_statement: bool


@dataclass(frozen=True, slots=True)
class RAGEvaluationCandidate:
    rank: int
    distance: float
    metadata: dict[str, Any]
    expected_document_matches: list[str]
    forbidden_cluster_matches: list[str]


@dataclass(frozen=True, slots=True)
class RAGEvaluationCapture:
    case_id: str
    question: str
    conversation_history: list[str]
    retrieval_query: str
    top_n: int
    candidates: list[RAGEvaluationCandidate]
    expected_document_ranks: dict[str, int | None]
    fallback_used: bool
    fallback_reason: str | None
    retrieval_time_ms: int
    generation_time_ms: int
    total_time_ms: int
    final_answer: str


SearchFn = Callable[..., list[RetrievedDocument]]
AnswerFn = Callable[[str, list[RetrievedDocument], list[str] | None], str]


def _metadata_haystack(metadata: dict[str, Any]) -> str:
    values = [
        metadata.get("document_id"),
        metadata.get("chunk_id"),
        metadata.get("source"),
        metadata.get("title"),
        metadata.get("url"),
    ]
    return " ".join(str(value).casefold() for value in values if value is not None)


def _matches_any(metadata: dict[str, Any], patterns: Sequence[str]) -> list[str]:
    haystack = _metadata_haystack(metadata)
    return [pattern for pattern in patterns if pattern.casefold() in haystack]


def load_evaluation_cases(path: Path = DEFAULT_EVALUATION_CASES_PATH) -> list[RAGEvaluationCase]:
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    return [RAGEvaluationCase(**raw_case) for raw_case in raw_cases["cases"]]


def capture_rag_evaluation_case(
    case: RAGEvaluationCase,
    *,
    search_fn: SearchFn | None = None,
    answer_fn: AnswerFn | None = None,
    top_n: int = 5,
) -> RAGEvaluationCapture:
    total_started = perf_counter()
    resolved_search_fn = search_fn or (lambda query, *, k: similarity_search(query, k=k))

    retrieval_query = rag.build_retrieval_query(case.question, case.conversation_history)
    retrieval_started = perf_counter()
    retrieved_documents = resolved_search_fn(retrieval_query, k=top_n)
    retrieval_elapsed = perf_counter() - retrieval_started

    candidates: list[RAGEvaluationCandidate] = []
    expected_document_ranks: dict[str, int | None] = {
        expected_document: None for expected_document in case.expected_documents
    }
    for rank, retrieved in enumerate(retrieved_documents, start=1):
        metadata = rag.normalize_source_metadata(retrieved.document.metadata)
        expected_matches = _matches_any(metadata, case.expected_documents)
        forbidden_matches = _matches_any(metadata, case.forbidden_clusters)
        for expected_document in expected_matches:
            expected_document_ranks.setdefault(expected_document, rank)
            if expected_document_ranks[expected_document] is None:
                expected_document_ranks[expected_document] = rank
        candidates.append(
            RAGEvaluationCandidate(
                rank=rank,
                distance=float(retrieved.distance),
                metadata=metadata,
                expected_document_matches=expected_matches,
                forbidden_cluster_matches=forbidden_matches,
            )
        )

    fallback_used = False
    fallback_reason: str | None = None
    generation_started = perf_counter()
    try:
        compiled_prompt = rag._prompt_compiler.compile(
            question=case.question,
            retrieved_documents=retrieved_documents,
            conversation_history=case.conversation_history,
        )
        bounded_documents = list(compiled_prompt.retrieved_documents)
        if answer_fn is None:
            fallback_used = True
            fallback_reason = "evaluation_answer_generator_not_configured"
            final_answer = rag.build_fallback_answer(bounded_documents)
        else:
            final_answer = answer_fn(case.question, bounded_documents, case.conversation_history)
            sources = rag.deduplicate_sources(bounded_documents)
            if answer_violates_policy(
                final_answer,
                source_count=len(sources),
                source_allowlist=build_source_allowlist(sources),
            ):
                fallback_used = True
                fallback_reason = "policy_output_violation"
                final_answer = rag.SAFE_POLICY_REFUSAL
    except Exception as exc:
        fallback_used = True
        fallback_reason = f"evaluation_capture_failed:{type(exc).__name__}"
        final_answer = rag.build_fallback_answer(retrieved_documents)
    generation_elapsed = perf_counter() - generation_started
    total_elapsed = perf_counter() - total_started

    return RAGEvaluationCapture(
        case_id=case.id,
        question=case.question,
        conversation_history=case.conversation_history,
        retrieval_query=retrieval_query,
        top_n=top_n,
        candidates=candidates,
        expected_document_ranks=expected_document_ranks,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        retrieval_time_ms=round(retrieval_elapsed * 1000),
        generation_time_ms=round(generation_elapsed * 1000),
        total_time_ms=round(total_elapsed * 1000),
        final_answer=final_answer,
    )


def capture_rag_evaluation_suite(
    cases: Sequence[RAGEvaluationCase],
    *,
    search_fn: SearchFn | None = None,
    answer_fn: AnswerFn | None = None,
    top_n: int = 5,
) -> list[RAGEvaluationCapture]:
    return [
        capture_rag_evaluation_case(
            case,
            search_fn=search_fn,
            answer_fn=answer_fn,
            top_n=top_n,
        )
        for case in cases
    ]


def _json_default(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_capture_report(path: Path, captures: Sequence[RAGEvaluationCapture]) -> None:
    payload = {
        "schema_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "rag_top_k": config.RAG_TOP_K,
        "baseline_top_n": captures[0].top_n if captures else None,
        "embedding_model": config.HF_EMBEDDING_MODEL,
        "llm_model": config.LLM_MODEL,
        "captures": captures,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture deterministic RAG evaluation baseline.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_EVALUATION_CASES_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use the configured local LLM for final answers instead of deterministic fallback.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    answer_fn = rag.invoke_llm if args.use_llm else None
    captures = capture_rag_evaluation_suite(
        load_evaluation_cases(args.cases),
        answer_fn=answer_fn,
        top_n=args.top_n,
    )
    write_capture_report(args.output, captures)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from .rag_evaluation import RAGEvaluationCase, _matches_any, load_evaluation_cases
from .rag_stage7_evaluation import (
    _apply_frozen_index,
    _clone_retrieved_documents,
    _dense_similarity_search_direct,
    _rank_documents,
)

if TYPE_CHECKING:
    from .vector import RetrievedDocument


DENSE_LIMIT = 16
LEXICAL_LIMIT = 16
RRF_LIMIT = 16
RAW_UNION_LIMIT = 32


@dataclass(frozen=True, slots=True)
class RetrievalProbeCapture:
    case_id: str
    question: str
    conversation_history: list[str]
    retrieval_query: str
    expected_documents: list[str]
    forbidden_clusters: list[str]
    dense_top16: list[dict[str, Any]]
    lexical_top16: list[dict[str, Any]]
    raw_union_top32: list[dict[str, Any]]
    rrf_selected_top16: list[dict[str, Any]]
    baseline_top4: list[dict[str, Any]]
    retrieval_diagnostics: dict[str, Any]
    metrics: dict[str, Any]
    timings_ms: dict[str, int]


def _metadata_for(item: "RetrievedDocument") -> dict[str, Any]:
    from . import rag

    return rag.normalize_source_metadata(item.document.metadata)


def _configure_frozen_index_runtime(index_dir: Path) -> Path:
    from . import config

    resolved = index_dir.expanduser().resolve()
    _apply_frozen_index(resolved)
    config.VECTOR_DB_DIR = resolved
    config.LEXICAL_INDEX_PATH = (resolved / "lexical_index.sqlite3").resolve()
    return resolved


def _candidate_payload(
    item: "RetrievedDocument",
    *,
    rank: int,
    expected_documents: Sequence[str],
    forbidden_clusters: Sequence[str],
    channel_ranks: dict[str, int] | None = None,
) -> dict[str, Any]:
    metadata = _metadata_for(item)
    expected_matches = _matches_any(metadata, expected_documents)
    forbidden_matches = _matches_any(metadata, forbidden_clusters)
    payload = {
        "rank": rank,
        "chunk_id": metadata.get("chunk_id"),
        "document_id": metadata.get("document_id"),
        "distance": float(item.distance),
        "metadata": metadata,
        "text": item.document.page_content,
        "content_preview": " ".join(item.document.page_content.split())[:500],
        "retrieval_diagnostics": dict(item._retrieval_diagnostics),
        "expected_document_matches": expected_matches,
        "forbidden_cluster_matches": forbidden_matches,
        "expected_source_hit": bool(expected_matches),
    }
    if channel_ranks is not None:
        payload["channel_ranks"] = dict(channel_ranks)
    return payload


def _candidate_payloads(
    items: Sequence["RetrievedDocument"],
    *,
    expected_documents: Sequence[str],
    forbidden_clusters: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        _candidate_payload(
            item,
            rank=rank,
            expected_documents=expected_documents,
            forbidden_clusters=forbidden_clusters,
        )
        for rank, item in enumerate(items, start=1)
    ]


def _raw_union_payloads(
    *,
    dense_documents: Sequence["RetrievedDocument"],
    lexical_documents: Sequence["RetrievedDocument"],
    expected_documents: Sequence[str],
    forbidden_clusters: Sequence[str],
    limit: int = RAW_UNION_LIMIT,
) -> list[dict[str, Any]]:
    from . import rag

    entries: dict[tuple[Any, ...], dict[str, Any]] = {}
    ordered_keys: list[tuple[Any, ...]] = []

    def add(channel: str, rank: int, item: "RetrievedDocument") -> None:
        key = rag._chunk_key(item)
        if key not in entries:
            entries[key] = {
                "item": item,
                "channel_ranks": {},
            }
            ordered_keys.append(key)
        entries[key]["channel_ranks"][channel] = rank

    for rank, item in enumerate(dense_documents, start=1):
        add("dense", rank, item)
    for rank, item in enumerate(lexical_documents, start=1):
        add("lexical", rank, item)

    payloads: list[dict[str, Any]] = []
    for union_rank, key in enumerate(ordered_keys[:limit], start=1):
        entry = entries[key]
        payloads.append(
            _candidate_payload(
                entry["item"],
                rank=union_rank,
                expected_documents=expected_documents,
                forbidden_clusters=forbidden_clusters,
                channel_ranks=entry["channel_ranks"],
            )
        )
    return payloads


def _hit_documents(payloads: Sequence[dict[str, Any]]) -> list[str]:
    found: list[str] = []
    for payload in payloads:
        for match in payload["expected_document_matches"]:
            if match not in found:
                found.append(match)
    return found


def _first_expected_rank(payloads: Sequence[dict[str, Any]]) -> dict[str, int | None]:
    ranks: dict[str, int | None] = {}
    for payload in payloads:
        for match in payload["expected_document_matches"]:
            ranks.setdefault(match, payload["rank"])
    return ranks


def _metrics(
    *,
    case: RAGEvaluationCase,
    baseline_top4: Sequence[dict[str, Any]],
    rrf_top16: Sequence[dict[str, Any]],
    dense_top16: Sequence[dict[str, Any]],
    lexical_top16: Sequence[dict[str, Any]],
    raw_union_top32: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    baseline_found = _hit_documents(baseline_top4)
    rrf_found = _hit_documents(rrf_top16)
    return {
        "baseline_hit_at_4": bool(baseline_found),
        "rrf_hit_at_16": bool(rrf_found),
        "dense_hit_at_16": any(item["expected_source_hit"] for item in dense_top16),
        "lexical_hit_at_16": any(item["expected_source_hit"] for item in lexical_top16),
        "raw_union_hit_at_32": any(item["expected_source_hit"] for item in raw_union_top32),
        "recoverable_rrf_top16": (not baseline_found) and bool(rrf_found),
        "baseline_expected_documents_found": baseline_found,
        "rrf_expected_documents_found": rrf_found,
        "rrf_expected_document_ranks": {
            expected: _first_expected_rank(rrf_top16).get(expected)
            for expected in case.expected_documents
        },
        "baseline_expected_document_ranks": {
            expected: _first_expected_rank(baseline_top4).get(expected)
            for expected in case.expected_documents
        },
    }


def capture_retrieval_probe_case(
    case: RAGEvaluationCase,
    *,
    index_dir: Path,
) -> RetrievalProbeCapture:
    from . import rag

    total_started = perf_counter()
    retrieval_query = rag.build_retrieval_query(case.question, case.conversation_history)

    dense_started = perf_counter()
    dense_documents = _dense_similarity_search_direct(
        retrieval_query,
        k=DENSE_LIMIT,
        index_dir=index_dir,
    )
    dense_ms = round((perf_counter() - dense_started) * 1000)

    lexical_started = perf_counter()
    lexical_documents, lexical_available = rag.lexical_similarity_search(
        retrieval_query,
        k=LEXICAL_LIMIT,
    )
    lexical_ms = round((perf_counter() - lexical_started) * 1000)

    rank_started = perf_counter()
    rrf_documents, retrieval_diagnostics = _rank_documents(
        dense_documents=_clone_retrieved_documents(dense_documents),
        lexical_documents=_clone_retrieved_documents(lexical_documents),
        top_k=RRF_LIMIT,
    )
    baseline_documents, baseline_diagnostics = _rank_documents(
        dense_documents=_clone_retrieved_documents(dense_documents),
        lexical_documents=_clone_retrieved_documents(lexical_documents),
        top_k=4,
    )
    rank_ms = round((perf_counter() - rank_started) * 1000)

    dense_top16 = _candidate_payloads(
        dense_documents,
        expected_documents=case.expected_documents,
        forbidden_clusters=case.forbidden_clusters,
    )
    lexical_top16 = _candidate_payloads(
        lexical_documents,
        expected_documents=case.expected_documents,
        forbidden_clusters=case.forbidden_clusters,
    )
    raw_union_top32 = _raw_union_payloads(
        dense_documents=dense_documents,
        lexical_documents=lexical_documents,
        expected_documents=case.expected_documents,
        forbidden_clusters=case.forbidden_clusters,
    )
    rrf_top16 = _candidate_payloads(
        rrf_documents,
        expected_documents=case.expected_documents,
        forbidden_clusters=case.forbidden_clusters,
    )
    baseline_top4 = _candidate_payloads(
        baseline_documents,
        expected_documents=case.expected_documents,
        forbidden_clusters=case.forbidden_clusters,
    )
    total_ms = round((perf_counter() - total_started) * 1000)

    return RetrievalProbeCapture(
        case_id=case.id,
        question=case.question,
        conversation_history=list(case.conversation_history),
        retrieval_query=retrieval_query,
        expected_documents=list(case.expected_documents),
        forbidden_clusters=list(case.forbidden_clusters),
        dense_top16=dense_top16,
        lexical_top16=lexical_top16,
        raw_union_top32=raw_union_top32,
        rrf_selected_top16=rrf_top16,
        baseline_top4=baseline_top4,
        retrieval_diagnostics={
            **retrieval_diagnostics,
            "lexical_available": lexical_available,
            "query_rewrite_used": False,
            "baseline_top4": baseline_diagnostics,
        },
        metrics=_metrics(
            case=case,
            baseline_top4=baseline_top4,
            rrf_top16=rrf_top16,
            dense_top16=dense_top16,
            lexical_top16=lexical_top16,
            raw_union_top32=raw_union_top32,
        ),
        timings_ms={
            "dense": dense_ms,
            "lexical": lexical_ms,
            "rank": rank_ms,
            "total": total_ms,
        },
    )


def capture_retrieval_probe_suite(
    *,
    index_dir: Path,
    cases_path: Path,
    limit: int | None = None,
) -> list[RetrievalProbeCapture]:
    from .vector import clear_vector_cache

    resolved_index_dir = _configure_frozen_index_runtime(index_dir)
    clear_vector_cache()
    cases = load_evaluation_cases(cases_path)
    selected_cases = cases[:limit] if limit is not None else cases
    return [
        capture_retrieval_probe_case(case, index_dir=resolved_index_dir) for case in selected_cases
    ]


def _summarize(captures: Sequence[RetrievalProbeCapture]) -> dict[str, Any]:
    total = len(captures)
    baseline_misses = [capture for capture in captures if not capture.metrics["baseline_hit_at_4"]]
    recoverable = [
        capture for capture in baseline_misses if capture.metrics["recoverable_rrf_top16"]
    ]
    return {
        "cases": total,
        "baseline_hit_at_4": sum(1 for capture in captures if capture.metrics["baseline_hit_at_4"]),
        "rrf_hit_at_16": sum(1 for capture in captures if capture.metrics["rrf_hit_at_16"]),
        "dense_hit_at_16": sum(1 for capture in captures if capture.metrics["dense_hit_at_16"]),
        "lexical_hit_at_16": sum(1 for capture in captures if capture.metrics["lexical_hit_at_16"]),
        "raw_union_hit_at_32": sum(
            1 for capture in captures if capture.metrics["raw_union_hit_at_32"]
        ),
        "baseline_misses": len(baseline_misses),
        "recoverable_rrf_top16": len(recoverable),
        "recoverable_case_ids": [capture.case_id for capture in recoverable],
        "avg_total_time_ms": (
            round(sum(capture.timings_ms["total"] for capture in captures) / total) if total else 0
        ),
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_retrieval_probe_report(
    path: Path,
    *,
    captures: Sequence[RetrievalProbeCapture],
    cases_path: Path,
    index_dir: Path,
) -> None:
    from . import config

    payload = {
        "schema_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "cases_path": str(cases_path),
        "index_dir": str(index_dir),
        "settings": {
            "rag_top_k": config.RAG_TOP_K,
            "dense_limit": DENSE_LIMIT,
            "lexical_limit": LEXICAL_LIMIT,
            "rrf_limit": RRF_LIMIT,
            "raw_union_limit": RAW_UNION_LIMIT,
            "rag_rrf_k": config.RAG_RRF_K,
            "rag_max_chunks_per_document": config.RAG_MAX_CHUNKS_PER_DOCUMENT,
            "embedding_model": config.HF_EMBEDDING_MODEL,
            "vector_db_dir": str(config.VECTOR_DB_DIR),
            "lexical_index_path": str(config.LEXICAL_INDEX_PATH),
            "query_rewrite_enabled": False,
            "generation_enabled": False,
        },
        "summary": _summarize(captures),
        "captures": list(captures),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture retrieval-only RAG context evidence without generation."
    )
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional prefix case limit for staged smoke runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    captures = capture_retrieval_probe_suite(
        index_dir=args.index_dir,
        cases_path=args.cases,
        limit=args.limit,
    )
    write_retrieval_probe_report(
        args.output,
        captures=captures,
        cases_path=args.cases,
        index_dir=args.index_dir,
    )


if __name__ == "__main__":
    main()

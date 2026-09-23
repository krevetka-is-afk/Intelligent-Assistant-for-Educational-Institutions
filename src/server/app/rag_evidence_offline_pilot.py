"""Capture paired, local-only RAG evidence experiments on frozen indexes.

This module intentionally writes answers and source text to a private local
artifact. It does not send those values to application logs or API metadata.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any


def _configure_runtime(index_dir: Path, flags: list[str]) -> None:
    resolved = index_dir.expanduser().resolve()
    os.environ["VECTOR_DB_DIR"] = str(resolved)
    os.environ["LEXICAL_INDEX_PATH"] = str(resolved / "lexical_index.sqlite3")
    os.environ["RAG_RETRIEVAL_MODE"] = "hybrid"
    os.environ["RAG_CONTEXT_EXPANSION_ENABLED"] = "false"
    os.environ["RAG_EVIDENCE_OFFLINE_CAPTURE_ENABLED"] = "true"
    os.environ["HF_EMBEDDING_MODEL"] = "cointegrated/rubert-tiny2"
    os.environ["HF_EMBEDDING_NORMALIZE"] = "false"
    os.environ["RAG_TOTAL_TIMEOUT_SECONDS"] = "420"
    os.environ["LLM_TIMEOUT_SECONDS"] = "360"
    for flag in flags:
        name, separator, value = flag.partition("=")
        if not separator or not name.startswith("RAG_"):
            raise ValueError(f"invalid RAG flag assignment: {flag!r}")
        if name == "RAG_EVIDENCE_OFFLINE_CAPTURE_ENABLED" and value.lower() != "true":
            raise ValueError("offline capture must remain enabled in the pilot runner")
        os.environ[name] = value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document_payload(retrieved: Any, rank: int) -> dict[str, Any]:
    document = retrieved.document
    return {
        "rank": rank,
        "metadata": dict(document.metadata or {}),
        "content": str(document.page_content),
        "distance": float(retrieved.distance),
    }


def _diagnostic_document_payload(payload: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "metadata": dict(payload.get("metadata") or {}),
        "content": str(payload.get("content") or payload.get("text") or ""),
        "distance": float(payload.get("distance") or 0.0),
    }


def _percentile_nearest_rank(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    import math

    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


async def _capture_suite(
    cases_path: Path,
    *,
    limit: int | None,
    probe_path: Path | None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    setup_started = perf_counter()
    from . import config, rag
    from .prompt_policy import build_source_allowlist, evaluate_answer_policy
    from .rag_evaluation import load_evaluation_cases
    from .rag_structural_pilot import load_gold_span_groups, score_capture

    cases = load_evaluation_cases(cases_path)
    selected = cases[:limit] if limit is not None else cases
    gold_groups = load_gold_span_groups(cases_path)
    probes: dict[str, dict[str, Any]] = {}
    if probe_path is not None:
        probe_report = json.loads(probe_path.read_text(encoding="utf-8"))
        probes = {
            capture["case_id"]: capture
            for capture in probe_report["captures"]
            if isinstance(capture, dict) and isinstance(capture.get("case_id"), str)
        }
    setup_ms = round((perf_counter() - setup_started) * 1000)

    captures: list[dict[str, Any]] = []
    for case in selected:
        response = await rag.ask_question(case.question, case.conversation_history)
        final_documents = [
            _document_payload(document, rank)
            for rank, document in enumerate(response.retrieved_documents, start=1)
        ]
        prompt_documents = getattr(response, "prompt_documents", None)
        if prompt_documents is None:
            prompt_documents = getattr(response, "generation_documents", None)
        diagnostic_prompt_documents = response.retrieval_diagnostics.get("actual_prompt_documents")
        if prompt_documents is None and isinstance(diagnostic_prompt_documents, list):
            actual_prompt_documents = [
                _diagnostic_document_payload(document, rank)
                for rank, document in enumerate(diagnostic_prompt_documents, start=1)
                if isinstance(document, dict)
            ]
            prompt_capture_available = True
        else:
            actual_prompt_documents = [
                _document_payload(document, rank)
                for rank, document in enumerate(prompt_documents or [], start=1)
            ]
            prompt_capture_available = prompt_documents is not None
        passage_score = (
            asdict(
                score_capture(
                    {"case_id": case.id, "bounded_prompt_documents": actual_prompt_documents},
                    gold_groups[case.id],
                    variant="actual_prompt",
                )
            )
            if prompt_capture_available and case.id in gold_groups
            else None
        )
        probe = probes.get(case.id)
        candidate_score = (
            asdict(
                score_capture(
                    {
                        "case_id": case.id,
                        "bounded_prompt_documents": probe["rrf_selected_top16"],
                    },
                    gold_groups[case.id],
                    variant="rrf_top16",
                )
            )
            if probe is not None and case.id in gold_groups
            else None
        )
        final_policy = evaluate_answer_policy(
            response.answer,
            source_count=len(response.sources),
            source_allowlist=build_source_allowlist(response.sources),
        )
        captures.append(
            {
                "case_id": case.id,
                "question": case.question,
                "conversation_history": case.conversation_history,
                "final_top4": final_documents,
                "actual_prompt_documents": actual_prompt_documents,
                "prompt_capture_available": prompt_capture_available,
                "candidate_exact_passage": candidate_score,
                "actual_prompt_exact_passage": passage_score,
                "answer": response.answer,
                "sources": response.sources,
                "final_policy_violated": final_policy.violated,
                "final_policy_reasons": sorted(final_policy.reasons),
                "metadata": response.metadata,
                "retrieval_diagnostics": response.retrieval_diagnostics,
            }
        )
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            staged = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            staged.write_text(
                json.dumps(
                    {"cases_sha256": _sha256_file(cases_path), "captures": captures},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            staged.replace(checkpoint_path)
        print(f"[evidence-pilot] captured {case.id}", flush=True)

    timings = [int(capture["metadata"].get("total_time_ms", 0)) for capture in captures]
    warm_timings = timings[1:]
    fallback_reasons = Counter(
        str(capture["metadata"].get("fallback_reason"))
        for capture in captures
        if capture["metadata"].get("fallback_used")
    )
    return {
        "schema_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "cases_path": str(cases_path.resolve()),
        "cases_sha256": _sha256_file(cases_path),
        "probe_path": str(probe_path.resolve()) if probe_path else None,
        "probe_sha256": _sha256_file(probe_path) if probe_path else None,
        "settings": {
            "rag_top_k": config.RAG_TOP_K,
            "rag_max_context_documents": config.RAG_MAX_CONTEXT_DOCUMENTS,
            "rag_max_document_chars": config.RAG_MAX_DOCUMENT_CHARS,
            "rag_max_total_context_chars": config.RAG_MAX_TOTAL_CONTEXT_CHARS,
            "rag_total_timeout_seconds": config.RAG_TOTAL_TIMEOUT_SECONDS,
            "llm_timeout_seconds": config.LLM_TIMEOUT_SECONDS,
            "rag_context_expansion_enabled": config.RAG_CONTEXT_EXPANSION_ENABLED,
            "rag_evidence_rerank_enabled": config.RAG_EVIDENCE_RERANK_ENABLED,
            "rag_evidence_rerank_method": config.RAG_EVIDENCE_RERANK_METHOD,
            "rag_evidence_structural_window_enabled": config.RAG_EVIDENCE_STRUCTURAL_WINDOW_ENABLED,
            "rag_evidence_sufficiency_enabled": config.RAG_EVIDENCE_SUFFICIENCY_ENABLED,
            "rag_evidence_retry_enabled": config.RAG_EVIDENCE_RETRY_ENABLED,
            "rag_evidence_marker_repair_enabled": config.RAG_EVIDENCE_MARKER_REPAIR_ENABLED,
            "rag_evidence_offline_capture_enabled": config.RAG_EVIDENCE_OFFLINE_CAPTURE_ENABLED,
            "llm_model": config.LLM_MODEL,
            "embedding_model": config.HF_EMBEDDING_MODEL,
            "embedding_normalize": config.HF_EMBEDDING_NORMALIZE,
            "rag_evidence_judge_model": config.RAG_EVIDENCE_JUDGE_MODEL,
            "offline_generation_seed": config.RAG_OFFLINE_GENERATION_SEED,
            "offline_generation_temperature": config.RAG_OFFLINE_GENERATION_TEMPERATURE,
        },
        "summary": {
            "cases": len(captures),
            "prompt_capture_cases": sum(
                bool(capture["prompt_capture_available"]) for capture in captures
            ),
            "candidate_all_passages_at_16": sum(
                bool((capture["candidate_exact_passage"] or {}).get("all_passages_covered"))
                for capture in captures
            ),
            "actual_prompt_all_passages": sum(
                bool((capture["actual_prompt_exact_passage"] or {}).get("all_passages_covered"))
                for capture in captures
            ),
            "cold_first_total_ms": timings[0] if timings else None,
            "setup_ms": setup_ms,
            "warm_p50_total_ms": round(median(warm_timings)) if warm_timings else None,
            "warm_p95_total_ms": _percentile_nearest_rank(warm_timings, 0.95),
            "fallbacks": sum(
                bool(capture["metadata"].get("fallback_used")) for capture in captures
            ),
            "fallback_reasons": dict(sorted(fallback_reasons.items())),
            "retry_attempts": sum(
                bool(capture["metadata"].get("rag_evidence_retry_attempted"))
                for capture in captures
            ),
            "retry_successes": sum(
                bool(capture["metadata"].get("rag_evidence_retry_succeeded"))
                for capture in captures
            ),
            "policy_repair_attempts": sum(
                bool(capture["metadata"].get("policy_output_repair_attempted"))
                for capture in captures
            ),
            "marker_sanitizations": sum(
                bool(capture["metadata"].get("policy_output_marker_sanitization_changed"))
                for capture in captures
            ),
            "final_policy_violations": sum(
                bool(capture["final_policy_violated"]) for capture in captures
            ),
        },
        "captures": captures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--flag", action="append", default=[])
    parser.add_argument("--prewarm-cross-encoder", action="store_true")
    args = parser.parse_args()
    _configure_runtime(args.index_dir, args.flag)
    prewarm_ms: int | None = None
    if args.prewarm_cross_encoder:
        os.environ["HF_HUB_OFFLINE"] = "1"
        from .rag_context_reranker_pilot import DEVICE, MAX_LENGTH, MODEL_NAME, _load_cross_encoder

        prewarm_started = perf_counter()
        _load_cross_encoder(MODEL_NAME, MAX_LENGTH, DEVICE)
        prewarm_ms = round((perf_counter() - prewarm_started) * 1000)
        print(f"[evidence-pilot] cross-encoder prewarm {prewarm_ms} ms", flush=True)
    checkpoint_path = args.output.with_suffix(args.output.suffix + ".checkpoint.json")
    report = asyncio.run(
        _capture_suite(
            args.cases,
            limit=args.limit,
            probe_path=args.probe,
            checkpoint_path=checkpoint_path,
        )
    )
    report["index_dir"] = str(args.index_dir.resolve())
    report["flags"] = args.flag
    report["cross_encoder_prewarm_ms"] = prewarm_ms
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checkpoint_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

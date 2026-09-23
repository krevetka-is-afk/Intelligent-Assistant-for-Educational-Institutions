from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

from .rag_context_selector_pilot import PILOT_CASE_IDS

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MAX_LENGTH = 384
BATCH_SIZE = 8
DEVICE = "cpu"
_MAX_SELECTED = 4
_MAX_CANDIDATES = 16
_PREVIEW_CHARS = 500

ScoreFn = Callable[[str, Sequence[Mapping[str, Any]]], Sequence[float]]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_captures(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [capture for capture in payload if isinstance(capture, Mapping)]
    if isinstance(payload, Mapping):
        captures = payload.get("captures")
        if isinstance(captures, list):
            return [capture for capture in captures if isinstance(capture, Mapping)]
    raise ValueError("probe JSON must be a capture list or contain a captures list")


def _candidate_metadata(candidate: Mapping[str, Any]) -> dict[str, Any]:
    metadata = candidate.get("metadata")
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _candidate_chunk_id(candidate: Mapping[str, Any]) -> str | None:
    for key in ("chunk_id", "id"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = _candidate_metadata(candidate).get("chunk_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _candidate_rank(candidate: Mapping[str, Any], fallback_rank: int) -> int:
    value = candidate.get("rank") or candidate.get("rrf_rank")
    if value is None:
        return fallback_rank
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback_rank
    return parsed if parsed > 0 else fallback_rank


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    for key in ("text", "content", "page_content", "content_preview", "snippet"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value
    document = candidate.get("document")
    if isinstance(document, Mapping):
        for key in ("page_content", "content", "text"):
            value = document.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _text_preview(text: str) -> str:
    return " ".join(text.split())[:_PREVIEW_CHARS]


def _metadata_haystack(candidate: Mapping[str, Any]) -> str:
    metadata = _candidate_metadata(candidate)
    values = [
        metadata.get("document_id"),
        metadata.get("chunk_id"),
        metadata.get("source"),
        metadata.get("title"),
        metadata.get("url"),
        candidate.get("document_id"),
        candidate.get("chunk_id"),
        candidate.get("source"),
        candidate.get("title"),
    ]
    return " ".join(str(value).casefold() for value in values if value is not None)


def _matches(candidate: Mapping[str, Any], patterns: Sequence[str]) -> list[str]:
    haystack = _metadata_haystack(candidate)
    return [pattern for pattern in patterns if pattern.casefold() in haystack]


def _metadata_metrics(
    candidates: Sequence[Mapping[str, Any]],
    *,
    expected_documents: Sequence[str],
    forbidden_clusters: Sequence[str],
) -> dict[str, Any]:
    expected_ranks: dict[str, int | None] = {
        pattern: None for pattern in expected_documents if isinstance(pattern, str)
    }
    forbidden_hits: dict[str, int] = {}
    for rank, candidate in enumerate(candidates, start=1):
        for pattern in _matches(candidate, list(expected_ranks)):
            if expected_ranks[pattern] is None:
                expected_ranks[pattern] = rank
        for pattern in _matches(
            candidate, [item for item in forbidden_clusters if isinstance(item, str)]
        ):
            forbidden_hits.setdefault(pattern, rank)
    return {
        "expected_hit_top4": any(rank is not None for rank in expected_ranks.values()),
        "expected_document_ranks": expected_ranks,
        "forbidden_cluster_hits": forbidden_hits,
    }


def _select_capture(
    captures: Sequence[Mapping[str, Any]], case_id: str
) -> Mapping[str, Any] | None:
    for capture in captures:
        if capture.get("case_id") == case_id:
            return capture
    return None


def _summarize_candidate(
    candidate: Mapping[str, Any],
    *,
    old_rank: int,
    new_rank: int | None,
    score: float,
) -> dict[str, Any]:
    metadata = _candidate_metadata(candidate)
    text = _candidate_text(candidate)
    return {
        "chunk_id": _candidate_chunk_id(candidate),
        "old_rank": old_rank,
        "new_rank": new_rank,
        "score": score,
        "document_id": metadata.get("document_id"),
        "title": metadata.get("title") or metadata.get("source"),
        "source": metadata.get("source"),
        "text_preview": _text_preview(text),
    }


@lru_cache(maxsize=4)
def _load_cross_encoder(
    model_name: str,
    max_length: int,
    device: str,
) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        model_name,
        max_length=max_length,
        device=device,
        trust_remote_code=False,
    )


def score_with_cross_encoder(
    retrieval_query: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    model_name: str = MODEL_NAME,
    max_length: int = MAX_LENGTH,
    batch_size: int = BATCH_SIZE,
    device: str = DEVICE,
) -> list[float]:
    model = _load_cross_encoder(model_name, max_length, device)
    pairs = [(retrieval_query, _candidate_text(candidate)) for candidate in candidates]
    raw_scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return [float(score) for score in raw_scores]


def _rerank_case(
    capture: Mapping[str, Any],
    *,
    source_probe: str,
    scorer: ScoreFn,
) -> dict[str, Any]:
    started = perf_counter()
    candidates = [
        candidate
        for candidate in capture.get("rrf_selected_top16", [])
        if isinstance(candidate, Mapping)
    ][:_MAX_CANDIDATES]
    baseline_top4 = [
        candidate
        for candidate in capture.get("baseline_top4", [])
        if isinstance(candidate, Mapping)
    ][:_MAX_SELECTED]
    retrieval_query = capture.get("retrieval_query")
    expected_documents = [
        item for item in capture.get("expected_documents", []) if isinstance(item, str)
    ]
    forbidden_clusters = [
        item for item in capture.get("forbidden_clusters", []) if isinstance(item, str)
    ]

    if not isinstance(retrieval_query, str) or not retrieval_query.strip():
        return {
            "case_id": capture.get("case_id"),
            "source_probe": source_probe,
            "status": "invalid_input",
            "reason": "missing_retrieval_query",
        }
    if len(candidates) < _MAX_CANDIDATES:
        return {
            "case_id": capture.get("case_id"),
            "source_probe": source_probe,
            "status": "invalid_input",
            "reason": "rrf_selected_top16_missing_or_incomplete",
            "candidate_count": len(candidates),
        }

    scores = [float(score) for score in scorer(retrieval_query, candidates)]
    if len(scores) != len(candidates):
        raise ValueError("scorer returned a score count that does not match candidates")

    old_ranks = [_candidate_rank(candidate, index) for index, candidate in enumerate(candidates, 1)]
    ranked = sorted(
        zip(candidates, scores, old_ranks, strict=True),
        key=lambda item: (-item[1], item[2]),
    )
    selected_top4 = [candidate for candidate, _score, _old_rank in ranked[:_MAX_SELECTED]]
    selected_ids = [_candidate_chunk_id(candidate) for candidate in selected_top4]
    new_rank_by_id = {
        _candidate_chunk_id(candidate): rank
        for rank, (candidate, _score, _old_rank) in enumerate(ranked, start=1)
    }
    score_by_id = {_candidate_chunk_id(candidate): score for candidate, score, _old_rank in ranked}
    candidate_details = [
        _summarize_candidate(
            candidate,
            old_rank=old_rank,
            new_rank=new_rank_by_id.get(_candidate_chunk_id(candidate)),
            score=score_by_id[_candidate_chunk_id(candidate)],
        )
        for candidate, _score, old_rank in zip(candidates, scores, old_ranks, strict=True)
    ]

    return {
        "case_id": capture.get("case_id"),
        "source_probe": source_probe,
        "question": capture.get("question"),
        "history_count": len(capture.get("conversation_history") or []),
        "status": "reranked",
        "candidate_count": len(candidates),
        "baseline_chunk_ids": [_candidate_chunk_id(candidate) for candidate in baseline_top4],
        "selected_chunk_ids": selected_ids,
        "baseline_top4": [
            _summarize_candidate(
                candidate,
                old_rank=_candidate_rank(candidate, index),
                new_rank=new_rank_by_id.get(_candidate_chunk_id(candidate)),
                score=score_by_id.get(_candidate_chunk_id(candidate), 0.0),
            )
            for index, candidate in enumerate(baseline_top4, start=1)
        ],
        "selected_top4": [
            _summarize_candidate(
                candidate,
                old_rank=_candidate_rank(candidate, index),
                new_rank=index,
                score=score_by_id[_candidate_chunk_id(candidate)],
            )
            for index, candidate in enumerate(selected_top4, start=1)
        ],
        "candidates": candidate_details,
        "frozen_trace_metrics": capture.get("metrics"),
        "baseline_metadata_metrics": _metadata_metrics(
            baseline_top4,
            expected_documents=expected_documents,
            forbidden_clusters=forbidden_clusters,
        ),
        "selected_metadata_metrics": _metadata_metrics(
            selected_top4,
            expected_documents=expected_documents,
            forbidden_clusters=forbidden_clusters,
        ),
        "timings_ms": {"rerank": round((perf_counter() - started) * 1000)},
    }


def run_reranker_pilot(
    *,
    main_probe: Path,
    holdout_probe: Path,
    output: Path,
    limit: int | None = None,
    all_cases: bool = False,
    scorer: ScoreFn | None = None,
) -> dict[str, Any]:
    main_payload = _read_json(main_probe)
    holdout_payload = _read_json(holdout_probe)
    probes = {
        "main": _probe_captures(main_payload),
        "holdout": _probe_captures(holdout_payload),
    }
    case_sources = {
        PILOT_CASE_IDS[0]: "main",
        PILOT_CASE_IDS[1]: "main",
        PILOT_CASE_IDS[2]: "holdout",
        PILOT_CASE_IDS[3]: "holdout",
    }
    if all_cases:
        case_sources = {
            str(capture["case_id"]): source
            for source, captures in probes.items()
            for capture in captures
            if isinstance(capture.get("case_id"), str)
        }
    case_ids = list(case_sources)
    if limit is not None:
        case_ids = case_ids[:limit]
    scorer = scorer or (
        lambda query, candidates: score_with_cross_encoder(
            query,
            candidates,
            model_name=MODEL_NAME,
            max_length=MAX_LENGTH,
            batch_size=BATCH_SIZE,
            device=DEVICE,
        )
    )

    results: list[dict[str, Any]] = []
    total_started = perf_counter()
    for case_id in case_ids:
        source_probe = case_sources[case_id]
        capture = _select_capture(probes[source_probe], case_id)
        if capture is None:
            results.append(
                {
                    "case_id": case_id,
                    "source_probe": source_probe,
                    "status": "invalid_input",
                    "reason": "case_capture_missing",
                }
            )
            continue
        results.append(_rerank_case(capture, source_probe=source_probe, scorer=scorer))

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "main_probe": str(main_probe),
            "main_probe_sha256": _sha256(main_probe),
            "holdout_probe": str(holdout_probe),
            "holdout_probe_sha256": _sha256(holdout_probe),
        },
        "model": {
            "name": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "device": DEVICE,
            "trust_remote_code": False,
        },
        "case_selection": {
            "all_cases": all_cases,
            "limit": limit,
            "pilot_case_ids": list(PILOT_CASE_IDS),
        },
        "notes": [
            "Offline frozen-trace reranker only; it does not change production retrieval.",
            "Expected-document and forbidden-cluster metrics are metadata checks, "
            "not answer-quality proof.",
        ],
        "cases": results,
        "timings_ms": {"total": round((perf_counter() - total_started) * 1000)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline frozen-trace CrossEncoder reranker pilot."
    )
    parser.add_argument("--main-probe", type=Path, required=True)
    parser.add_argument("--holdout-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--all", action="store_true", dest="all_cases")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_reranker_pilot(
        main_probe=args.main_probe,
        holdout_probe=args.holdout_probe,
        output=args.output,
        limit=args.limit,
        all_cases=args.all_cases,
    )


if __name__ == "__main__":
    main()

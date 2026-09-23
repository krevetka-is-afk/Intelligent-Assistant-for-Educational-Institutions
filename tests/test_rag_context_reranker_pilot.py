from __future__ import annotations

import json
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from src.server.app import rag_context_reranker_pilot as reranker_pilot
from src.server.app.rag_context_reranker_pilot import (
    MODEL_NAME,
    _candidate_chunk_id,
    run_reranker_pilot,
)
from src.server.app.rag_context_selector_pilot import PILOT_CASE_IDS


def _candidate(case_id: str, rank: int, *, expected: bool = False) -> dict[str, Any]:
    document_id = f"doc-{case_id}" if expected else f"other-doc-{rank}"
    return {
        "rank": rank,
        "chunk_id": f"{case_id}-chunk-{rank}",
        "text": f"Full candidate text for {case_id} rank {rank}",
        "content_preview": f"Preview {rank}",
        "metadata": {
            "chunk_id": f"{case_id}-chunk-{rank}",
            "document_id": document_id,
            "title": f"Title {rank}",
            "source": f"source/{document_id}.pdf",
        },
    }


def _capture(case_id: str) -> dict[str, Any]:
    candidates = [_candidate(case_id, rank, expected=rank == 6) for rank in range(1, 17)]
    return {
        "case_id": case_id,
        "question": f"Question for {case_id}?",
        "conversation_history": ["old message"],
        "retrieval_query": f"Better retrieval query for {case_id}",
        "expected_documents": [f"doc-{case_id}"],
        "forbidden_clusters": ["forbidden-doc"],
        "rrf_selected_top16": candidates,
        "baseline_top4": candidates[:4],
        "metrics": {"baseline_hit_at_4": False, "rrf_hit_at_16": True},
    }


def _probe(path: Path, case_ids: Sequence[str]) -> Path:
    payload = {
        "schema_version": 1,
        "captures": [_capture(case_id) for case_id in case_ids],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_reranker_pilot_scores_rrf_top16_and_selects_stable_top4(tmp_path: Path) -> None:
    main_probe = _probe(tmp_path / "main.json", PILOT_CASE_IDS[:2])
    holdout_probe = _probe(tmp_path / "holdout.json", PILOT_CASE_IDS[2:])
    seen_queries: list[str] = []

    def scorer(query: str, candidates: Sequence[Mapping[str, Any]]) -> list[float]:
        seen_queries.append(query)
        scores = [0.0] * len(candidates)
        scores[4] = 10.0
        scores[5] = 9.0
        scores[6] = 9.0
        scores[7] = 8.0
        return scores

    report = run_reranker_pilot(
        main_probe=main_probe,
        holdout_probe=holdout_probe,
        output=tmp_path / "out.json",
        scorer=scorer,
    )

    assert seen_queries == [f"Better retrieval query for {case_id}" for case_id in PILOT_CASE_IDS]
    assert report["model"]["name"] == MODEL_NAME
    assert report["model"]["max_length"] == 384
    assert report["model"]["batch_size"] == 8
    assert report["model"]["device"] == "cpu"
    assert report["model"]["trust_remote_code"] is False
    first = report["cases"][0]
    assert first["status"] == "reranked"
    assert first["baseline_chunk_ids"] == [
        f"{PILOT_CASE_IDS[0]}-chunk-1",
        f"{PILOT_CASE_IDS[0]}-chunk-2",
        f"{PILOT_CASE_IDS[0]}-chunk-3",
        f"{PILOT_CASE_IDS[0]}-chunk-4",
    ]
    assert first["selected_chunk_ids"] == [
        f"{PILOT_CASE_IDS[0]}-chunk-5",
        f"{PILOT_CASE_IDS[0]}-chunk-6",
        f"{PILOT_CASE_IDS[0]}-chunk-7",
        f"{PILOT_CASE_IDS[0]}-chunk-8",
    ]
    assert first["selected_metadata_metrics"]["expected_hit_top4"] is True
    assert first["baseline_metadata_metrics"]["expected_hit_top4"] is False
    assert first["frozen_trace_metrics"] == {
        "baseline_hit_at_4": False,
        "rrf_hit_at_16": True,
    }
    assert first["candidates"][5]["old_rank"] == 6
    assert first["candidates"][5]["new_rank"] == 2
    assert "Full candidate text" in first["candidates"][5]["text_preview"]


def test_reranker_pilot_limit_keeps_prefix_cases(tmp_path: Path) -> None:
    main_probe = _probe(tmp_path / "main.json", PILOT_CASE_IDS[:2])
    holdout_probe = _probe(tmp_path / "holdout.json", PILOT_CASE_IDS[2:])

    def scorer(_query: str, candidates: Sequence[Mapping[str, Any]]) -> list[float]:
        return [float(index) for index, _candidate in enumerate(candidates)]

    report = run_reranker_pilot(
        main_probe=main_probe,
        holdout_probe=holdout_probe,
        output=tmp_path / "out.json",
        limit=2,
        scorer=scorer,
    )

    assert [case["case_id"] for case in report["cases"]] == list(PILOT_CASE_IDS[:2])
    assert report["case_selection"]["limit"] == 2


def test_reranker_pilot_rejects_incomplete_rrf_pool(tmp_path: Path) -> None:
    main_probe = _probe(tmp_path / "main.json", PILOT_CASE_IDS[:2])
    holdout_probe = _probe(tmp_path / "holdout.json", PILOT_CASE_IDS[2:])
    payload = json.loads(main_probe.read_text(encoding="utf-8"))
    payload["captures"][0]["rrf_selected_top16"] = payload["captures"][0]["rrf_selected_top16"][:3]
    main_probe.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def scorer(_query: str, candidates: Sequence[Mapping[str, Any]]) -> list[float]:
        return [0.0] * len(candidates)

    report = run_reranker_pilot(
        main_probe=main_probe,
        holdout_probe=holdout_probe,
        output=tmp_path / "out.json",
        scorer=scorer,
    )

    first = report["cases"][0]
    assert first["status"] == "invalid_input"
    assert first["reason"] == "rrf_selected_top16_missing_or_incomplete"


def test_reranker_pilot_accepts_saved_probe_schema(tmp_path: Path) -> None:
    main_probe = Path(".omx/reports/rag-context-selection-2026-09-23/probe.main.json")
    holdout_probe = Path(".omx/reports/rag-context-selection-2026-09-23/probe.holdout.json")
    if not main_probe.exists() or not holdout_probe.exists():
        pytest.skip("saved probe reports are not available in this checkout")

    def scorer(_query: str, candidates: Sequence[Mapping[str, Any]]) -> list[float]:
        return [float(len(candidates) - index) for index, _candidate in enumerate(candidates)]

    report = run_reranker_pilot(
        main_probe=main_probe,
        holdout_probe=holdout_probe,
        output=tmp_path / "out.json",
        scorer=scorer,
    )

    assert [case["case_id"] for case in report["cases"]] == list(PILOT_CASE_IDS)
    assert {case["status"] for case in report["cases"]} == {"reranked"}
    assert all(len(case["candidates"]) == 16 for case in report["cases"])
    assert all(len(case["selected_top4"]) == 4 for case in report["cases"])
    assert all(
        _candidate_chunk_id(case["selected_top4"][0]) is not None for case in report["cases"]
    )


def test_default_cross_encoder_scorer_reuses_one_model_per_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main_probe = _probe(tmp_path / "main.json", PILOT_CASE_IDS[:2])
    holdout_probe = _probe(tmp_path / "holdout.json", PILOT_CASE_IDS[2:])

    class FakeCrossEncoder:
        init_calls: list[dict[str, Any]] = []
        predict_calls = 0

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.init_calls.append({"args": args, "kwargs": kwargs})

        def predict(
            self,
            pairs: Sequence[tuple[str, str]],
            *,
            batch_size: int,
            show_progress_bar: bool,
        ) -> list[float]:
            type(self).predict_calls += 1
            assert batch_size == 8
            assert show_progress_bar is False
            return [float(len(pairs) - index) for index, _pair in enumerate(pairs)]

    fake_sentence_transformers = types.SimpleNamespace(CrossEncoder=FakeCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)
    reranker_pilot._load_cross_encoder.cache_clear()

    try:
        report = run_reranker_pilot(
            main_probe=main_probe,
            holdout_probe=holdout_probe,
            output=tmp_path / "out.json",
        )
    finally:
        reranker_pilot._load_cross_encoder.cache_clear()

    assert {case["status"] for case in report["cases"]} == {"reranked"}
    assert len(FakeCrossEncoder.init_calls) == 1
    assert FakeCrossEncoder.predict_calls == 4
    init_call = FakeCrossEncoder.init_calls[0]
    assert init_call["args"] == (MODEL_NAME,)
    assert init_call["kwargs"] == {
        "max_length": 384,
        "device": "cpu",
        "trust_remote_code": False,
    }

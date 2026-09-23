from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.server.app.rag_structural_pilot import (
    compare_passage_captures,
    load_gold_span_groups,
    main,
    score_capture,
)


def _bounded_document(
    source: str,
    page: int,
    content: str,
    *,
    rank: int = 1,
    chunk_id: str = "opaque-chunk",
) -> dict[str, Any]:
    return {
        "rank": rank,
        "metadata": {
            "source": source,
            "page": page,
            "chunk_id": chunk_id,
        },
        "content": content,
    }


def _capture(
    case_id: str, documents: list[dict[str, Any]], *, mode: str = "hybrid"
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "mode": mode,
        "question": "redacted in structural scorer",
        "bounded_prompt_documents": documents,
    }


def _write_capture(path: Path, captures: list[dict[str, Any]]) -> Path:
    path.write_text(
        json.dumps({"schema_version": 1, "captures": captures}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _write_gold(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "id": "case-1",
                        "question": "Q",
                        "conversation_history": [],
                        "expected_documents": ["policy.pdf"],
                        "forbidden_clusters": [],
                        "minimum_answer_points": [],
                        "allow_no_calendar_dates_statement": False,
                    }
                ],
                "gold_spans_by_case": {
                    "case-1": {
                        "tags": ["pilot"],
                        "required": [
                            {
                                "id": "must-have",
                                "source": "policy.pdf",
                                "page": 4,
                                "quote": "First required passage",
                            },
                            {
                                "id": "alternative-rule",
                                "source": "policy.pdf",
                                "page": 5,
                                "quote": "Alternative wording A",
                                "alternatives": [
                                    {
                                        "source": "policy.pdf",
                                        "page": 6,
                                        "quote": "Alternative wording B",
                                    },
                                ],
                            },
                        ],
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_load_gold_span_groups_accepts_required_groups_and_alternatives(tmp_path: Path) -> None:
    groups = load_gold_span_groups(_write_gold(tmp_path / "gold.json"))

    assert list(groups) == ["case-1"]
    assert [group.id for group in groups["case-1"]] == ["must-have", "alternative-rule"]
    assert len(groups["case-1"][1].alternatives) == 2


def test_load_gold_span_groups_accepts_answer_bearing_quotes_shape(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(
        json.dumps(
            {
                "cases": [{"id": "case-1"}],
                "gold_spans_by_case": {
                    "case-1": {
                        "source": "policy.docx",
                        "answer_bearing_quotes": [
                            {"normalized_quote": "Primary quote"},
                        ],
                        "alternative_sources": ["policy.pdf"],
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    groups = load_gold_span_groups(gold_path)

    assert groups["case-1"][0].id == "quote-1"
    assert [span.source for span in groups["case-1"][0].alternatives] == [
        "policy.docx",
        "policy.pdf",
    ]


def test_score_capture_matches_normalized_quote_by_source_and_page_not_chunk_id(
    tmp_path: Path,
) -> None:
    groups = load_gold_span_groups(_write_gold(tmp_path / "gold.json"))["case-1"]
    capture = _capture(
        "case-1",
        [
            _bounded_document(
                "policy.pdf",
                4,
                "Noise before FIRST   required\npassage and after.",
                chunk_id="different-from-gold",
            ),
            _bounded_document(
                "policy.pdf",
                6,
                "This chunk contains alternative wording b.",
                rank=2,
            ),
        ],
    )

    score = score_capture(capture, groups, variant="C")

    assert score.all_passages_covered is True
    assert score.covered_gold_groups == 2
    assert score.document_hit_all is True
    assert score.wrong_page_hits == 0
    assert score.matches[0].rank == 1
    assert score.matches[1].rank == 2
    assert 0 < score.useful_chars_ratio <= 1


def test_score_capture_counts_document_hit_and_wrong_page_without_passage_coverage(
    tmp_path: Path,
) -> None:
    groups = load_gold_span_groups(_write_gold(tmp_path / "gold.json"))["case-1"]
    capture = _capture(
        "case-1",
        [
            _bounded_document("policy.pdf", 7, "First required passage", rank=1),
            _bounded_document("policy.pdf", 5, "Unhelpful page from same document", rank=2),
        ],
    )

    score = score_capture(capture, groups, variant="A")

    assert score.all_passages_covered is False
    assert score.any_passage_covered is False
    assert score.document_hit_any is True
    assert score.wrong_page_hits == 1
    assert score.useful_chars_ratio == 0


def test_compare_passage_captures_reports_aggregate_and_c_vs_baseline_pairs(
    tmp_path: Path,
) -> None:
    gold_path = _write_gold(tmp_path / "gold.json")
    no_passage = [_bounded_document("policy.pdf", 4, "Same document but no exact quote")]
    one_passage = [_bounded_document("policy.pdf", 4, "First required passage")]
    all_passages = [
        _bounded_document("policy.pdf", 4, "First required passage", rank=1),
        _bounded_document("policy.pdf", 6, "Alternative wording B", rank=2),
    ]
    a_path = _write_capture(tmp_path / "a.json", [_capture("case-1", no_passage)])
    b_path = _write_capture(tmp_path / "b.json", [_capture("case-1", one_passage)])
    c_path = _write_capture(tmp_path / "c.json", [_capture("case-1", all_passages)])

    report = compare_passage_captures(
        gold_cases_path=gold_path,
        variants={"A": a_path, "B": b_path, "C": c_path},
    )

    assert report["summary"]["A"]["any_passage_covered"] == 0
    assert report["summary"]["B"]["any_passage_covered"] == 1
    assert report["summary"]["C"]["all_passages_covered"] == 1
    assert report["cases"][0]["paired"]["A"]["outcome"] == "c_only_all"
    assert report["cases"][0]["paired"]["B"]["outcome"] == "c_only_all"
    assert report["paired_summary"] == {
        "A": {"c_only_all": 1, "baseline_only_all": 0, "both_all": 0, "neither_all": 0},
        "B": {"c_only_all": 1, "baseline_only_all": 0, "both_all": 0, "neither_all": 0},
    }


def test_compare_passage_captures_counts_structural_losses(tmp_path: Path) -> None:
    gold_path = _write_gold(tmp_path / "gold.json")
    all_passages = [
        _bounded_document("policy.pdf", 4, "First required passage", rank=1),
        _bounded_document("policy.pdf", 6, "Alternative wording B", rank=2),
    ]
    no_passage = [_bounded_document("policy.pdf", 4, "Same document but no exact quote")]
    a_path = _write_capture(tmp_path / "a.json", [_capture("case-1", all_passages)])
    b_path = _write_capture(tmp_path / "b.json", [_capture("case-1", no_passage)])
    c_path = _write_capture(tmp_path / "c.json", [_capture("case-1", no_passage)])

    report = compare_passage_captures(
        gold_cases_path=gold_path,
        variants={"A": a_path, "B": b_path, "C": c_path},
    )

    assert report["cases"][0]["paired"]["A"]["outcome"] == "baseline_only_all"
    assert report["cases"][0]["paired"]["B"]["outcome"] == "neither_all"
    assert report["paired_summary"] == {
        "A": {"c_only_all": 0, "baseline_only_all": 1, "both_all": 0, "neither_all": 0},
        "B": {"c_only_all": 0, "baseline_only_all": 0, "both_all": 0, "neither_all": 1},
    }


def test_compare_passage_captures_does_not_count_partial_gain_as_go_win(
    tmp_path: Path,
) -> None:
    gold_path = _write_gold(tmp_path / "gold.json")
    no_passage = [_bounded_document("policy.pdf", 4, "Same document but no exact quote")]
    one_passage = [_bounded_document("policy.pdf", 4, "First required passage")]
    a_path = _write_capture(tmp_path / "a.json", [_capture("case-1", no_passage)])
    b_path = _write_capture(tmp_path / "b.json", [_capture("case-1", no_passage)])
    c_path = _write_capture(tmp_path / "c.json", [_capture("case-1", one_passage)])

    report = compare_passage_captures(
        gold_cases_path=gold_path,
        variants={"A": a_path, "B": b_path, "C": c_path},
    )

    c_vs_a = report["cases"][0]["paired"]["A"]
    assert c_vs_a["outcome"] == "neither_all"
    assert c_vs_a["passage_coverage_delta"] == 0.5
    assert report["paired_summary"]["A"]["c_only_all"] == 0
    assert report["paired_summary"]["A"]["neither_all"] == 1


def test_cli_writes_structural_pilot_report(tmp_path: Path, monkeypatch: Any) -> None:
    gold_path = _write_gold(tmp_path / "gold.json")
    capture_path = _write_capture(
        tmp_path / "capture.json",
        [
            _capture(
                "case-1",
                [
                    _bounded_document("policy.pdf", 4, "First required passage", rank=1),
                    _bounded_document("policy.pdf", 6, "Alternative wording B", rank=2),
                ],
            )
        ],
    )
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "rag_structural_pilot",
            "--gold-cases",
            str(gold_path),
            "--a",
            str(capture_path),
            "--b",
            str(capture_path),
            "--c",
            str(capture_path),
            "--output",
            str(output),
        ],
    )

    main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["C"]["all_passages_covered"] == 1

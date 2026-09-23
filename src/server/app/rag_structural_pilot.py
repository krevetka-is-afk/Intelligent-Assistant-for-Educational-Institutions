from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_GOLD_CASES_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "rag_eval"
    / "structural_chunking_pilot.v1.json"
)
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class GoldSpan:
    source: str
    page: int | None
    quote: str
    label: str | None = None


@dataclass(frozen=True, slots=True)
class GoldSpanGroup:
    id: str
    alternatives: tuple[GoldSpan, ...]


@dataclass(frozen=True, slots=True)
class PassageMatch:
    group_id: str
    source: str
    page: int | None
    quote_chars: int
    rank: int | None
    matched: bool
    document_hit: bool
    wrong_page_hit: bool


@dataclass(frozen=True, slots=True)
class CasePassageScore:
    case_id: str
    variant: str
    total_gold_groups: int
    covered_gold_groups: int
    all_passages_covered: bool
    any_passage_covered: bool
    passage_coverage_ratio: float
    document_hit_any: bool
    document_hit_all: bool
    wrong_page_hits: int
    prompt_chars: int
    useful_chars: int
    useful_chars_ratio: float
    matches: list[PassageMatch]


def _normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value.replace("\u00a0", " ")).strip().casefold()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _metadata(document: Mapping[str, Any]) -> Mapping[str, Any]:
    value = document.get("metadata")
    return value if isinstance(value, Mapping) else {}


def _document_source(document: Mapping[str, Any]) -> str | None:
    metadata = _metadata(document)
    for key in ("source", "document_id", "title", "url"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("source", "document_id", "title", "url"):
        value = document.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _document_page(document: Mapping[str, Any]) -> int | None:
    metadata = _metadata(document)
    for key in (
        "page",
        "page_number",
        "pdf_page",
        "source_pdf_page",
        "page_index",
    ):
        if key in metadata:
            return _as_int_or_none(metadata.get(key))
    for key in ("page", "page_number", "pdf_page", "source_pdf_page", "page_index"):
        if key in document:
            return _as_int_or_none(document.get(key))
    return None


def _document_content(document: Mapping[str, Any]) -> str:
    for key in ("content", "page_content", "text", "content_preview", "snippet"):
        value = document.get(key)
        if isinstance(value, str):
            return value
    nested = document.get("document")
    if isinstance(nested, Mapping):
        for key in ("page_content", "content", "text"):
            value = nested.get(key)
            if isinstance(value, str):
                return value
    return ""


def _source_matches(actual: str | None, expected: str) -> bool:
    if actual is None:
        return False
    normalized_actual = _normalize_text(actual)
    normalized_expected = _normalize_text(expected)
    actual_name = _normalize_text(Path(actual).name)
    expected_name = _normalize_text(Path(expected).name)
    return (
        normalized_actual == normalized_expected
        or normalized_expected in normalized_actual
        or normalized_actual in normalized_expected
        or (actual_name == expected_name and actual_name != "")
    )


def _page_matches(actual: int | None, expected: int | None) -> bool:
    return expected is None or actual == expected


def _load_captures(path: Path) -> list[Mapping[str, Any]]:
    payload = _read_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        captures = payload.get("captures")
        if isinstance(captures, list):
            return [item for item in captures if isinstance(item, Mapping)]
    raise ValueError(f"{path} must be a capture list or contain a captures list")


def load_gold_span_groups(path: Path = DEFAULT_GOLD_CASES_PATH) -> dict[str, list[GoldSpanGroup]]:
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("gold fixture must be a JSON object")
    raw_groups_by_case = payload.get("gold_spans_by_case")
    if not isinstance(raw_groups_by_case, Mapping):
        raise ValueError("gold fixture must contain gold_spans_by_case object")

    groups_by_case: dict[str, list[GoldSpanGroup]] = {}
    for case_id, raw_case_gold in raw_groups_by_case.items():
        if not isinstance(case_id, str):
            continue
        raw_groups = _required_gold_groups(raw_case_gold)
        groups: list[GoldSpanGroup] = []
        for index, raw_group in enumerate(raw_groups, start=1):
            group = _parse_gold_group(raw_group, fallback_id=f"span-{index}")
            if group.alternatives:
                groups.append(group)
        groups_by_case[case_id] = groups
    return groups_by_case


def _required_gold_groups(raw_case_gold: Any) -> list[Any]:
    if isinstance(raw_case_gold, Mapping):
        required = raw_case_gold.get("required")
        if isinstance(required, list):
            return required
        answer_bearing_quotes = raw_case_gold.get("answer_bearing_quotes")
        if isinstance(answer_bearing_quotes, list):
            return _answer_bearing_quote_groups(raw_case_gold, answer_bearing_quotes)
        raise ValueError("gold case object must contain required list")
    if isinstance(raw_case_gold, list):
        return raw_case_gold
    raise ValueError("gold_spans_by_case values must be objects or lists")


def _answer_bearing_quote_groups(
    raw_case_gold: Mapping[str, Any], answer_bearing_quotes: Sequence[Any]
) -> list[dict[str, Any]]:
    source = raw_case_gold.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("answer_bearing_quotes gold case must contain source")
    alternative_sources = raw_case_gold.get("alternative_sources")
    alternatives = (
        [item.strip() for item in alternative_sources if isinstance(item, str) and item.strip()]
        if isinstance(alternative_sources, list)
        else []
    )
    groups: list[dict[str, Any]] = []
    for index, raw_quote in enumerate(answer_bearing_quotes, start=1):
        if not isinstance(raw_quote, Mapping):
            raise ValueError("answer_bearing_quotes entries must be objects")
        quote = raw_quote.get("quote") or raw_quote.get("normalized_quote")
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError("answer_bearing_quotes entries must contain quote")
        page = (
            raw_quote.get("page")
            or raw_quote.get("source_pdf_page")
            or raw_case_gold.get("source_pdf_page")
            or raw_case_gold.get("page")
        )
        groups.append(
            {
                "id": str(raw_quote.get("id") or raw_quote.get("label") or f"quote-{index}"),
                "source": source,
                "page": page,
                "quote": quote,
                "alternatives": [
                    {"source": alternative_source, "page": page, "quote": quote}
                    for alternative_source in alternatives
                ],
            }
        )
    return groups


def _parse_gold_group(raw_group: Any, *, fallback_id: str) -> GoldSpanGroup:
    if not isinstance(raw_group, Mapping):
        raise ValueError("gold span group must be an object")
    raw_alternatives = raw_group.get("alternatives") or raw_group.get("any_of")
    if isinstance(raw_alternatives, list):
        alternatives = _parse_gold_group_alternatives(raw_group, raw_alternatives)
    else:
        alternatives = (_parse_gold_span(raw_group),)
    group_id = raw_group.get("id") or raw_group.get("label") or fallback_id
    return GoldSpanGroup(id=str(group_id), alternatives=alternatives)


def _parse_gold_group_alternatives(
    raw_group: Mapping[str, Any], raw_alternatives: Sequence[Any]
) -> tuple[GoldSpan, ...]:
    alternatives: list[GoldSpan] = []
    if isinstance(raw_group.get("source"), str) or isinstance(raw_group.get("quote"), str):
        alternatives.append(_parse_gold_span(raw_group))
    alternatives.extend(_parse_gold_span(item) for item in raw_alternatives)
    return tuple(alternatives)


def _parse_gold_span(raw_span: Any) -> GoldSpan:
    if not isinstance(raw_span, Mapping):
        raise ValueError("gold span must be an object")
    source = raw_span.get("source")
    quote = raw_span.get("quote")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("gold span must contain non-empty source")
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError("gold span must contain non-empty quote")
    label = raw_span.get("label")
    return GoldSpan(
        source=source.strip(),
        page=_as_int_or_none(raw_span.get("page")),
        quote=quote.strip(),
        label=label.strip() if isinstance(label, str) and label.strip() else None,
    )


def score_capture(
    capture: Mapping[str, Any],
    gold_groups: Sequence[GoldSpanGroup],
    *,
    variant: str,
) -> CasePassageScore:
    case_id = str(capture.get("case_id") or "")
    raw_documents = capture.get("bounded_prompt_documents")
    documents = (
        [item for item in raw_documents if isinstance(item, Mapping)]
        if isinstance(raw_documents, list)
        else []
    )

    prompt_chars = sum(len(_document_content(document)) for document in documents)
    useful_document_indexes: set[int] = set()
    matches: list[PassageMatch] = []

    for group in gold_groups:
        group_match = _best_group_match(group, documents)
        matches.append(group_match)
        if group_match.matched and group_match.rank is not None:
            useful_document_indexes.add(group_match.rank - 1)

    covered = sum(1 for match in matches if match.matched)
    useful_chars = sum(
        len(_document_content(document))
        for index, document in enumerate(documents)
        if index in useful_document_indexes
    )
    total_groups = len(gold_groups)
    document_hits = [match.document_hit for match in matches]
    return CasePassageScore(
        case_id=case_id,
        variant=variant,
        total_gold_groups=total_groups,
        covered_gold_groups=covered,
        all_passages_covered=total_groups > 0 and covered == total_groups,
        any_passage_covered=covered > 0,
        passage_coverage_ratio=(covered / total_groups) if total_groups else 0.0,
        document_hit_any=any(document_hits),
        document_hit_all=bool(document_hits) and all(document_hits),
        wrong_page_hits=sum(1 for match in matches if match.wrong_page_hit),
        prompt_chars=prompt_chars,
        useful_chars=useful_chars,
        useful_chars_ratio=(useful_chars / prompt_chars) if prompt_chars else 0.0,
        matches=matches,
    )


def _best_group_match(group: GoldSpanGroup, documents: Sequence[Mapping[str, Any]]) -> PassageMatch:
    fallback = group.alternatives[0]
    best_document_hit: PassageMatch | None = None
    best_wrong_page: PassageMatch | None = None
    for span in group.alternatives:
        normalized_quote = _normalize_text(span.quote)
        for rank, document in enumerate(documents, start=1):
            source_matches = _source_matches(_document_source(document), span.source)
            page_matches = _page_matches(_document_page(document), span.page)
            if not source_matches:
                continue
            content_matches = normalized_quote in _normalize_text(_document_content(document))
            candidate = PassageMatch(
                group_id=group.id,
                source=span.source,
                page=span.page,
                quote_chars=len(span.quote),
                rank=rank,
                matched=content_matches and page_matches,
                document_hit=True,
                wrong_page_hit=content_matches and not page_matches,
            )
            if candidate.matched:
                return candidate
            if candidate.wrong_page_hit and best_wrong_page is None:
                best_wrong_page = candidate
            if best_document_hit is None:
                best_document_hit = candidate
    if best_wrong_page is not None:
        return best_wrong_page
    if best_document_hit is not None:
        return best_document_hit
    return PassageMatch(
        group_id=group.id,
        source=fallback.source,
        page=fallback.page,
        quote_chars=len(fallback.quote),
        rank=None,
        matched=False,
        document_hit=False,
        wrong_page_hit=False,
    )


def _captures_by_case(captures: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    by_case: dict[str, Mapping[str, Any]] = {}
    for capture in captures:
        case_id = capture.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            continue
        if capture.get("mode") == "hybrid" or case_id not in by_case:
            by_case[case_id] = capture
    return by_case


def compare_passage_captures(
    *,
    gold_cases_path: Path,
    variants: Mapping[str, Path],
) -> dict[str, Any]:
    gold_by_case = load_gold_span_groups(gold_cases_path)
    captures_by_variant = {
        variant: _captures_by_case(_load_captures(path)) for variant, path in variants.items()
    }
    case_ids = sorted(gold_by_case)
    case_reports: list[dict[str, Any]] = []
    score_by_variant: dict[str, list[CasePassageScore]] = {variant: [] for variant in variants}

    for case_id in case_ids:
        per_variant: dict[str, CasePassageScore | None] = {}
        for variant in variants:
            capture = captures_by_variant[variant].get(case_id)
            if capture is None:
                per_variant[variant] = None
                continue
            score = score_capture(capture, gold_by_case[case_id], variant=variant)
            per_variant[variant] = score
            score_by_variant[variant].append(score)
        case_reports.append(
            {
                "case_id": case_id,
                "gold_groups": len(gold_by_case[case_id]),
                "variants": {
                    variant: _score_payload(score) if score is not None else None
                    for variant, score in per_variant.items()
                },
                "paired": _paired_payload(per_variant),
            }
        )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "gold_cases_path": str(gold_cases_path),
        "variants": {variant: str(path) for variant, path in variants.items()},
        "summary": {
            variant: _aggregate_scores(scores) for variant, scores in score_by_variant.items()
        },
        "paired_summary": _aggregate_paired(case_reports),
        "cases": case_reports,
    }


def _score_payload(score: CasePassageScore) -> dict[str, Any]:
    payload = asdict(score)
    payload["matches"] = [asdict(match) for match in score.matches]
    return payload


def _aggregate_scores(scores: Sequence[CasePassageScore]) -> dict[str, Any]:
    case_count = len(scores)
    total_prompt_chars = sum(score.prompt_chars for score in scores)
    total_useful_chars = sum(score.useful_chars for score in scores)
    return {
        "cases": case_count,
        "all_passages_covered": sum(score.all_passages_covered for score in scores),
        "any_passage_covered": sum(score.any_passage_covered for score in scores),
        "document_hit_any": sum(score.document_hit_any for score in scores),
        "document_hit_all": sum(score.document_hit_all for score in scores),
        "wrong_page_hits": sum(score.wrong_page_hits for score in scores),
        "avg_passage_coverage_ratio": (
            sum(score.passage_coverage_ratio for score in scores) / case_count
            if case_count
            else 0.0
        ),
        "useful_chars_ratio": (
            total_useful_chars / total_prompt_chars if total_prompt_chars else 0.0
        ),
        "prompt_chars": total_prompt_chars,
        "useful_chars": total_useful_chars,
    }


def _paired_payload(scores: Mapping[str, CasePassageScore | None]) -> dict[str, Any]:
    structural = scores.get("C")
    if structural is None:
        return {}
    return {
        baseline: _compare_pair(structural, baseline_score)
        for baseline, baseline_score in scores.items()
        if baseline != "C" and baseline_score is not None
    }


def _compare_pair(structural: CasePassageScore, baseline: CasePassageScore) -> dict[str, Any]:
    if structural.all_passages_covered and not baseline.all_passages_covered:
        outcome = "c_only_all"
    elif baseline.all_passages_covered and not structural.all_passages_covered:
        outcome = "baseline_only_all"
    elif structural.all_passages_covered and baseline.all_passages_covered:
        outcome = "both_all"
    else:
        outcome = "neither_all"
    return {
        "outcome": outcome,
        "passage_coverage_delta": (
            structural.passage_coverage_ratio - baseline.passage_coverage_ratio
        ),
        "useful_chars_ratio_delta": structural.useful_chars_ratio - baseline.useful_chars_ratio,
        "wrong_page_delta": structural.wrong_page_hits - baseline.wrong_page_hits,
    }


def _aggregate_paired(case_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcome_keys = {
        "c_only_all": "c_only_all",
        "baseline_only_all": "baseline_only_all",
        "both_all": "both_all",
        "neither_all": "neither_all",
    }
    aggregate: dict[str, dict[str, int]] = {}
    for report in case_reports:
        paired = report.get("paired")
        if not isinstance(paired, Mapping):
            continue
        for baseline, payload in paired.items():
            if not isinstance(payload, Mapping):
                continue
            outcome = payload.get("outcome")
            if outcome not in outcome_keys:
                continue
            bucket = aggregate.setdefault(
                str(baseline),
                {
                    "c_only_all": 0,
                    "baseline_only_all": 0,
                    "both_all": 0,
                    "neither_all": 0,
                },
            )
            bucket[outcome_keys[outcome]] += 1
    return aggregate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score structural chunking pilot Stage7 captures against exact gold passages."
    )
    parser.add_argument("--gold-cases", type=Path, default=DEFAULT_GOLD_CASES_PATH)
    parser.add_argument("--a", type=Path, required=True, help="Variant A Stage7 capture JSON.")
    parser.add_argument("--b", type=Path, required=True, help="Variant B Stage7 capture JSON.")
    parser.add_argument("--c", type=Path, required=True, help="Variant C Stage7 capture JSON.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = compare_passage_captures(
        gold_cases_path=args.gold_cases,
        variants={"A": args.a, "B": args.b, "C": args.c},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

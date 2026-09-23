from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document

from . import config
from .rag_evidence_models import EvidenceWindowResult
from .vector import RetrievedDocument, get_chunks_by_ids

_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{3,}")


def _tokens(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _TOKEN_RE.finditer(text)}


def _metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved


def _chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}:{chunk_index:05d}"


def _same_version(metadata: dict[str, Any], anchor: dict[str, Any], chunk_index: int) -> bool:
    return (
        metadata.get("document_id") == anchor.get("document_id")
        and metadata.get("source_sha256") == anchor.get("source_sha256")
        and metadata.get("page") == anchor.get("page")
        and _metadata_int(metadata, "chunk_index") == chunk_index
    )


def _range(metadata: dict[str, Any]) -> tuple[int, int] | None:
    start = _metadata_int(metadata, "char_start")
    end = _metadata_int(metadata, "char_end")
    if start is None or end is None or start < 0 or end <= start:
        return None
    return start, end


def _trusted_neighbor(
    *,
    candidate: Document | None,
    anchor_metadata: dict[str, Any],
    expected_chunk_index: int,
    before_anchor: bool,
) -> Document | None:
    if candidate is None:
        return None
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    if not _same_version(metadata, anchor_metadata, expected_chunk_index):
        return None
    neighbor_range = _range(metadata)
    anchor_range = _range(anchor_metadata)
    if neighbor_range is None or anchor_range is None:
        return None
    if before_anchor and not (
        neighbor_range[0] < anchor_range[0] and neighbor_range[1] <= anchor_range[0]
    ):
        return None
    if not before_anchor and not (
        neighbor_range[0] >= anchor_range[1] and neighbor_range[1] > anchor_range[1]
    ):
        return None
    return candidate


def _clip_head(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."


def _clip_tail(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[-limit:]
    return "..." + text[-(limit - 3) :].lstrip()


def _window_text(
    *,
    previous: Document | None,
    anchor: str,
    next_document: Document | None,
    content_limit: int,
) -> str | None:
    if len(anchor) > content_limit:
        return None

    previous_text = str(previous.page_content or "").strip() if previous is not None else ""
    next_text = str(next_document.page_content or "").strip() if next_document is not None else ""
    side_count = int(bool(previous_text)) + int(bool(next_text))
    if side_count == 0:
        return anchor

    separator_budget = 2 * side_count
    side_budget = content_limit - len(anchor) - separator_budget
    if side_budget <= 0:
        return anchor

    if previous_text and next_text:
        previous_budget = side_budget // 2
        next_budget = side_budget - previous_budget
    elif previous_text:
        previous_budget = side_budget
        next_budget = 0
    else:
        previous_budget = 0
        next_budget = side_budget

    parts: list[str] = []
    if previous_text and previous_budget > 0:
        parts.append(_clip_tail(previous_text, previous_budget))
    parts.append(anchor)
    if next_text and next_budget > 0:
        parts.append(_clip_head(next_text, next_budget))
    return "\n\n".join(parts)


def _useful_neighbor(candidate: Document | None, question_terms: set[str]) -> Document | None:
    if candidate is None or not question_terms:
        return candidate
    content_terms = _tokens(str(candidate.page_content or ""))
    return candidate if question_terms & content_terms else None


def _expand_one(
    retrieved: RetrievedDocument,
    *,
    content_limit: int,
    used_fragment_ids: set[str],
    question_terms: set[str],
) -> tuple[RetrievedDocument, dict[str, Any]]:
    metadata = retrieved.document.metadata if isinstance(retrieved.document.metadata, dict) else {}
    document_id = metadata.get("document_id")
    chunk_index = _metadata_int(metadata, "chunk_index")
    source_sha256 = metadata.get("source_sha256")
    if (
        not isinstance(document_id, str)
        or not document_id
        or chunk_index is None
        or not isinstance(source_sha256, str)
        or not source_sha256
        or _range(metadata) is None
    ):
        return retrieved, {"status": "skipped", "reason": "untrusted_anchor_metadata"}

    anchor_id = metadata.get("chunk_id") or _chunk_id(document_id, chunk_index)
    ids = [
        _chunk_id(document_id, chunk_index - 2),
        _chunk_id(document_id, chunk_index - 1),
        str(anchor_id),
        _chunk_id(document_id, chunk_index + 1),
        _chunk_id(document_id, chunk_index + 2),
    ]
    try:
        chunks = get_chunks_by_ids(ids)
    except Exception:
        return retrieved, {"status": "skipped", "reason": "index_read_failed"}

    previous = _trusted_neighbor(
        candidate=chunks.get(ids[1]),
        anchor_metadata=metadata,
        expected_chunk_index=chunk_index - 1,
        before_anchor=True,
    )
    anchor = chunks.get(ids[2])
    next_document = _trusted_neighbor(
        candidate=chunks.get(ids[3]),
        anchor_metadata=metadata,
        expected_chunk_index=chunk_index + 1,
        before_anchor=False,
    )
    if anchor is not None and str(anchor.page_content or "") != str(
        retrieved.document.page_content or ""
    ):
        return retrieved, {"status": "skipped", "reason": "anchor_text_mismatch"}
    if ids[1] in used_fragment_ids:
        previous = None
    if ids[3] in used_fragment_ids:
        next_document = None
    previous = _useful_neighbor(previous, question_terms)
    next_document = _useful_neighbor(next_document, question_terms)

    neighbor_ids: list[str] = []
    if previous is not None:
        neighbor_ids.append(ids[1])
    if next_document is not None:
        neighbor_ids.append(ids[3])
    if not neighbor_ids:
        return retrieved, {"status": "skipped", "reason": "no_trusted_neighbors"}

    edge_candidates: list[tuple[int, str, Document, bool]] = []
    if previous is not None and ids[0] not in used_fragment_ids:
        previous_edge = _trusted_neighbor(
            candidate=chunks.get(ids[0]),
            anchor_metadata=previous.metadata if isinstance(previous.metadata, dict) else {},
            expected_chunk_index=chunk_index - 2,
            before_anchor=True,
        )
        previous_edge = _useful_neighbor(previous_edge, question_terms)
        if previous_edge is not None:
            overlap = len(question_terms & _tokens(str(previous_edge.page_content or "")))
            edge_candidates.append((overlap, ids[0], previous_edge, True))
    if next_document is not None and ids[4] not in used_fragment_ids:
        next_metadata = next_document.metadata if isinstance(next_document.metadata, dict) else {}
        next_edge = _trusted_neighbor(
            candidate=chunks.get(ids[4]),
            anchor_metadata=next_metadata,
            expected_chunk_index=chunk_index + 2,
            before_anchor=False,
        )
        next_edge = _useful_neighbor(next_edge, question_terms)
        if next_edge is not None:
            overlap = len(question_terms & _tokens(str(next_edge.page_content or "")))
            edge_candidates.append((overlap, ids[4], next_edge, False))
    if edge_candidates:
        _overlap, edge_id, edge_document, before_anchor = max(
            edge_candidates,
            key=lambda item: (item[0], item[1]),
        )
        if before_anchor and previous is not None:
            previous = Document(
                page_content=(
                    str(edge_document.page_content or "").strip()
                    + "\n\n"
                    + str(previous.page_content or "").strip()
                ),
                metadata=dict(previous.metadata),
            )
            neighbor_ids.insert(0, edge_id)
        elif not before_anchor and next_document is not None:
            next_document = Document(
                page_content=(
                    str(next_document.page_content or "").strip()
                    + "\n\n"
                    + str(edge_document.page_content or "").strip()
                ),
                metadata=dict(next_document.metadata),
            )
            neighbor_ids.append(edge_id)

    expanded_text = _window_text(
        previous=previous,
        anchor=str(retrieved.document.page_content or "").strip(),
        next_document=next_document,
        content_limit=content_limit,
    )
    if expanded_text is None:
        return retrieved, {"status": "skipped", "reason": "anchor_exceeds_budget"}
    expanded_metadata = dict(metadata)
    expanded_metadata["evidence_window_anchor_chunk_id"] = str(anchor_id)
    expanded_metadata["evidence_window_neighbor_chunk_ids"] = neighbor_ids
    expanded = RetrievedDocument(
        document=Document(
            page_content=expanded_text,
            metadata=expanded_metadata,
            id=getattr(retrieved.document, "id", None),
        ),
        distance=retrieved.distance,
        _retrieval_diagnostics=dict(retrieved._retrieval_diagnostics),
    )
    return expanded, {
        "status": "expanded",
        "anchor_id": str(anchor_id),
        "neighbor_ids": neighbor_ids,
    }


def build_structural_windows(
    retrieved_documents: list[RetrievedDocument],
    *,
    question: str = "",
) -> EvidenceWindowResult:
    if not retrieved_documents:
        return EvidenceWindowResult(
            prompt_documents=[],
            diagnostics={"enabled": True, "documents": []},
        )

    per_document_limit = max(
        1,
        min(
            int(config.RAG_MAX_DOCUMENT_CHARS),
            int(config.RAG_MAX_TOTAL_CONTEXT_CHARS) // max(1, len(retrieved_documents)),
        ),
    )
    prompt_documents: list[RetrievedDocument] = []
    diagnostics: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    used_fragment_ids: set[str] = set()
    question_terms = _tokens(question)
    for retrieved in retrieved_documents:
        expanded, item_diagnostics = _expand_one(
            retrieved,
            content_limit=per_document_limit,
            used_fragment_ids=used_fragment_ids,
            question_terms=question_terms,
        )
        metadata = (
            expanded.document.metadata if isinstance(expanded.document.metadata, dict) else {}
        )
        neighbor_ids = metadata.get("evidence_window_neighbor_chunk_ids")
        anchor_id = metadata.get("evidence_window_anchor_chunk_id") or metadata.get("chunk_id")
        key = (
            metadata.get("chunk_id"),
            metadata.get("document_id"),
            metadata.get("page"),
            metadata.get("chunk_index"),
        )
        if key in seen:
            item_diagnostics = {**item_diagnostics, "status": "skipped", "reason": "duplicate"}
        else:
            seen.add(key)
            prompt_documents.append(expanded)
            if isinstance(anchor_id, str):
                used_fragment_ids.add(anchor_id)
            if isinstance(neighbor_ids, list):
                used_fragment_ids.update(str(item) for item in neighbor_ids)
        diagnostics.append(item_diagnostics)

    return EvidenceWindowResult(
        prompt_documents=prompt_documents,
        diagnostics={
            "enabled": True,
            "per_document_limit": per_document_limit,
            "documents": diagnostics,
        },
    )

import asyncio
import json
import logging
from typing import Any

import pytest
from langchain_core.documents import Document

from src.server.app import rag
from src.server.app.vector import RetrievedDocument


def _chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}:{chunk_index:05d}"


def _metadata(
    *,
    document_id: str = "doc",
    source_sha256: str = "sha",
    page: int = 7,
    chunk_index: int = 1,
    char_start: int = 7,
    char_end: int = 19,
    chunk_id: str | None = None,
) -> dict[str, object]:
    return {
        "document_id": document_id,
        "source_sha256": source_sha256,
        "page": page,
        "chunk_index": chunk_index,
        "char_start": char_start,
        "char_end": char_end,
        "chunk_id": chunk_id or _chunk_id(document_id, chunk_index),
        "source": f"{document_id}.txt",
        "title": f"Document {document_id}",
    }


def _document(content: str, **metadata: Any) -> Document:
    resolved = _metadata(**metadata)
    return Document(page_content=content, metadata=resolved, id=str(resolved["chunk_id"]))


def _retrieved(
    content: str = "second third",
    *,
    distance: float = 0.2,
    diagnostics: dict[str, object] | None = None,
    **metadata: Any,
) -> RetrievedDocument:
    return RetrievedDocument(
        document=_document(content, **metadata),
        distance=distance,
        _retrieval_diagnostics=dict(diagnostics or {}),
    )


def _neighbor_map(anchor: RetrievedDocument, *, previous: Document, next_document: Document):
    anchor_id = str(anchor.document.metadata["chunk_id"])
    previous_id = str(previous.metadata["chunk_id"])
    next_id = str(next_document.metadata["chunk_id"])
    return {previous_id: previous, anchor_id: anchor.document, next_id: next_document}


def _valid_previous(**metadata: object) -> Document:
    return _document(
        "first second",
        chunk_index=0,
        char_start=0,
        char_end=12,
        **metadata,
    )


def _valid_next(**metadata: object) -> Document:
    return _document(
        "third fourth",
        chunk_index=2,
        char_start=14,
        char_end=26,
        **metadata,
    )


def _enable_expansion(monkeypatch: pytest.MonkeyPatch, *, per_doc: int = 1000, total: int = 4000):
    monkeypatch.setattr(rag.config, "RAG_CONTEXT_EXPANSION_ENABLED", True)
    monkeypatch.setattr(rag.config, "RAG_MAX_CONTEXT_DOCUMENTS", 4)
    monkeypatch.setattr(rag.config, "RAG_MAX_DOCUMENT_CHARS", per_doc)
    monkeypatch.setattr(rag.config, "RAG_MAX_TOTAL_CONTEXT_CHARS", total)


def test_expand_context_documents_returns_original_list_when_flag_is_off(monkeypatch):
    monkeypatch.setattr(rag.config, "RAG_CONTEXT_EXPANSION_ENABLED", False)
    retrieved = [_retrieved()]

    def _unexpected_fetch(ids):  # pragma: no cover - assertion path
        raise AssertionError(f"index read was not expected: {ids}")

    monkeypatch.setattr(rag, "get_chunks_by_ids", _unexpected_fetch)

    expanded = rag.expand_context_documents_for_generation(retrieved)

    assert expanded is retrieved


def test_expand_context_documents_orders_same_page_neighbors_and_removes_overlap(monkeypatch):
    _enable_expansion(monkeypatch)
    anchor = _retrieved(distance=0.11, diagnostics={"rank": 1})
    previous = _valid_previous()
    next_document = _valid_next()
    monkeypatch.setattr(
        rag,
        "get_chunks_by_ids",
        lambda ids: _neighbor_map(anchor, previous=previous, next_document=next_document),
    )

    expanded = rag.expand_context_documents_for_generation([anchor])

    assert expanded[0].document.page_content == "first second third fourth"
    assert expanded[0].distance == anchor.distance
    assert expanded[0]._retrieval_diagnostics == {"rank": 1}


def test_expand_context_documents_keeps_source_metadata_from_anchor(monkeypatch):
    _enable_expansion(monkeypatch)
    anchor = _retrieved(source_sha256="anchor-sha", page=3)
    previous = _valid_previous(source_sha256="anchor-sha", page=3)
    next_document = _valid_next(source_sha256="anchor-sha", page=3)
    monkeypatch.setattr(
        rag,
        "get_chunks_by_ids",
        lambda ids: _neighbor_map(anchor, previous=previous, next_document=next_document),
    )

    expanded = rag.expand_context_documents_for_generation([anchor])

    assert expanded[0].document.metadata == anchor.document.metadata


def test_expand_context_documents_falls_back_when_previous_page_differs(monkeypatch):
    _enable_expansion(monkeypatch)
    anchor = _retrieved(page=3)
    previous = _valid_previous(page=2)
    next_document = _valid_next(page=3)
    monkeypatch.setattr(
        rag,
        "get_chunks_by_ids",
        lambda ids: _neighbor_map(anchor, previous=previous, next_document=next_document),
    )

    expanded = rag.expand_context_documents_for_generation([anchor])

    assert expanded[0] is anchor


def test_expand_context_documents_falls_back_when_next_document_differs(monkeypatch):
    _enable_expansion(monkeypatch)
    anchor = _retrieved(document_id="doc-a")
    previous = _valid_previous(document_id="doc-a")
    next_document = _valid_next(document_id="doc-b")
    monkeypatch.setattr(
        rag,
        "get_chunks_by_ids",
        lambda ids: _neighbor_map(anchor, previous=previous, next_document=next_document),
    )

    expanded = rag.expand_context_documents_for_generation([anchor])

    assert expanded[0] is anchor


def test_expand_context_documents_falls_back_when_source_version_differs(monkeypatch):
    _enable_expansion(monkeypatch)
    anchor = _retrieved(source_sha256="sha-a")
    previous = _valid_previous(source_sha256="sha-a")
    next_document = _valid_next(source_sha256="sha-b")
    monkeypatch.setattr(
        rag,
        "get_chunks_by_ids",
        lambda ids: _neighbor_map(anchor, previous=previous, next_document=next_document),
    )

    expanded = rag.expand_context_documents_for_generation([anchor])

    assert expanded[0] is anchor


def test_expand_context_documents_falls_back_when_neighbor_offsets_have_positive_gap(monkeypatch):
    _enable_expansion(monkeypatch)
    anchor = _retrieved(char_start=20, char_end=32)
    previous = _document("first second", chunk_index=0, char_start=0, char_end=12)
    next_document = _document("third fourth", chunk_index=2, char_start=32, char_end=44)
    monkeypatch.setattr(
        rag,
        "get_chunks_by_ids",
        lambda ids: _neighbor_map(anchor, previous=previous, next_document=next_document),
    )

    expanded = rag.expand_context_documents_for_generation([anchor])

    assert expanded[0] is anchor


def test_expand_context_documents_falls_back_when_neighbor_is_missing(monkeypatch):
    _enable_expansion(monkeypatch)
    anchor = _retrieved()
    previous = _valid_previous()
    monkeypatch.setattr(
        rag,
        "get_chunks_by_ids",
        lambda ids: {str(previous.metadata["chunk_id"]): previous},
    )

    expanded = rag.expand_context_documents_for_generation([anchor])

    assert expanded[0] is anchor


def test_expand_context_documents_falls_back_when_index_read_fails(monkeypatch):
    _enable_expansion(monkeypatch)
    anchor = _retrieved()

    def _raise_fetch(ids):
        raise RuntimeError("read failed")

    monkeypatch.setattr(rag, "get_chunks_by_ids", _raise_fetch)

    expanded = rag.expand_context_documents_for_generation([anchor])

    assert expanded[0] is anchor


@pytest.mark.parametrize(
    "bad_metadata",
    [
        {"document_id": ""},
        {"source_sha256": ""},
        {"chunk_index": "not-int"},
        {"chunk_index": 1.2},
        {"chunk_index": True},
        {"char_start": "not-int"},
        {"char_start": 7.5},
        {"char_start": False},
        {"char_end": 7},
        {"char_end": 19.5},
        {"char_end": True},
    ],
)
def test_expand_context_documents_falls_back_when_anchor_metadata_is_malformed(
    monkeypatch, bad_metadata
):
    _enable_expansion(monkeypatch)
    anchor_metadata = _metadata() | bad_metadata
    anchor = RetrievedDocument(
        document=Document(page_content="second third", metadata=anchor_metadata),
        distance=0.2,
    )
    calls: list[list[str]] = []

    def _record_fetch(ids):
        calls.append(list(ids))
        return {}

    monkeypatch.setattr(rag, "get_chunks_by_ids", _record_fetch)

    expanded = rag.expand_context_documents_for_generation([anchor])

    assert expanded[0] is anchor
    assert calls == []


def test_expand_context_documents_falls_back_when_anchor_chunk_id_does_not_match_index(
    monkeypatch,
):
    _enable_expansion(monkeypatch)
    anchor = _retrieved(chunk_id="unexpected-anchor-id")
    previous = _valid_previous()
    next_document = _valid_next()
    fetched_anchor = _document("second third")
    monkeypatch.setattr(
        rag,
        "get_chunks_by_ids",
        lambda ids: {
            str(previous.metadata["chunk_id"]): previous,
            "unexpected-anchor-id": fetched_anchor,
            str(next_document.metadata["chunk_id"]): next_document,
        },
    )

    expanded = rag.expand_context_documents_for_generation([anchor])

    assert expanded[0] is anchor


def test_expand_context_documents_falls_back_when_fetched_anchor_text_differs(monkeypatch):
    _enable_expansion(monkeypatch)
    anchor = _retrieved()
    fetched_anchor = _document("changed anchor text")
    previous = _valid_previous()
    next_document = _valid_next()
    monkeypatch.setattr(
        rag,
        "get_chunks_by_ids",
        lambda ids: {
            str(previous.metadata["chunk_id"]): previous,
            str(fetched_anchor.metadata["chunk_id"]): fetched_anchor,
            str(next_document.metadata["chunk_id"]): next_document,
        },
    )

    expanded = rag.expand_context_documents_for_generation([anchor])

    assert expanded[0] is anchor


def test_expand_context_documents_falls_back_when_text_overlap_contradicts_offsets(monkeypatch):
    _enable_expansion(monkeypatch)
    anchor = _retrieved(content="second third")
    previous = _document("unrelated left", chunk_index=0, char_start=0, char_end=12)
    next_document = _document("unrelated right", chunk_index=2, char_start=14, char_end=28)
    monkeypatch.setattr(
        rag,
        "get_chunks_by_ids",
        lambda ids: _neighbor_map(anchor, previous=previous, next_document=next_document),
    )

    expanded = rag.expand_context_documents_for_generation([anchor])

    assert expanded[0] is anchor


def test_expand_context_documents_falls_back_when_right_text_overlap_is_missing(monkeypatch):
    _enable_expansion(monkeypatch)
    anchor = _retrieved()
    previous = _valid_previous()
    next_document = _document("unrelated right", chunk_index=2, char_start=14, char_end=28)
    monkeypatch.setattr(
        rag,
        "get_chunks_by_ids",
        lambda ids: _neighbor_map(anchor, previous=previous, next_document=next_document),
    )

    expanded = rag.expand_context_documents_for_generation([anchor])

    assert expanded[0] is anchor


def test_expand_context_documents_preserves_anchor_when_clipping_expanded_context(monkeypatch):
    _enable_expansion(monkeypatch, per_doc=14, total=14)
    anchor = _retrieved(content="ANCHOR-CORE", char_start=90, char_end=101)
    previous = _document("L" * 90 + "ANCH", chunk_index=0, char_start=0, char_end=94)
    next_document = _document("CORE" + "R" * 90, chunk_index=2, char_start=97, char_end=191)
    monkeypatch.setattr(
        rag,
        "get_chunks_by_ids",
        lambda ids: _neighbor_map(anchor, previous=previous, next_document=next_document),
    )

    expanded = rag.expand_context_documents_for_generation([anchor])

    assert "ANCHOR-CORE" in expanded[0].document.page_content
    assert len(expanded[0].document.page_content) <= 14


def test_expand_context_documents_falls_back_when_anchor_exceeds_fair_budget(monkeypatch):
    _enable_expansion(monkeypatch, per_doc=100, total=8)
    anchor = _retrieved(content="second third")
    previous = _valid_previous()
    next_document = _valid_next()
    monkeypatch.setattr(
        rag,
        "get_chunks_by_ids",
        lambda ids: _neighbor_map(anchor, previous=previous, next_document=next_document),
    )

    expanded = rag.expand_context_documents_for_generation([anchor])

    assert expanded[0] is anchor


def test_compile_generation_prompt_keeps_four_sources_under_total_budget(monkeypatch):
    _enable_expansion(monkeypatch, per_doc=100, total=80)
    anchors = [
        _retrieved(
            content=f"anchor-{index}",
            distance=0.1 + index,
            document_id=f"doc-{index}",
            chunk_index=1,
            char_start=10,
            char_end=18,
        )
        for index in range(4)
    ]

    def _fetch(ids):
        anchor_id = ids[1]
        document_id = anchor_id.rsplit(":", 1)[0]
        anchor = next(
            item for item in anchors if item.document.metadata["document_id"] == document_id
        )
        return _neighbor_map(
            anchor,
            previous=_document(
                f"before-{document_id} anchor",
                document_id=document_id,
                chunk_index=0,
                char_start=0,
                char_end=17,
            ),
            next_document=_document(
                f"anchor-{document_id.removeprefix('doc-')} after-{document_id}",
                document_id=document_id,
                chunk_index=2,
                char_start=12,
                char_end=30,
            ),
        )

    monkeypatch.setattr(rag, "get_chunks_by_ids", _fetch)

    compiled = rag.compile_generation_prompt(
        question="question",
        retrieved_documents=anchors,
    )

    assert len(compiled.retrieved_documents) == 4
    assert compiled.context_char_count <= 80
    assert [item.distance for item in compiled.retrieved_documents] == [
        item.distance for item in anchors
    ]


def test_ask_question_keeps_public_sources_unexpanded_and_generation_prompt_expanded(
    monkeypatch, caplog
):
    _enable_expansion(monkeypatch, per_doc=1000, total=4000)
    anchors = [
        _retrieved(
            content=f"anchor-{index}",
            distance=0.1 + index,
            document_id=f"doc-{index}",
            chunk_index=1,
            char_start=10,
            char_end=18,
        )
        for index in range(5)
    ]

    def _fetch(ids):
        anchor_id = ids[1]
        document_id = anchor_id.rsplit(":", 1)[0]
        anchor = next(
            item for item in anchors if item.document.metadata["document_id"] == document_id
        )
        return _neighbor_map(
            anchor,
            previous=_document(
                f"before-{document_id} anchor",
                document_id=document_id,
                chunk_index=0,
                char_start=0,
                char_end=17,
            ),
            next_document=_document(
                f"anchor-{document_id.removeprefix('doc-')} after-{document_id}",
                document_id=document_id,
                chunk_index=2,
                char_start=12,
                char_end=30,
            ),
        )

    llm_documents: list[RetrievedDocument] = []

    def _invoke_llm(question, retrieved_documents, conversation_history=None):
        del question, conversation_history
        llm_documents.extend(retrieved_documents)
        return "Ответ подтверждён источниками [1] [4]."

    monkeypatch.setattr(rag.config, "RAG_TOP_K", 5)
    monkeypatch.setattr(rag, "retrieve_documents", lambda *args, **kwargs: (anchors, {}, {}))
    monkeypatch.setattr(
        rag.query_rewrite,
        "rewrite_retrieval_query",
        lambda *args, **kwargs: rag.query_rewrite.QueryRewriteResult(
            query=args[0],
            used=False,
            fallback_reason=None,
            history_used=False,
            diagnostics={},
        ),
    )
    monkeypatch.setattr(rag, "get_chunks_by_ids", _fetch)
    monkeypatch.setattr(rag, "invoke_llm", _invoke_llm)

    with caplog.at_level(logging.INFO):
        result = asyncio.run(rag.ask_question("question"))

    assert [item.distance for item in llm_documents] == [item.distance for item in anchors[:4]]
    assert all("before-doc" in item.document.page_content for item in llm_documents)
    assert [source["content"] for source in result.sources] == [
        f"anchor-{index}" for index in range(4)
    ]
    assert [item.distance for item in result.retrieved_documents] == [
        item.distance for item in anchors[:4]
    ]
    assert "before-doc-0" not in json.dumps(result.sources, ensure_ascii=False)
    assert "before-doc-0" not in caplog.text

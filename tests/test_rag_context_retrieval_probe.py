import json

from langchain_core.documents import Document

from src.server.app import rag_context_retrieval_probe as probe
from src.server.app.rag_evaluation import RAGEvaluationCase
from src.server.app.vector import RetrievedDocument


def _doc(
    chunk_id: str,
    document_id: str,
    *,
    text: str = "content",
    distance: float = 0.2,
    title: str | None = None,
) -> RetrievedDocument:
    metadata = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "source": f"{document_id}.txt",
        "title": title or document_id,
        "chunk_index": 0,
    }
    if chunk_id:
        metadata["chunk_id"] = chunk_id
    return RetrievedDocument(
        document=Document(
            page_content=text,
            metadata=metadata,
        ),
        distance=distance,
        _retrieval_diagnostics={"fixture": True},
    )


def _case() -> RAGEvaluationCase:
    return RAGEvaluationCase(
        id="case-1",
        question="Когда пересдача?",
        conversation_history=[],
        expected_documents=["target-doc"],
        forbidden_clusters=["forbidden-doc"],
        minimum_answer_points=[],
        allow_no_calendar_dates_statement=True,
    )


def test_capture_retrieval_probe_records_full_channels_and_recoverability(monkeypatch, tmp_path):
    dense = [_doc("dense-1", "offtopic-doc", distance=0.1)]
    lexical = [_doc("lex-1", "target-doc", distance=0.9, title="target-doc")]

    monkeypatch.setattr(probe, "_dense_similarity_search_direct", lambda *args, **kwargs: dense)

    def _lexical(query: str, *, k: int):
        assert query == "Когда пересдача?"
        assert k == probe.LEXICAL_LIMIT
        return lexical, True

    monkeypatch.setattr("src.server.app.rag.lexical_similarity_search", _lexical)

    rank_calls: list[int] = []

    def _rank_documents(**kwargs):
        rank_calls.append(kwargs["top_k"])
        rrf_ranked = [
            _doc("dense-1", "offtopic-doc", distance=0.1),
            _doc("rrf-2", "another-doc", distance=0.2),
            _doc("rrf-3", "third-doc", distance=0.3),
            _doc("rrf-4", "fourth-doc", distance=0.4),
            _doc("lex-1", "target-doc", distance=0.5, title="target-doc"),
        ]
        if kwargs["top_k"] == 4:
            return rrf_ranked[:4], {"candidate_count": 2, "selected": [{"rank": 1}]}
        assert kwargs["top_k"] == probe.RRF_LIMIT
        return rrf_ranked, {"candidate_count": 2, "selected": [{"rank": 1}]}

    monkeypatch.setattr(probe, "_rank_documents", _rank_documents)

    capture = probe.capture_retrieval_probe_case(_case(), index_dir=tmp_path)

    assert capture.retrieval_query == "Когда пересдача?"
    assert len(capture.dense_top16) == 1
    assert len(capture.lexical_top16) == 1
    assert len(capture.raw_union_top32) == 2
    assert capture.raw_union_top32[1]["channel_ranks"] == {"lexical": 1}
    assert len(capture.rrf_selected_top16) == 5
    assert len(capture.baseline_top4) == 4
    assert rank_calls == [probe.RRF_LIMIT, 4]
    assert capture.metrics["baseline_hit_at_4"] is False
    assert capture.metrics["rrf_hit_at_16"] is True
    assert capture.metrics["recoverable_rrf_top16"] is True
    assert capture.metrics["rrf_expected_document_ranks"] == {"target-doc": 5}
    assert capture.retrieval_diagnostics["query_rewrite_used"] is False
    assert capture.retrieval_diagnostics["lexical_available"] is True
    assert "baseline_top4" in capture.retrieval_diagnostics


def test_raw_union_uses_production_chunk_key_for_deduplication():
    dense = _doc("", "doc-a", distance=0.1)
    lexical = _doc("", "doc-a", distance=0.9)
    dense.document.metadata.pop("chunk_id", None)
    lexical.document.metadata.pop("chunk_id", None)

    payloads = probe._raw_union_payloads(
        dense_documents=[dense],
        lexical_documents=[lexical],
        expected_documents=[],
        forbidden_clusters=[],
    )

    assert len(payloads) == 1
    assert payloads[0]["channel_ranks"] == {"dense": 1, "lexical": 1}


def test_suite_updates_already_loaded_config_before_lexical_lookup(monkeypatch, tmp_path):
    from src.server.app import config, rag

    cases_path = tmp_path / "cases.json"
    index_dir = tmp_path / "copied-index"
    index_dir.mkdir()
    cases_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "id": "case-1",
                        "question": "Что регулирует target-doc?",
                        "conversation_history": [],
                        "expected_documents": ["target-doc"],
                        "forbidden_clusters": [],
                        "minimum_answer_points": [],
                        "allow_no_calendar_dates_statement": False,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        probe,
        "_dense_similarity_search_direct",
        lambda *args, **kwargs: [_doc("dense-1", "target-doc", title="target-doc")],
    )

    observed: dict[str, object] = {}

    def _lexical(query: str, *, k: int):
        observed["vector_db_dir"] = config.VECTOR_DB_DIR
        observed["lexical_index_path"] = config.LEXICAL_INDEX_PATH
        observed["rag_config_is_config"] = rag.config is config
        return [], True

    monkeypatch.setattr(rag, "lexical_similarity_search", _lexical)
    monkeypatch.setattr(
        probe,
        "_rank_documents",
        lambda **kwargs: (kwargs["dense_documents"][: kwargs["top_k"]], {}),
    )

    captures = probe.capture_retrieval_probe_suite(
        index_dir=index_dir,
        cases_path=cases_path,
        limit=1,
    )

    assert len(captures) == 1
    assert observed == {
        "vector_db_dir": index_dir.resolve(),
        "lexical_index_path": (index_dir / "lexical_index.sqlite3").resolve(),
        "rag_config_is_config": True,
    }


def test_main_writes_report_without_generation(monkeypatch, tmp_path):
    cases_path = tmp_path / "cases.json"
    output_path = tmp_path / "probe.json"
    index_dir = tmp_path / "index"
    cases_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "id": "case-1",
                        "question": "Что регулирует target-doc?",
                        "conversation_history": [],
                        "expected_documents": ["target-doc"],
                        "forbidden_clusters": [],
                        "minimum_answer_points": [],
                        "allow_no_calendar_dates_statement": False,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    dense = [_doc("dense-1", "target-doc", text="Target evidence", title="target-doc")]
    monkeypatch.setattr(probe, "_dense_similarity_search_direct", lambda *args, **kwargs: dense)
    monkeypatch.setattr(
        "src.server.app.rag.lexical_similarity_search",
        lambda query, *, k: ([], True),
    )

    def _rank_documents(**kwargs):
        return kwargs["dense_documents"][: kwargs["top_k"]], {
            "candidate_count": 1,
            "selected": [],
        }

    monkeypatch.setattr(probe, "_rank_documents", _rank_documents)

    def _unexpected_generation(*args, **kwargs):
        raise AssertionError("retrieval probe must not call generation")

    monkeypatch.setattr("src.server.app.rag.invoke_llm", _unexpected_generation)
    monkeypatch.setattr(
        "src.server.app.rag._prompt_compiler.compile",
        _unexpected_generation,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "rag_context_retrieval_probe",
            "--index-dir",
            str(index_dir),
            "--cases",
            str(cases_path),
            "--output",
            str(output_path),
            "--limit",
            "1",
        ],
    )

    probe.main()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["settings"]["generation_enabled"] is False
    assert report["settings"]["query_rewrite_enabled"] is False
    assert report["summary"]["cases"] == 1
    assert report["summary"]["baseline_hit_at_4"] == 1
    assert report["captures"][0]["dense_top16"][0]["text"] == "Target evidence"

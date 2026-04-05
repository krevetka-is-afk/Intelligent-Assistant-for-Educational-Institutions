def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_metrics_endpoint_requires_api_key(client):
    r = client.get("/metrics")
    assert r.status_code == 401


def test_metrics_endpoint_available_with_api_key(client, auth_headers):
    r = client.get("/metrics", headers=auth_headers)
    assert r.status_code == 200
    assert "rag_requests_total" in r.text


def test_prepare_rag_runtime_indexes_empty_store(monkeypatch):
    from src.server.app import main
    from src.server.app.document_ingestion import IndexingSummary
    from src.server.app.vector import EmptyVectorStoreError

    monkeypatch.setattr(main.config, "PREPARE_RAG_ON_STARTUP", True)
    monkeypatch.setattr(main.config, "AUTO_INDEX_ON_STARTUP", True)

    calls: list[str] = []
    state = {"attempt": 0}

    def fake_ensure_vector_store_ready():
        state["attempt"] += 1
        if state["attempt"] == 1:
            raise EmptyVectorStoreError("Vector index is empty")
        return 12

    def fake_index_directory(*args, **kwargs):
        calls.append("index")
        return IndexingSummary(
            files_seen=3,
            indexed_files=3,
            skipped_files=0,
            failed_files=0,
            chunks_written=12,
        )

    monkeypatch.setattr(main, "ensure_vector_store_ready", fake_ensure_vector_store_ready)
    monkeypatch.setattr(main, "index_directory", fake_index_directory)
    monkeypatch.setattr(main, "clear_vector_cache", lambda: calls.append("clear"))

    main._prepare_rag_runtime()

    assert calls == ["clear", "index", "clear"]

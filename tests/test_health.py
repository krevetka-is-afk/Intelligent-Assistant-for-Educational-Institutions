import asyncio

import pytest
from starlette.testclient import TestClient


def _mark_web_auth_ready():
    from src.server.app.readiness import readiness_state

    readiness_state.mark_web_auth_ready()


def _mark_service_ready(indexed_chunks: int = 4):
    from src.server.app.readiness import readiness_state

    readiness_state.mark_web_auth_ready()
    readiness_state.mark_rag_ready(indexed_chunks=indexed_chunks)


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_liveness_endpoints_do_not_check_dependencies(client, monkeypatch):
    from src.server.app import main

    def fail_if_called():
        raise AssertionError("liveness must not check the vector store")

    monkeypatch.setattr(main, "ensure_vector_store_ready", fail_if_called)

    for endpoint in ("/health", "/live"):
        response = client.get(endpoint)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_ready_returns_503_while_initializing(client):
    from src.server.app.readiness import readiness_state

    readiness_state.reset()

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "initializing"
    assert response.json()["checks"] == {"rag": "initializing", "web_auth": "initializing"}


def test_web_auth_db_init_success_marks_web_auth_ready(monkeypatch):
    from src.server.app import main
    from src.server.app.readiness import readiness_state

    calls: list[str] = []

    async def fake_init_auth_db():
        calls.append("init")

    async def fake_dispose_auth_db():
        calls.append("dispose")

    async def fake_rag_startup_worker(generation=None):
        readiness_state.mark_rag_initializing(
            generation=generation,
            preparation_skipped=True,
        )

    monkeypatch.setattr(main, "init_auth_db", fake_init_auth_db)
    monkeypatch.setattr(main, "dispose_auth_db", fake_dispose_auth_db)
    monkeypatch.setattr(main, "_rag_startup_worker", fake_rag_startup_worker)

    async def exercise_lifespan():
        async with main.lifespan(main.app):
            await asyncio.sleep(0)
            return readiness_state.snapshot()

    payload = asyncio.run(exercise_lifespan())

    assert calls == ["init", "dispose"]
    assert payload["status"] == "initializing"
    assert payload["checks"] == {"rag": "initializing", "web_auth": "ready"}


def test_web_auth_db_init_failure_keeps_live_ok_and_ready_failed(monkeypatch, caplog):
    from src.server.app import main

    async def fail_init_auth_db():
        raise RuntimeError("auth db unavailable")

    async def fake_dispose_auth_db():
        return None

    async def fake_rag_startup_worker(generation=None):
        from src.server.app.readiness import readiness_state

        readiness_state.mark_rag_initializing(
            generation=generation,
            preparation_skipped=True,
        )

    monkeypatch.setattr(main, "init_auth_db", fail_init_auth_db)
    monkeypatch.setattr(main, "dispose_auth_db", fake_dispose_auth_db)
    monkeypatch.setattr(main, "_rag_startup_worker", fake_rag_startup_worker)

    with caplog.at_level("ERROR", logger="server"):
        with TestClient(main.app) as local_client:
            live_response = local_client.get("/live")
            ready_response = local_client.get("/ready")

    assert live_response.status_code == 200
    assert live_response.json() == {"status": "ok"}
    assert ready_response.status_code == 503
    assert ready_response.json()["status"] == "failed"
    assert ready_response.json()["checks"] == {"rag": "initializing", "web_auth": "failed"}
    assert ready_response.json()["reason"] == "web_auth_db_init_failed"
    assert any(
        record.message == "Web auth database initialization failed"
        and getattr(record, "dependency", None) == "web_auth_db"
        and getattr(record, "error_type", None) == "web_auth_db_init_failed"
        for record in caplog.records
    )


def test_successful_preparation_marks_service_ready(client, monkeypatch):
    from src.server.app import main
    from src.server.app.readiness import readiness_state

    readiness_state.reset()
    _mark_web_auth_ready()
    monkeypatch.setattr(main.config, "PREPARE_RAG_ON_STARTUP", True)
    monkeypatch.setattr(main, "ensure_vector_store_ready", lambda: 7)

    main._prepare_rag_runtime()
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["checks"] == {"rag": "ready", "web_auth": "ready"}
    assert response.json()["details"]["rag"]["indexed_chunks"] == 7


def test_skipped_startup_preparation_does_not_mark_ready(client, monkeypatch):
    from src.server.app import main
    from src.server.app.readiness import readiness_state

    readiness_state.reset()
    _mark_web_auth_ready()
    monkeypatch.setattr(main.config, "PREPARE_RAG_ON_STARTUP", False)

    main._prepare_rag_runtime()
    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "initializing"
    assert response.json()["checks"] == {"rag": "initializing", "web_auth": "ready"}
    assert response.json()["details"]["rag"]["preparation_skipped"] is True


def test_startup_preparation_failure_marks_service_failed(client, monkeypatch):
    from src.server.app import main
    from src.server.app.readiness import readiness_state

    readiness_state.reset()
    _mark_web_auth_ready()

    def fail_preparation():
        raise RuntimeError("test startup failure")

    monkeypatch.setattr(main, "_prepare_rag_runtime", fail_preparation)

    asyncio.run(main._rag_startup_worker())
    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "failed"
    assert response.json()["reason"] == "rag_startup_preparation_failed"
    assert response.json()["checks"]["rag"] == "failed"
    assert response.json()["checks"]["web_auth"] == "ready"


def test_empty_vector_index_startup_failure_marks_service_failed(client, monkeypatch):
    from src.server.app import main
    from src.server.app.readiness import readiness_state
    from src.server.app.vector import EmptyVectorStoreError

    readiness_state.reset()
    _mark_web_auth_ready()
    monkeypatch.setattr(main.config, "PREPARE_RAG_ON_STARTUP", True)
    monkeypatch.setattr(main.config, "AUTO_INDEX_ON_STARTUP", False)

    def fail_vector_store_check():
        raise EmptyVectorStoreError("Vector index is empty")

    monkeypatch.setattr(main, "ensure_vector_store_ready", fail_vector_store_check)

    asyncio.run(main._rag_startup_worker())
    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "failed"
    assert response.json()["reason"] == "rag_startup_preparation_failed"


def test_runtime_vector_store_failure_marks_service_failed(client, monkeypatch):
    from src.server.app import main
    from src.server.app.readiness import readiness_state

    readiness_state.reset()
    _mark_web_auth_ready()
    monkeypatch.setattr(main.config, "PREPARE_RAG_ON_STARTUP", True)

    def fail_vector_store_check():
        raise RuntimeError("vector runtime unavailable")

    monkeypatch.setattr(main, "ensure_vector_store_ready", fail_vector_store_check)

    asyncio.run(main._rag_startup_worker())
    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "failed"
    assert response.json()["reason"] == "rag_startup_preparation_failed"


@pytest.mark.parametrize(
    ("exception", "expected_code", "expected_reason"),
    [
        (
            "empty",
            "vector_index_empty",
            "rag_runtime_vector_index_empty",
        ),
        (
            "unavailable",
            "vector_store_unavailable",
            "rag_runtime_vector_store_unavailable",
        ),
    ],
)
def test_runtime_vector_store_failure_from_request_marks_readiness_failed(
    client, monkeypatch, auth_headers, exception, expected_code, expected_reason
):
    from src.server.app import rag
    from src.server.app.vector import EmptyVectorStoreError, VectorStoreUnavailableError

    _mark_service_ready(indexed_chunks=4)

    def fail_similarity_search(*args, **kwargs):
        if exception == "empty":
            raise EmptyVectorStoreError("Vector index is empty")
        raise VectorStoreUnavailableError("Similarity search failed")

    monkeypatch.setattr(rag, "similarity_search", fail_similarity_search)

    failed_request = client.post(
        "/ask",
        json={"question": "Что есть в документах?"},
        headers=auth_headers,
    )
    response = client.get("/ready")
    live_response = client.get("/live")

    assert failed_request.status_code == 503
    assert failed_request.json()["code"] == expected_code
    assert response.status_code == 503
    assert response.json()["status"] == "failed"
    assert response.json()["reason"] == expected_reason
    assert response.json()["checks"] == {"rag": "failed", "web_auth": "ready"}
    assert live_response.status_code == 200
    assert live_response.json() == {"status": "ok"}


def test_stale_rag_startup_worker_does_not_update_new_lifecycle(client, monkeypatch):
    from src.server.app import main
    from src.server.app.readiness import readiness_state

    stale_generation = readiness_state.reset()
    _mark_web_auth_ready()
    readiness_state.reset()
    _mark_web_auth_ready()
    monkeypatch.setattr(main.config, "PREPARE_RAG_ON_STARTUP", True)
    monkeypatch.setattr(main, "ensure_vector_store_ready", lambda: 7)

    asyncio.run(main._rag_startup_worker(stale_generation))
    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "initializing"
    assert response.json()["checks"] == {"rag": "initializing", "web_auth": "ready"}
    assert "details" not in response.json()


def test_late_rag_ready_does_not_overwrite_web_auth_failure(client):
    from src.server.app.readiness import readiness_state

    readiness_state.reset()
    readiness_state.mark_web_auth_failed("web_auth_db_init_failed")
    readiness_state.mark_rag_ready(indexed_chunks=7)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "failed"
    assert response.json()["checks"] == {"rag": "ready", "web_auth": "failed"}
    assert response.json()["reason"] == "web_auth_db_init_failed"


def test_llm_fallback_degrades_readiness_and_success_recovers(client, monkeypatch):
    from src.server.app import main
    from src.server.app.rag import RAGResponse

    _mark_service_ready(indexed_chunks=4)

    async def fallback_response(*args, **kwargs):
        return RAGResponse(
            answer="fallback",
            sources=[],
            metadata={"fallback_used": True, "fallback_reason": "llm_unavailable"},
            retrieved_documents=[object()],
        )

    monkeypatch.setattr(main, "ask_question", fallback_response)
    asyncio.run(main._call_ask_question("question"))

    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["checks"] == {"rag": "degraded", "web_auth": "ready"}
    assert response.json()["reason"] == "llm_unavailable"

    async def successful_response(*args, **kwargs):
        return RAGResponse(
            answer="answer",
            sources=[],
            metadata={"fallback_used": False, "fallback_reason": None},
            retrieved_documents=[object()],
        )

    monkeypatch.setattr(main, "ask_question", successful_response)
    asyncio.run(main._call_ask_question("question"))

    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["checks"] == {"rag": "ready", "web_auth": "ready"}
    assert "reason" not in response.json()


def test_liveness_endpoints_stay_ok_when_readiness_is_degraded(client, monkeypatch):
    from src.server.app import main
    from src.server.app.rag import RAGResponse

    _mark_service_ready(indexed_chunks=4)

    async def fallback_response(*args, **kwargs):
        return RAGResponse(
            answer="fallback",
            sources=[],
            metadata={"fallback_used": True, "fallback_reason": "llm_unavailable"},
            retrieved_documents=[object()],
        )

    monkeypatch.setattr(main, "ask_question", fallback_response)
    asyncio.run(main._call_ask_question("question"))

    ready_response = client.get("/ready")
    assert ready_response.status_code == 200
    assert ready_response.json()["status"] == "degraded"

    for endpoint in ("/health", "/live"):
        response = client.get(endpoint)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_policy_fallback_does_not_degrade_readiness(client, monkeypatch):
    from src.server.app import main
    from src.server.app.rag import RAGResponse

    _mark_service_ready(indexed_chunks=4)

    async def policy_fallback_response(*args, **kwargs):
        return RAGResponse(
            answer="refusal",
            sources=[],
            metadata={
                "fallback_used": True,
                "fallback_reason": "policy_output_violation",
            },
            retrieved_documents=[object()],
        )

    monkeypatch.setattr(main, "ask_question", policy_fallback_response)
    asyncio.run(main._call_ask_question("question"))

    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_llm_fallback_without_search_result_does_not_degrade(client, monkeypatch):
    from src.server.app import main
    from src.server.app.rag import RAGResponse

    _mark_service_ready(indexed_chunks=4)

    async def empty_search_fallback_response(*args, **kwargs):
        return RAGResponse(
            answer="fallback",
            sources=[],
            metadata={"fallback_used": True, "fallback_reason": "llm_unavailable"},
            retrieved_documents=[],
        )

    monkeypatch.setattr(main, "ask_question", empty_search_fallback_response)
    asyncio.run(main._call_ask_question("question"))

    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


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

from langchain_core.documents import Document

from src.server.app.rag import RAGResponse
from src.server.app.vector import EmptyVectorStoreError


async def _fake_ask_question(question: str) -> RAGResponse:
    assert question == "Hello world"
    return RAGResponse(
        answer="Ответ найден.",
        sources=[
            {
                "content": "Расписание пересдач опубликовано на портале.",
                "metadata": {"title": "faq", "source": "faq", "page": 2, "chunk_index": 0},
            }
        ],
        metadata={
            "model": "mistral:7b",
            "embedding_model": "cointegrated/rubert-tiny2",
            "num_sources": 1,
            "confidence": 0.91,
            "fallback_used": False,
            "fallback_reason": None,
            "retrieval_time_ms": 5,
            "generation_time_ms": 40,
            "total_time_ms": 45,
        },
        retrieved_documents=[
            type(
                "_Retrieved",
                (),
                {
                    "document": Document(
                        page_content="Расписание пересдач опубликовано на портале.",
                        metadata={"source": "faq", "page": 2},
                    ),
                    "distance": 0.1,
                },
            )()
        ],
    )


def test_ask_returns_compatible_contract(client, monkeypatch):
    monkeypatch.setattr("src.server.app.main.ask_question", _fake_ask_question)

    response = client.post(
        "/ask",
        json={"question": "Hello world"},
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Ответ найден.",
        "sources": [
            {
                "content": "Расписание пересдач опубликовано на портале.",
                "metadata": {"title": "faq", "source": "faq", "page": 2, "chunk_index": 0},
            }
        ],
        "metadata": {
            "model": "mistral:7b",
            "embedding_model": "cointegrated/rubert-tiny2",
            "num_sources": 1,
            "confidence": 0.91,
            "fallback_used": False,
            "fallback_reason": None,
            "retrieval_time_ms": 5,
            "generation_time_ms": 40,
            "total_time_ms": 45,
        },
    }


def test_ask_rejects_empty_question(client, auth_headers):
    response = client.post("/ask", json={"question": "   "}, headers=auth_headers)

    assert response.status_code == 400
    assert response.json() == {"error": "Question must be a non-empty string"}


async def _raise_empty_index(question: str) -> RAGResponse:
    raise EmptyVectorStoreError("Vector index is empty. Run indexing first.")


def test_ask_returns_503_for_empty_index(client, monkeypatch):
    monkeypatch.setattr("src.server.app.main.ask_question", _raise_empty_index)

    response = client.post(
        "/ask",
        json={"question": "Hello world"},
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 503
    assert response.json() == {"error": "Vector index is empty. Run indexing first."}


def test_ask_returns_401_without_api_key(client):
    response = client.post("/ask", json={"question": "Hello world"})

    assert response.status_code == 401
    assert response.json() == {"error": "Unauthorized"}


def test_ask_returns_401_with_invalid_api_key(client):
    response = client.post("/ask", json={"question": "Hello world"}, headers={"X-API-Key": "wrong"})

    assert response.status_code == 401
    assert response.json() == {"error": "Unauthorized"}


def test_web_ask_proxy_does_not_require_api_key(client, monkeypatch):
    monkeypatch.setattr("src.server.app.main.ask_question", _fake_ask_question)

    response = client.post("/web/ask", json={"question": "Hello world"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Ответ найден."


def test_ask_rejects_malformed_json(client, auth_headers):
    response = client.post(
        "/ask",
        content='{"question": ',
        headers={**auth_headers, "Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json() == {"error": "Invalid JSON in request body"}

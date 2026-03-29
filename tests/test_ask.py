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


def test_ask_returns_compatible_contract(client, auth_headers, monkeypatch):
    monkeypatch.setattr("src.server.app.main.ask_question", _fake_ask_question)

    response = client.post("/ask", json={"question": "Hello world"}, headers=auth_headers)

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


def test_ask_returns_503_for_empty_index(client, auth_headers, monkeypatch):
    monkeypatch.setattr("src.server.app.main.ask_question", _raise_empty_index)

    response = client.post("/ask", json={"question": "Hello world"}, headers=auth_headers)

    assert response.status_code == 503
    assert response.json() == {"error": "Vector index is empty. Run indexing first."}

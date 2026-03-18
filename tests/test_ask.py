from langchain_core.documents import Document


class _FakeRetriever:
    def invoke(self, question: str):
        assert question == "Hello world"
        return [
            Document(
                page_content="Расписание пересдач опубликовано на портале.",
                metadata={"source": "faq", "page": 2, "ignored": "drop-me"},
            )
        ]


class _FakeChain:
    def invoke(self, payload):
        assert payload["question"] == "Hello world"
        return "Ответ найден."


def test_ask_returns_compatible_contract(client, monkeypatch):
    monkeypatch.setattr("src.server.app.main.get_retriever", lambda: _FakeRetriever())
    monkeypatch.setattr("src.server.app.main.chain", _FakeChain())

    response = client.post("/ask", json={"question": "Hello world"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Ответ найден.",
        "sources": [
            {
                "content": "Расписание пересдач опубликовано на портале.",
                "metadata": {"title": "faq", "source": "faq", "page": 2},
            }
        ],
        "metadata": {
            "model": "gemma2:2b",
            "num_sources": 1,
        },
    }


def test_ask_rejects_empty_question(client):
    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 400
    assert response.json() == {"error": "Question must be a non-empty string"}

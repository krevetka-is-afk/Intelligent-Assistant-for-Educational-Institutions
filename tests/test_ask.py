import re

from langchain_core.documents import Document
from starlette.testclient import TestClient

from src.server.app.main import app
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


def _bootstrap_admin(client, bootstrap_token: str):
    return client.post(
        "/web/bootstrap",
        data={
            "bootstrap_token": bootstrap_token,
            "username": "admin",
            "password": "admin-password",
        },
        follow_redirects=False,
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
    assert response.json() == {
        "error": "Vector index is empty. Run indexing first.",
        "code": "vector_index_empty",
    }


def test_ask_returns_401_without_api_key(client):
    response = client.post("/ask", json={"question": "Hello world"})

    assert response.status_code == 401
    assert response.json() == {"error": "Unauthorized"}


def test_ask_returns_401_with_invalid_api_key(client):
    response = client.post("/ask", json={"question": "Hello world"}, headers={"X-API-Key": "wrong"})

    assert response.status_code == 401
    assert response.json() == {"error": "Unauthorized"}


def test_web_ask_requires_authentication(client, monkeypatch):
    monkeypatch.setattr("src.server.app.main.ask_question", _fake_ask_question)

    response = client.post("/web/ask", json={"question": "Hello world"})

    assert response.status_code == 401
    assert response.json() == {"error": "Unauthorized"}


def test_web_login_requires_bootstrap_completion(client):
    response = client.post(
        "/web/login",
        data={"username": "admin", "password": "admin-password"},
        follow_redirects=False,
    )

    assert response.status_code == 503


def test_web_bootstrap_rejects_invalid_token(client):
    response = client.post(
        "/web/bootstrap",
        data={
            "bootstrap_token": "wrong-token",
            "username": "admin",
            "password": "admin-password",
        },
        follow_redirects=False,
    )

    assert response.status_code == 401
    assert "Неверный bootstrap token." in response.text


def test_web_bootstrap_creates_admin_session(client, bootstrap_token):
    response = _bootstrap_admin(client, bootstrap_token)

    assert response.status_code == 303
    assert "web_session=" in response.headers["set-cookie"]

    page = client.get("/web")
    assert page.status_code == 200
    assert "Signed in as <strong>admin</strong> (admin)" in page.text


def test_web_ask_accepts_authenticated_web_session(client, monkeypatch, bootstrap_token):
    monkeypatch.setattr("src.server.app.main.ask_question", _fake_ask_question)
    bootstrap_response = _bootstrap_admin(client, bootstrap_token)
    assert bootstrap_response.status_code == 303

    response = client.post("/web/ask", json={"question": "Hello world"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Ответ найден."


def test_web_page_hides_sources_ui_when_disabled(client, monkeypatch, bootstrap_token):
    monkeypatch.setattr("src.server.app.main.config.SHOW_SOURCES", False)

    bootstrap_response = _bootstrap_admin(client, bootstrap_token)
    assert bootstrap_response.status_code == 303

    page = client.get("/web")

    assert page.status_code == 200
    assert 'id="sources-toggle"' not in page.text
    assert "const showSourcesEnabled = false;" in page.text


def test_web_ask_keeps_sources_in_json_when_ui_disabled(client, monkeypatch, bootstrap_token):
    monkeypatch.setattr("src.server.app.main.ask_question", _fake_ask_question)
    monkeypatch.setattr("src.server.app.main.config.SHOW_SOURCES", False)

    bootstrap_response = _bootstrap_admin(client, bootstrap_token)
    assert bootstrap_response.status_code == 303

    response = client.post("/web/ask", json={"question": "Hello world"})

    assert response.status_code == 200
    assert response.json()["sources"] == [
        {
            "content": "Расписание пересдач опубликовано на портале.",
            "metadata": {"title": "faq", "source": "faq", "page": 2, "chunk_index": 0},
        }
    ]


def test_web_invite_activation_creates_user_session(client, monkeypatch, bootstrap_token):
    monkeypatch.setattr("src.server.app.main.ask_question", _fake_ask_question)
    bootstrap_response = _bootstrap_admin(client, bootstrap_token)
    assert bootstrap_response.status_code == 303

    invite_response = client.post(
        "/web/admin/invites",
        data={"recipient_label": "ivan.petrov", "expires_in_hours": 24},
    )
    assert invite_response.status_code == 200
    invite_code_match = re.search(r'<code id="invite-code">([^<]+)</code>', invite_response.text)
    assert invite_code_match is not None
    invite_code = invite_code_match.group(1)

    with TestClient(app) as invited_client:
        accept_response = invited_client.post(
            "/web/invite/accept",
            data={
                "invite_code": invite_code,
                "username": "ivan.petrov",
                "password": "invite-password",
            },
            follow_redirects=False,
        )
        assert accept_response.status_code == 303
        assert "web_session=" in accept_response.headers["set-cookie"]

        ask_response = invited_client.post("/web/ask", json={"question": "Hello world"})
        assert ask_response.status_code == 200
        assert ask_response.json()["answer"] == "Ответ найден."


def test_non_admin_cannot_create_invites(client, bootstrap_token):
    bootstrap_response = _bootstrap_admin(client, bootstrap_token)
    assert bootstrap_response.status_code == 303

    invite_response = client.post(
        "/web/admin/invites",
        data={"recipient_label": "ivan.petrov", "expires_in_hours": 24},
    )
    invite_code_match = re.search(r'<code id="invite-code">([^<]+)</code>', invite_response.text)
    assert invite_code_match is not None
    invite_code = invite_code_match.group(1)

    with TestClient(app) as invited_client:
        accept_response = invited_client.post(
            "/web/invite/accept",
            data={
                "invite_code": invite_code,
                "username": "ivan.petrov",
                "password": "invite-password",
            },
            follow_redirects=False,
        )
        assert accept_response.status_code == 303

        forbidden_response = invited_client.post(
            "/web/admin/invites",
            data={"recipient_label": "petr", "expires_in_hours": 24},
        )
        assert forbidden_response.status_code == 403
        assert "Только администратор может создавать инвайты." in forbidden_response.text


def test_web_logout_invalidates_session(client, monkeypatch, bootstrap_token):
    monkeypatch.setattr("src.server.app.main.ask_question", _fake_ask_question)
    bootstrap_response = _bootstrap_admin(client, bootstrap_token)
    assert bootstrap_response.status_code == 303

    logout_response = client.post("/web/logout", follow_redirects=False)

    assert logout_response.status_code == 303

    response = client.post("/web/ask", json={"question": "Hello world"})

    assert response.status_code == 401


def test_web_ask_accepts_api_key(client, monkeypatch, auth_headers):
    monkeypatch.setattr("src.server.app.main.ask_question", _fake_ask_question)

    response = client.post("/web/ask", json={"question": "Hello world"}, headers=auth_headers)

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

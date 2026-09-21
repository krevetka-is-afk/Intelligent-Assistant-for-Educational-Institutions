import asyncio
import re

import pytest
from langchain_core.documents import Document
from starlette.testclient import TestClient

from src.server.app.main import app, conversation_memory_store
from src.server.app.rag import RAGResponse
from src.server.app.vector import EmptyVectorStoreError


def _rag_metadata(**overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "model": "qwen2.5:3b",
        "embedding_model": "cointegrated/rubert-tiny2",
        "num_sources": 0,
        "confidence": 0.0,
        "fallback_used": False,
        "fallback_reason": None,
        "retrieval_time_ms": 1,
        "generation_time_ms": 1,
        "total_time_ms": 2,
    }
    metadata.update(overrides)
    return metadata


@pytest.fixture(autouse=True)
def reset_conversation_memory_store():
    asyncio.run(conversation_memory_store.clear_all())
    yield
    asyncio.run(conversation_memory_store.clear_all())


async def _fake_ask_question(
    question: str, conversation_history: list[str] | None = None
) -> RAGResponse:
    del conversation_history
    assert question == "Hello world"
    return RAGResponse(
        answer="Ответ найден.",
        sources=[
            {
                "content": "Расписание пересдач опубликовано на портале.",
                "metadata": {"title": "faq", "source": "faq", "page": 2, "chunk_index": 0},
            }
        ],
        metadata=_rag_metadata(
            num_sources=1,
            confidence=0.91,
            retrieval_time_ms=5,
            generation_time_ms=40,
            total_time_ms=45,
        ),
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
        "metadata": _rag_metadata(
            num_sources=1,
            confidence=0.91,
            retrieval_time_ms=5,
            generation_time_ms=40,
            total_time_ms=45,
        ),
    }


def test_ask_rejects_empty_question(client, auth_headers):
    response = client.post("/ask", json={"question": "   "}, headers=auth_headers)

    assert response.status_code == 400
    assert response.json() == {"error": "Question must be a non-empty string"}


@pytest.mark.parametrize(
    "question",
    [
        " ",
        "",
        "\n\t",
        None,
        123,
        [],
        {},
    ],
)
def test_ask_rejects_invalid_question_before_rag(client, auth_headers, monkeypatch, question):
    rag_calls = []

    async def unexpected_rag_call(question, conversation_history=None):
        rag_calls.append(question)
        raise AssertionError("Invalid input must not reach RAG")

    monkeypatch.setattr("src.server.app.main.ask_question", unexpected_rag_call)

    response = client.post(
        "/ask",
        json={
            "question": question,
        },
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert response.json() == {"error": "Question must be a non-empty string"}
    assert rag_calls == []


@pytest.mark.parametrize(
    "question",
    [
        " " + "a" * 501 + " ",
        "a" * 501,
    ],
    ids=["trim-over-limit", "over-limit"],
)
def test_ask_rejects_question_length_over_limit_before_rag(
    client, auth_headers, monkeypatch, question
):
    rag_calls = []

    async def unexpected_rag_call(question, conversation_history=None):
        rag_calls.append(question)
        raise AssertionError("Invalid input must not reach RAG")

    monkeypatch.setattr("src.server.app.main.ask_question", unexpected_rag_call)

    response = client.post("/ask", json={"question": question}, headers=auth_headers)

    assert response.status_code == 400
    assert response.json() == {"error": "Question must not exceed 500 characters"}
    assert rag_calls == []


@pytest.mark.parametrize(
    ("question", "expected_question"),
    [
        ("a" * 499, "a" * 499),
        ("a" * 500, "a" * 500),
        (" " + "a" * 500 + " ", "a" * 500),
    ],
    ids=["below-limit", "at-limit", "trim-before-limit"],
)
def test_ask_accepts_question_length_boundary(
    client, auth_headers, monkeypatch, question, expected_question
):
    rag_calls = []

    async def capture_question(question, conversation_history=None):
        rag_calls.append(question)
        return RAGResponse(
            answer="test-answer",
            sources=[],
            metadata={
                "model": "test",
                "embedding_model": "test",
                "num_sources": 0,
                "confidence": 0.0,
                "fallback_used": False,
                "fallback_reason": None,
                "retrieval_time_ms": 0,
                "generation_time_ms": 0,
                "total_time_ms": 0,
            },
            retrieved_documents=[],
        )

    monkeypatch.setattr("src.server.app.main.ask_question", capture_question)

    response = client.post("/ask", json={"question": question}, headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["answer"] == "test-answer"
    assert rag_calls == [expected_question]


async def _raise_empty_index(
    question: str, conversation_history: list[str] | None = None
) -> RAGResponse:
    del question, conversation_history
    raise EmptyVectorStoreError("Vector index is empty. Run indexing first.")


def test_ask_returns_503_for_empty_index(client, auth_headers, monkeypatch):
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


def test_web_ask_memory_isolated_between_users(client, monkeypatch, bootstrap_token):
    captured_histories: list[list[str]] = []

    async def _capture_ask(
        question: str, conversation_history: list[str] | None = None
    ) -> RAGResponse:
        captured_histories.append(list(conversation_history or []))
        return RAGResponse(
            answer=f"ok: {question}",
            sources=[],
            metadata=_rag_metadata(),
            retrieved_documents=[],
        )

    monkeypatch.setattr("src.server.app.main.ask_question", _capture_ask)
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

        a1 = client.post("/web/ask", json={"question": "A1"})
        assert a1.status_code == 200
        assert captured_histories == [[]]

        b1 = invited_client.post("/web/ask", json={"question": "B1"})
        assert b1.status_code == 200
        assert captured_histories == [[], []]

        a2 = client.post("/web/ask", json={"question": "A2"})
        assert a2.status_code == 200
        assert captured_histories == [[], [], ["A1"]]

        b2 = invited_client.post("/web/ask", json={"question": "B2"})
        assert b2.status_code == 200
        assert captured_histories == [[], [], ["A1"], ["B1"]]


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


def test_ask_rejects_invalid_session_id_without_exception_details(client, auth_headers):
    response = client.post(
        "/ask",
        json={"question": "Hello world", "session_id": {"unexpected": "object"}},
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "session_id must be a non-empty string up to 128 characters"
    }


def test_web_ask_session_memory_keeps_last_five_messages(client, monkeypatch, bootstrap_token):
    captured_histories: list[list[str]] = []

    async def _capture_ask(
        question: str, conversation_history: list[str] | None = None
    ) -> RAGResponse:
        captured_histories.append(list(conversation_history or []))
        return RAGResponse(
            answer=f"Ответ на {question}",
            sources=[],
            metadata=_rag_metadata(),
            retrieved_documents=[],
        )

    monkeypatch.setattr("src.server.app.main.ask_question", _capture_ask)

    bootstrap_response = _bootstrap_admin(client, bootstrap_token)
    assert bootstrap_response.status_code == 303

    for index in range(1, 8):
        response = client.post(
            "/web/ask",
            json={
                "question": f"Q{index}",
            },
        )
        assert response.status_code == 200

    assert captured_histories == [
        [],
        ["Q1"],
        ["Q1", "Q2"],
        ["Q1", "Q2", "Q3"],
        ["Q1", "Q2", "Q3", "Q4"],
        ["Q1", "Q2", "Q3", "Q4", "Q5"],
        ["Q2", "Q3", "Q4", "Q5", "Q6"],
    ]


def test_ask_does_not_use_caller_controlled_session_memory(client, auth_headers, monkeypatch):
    captured_histories: list[list[str]] = []

    async def _capture_ask(
        question: str, conversation_history: list[str] | None = None
    ) -> RAGResponse:
        captured_histories.append(list(conversation_history or []))
        return RAGResponse(
            answer=f"Ответ на {question}",
            sources=[],
            metadata=_rag_metadata(),
            retrieved_documents=[],
        )

    monkeypatch.setattr("src.server.app.main.ask_question", _capture_ask)

    for index in range(2):
        response = client.post(
            "/ask",
            json={"question": f"Q{index}", "session_id": "tg:42"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    assert captured_histories == [
        [],
        [],
    ]


def test_web_ask_with_api_key_does_not_use_caller_controlled_session_memory(
    client, auth_headers, monkeypatch
):
    captured_histories: list[list[str]] = []

    async def _capture_ask(
        question: str, conversation_history: list[str] | None = None
    ) -> RAGResponse:
        captured_histories.append(list(conversation_history or []))
        return RAGResponse(
            answer=f"Ответ на {question}",
            sources=[],
            metadata=_rag_metadata(),
            retrieved_documents=[],
        )

    monkeypatch.setattr("src.server.app.main.ask_question", _capture_ask)

    for index in range(2):
        response = client.post(
            "/web/ask", json={"question": f"Q{index}", "session_id": "52"}, headers=auth_headers
        )
        assert response.status_code == 200

    assert captured_histories == [
        [],
        [],
    ]


def test_web_ask_with_cookie_and_api_key_uses_verified_web_memory(
    client, auth_headers, monkeypatch, bootstrap_token
):
    captured_histories: list[list[str]] = []

    async def _capture_ask(
        question: str, conversation_history: list[str] | None = None
    ) -> RAGResponse:
        captured_histories.append(list(conversation_history or []))
        return RAGResponse(
            answer=f"ok: {question}",
            sources=[],
            metadata=_rag_metadata(),
            retrieved_documents=[],
        )

    monkeypatch.setattr("src.server.app.main.ask_question", _capture_ask)
    bootstrap_response = _bootstrap_admin(client, bootstrap_token)
    assert bootstrap_response.status_code == 303

    first = client.post("/web/ask", json={"question": "Web A1"}, headers=auth_headers)
    second = client.post("/web/ask", json={"question": "Web A2"}, headers=auth_headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert captured_histories == [[], ["Web A1"]]


def test_web_ask_uses_web_user_memory_key(client, monkeypatch, bootstrap_token):
    captured_histories: list[list[str]] = []

    async def _capture_ask(
        question: str, conversation_history: list[str] | None = None
    ) -> RAGResponse:
        captured_histories.append(list(conversation_history or []))
        return RAGResponse(
            answer=f"ok: {question}",
            sources=[],
            metadata=_rag_metadata(),
            retrieved_documents=[],
        )

    monkeypatch.setattr("src.server.app.main.ask_question", _capture_ask)
    bootstrap_response = _bootstrap_admin(client, bootstrap_token)
    assert bootstrap_response.status_code == 303

    first = client.post("/web/ask", json={"question": "Первый вопрос"})
    second = client.post("/web/ask", json={"question": "Второй вопрос"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert captured_histories == [[], ["Первый вопрос"]]


def test_telegram_ask_memory_isolated_between_users(client, telegram_service_headers, monkeypatch):
    captured_histories: list[list[str]] = []

    async def _capture_ask(
        question: str, conversation_history: list[str] | None = None
    ) -> RAGResponse:
        captured_histories.append(list(conversation_history or []))
        return RAGResponse(
            answer=f"ok: {question}",
            sources=[],
            metadata=_rag_metadata(),
            retrieved_documents=[],
        )

    monkeypatch.setattr("src.server.app.main.ask_question", _capture_ask)

    requests = [
        {"telegram_user_id": 101, "question": "A1"},
        {"telegram_user_id": 202, "question": "B1"},
        {"telegram_user_id": 101, "question": "A2"},
        {"telegram_user_id": 202, "question": "B2"},
    ]
    for payload in requests:
        response = client.post(
            "/telegram/ask",
            json=payload,
            headers=telegram_service_headers,
        )
        assert response.status_code == 200

    assert captured_histories == [[], [], ["A1"], ["B1"]]


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-Telegram-Service-Key": "wrong"},
        {"X-API-Key": "test-api-key"},
    ],
    ids=["missing", "wrong", "generic-api-key"],
)
def test_telegram_ask_rejects_invalid_service_key_before_rag(client, monkeypatch, headers):
    rag_calls = []

    async def unexpected_rag_call(question, conversation_history=None):
        rag_calls.append(question)
        raise AssertionError("Unauthorized Telegram request must not reach RAG")

    monkeypatch.setattr("src.server.app.main.ask_question", unexpected_rag_call)

    response = client.post(
        "/telegram/ask",
        json={"telegram_user_id": 101, "question": "Hello world"},
        headers=headers,
    )

    assert response.status_code == 401
    assert response.json() == {"error": "Unauthorized"}
    assert rag_calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "Hello world"},
        {"telegram_user_id": 0, "question": "Hello world"},
        {"telegram_user_id": -1, "question": "Hello world"},
        {"telegram_user_id": "101", "question": "Hello world"},
        {"telegram_user_id": True, "question": "Hello world"},
    ],
    ids=["missing", "zero", "negative", "string", "bool"],
)
def test_telegram_ask_rejects_invalid_telegram_user_id(
    client, telegram_service_headers, monkeypatch, payload
):
    rag_calls = []

    async def unexpected_rag_call(question, conversation_history=None):
        rag_calls.append(question)
        raise AssertionError("Invalid Telegram user id must not reach RAG")

    monkeypatch.setattr("src.server.app.main.ask_question", unexpected_rag_call)

    response = client.post("/telegram/ask", json=payload, headers=telegram_service_headers)

    assert response.status_code == 400
    assert response.json() == {"error": "telegram_user_id must be a positive integer"}
    assert rag_calls == []

import asyncio
import importlib

import httpx
from sqlalchemy import select


def _load_bot_modules(monkeypatch, tmp_path):
    db_path = tmp_path / "bot.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")

    database = importlib.import_module("src.bot.core.database")
    database = importlib.reload(database)
    crud = importlib.import_module("src.bot.core.crud")
    crud = importlib.reload(crud)
    api_client = importlib.import_module("src.bot.api_client")
    api_client = importlib.reload(api_client)
    service = importlib.import_module("src.bot.service")
    service = importlib.reload(service)
    models = importlib.import_module("src.bot.core.models")

    return database, crud, api_client, service, models


def test_ask_api_client_accepts_answer_only(monkeypatch, tmp_path):
    database, _, api_client, _, _ = _load_bot_modules(monkeypatch, tmp_path)

    async def scenario():
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "answer": "Ответ из API",
                    "sources": [{"content": "Документ", "metadata": {"page": 7}}],
                },
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = api_client.AskAPIClient(base_url="http://test", client=http_client)
            result = await client.ask("question")

        assert result.answer == "Ответ из API"
        assert result.sources[0].content == "Документ"
        assert result.sources[0].metadata == {"page": 7}

    try:
        asyncio.run(scenario())
    finally:
        asyncio.run(database.engine.dispose())


def test_process_text_question_saves_history_and_sends_reply(monkeypatch, tmp_path):
    database, _, api_client, service, models = _load_bot_modules(monkeypatch, tmp_path)

    class _FakeAPIClient:
        async def ask(self, question: str):
            assert question == "Когда дедлайн?"
            return api_client.AskResult(
                answer="Дедлайн указан в LMS.",
                sources=[api_client.AskSource(content="LMS", metadata={"source": "portal"})],
            )

    async def scenario():
        await database.init_db()
        sent_messages: list[str] = []

        async def send_reply(text: str) -> None:
            sent_messages.append(text)

        reply = await service.process_text_question(
            telegram_id=101,
            username="student",
            question="Когда дедлайн?",
            send_reply=send_reply,
            api_client=_FakeAPIClient(),
        )

        async with database.async_session_factory() as session:
            stored_request = await session.scalar(select(models.Request))

        assert reply.message == "Дедлайн указан в LMS."
        assert len(reply.sources) == 1
        assert sent_messages == ["Дедлайн указан в LMS."]
        assert stored_request is not None
        assert stored_request.raw_content == "Когда дедлайн?"
        assert stored_request.ai_response == "Дедлайн указан в LMS."

    try:
        asyncio.run(scenario())
    finally:
        asyncio.run(database.engine.dispose())


def test_process_text_question_handles_timeout(monkeypatch, tmp_path, caplog):
    database, _, _, service, models = _load_bot_modules(monkeypatch, tmp_path)
    service.logger.propagate = True

    class _TimeoutAPIClient:
        async def ask(self, question: str):
            raise service.AskAPITimeoutError("timeout")

    async def scenario():
        await database.init_db()
        sent_messages: list[str] = []

        async def send_reply(text: str) -> None:
            sent_messages.append(text)

        await service.process_text_question(
            telegram_id=202,
            username="student",
            question="Есть ли аудитория?",
            send_reply=send_reply,
            api_client=_TimeoutAPIClient(),
        )

        async with database.async_session_factory() as session:
            stored_request = await session.scalar(select(models.Request))

        assert stored_request is not None
        assert stored_request.ai_response == service.TIMEOUT_REPLY_TEXT
        assert sent_messages == [service.TIMEOUT_REPLY_TEXT]

    try:
        with caplog.at_level("WARNING", logger="bot.service"):
            asyncio.run(scenario())
    finally:
        asyncio.run(database.engine.dispose())

    assert "Timed out while processing question" in caplog.text


def test_process_text_question_handles_api_unavailable(monkeypatch, tmp_path, caplog):
    database, _, _, service, models = _load_bot_modules(monkeypatch, tmp_path)
    service.logger.propagate = True

    class _UnavailableAPIClient:
        async def ask(self, question: str):
            raise service.AskAPIUnavailableError("down")

    async def scenario():
        await database.init_db()
        sent_messages: list[str] = []

        async def send_reply(text: str) -> None:
            sent_messages.append(text)

        await service.process_text_question(
            telegram_id=303,
            username="student",
            question="Когда будет ответ?",
            send_reply=send_reply,
            api_client=_UnavailableAPIClient(),
        )

        async with database.async_session_factory() as session:
            stored_request = await session.scalar(select(models.Request))

        assert stored_request is not None
        assert stored_request.ai_response == service.UNAVAILABLE_REPLY_TEXT
        assert sent_messages == [service.UNAVAILABLE_REPLY_TEXT]

    try:
        with caplog.at_level("ERROR", logger="bot.service"):
            asyncio.run(scenario())
    finally:
        asyncio.run(database.engine.dispose())

    assert "API unavailable while processing question" in caplog.text

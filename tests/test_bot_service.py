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
        assert result.metadata == {}

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
                sources=[
                    api_client.AskSource(
                        content="LMS",
                        metadata={"title": "Портал LMS", "page": 3, "source": "portal"},
                    )
                ],
                metadata={"confidence": 0.82, "fallback_used": False},
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

        expected_message = (
            "Дедлайн указан в LMS.\n\n" "Уверенность: 0.82\n\n" "Источники:\n1. Портал LMS, стр. 3"
        )
        assert reply.message == expected_message
        assert len(reply.sources) == 1
        assert reply.metadata == {"confidence": 0.82, "fallback_used": False}
        assert sent_messages == [expected_message]
        assert stored_request is not None
        assert stored_request.raw_content == "Когда дедлайн?"
        assert stored_request.ai_response == expected_message

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


def test_process_question_saves_image_content_type(monkeypatch, tmp_path):
    database, _, api_client, service, models = _load_bot_modules(monkeypatch, tmp_path)

    class _FakeAPIClient:
        async def ask(self, question: str):
            assert question == "Извлеченный текст"
            return api_client.AskResult(
                answer="Ответ по фото.",
                sources=[
                    api_client.AskSource(
                        content="Doc",
                        metadata={"title": "Учебный регламент", "page": 1},
                    )
                ],
                metadata={"confidence": 0.65, "fallback_used": True},
            )

    async def scenario():
        await database.init_db()
        sent_messages: list[str] = []

        async def send_reply(text: str) -> None:
            sent_messages.append(text)

        reply = await service.process_question(
            telegram_id=404,
            username="student",
            question="Извлеченный текст",
            raw_content="Сырой OCR текст",
            content_type="image",
            send_reply=send_reply,
            api_client=_FakeAPIClient(),
        )

        async with database.async_session_factory() as session:
            stored_request = await session.scalar(select(models.Request))

        expected_message = (
            "Ответ по фото.\n\n"
            "Уверенность: 0.65\n"
            "Режим ответа: fallback по найденным документам.\n\n"
            "Источники:\n1. Учебный регламент, стр. 1"
        )
        assert reply.message == expected_message
        assert len(reply.sources) == 1
        assert reply.metadata == {"confidence": 0.65, "fallback_used": True}
        assert sent_messages == [expected_message]
        assert stored_request is not None
        assert stored_request.content_type == "image"
        assert stored_request.raw_content == "Сырой OCR текст"
        assert stored_request.ai_response == expected_message

    try:
        asyncio.run(scenario())
    finally:
        asyncio.run(database.engine.dispose())


def test_split_reply_text_breaks_long_messages(monkeypatch, tmp_path):
    database, _, _, service, _ = _load_bot_modules(monkeypatch, tmp_path)

    long_text = ("Очень длинный ответ " * 400).strip()

    chunks = service._split_reply_text(long_text, max_length=250)

    assert len(chunks) > 1
    assert all(len(chunk) <= 250 for chunk in chunks)
    assert chunks[0].startswith("Очень длинный ответ")
    assert chunks[-1].endswith("ответ")

    asyncio.run(database.engine.dispose())


def test_format_sources_list_deduplicates_title_and_page(monkeypatch, tmp_path):
    database, _, api_client, service, _ = _load_bot_modules(monkeypatch, tmp_path)

    duplicated_sources = [
        api_client.AskSource(content="Doc 1", metadata={"title": "Положение", "page": 5}),
        api_client.AskSource(content="Doc 2", metadata={"title": "Положение", "page": 5}),
        api_client.AskSource(content="Doc 3", metadata={"title": "Справка", "page": 8}),
    ]

    sources_block = service._format_sources_list(duplicated_sources)

    assert sources_block == "Источники:\n1. Положение, стр. 5\n2. Справка, стр. 8"
    asyncio.run(database.engine.dispose())

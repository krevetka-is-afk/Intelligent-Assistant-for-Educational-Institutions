import asyncio
import importlib

from sqlalchemy import func, select


def test_bot_crud_smoke(tmp_path, monkeypatch):
    db_path = tmp_path / "bot.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")

    database = importlib.import_module("src.bot.core.database")
    database = importlib.reload(database)
    crud = importlib.import_module("src.bot.core.crud")
    crud = importlib.reload(crud)
    models = importlib.import_module("src.bot.core.models")

    async def scenario():
        await database.init_db()

        user = await crud.get_or_create_user(telegram_id=42, username="student")
        same_user = await crud.get_or_create_user(telegram_id=42, username="student_updated")
        request = await crud.create_request(
            user_id=user.id,
            content_type="text",
            raw_content="Когда консультация?",
            ai_response="Смотрите расписание кафедры.",
        )

        async with database.async_session_factory() as session:
            user_count = await session.scalar(select(func.count()).select_from(models.User))
            request_count = await session.scalar(select(func.count()).select_from(models.Request))
            stored_user = await session.scalar(
                select(models.User).where(models.User.telegram_id == 42)
            )

        assert same_user.id == user.id
        assert stored_user.username == "student_updated"
        assert request.id is not None
        assert user_count == 1
        assert request_count == 1

        await database.engine.dispose()

    asyncio.run(scenario())

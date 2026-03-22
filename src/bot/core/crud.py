from core.database import async_session
from core.models import Query, User
from sqlalchemy import select


async def get_or_create_user(telegram_id: int, username: str | None = None) -> User:
    safe_username = username if username is not None else f"user_{telegram_id}"
    async with async_session() as session:
        result = await session.execute(select(User).where(User.telegram_id == telegram_id))
        user = result.scalar_one_or_none()
        if not user:
            user = User(telegram_id=telegram_id, username=safe_username)
            session.add(user)
            await session.commit()
            await session.refresh(user)
        return user


async def create_query(
    user_id: int,
    content_type: str,
    question: str,
    answer: str,
) -> Query:
    async with async_session() as session:
        query = Query(
            user_id=user_id,
            content_type=content_type,
            question=question[:4000],
            answer=(answer or "")[:8000],
        )
        session.add(query)
        await session.commit()
        await session.refresh(query)
        return query

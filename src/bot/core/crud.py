from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from .database import async_session_factory
from .models import Request, User


async def get_or_create_user(telegram_id: int, username: str | None = None) -> User:
    safe_username = username or f"user_{telegram_id}"

    async with async_session_factory() as session:
        result = await session.execute(select(User).where(User.telegram_id == telegram_id))
        user = result.scalar_one_or_none()
        if user is not None:
            if username and user.username != username:
                user.username = username
                await session.commit()
                await session.refresh(user)
            return user

        user = User(telegram_id=telegram_id, username=safe_username)
        session.add(user)

        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            result = await session.execute(select(User).where(User.telegram_id == telegram_id))
            existing_user = result.scalar_one()
            if username and existing_user.username != username:
                existing_user.username = username
                await session.commit()
                await session.refresh(existing_user)
            return existing_user

        await session.refresh(user)
        return user


async def create_request(
    user_id: int,
    content_type: str,
    raw_content: str | None,
    ai_response: str | None,
) -> Request:
    async with async_session_factory() as session:
        request = Request(
            user_id=user_id,
            content_type=content_type,
            raw_content=(raw_content or "")[:4000],
            ai_response=(ai_response or "")[:8000],
        )
        session.add(request)
        await session.commit()
        await session.refresh(request)
        return request

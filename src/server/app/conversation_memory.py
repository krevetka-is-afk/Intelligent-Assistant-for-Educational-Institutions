from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, func, select

from .auth_database import async_session_factory
from .auth_models import ConversationMemoryMessage


class ConversationMemoryStore:
    """DB-backed windowed memory with TTL and session eviction for server deployments."""

    def __init__(self, *, max_messages: int, ttl_seconds: float, max_sessions: int) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")

        self._max_messages = max_messages
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions

    async def get_recent_user_messages(self, key: str) -> list[str]:
        normalized_key = self._normalize_key(key)
        if not normalized_key:
            return []

        cutoff = self._ttl_cutoff()
        async with async_session_factory() as session:
            await session.execute(
                delete(ConversationMemoryMessage).where(
                    ConversationMemoryMessage.created_at < cutoff
                )
            )
            result = await session.execute(
                select(ConversationMemoryMessage.message)
                .where(
                    ConversationMemoryMessage.session_key == normalized_key,
                    ConversationMemoryMessage.created_at >= cutoff,
                )
                .order_by(
                    ConversationMemoryMessage.created_at.desc(),
                    ConversationMemoryMessage.id.desc(),
                )
                .limit(self._max_messages)
            )
            await session.commit()
            messages_desc = list(result.scalars())
        return list(reversed(messages_desc))

    async def append_user_message(self, key: str, message: str) -> None:
        normalized_key = self._normalize_key(key)
        normalized_message = self._normalize_message(message)
        if not normalized_key or not normalized_message:
            return

        cutoff = self._ttl_cutoff()
        now = datetime.now(UTC).replace(tzinfo=None)
        async with async_session_factory() as session:
            await session.execute(
                delete(ConversationMemoryMessage).where(
                    ConversationMemoryMessage.created_at < cutoff
                )
            )
            session.add(
                ConversationMemoryMessage(
                    session_key=normalized_key,
                    message=normalized_message,
                    created_at=now,
                )
            )
            await session.flush()

            keep_ids_result = await session.execute(
                select(ConversationMemoryMessage.id)
                .where(ConversationMemoryMessage.session_key == normalized_key)
                .order_by(
                    ConversationMemoryMessage.created_at.desc(),
                    ConversationMemoryMessage.id.desc(),
                )
                .limit(self._max_messages)
            )
            keep_ids = list(keep_ids_result.scalars())
            if keep_ids:
                await session.execute(
                    delete(ConversationMemoryMessage).where(
                        ConversationMemoryMessage.session_key == normalized_key,
                        ConversationMemoryMessage.id.not_in(keep_ids),
                    )
                )

            grouped_sessions_result = await session.execute(
                select(
                    ConversationMemoryMessage.session_key,
                    func.max(ConversationMemoryMessage.id),
                )
                .group_by(ConversationMemoryMessage.session_key)
                .order_by(func.max(ConversationMemoryMessage.id).asc())
            )
            session_keys = [row[0] for row in grouped_sessions_result.all()]
            overflow = len(session_keys) - self._max_sessions
            if overflow > 0:
                evicted_keys = session_keys[:overflow]
                await session.execute(
                    delete(ConversationMemoryMessage).where(
                        ConversationMemoryMessage.session_key.in_(evicted_keys)
                    )
                )

            await session.commit()

    async def clear_all(self) -> None:
        async with async_session_factory() as session:
            await session.execute(delete(ConversationMemoryMessage))
            await session.commit()

    def _ttl_cutoff(self) -> datetime:
        return datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=self._ttl_seconds)

    @staticmethod
    def _normalize_key(key: str) -> str:
        return key.strip()

    @staticmethod
    def _normalize_message(message: str) -> str:
        return message.strip()

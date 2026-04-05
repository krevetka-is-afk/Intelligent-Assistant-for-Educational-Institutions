from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

from passlib.context import CryptContext
from sqlalchemy import func, select

from .auth_database import async_session_factory
from .auth_models import WebInvite, WebSession, WebUser

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
SESSION_TTL_HOURS = 8


class AuthError(RuntimeError):
    """Base auth workflow error."""


class BootstrapAlreadyConfiguredError(AuthError):
    """Raised when bootstrap admin is already configured."""


class InvalidCredentialsError(AuthError):
    """Raised when user credentials are invalid."""


class UsernameAlreadyExistsError(AuthError):
    """Raised when username is already taken."""


class InvalidInviteError(AuthError):
    """Raised when invite code is invalid."""


class ExpiredInviteError(AuthError):
    """Raised when invite code expired or was already used."""


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _normalize_username(username: str) -> str:
    normalized = username.strip().lower()
    if not normalized:
        raise ValueError("Username must be a non-empty string")
    if len(normalized) > 128:
        raise ValueError("Username must not exceed 128 characters")
    return normalized


def _hash_password(password: str) -> str:
    stripped = password.strip()
    if len(stripped) < 8:
        raise ValueError("Password must contain at least 8 characters")
    if len(stripped) > 255:
        raise ValueError("Password must not exceed 255 characters")
    return pwd_context.hash(stripped)


def _verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def has_admin_user() -> bool:
    async with async_session_factory() as session:
        count = await session.scalar(
            select(func.count()).select_from(WebUser).where(WebUser.is_admin.is_(True))
        )
    return bool(count)


async def create_bootstrap_admin(username: str, password: str) -> WebUser:
    normalized_username = _normalize_username(username)
    password_hash = _hash_password(password)

    async with async_session_factory() as session:
        admin_exists = await session.scalar(
            select(func.count()).select_from(WebUser).where(WebUser.is_admin.is_(True))
        )
        if admin_exists:
            raise BootstrapAlreadyConfiguredError("Bootstrap admin is already configured")

        existing_user = await session.scalar(
            select(WebUser).where(WebUser.username == normalized_username)
        )
        if existing_user is not None:
            raise UsernameAlreadyExistsError("Username is already taken")

        user = WebUser(
            username=normalized_username,
            password_hash=password_hash,
            is_admin=True,
            is_active=True,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


async def authenticate_user(username: str, password: str) -> WebUser:
    normalized_username = _normalize_username(username)

    async with async_session_factory() as session:
        user = await session.scalar(
            select(WebUser).where(
                WebUser.username == normalized_username,
                WebUser.is_active.is_(True),
            )
        )
        if user is None or not _verify_password(password, user.password_hash):
            raise InvalidCredentialsError("Invalid username or password")

        user.last_login_at = _utcnow()
        await session.commit()
        await session.refresh(user)
        return user


async def create_invite(
    *,
    created_by_user_id: int,
    recipient_label: str | None,
    expires_in_hours: int,
) -> tuple[str, WebInvite]:
    if expires_in_hours <= 0:
        raise ValueError("Invite expiry must be positive")

    raw_code = secrets.token_urlsafe(24)
    code_hash = _hash_token(raw_code)
    invite = WebInvite(
        code_hash=code_hash,
        recipient_label=(recipient_label or "").strip() or None,
        created_by_user_id=created_by_user_id,
        expires_at=_utcnow() + timedelta(hours=expires_in_hours),
    )

    async with async_session_factory() as session:
        session.add(invite)
        await session.commit()
        await session.refresh(invite)
        return raw_code, invite


async def accept_invite(invite_code: str, username: str, password: str) -> WebUser:
    code_hash = _hash_token(invite_code.strip())
    normalized_username = _normalize_username(username)
    password_hash = _hash_password(password)

    async with async_session_factory() as session:
        invite = await session.scalar(select(WebInvite).where(WebInvite.code_hash == code_hash))
        if invite is None:
            raise InvalidInviteError("Invite code is invalid")
        if invite.used_at is not None or invite.expires_at <= _utcnow():
            raise ExpiredInviteError("Invite code expired or was already used")

        existing_user = await session.scalar(
            select(WebUser).where(WebUser.username == normalized_username)
        )
        if existing_user is not None:
            raise UsernameAlreadyExistsError("Username is already taken")

        user = WebUser(
            username=normalized_username,
            password_hash=password_hash,
            is_admin=False,
            is_active=True,
        )
        session.add(user)
        await session.flush()

        invite.used_at = _utcnow()
        invite.used_by_user_id = user.id

        await session.commit()
        await session.refresh(user)
        return user


async def create_web_session(*, user_id: int, user_agent: str | None) -> str:
    raw_token = secrets.token_urlsafe(32)
    session_record = WebSession(
        token_hash=_hash_token(raw_token),
        user_id=user_id,
        expires_at=_utcnow() + timedelta(hours=SESSION_TTL_HOURS),
        user_agent=(user_agent or "").strip() or None,
    )

    async with async_session_factory() as session:
        session.add(session_record)
        await session.commit()

    return raw_token


async def get_user_by_session_token(token: str) -> WebUser | None:
    token_hash = _hash_token(token)

    async with async_session_factory() as session:
        stmt = (
            select(WebUser)
            .join(WebSession, WebSession.user_id == WebUser.id)
            .where(
                WebSession.token_hash == token_hash,
                WebSession.revoked_at.is_(None),
                WebSession.expires_at > _utcnow(),
                WebUser.is_active.is_(True),
            )
        )
        return await session.scalar(stmt)


async def revoke_session(token: str) -> None:
    token_hash = _hash_token(token)

    async with async_session_factory() as session:
        session_record = await session.scalar(
            select(WebSession).where(
                WebSession.token_hash == token_hash,
                WebSession.revoked_at.is_(None),
            )
        )
        if session_record is None:
            return

        session_record.revoked_at = _utcnow()
        await session.commit()

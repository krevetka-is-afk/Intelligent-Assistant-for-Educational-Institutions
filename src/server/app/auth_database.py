from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from . import config
from .auth_models import Base

engine = create_async_engine(config.WEB_AUTH_DATABASE_URL, echo=False)
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_auth_db() -> None:
    if config.WEB_AUTH_DATABASE_URL.startswith("sqlite"):
        database_path = config.resolve_sqlite_path_from_url(config.WEB_AUTH_DATABASE_URL)
        if database_path is not None:
            database_path.parent.mkdir(parents=True, exist_ok=True)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def dispose_auth_db() -> None:
    await engine.dispose()

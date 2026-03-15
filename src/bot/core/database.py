import core.config as config
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

if not config.DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env")

engine = create_async_engine(config.DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    from core.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

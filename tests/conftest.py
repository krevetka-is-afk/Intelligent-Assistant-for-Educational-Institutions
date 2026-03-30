import asyncio
import os
from pathlib import Path

import pytest
from starlette.testclient import TestClient

os.environ["API_KEY"] = "test-api-key"
os.environ["SHOW_SOURCES"] = "1"
TEST_WEB_AUTH_DB_PATH = (
    Path(__file__).resolve().parent / ".tmp" / "intelligent_assistant_test_web_auth.db"
)
os.environ["WEB_BOOTSTRAP_ADMIN_TOKEN"] = "bootstrap-test-token"
os.environ["WEB_AUTH_DATABASE_URL"] = f"sqlite+aiosqlite:///{TEST_WEB_AUTH_DB_PATH}"
os.environ["PREPARE_RAG_ON_STARTUP"] = "0"
os.environ["AUTO_INDEX_ON_STARTUP"] = "0"

from src.server.app.auth_database import engine  # noqa: E402
from src.server.app.auth_models import Base  # noqa: E402
from src.server.app.main import app, limiter  # noqa: E402

TEST_API_KEY = "test-api-key"
TEST_BOOTSTRAP_TOKEN = "bootstrap-test-token"


async def _reset_web_auth_db() -> None:
    TEST_WEB_AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_web_auth_db():
    asyncio.run(_reset_web_auth_db())
    storage_reset = getattr(limiter._storage, "reset", None)
    if callable(storage_reset):
        storage_reset()
    yield
    asyncio.run(engine.dispose())
    TEST_WEB_AUTH_DB_PATH.unlink(missing_ok=True)


@pytest.fixture
def auth_headers():
    return {"X-API-Key": TEST_API_KEY}


@pytest.fixture
def bootstrap_token():
    return TEST_BOOTSTRAP_TOKEN

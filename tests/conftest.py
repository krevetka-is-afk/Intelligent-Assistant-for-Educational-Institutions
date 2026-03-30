import os

import pytest
from starlette.testclient import TestClient

os.environ["API_KEY"] = "test-api-key"
os.environ["WEB_UI_PASSWORD"] = "test-web-password"

from src.server.app.main import app  # noqa: E402

TEST_API_KEY = "test-api-key"
TEST_WEB_UI_PASSWORD = "test-web-password"


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers():
    return {"X-API-Key": TEST_API_KEY}


@pytest.fixture
def web_auth_password():
    return TEST_WEB_UI_PASSWORD

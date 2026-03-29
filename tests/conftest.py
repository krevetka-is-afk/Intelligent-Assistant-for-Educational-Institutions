# isort: skip_file
import os

os.environ["API_KEY"] = "test-api-key"

import pytest  # noqa: E402
import src.server.app.main as _server_main  # noqa: E402
from src.server.app.main import app  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

_server_main._API_KEY = "test-api-key"

TEST_API_KEY = "test-api-key"


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers():
    return {"X-API-Key": TEST_API_KEY}

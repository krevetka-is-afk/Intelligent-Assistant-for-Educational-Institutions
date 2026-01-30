import pytest
from starlette.testclient import TestClient

from server.app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

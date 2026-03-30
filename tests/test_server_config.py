from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def restore_shared_config_module():
    yield
    module = importlib.import_module("src.server.app.config")
    importlib.reload(module)


def _reload_config(monkeypatch, *, vector_db_dir: str | None, documents_dir: str | None):
    if vector_db_dir is None:
        monkeypatch.delenv("VECTOR_DB_DIR", raising=False)
    else:
        monkeypatch.setenv("VECTOR_DB_DIR", vector_db_dir)

    if documents_dir is None:
        monkeypatch.delenv("DOCUMENTS_DIR", raising=False)
    else:
        monkeypatch.setenv("DOCUMENTS_DIR", documents_dir)

    module = importlib.import_module("src.server.app.config")
    return importlib.reload(module)


def _reload_config_with_auth_url(
    monkeypatch,
    *,
    web_auth_database_url: str | None,
    in_container: bool,
):
    if web_auth_database_url is None:
        monkeypatch.delenv("WEB_AUTH_DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("WEB_AUTH_DATABASE_URL", web_auth_database_url)

    module = importlib.import_module("src.server.app.config")
    monkeypatch.setattr(
        module.Path, "exists", lambda self: str(self) == "/.dockerenv" if in_container else False
    )
    return importlib.reload(module)


def test_config_falls_back_from_docker_vector_path_for_local_runs(monkeypatch):
    config = _reload_config(monkeypatch, vector_db_dir="/data", documents_dir=None)

    assert config.VECTOR_DB_DIR == (Path.cwd() / "src" / "server" / "chrome_langchain_db").resolve()


def test_config_falls_back_from_docker_documents_path_for_local_runs(monkeypatch):
    config = _reload_config(monkeypatch, vector_db_dir=None, documents_dir="/data_and_documents")

    assert config.DOCUMENTS_DIR == (Path.cwd() / "data_and_documents").resolve()


def test_relative_web_auth_database_url_maps_to_data_dir_in_container(monkeypatch):
    config = _reload_config_with_auth_url(
        monkeypatch,
        web_auth_database_url="sqlite+aiosqlite:///./.web_auth.db",
        in_container=True,
    )

    assert config.WEB_AUTH_DATABASE_URL == "sqlite+aiosqlite:////data/.web_auth.db"

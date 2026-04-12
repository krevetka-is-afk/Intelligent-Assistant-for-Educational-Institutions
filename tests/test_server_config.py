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
        module.Path,
        "exists",
        lambda self: self.as_posix() == "/.dockerenv" if in_container else False,
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


def test_show_sources_flag_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SHOW_SOURCES", "0")
    config = _reload_config(monkeypatch, vector_db_dir=None, documents_dir=None)

    assert config.SHOW_SOURCES is False


def test_validate_runtime_config_accepts_memory_window_below_five(monkeypatch):
    monkeypatch.setenv("CONVERSATION_MEMORY_WINDOW", "4")
    config = _reload_config(monkeypatch, vector_db_dir=None, documents_dir=None)

    config.validate_runtime_config()


def test_default_chunk_settings_are_increased_for_rag_quality(monkeypatch):
    config = _reload_config(monkeypatch, vector_db_dir=None, documents_dir=None)

    assert config.CHUNK_SIZE > 500
    assert config.CHUNK_OVERLAP >= 100
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE


def test_chunk_settings_can_be_overridden_from_env(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "1200")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "180")
    config = _reload_config(monkeypatch, vector_db_dir=None, documents_dir=None)

    assert config.CHUNK_SIZE == 1200
    assert config.CHUNK_OVERLAP == 180
    config.validate_chunk_settings()


def test_hybrid_retrieval_settings_are_positive(monkeypatch):
    monkeypatch.setenv("RAG_HYBRID_DENSE_TOP_K", "12")
    monkeypatch.setenv("RAG_HYBRID_LEXICAL_TOP_K", "12")
    monkeypatch.setenv("RAG_HYBRID_RRF_K", "60")
    config = _reload_config(monkeypatch, vector_db_dir=None, documents_dir=None)

    assert config.RAG_HYBRID_DENSE_TOP_K == 12
    assert config.RAG_HYBRID_LEXICAL_TOP_K == 12
    assert config.RAG_HYBRID_RRF_K == 60
    config.validate_runtime_config()


def test_lexical_db_path_can_be_overridden_from_env(monkeypatch, tmp_path):
    custom_path = tmp_path / "custom_lexical.db"
    monkeypatch.setenv("RAG_LEXICAL_DB_PATH", str(custom_path))
    config = _reload_config(monkeypatch, vector_db_dir=None, documents_dir=None)

    assert config.RAG_LEXICAL_DB_PATH == custom_path.resolve()

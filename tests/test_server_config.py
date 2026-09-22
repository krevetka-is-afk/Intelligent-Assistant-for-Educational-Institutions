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


def _reload_config_for_env(monkeypatch, **env: str):
    keys = {
        "APP_ENV",
        "AUTO_INDEX_ON_STARTUP",
        "RAG_MAX_CONTEXT_DOCUMENTS",
        "RAG_MAX_DOCUMENT_CHARS",
        "RAG_MAX_TOTAL_CONTEXT_CHARS",
        "RAG_MAX_HISTORY_MESSAGES",
        "RAG_MAX_HISTORY_CHARS",
        "RAG_SOURCE_SNIPPET_CHARS",
    }
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
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


def test_auto_index_defaults_to_disabled_in_production(monkeypatch):
    config = _reload_config_for_env(monkeypatch, APP_ENV="production")

    assert config.AUTO_INDEX_ON_STARTUP is False


def test_rag_policy_limits_are_configurable(monkeypatch):
    config = _reload_config_for_env(
        monkeypatch,
        RAG_MAX_CONTEXT_DOCUMENTS="2",
        RAG_MAX_DOCUMENT_CHARS="300",
        RAG_MAX_TOTAL_CONTEXT_CHARS="500",
        RAG_MAX_HISTORY_MESSAGES="3",
        RAG_MAX_HISTORY_CHARS="400",
        RAG_SOURCE_SNIPPET_CHARS="120",
    )

    assert config.RAG_MAX_CONTEXT_DOCUMENTS == 2
    assert config.RAG_MAX_DOCUMENT_CHARS == 300
    assert config.RAG_MAX_TOTAL_CONTEXT_CHARS == 500
    assert config.RAG_MAX_HISTORY_MESSAGES == 3
    assert config.RAG_MAX_HISTORY_CHARS == 400
    assert config.RAG_SOURCE_SNIPPET_CHARS == 120


@pytest.mark.parametrize("variable", ["API_KEY", "TELEGRAM_SERVICE_KEY"])
def test_runtime_config_rejects_blank_required_secret(monkeypatch, variable):
    monkeypatch.setenv("API_KEY", "test-api-key")
    monkeypatch.setenv("TELEGRAM_SERVICE_KEY", "test-telegram-service-key")
    monkeypatch.setenv(variable, "  ")

    config = importlib.reload(importlib.import_module("src.server.app.config"))

    with pytest.raises(RuntimeError, match=rf"{variable} is not set"):
        config.validate_runtime_config()


def test_blank_bootstrap_token_disables_bootstrap(monkeypatch):
    monkeypatch.setenv("WEB_BOOTSTRAP_ADMIN_TOKEN", "  ")

    config = importlib.reload(importlib.import_module("src.server.app.config"))

    assert config.WEB_BOOTSTRAP_ADMIN_TOKEN is None

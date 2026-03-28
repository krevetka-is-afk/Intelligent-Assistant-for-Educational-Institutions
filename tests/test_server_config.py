from __future__ import annotations

import importlib
from pathlib import Path


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


def test_config_falls_back_from_docker_vector_path_for_local_runs(monkeypatch):
    config = _reload_config(monkeypatch, vector_db_dir="/data", documents_dir=None)

    assert config.VECTOR_DB_DIR == (Path.cwd() / "src" / "server" / "chrome_langchain_db").resolve()


def test_config_falls_back_from_docker_documents_path_for_local_runs(monkeypatch):
    config = _reload_config(monkeypatch, vector_db_dir=None, documents_dir="/data_and_documents")

    assert config.DOCUMENTS_DIR == (Path.cwd() / "data_and_documents").resolve()

from __future__ import annotations

import importlib

import pytest


def _reload_bot_config(monkeypatch, **env: str) -> object:
    for variable in (
        "DATABASE_URL",
        "API_BASE_URL",
        "RAG_API_URL",
        "TELEGRAM_SERVICE_KEY",
        "BOT_TOKEN",
    ):
        monkeypatch.delenv(variable, raising=False)
    for variable, value in env.items():
        monkeypatch.setenv(variable, value)
    return importlib.reload(importlib.import_module("src.bot.core.config"))


@pytest.mark.parametrize(
    ("variable", "message"),
    [
        ("DATABASE_URL", "DATABASE_URL is not set"),
        ("API_BASE_URL", "API_BASE_URL is not set"),
        ("TELEGRAM_SERVICE_KEY", "TELEGRAM_SERVICE_KEY is not set"),
        ("BOT_TOKEN", "BOT_TOKEN is not set"),
    ],
)
def test_bot_runtime_config_rejects_blank_required_value(monkeypatch, variable, message):
    env = {
        "DATABASE_URL": "sqlite+aiosqlite:///bot.db",
        "API_BASE_URL": "http://server:8000",
        "TELEGRAM_SERVICE_KEY": "telegram-service-key",
        "BOT_TOKEN": "bot-token",
    }
    env[variable] = "  "
    config = _reload_bot_config(monkeypatch, **env)

    with pytest.raises(RuntimeError, match=message):
        config.validate_runtime_config()


def test_bot_runtime_config_normalizes_api_base_url(monkeypatch):
    config = _reload_bot_config(
        monkeypatch,
        DATABASE_URL="sqlite+aiosqlite:///bot.db",
        API_BASE_URL="  http://server:8000/  ",
        TELEGRAM_SERVICE_KEY="telegram-service-key",
        BOT_TOKEN="bot-token",
    )

    assert config.resolve_api_base_url() == "http://server:8000"

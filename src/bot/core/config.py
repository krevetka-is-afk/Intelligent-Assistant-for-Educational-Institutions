import os

from dotenv import load_dotenv

load_dotenv()

APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
API_KEY = os.getenv("API_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
API_BASE_URL = os.getenv("API_BASE_URL")
RAG_API_URL = os.getenv("RAG_API_URL")


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


SHOW_SOURCES = _get_bool_env("SHOW_SOURCES", True)


def resolve_api_base_url() -> str | None:
    if API_BASE_URL:
        return API_BASE_URL.rstrip("/")
    if RAG_API_URL:
        return RAG_API_URL.removesuffix("/").removesuffix("/ask")
    return None


def validate_runtime_config(*, require_bot_token: bool = True) -> None:
    if DATABASE_URL is None:
        raise RuntimeError("DATABASE_URL is not set")
    if resolve_api_base_url() is None:
        raise RuntimeError("API_BASE_URL is not set")
    if API_KEY is None:
        raise RuntimeError("API_KEY is not set")
    if require_bot_token and BOT_TOKEN is None:
        raise RuntimeError("BOT_TOKEN is not set")

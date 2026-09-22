import os

from dotenv import load_dotenv

load_dotenv()

APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
TELEGRAM_SERVICE_KEY = os.getenv("TELEGRAM_SERVICE_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
API_BASE_URL = os.getenv("API_BASE_URL")
RAG_API_URL = os.getenv("RAG_API_URL")


def _has_nonempty_value(value: str | None) -> bool:
    return value is not None and bool(value.strip())


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


SHOW_SOURCES = _get_bool_env("SHOW_SOURCES", True)


def resolve_api_base_url() -> str | None:
    if _has_nonempty_value(API_BASE_URL):
        return API_BASE_URL.strip().rstrip("/")
    if _has_nonempty_value(RAG_API_URL):
        resolved = RAG_API_URL.strip().rstrip("/")
        for suffix in ("/telegram/ask", "/ask"):
            if resolved.endswith(suffix):
                return resolved[: -len(suffix)]
        return resolved
    return None


def validate_runtime_config(*, require_bot_token: bool = True) -> None:
    if not _has_nonempty_value(DATABASE_URL):
        raise RuntimeError("DATABASE_URL is not set")
    if resolve_api_base_url() is None:
        raise RuntimeError("API_BASE_URL is not set")
    if not _has_nonempty_value(TELEGRAM_SERVICE_KEY):
        raise RuntimeError("TELEGRAM_SERVICE_KEY is not set")
    if require_bot_token and not _has_nonempty_value(BOT_TOKEN):
        raise RuntimeError("BOT_TOKEN is not set")

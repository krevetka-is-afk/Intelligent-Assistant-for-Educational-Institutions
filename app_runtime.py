from __future__ import annotations

import logging
import os
from logging.config import dictConfig
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_LOGGING_CONFIGURED_FOR: str | None = None


class ContextDefaultsFilter(logging.Filter):
    def __init__(self, service: str, env: str) -> None:
        super().__init__()
        self.service = service
        self.env = env

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "service", None):
            record.service = self.service
        if not getattr(record, "env", None):
            record.env = self.env

        for field in ("request_id", "endpoint", "stage", "telegram_id", "error_type"):
            if not getattr(record, field, None):
                setattr(record, field, "-")
        if not getattr(record, "web_user_id", None):
            record.web_user_id = "-"
        return True


def getenv(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def require_env(name: str) -> str:
    value = getenv(name)
    if value is None:
        raise RuntimeError(f"{name} is not set")
    return value


def get_log_level(default: str = "INFO") -> str:
    return (getenv("LOG_LEVEL", default) or default).upper()


def get_app_env(default: str = "development") -> str:
    return getenv("APP_ENV", default) or default


def setup_logging(service_name: str, *, default_level: str = "INFO") -> None:
    global _LOGGING_CONFIGURED_FOR
    if _LOGGING_CONFIGURED_FOR == service_name:
        return

    env = get_app_env()
    level = get_log_level(default_level)
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "context": {
                    "()": "app_runtime.ContextDefaultsFilter",
                    "service": service_name,
                    "env": env,
                }
            },
            "formatters": {
                "structured": {
                    "format": (
                        "%(asctime)s %(levelname)s service=%(service)s env=%(env)s "
                        "logger=%(name)s request_id=%(request_id)s endpoint=%(endpoint)s "
                        "stage=%(stage)s telegram_id=%(telegram_id)s web_user_id=%(web_user_id)s "
                        "error_type=%(error_type)s "
                        "%(message)s"
                    )
                }
            },
            "handlers": {
                "stdout": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "filters": ["context"],
                    "formatter": "structured",
                }
            },
            "root": {
                "handlers": ["stdout"],
                "level": level,
            },
        }
    )
    _LOGGING_CONFIGURED_FOR = service_name


def log_extra(**values: Any) -> dict[str, Any]:
    return values


def fatal(message: str, *args: Any, logger_name: str, **extra: Any) -> None:
    setup_logging(logger_name)
    logger = logging.getLogger(logger_name)
    logger.critical(message, *args, extra=log_extra(stage="startup", error_type="config", **extra))
    raise RuntimeError(message % args if args else message)


__all__ = [
    "fatal",
    "get_app_env",
    "get_log_level",
    "getenv",
    "log_extra",
    "require_env",
    "setup_logging",
]

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .api_client import (
    DEFAULT_TIMEOUT_SECONDS,
    AskAPIClient,
    AskAPIResponseError,
    AskAPITimeoutError,
    AskAPIUnavailableError,
    AskSource,
)
from .core.crud import create_request, get_or_create_user

logger = logging.getLogger("bot.service")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.propagate = False

TIMEOUT_REPLY_TEXT = (
    f"Сервис отвечает дольше {int(DEFAULT_TIMEOUT_SECONDS)} секунд. Попробуйте позже."
)
UNAVAILABLE_REPLY_TEXT = "Сервис ответов сейчас недоступен. Попробуйте позже."
INVALID_RESPONSE_REPLY_TEXT = "Не удалось обработать ответ сервиса. Попробуйте позже."

ReplySender = Callable[[str], Awaitable[None]]


@dataclass(slots=True)
class BotReply:
    message: str
    sources: list[AskSource]
    request_id: int


async def process_text_question(
    telegram_id: int,
    username: str | None,
    question: str,
    send_reply: ReplySender,
    api_client: AskAPIClient | None = None,
) -> BotReply:
    client = api_client or AskAPIClient()
    user = await get_or_create_user(telegram_id=telegram_id, username=username)

    reply_text: str
    sources: list[AskSource] = []

    try:
        result = await client.ask(question)
        reply_text = result.answer
        sources = result.sources
    except AskAPITimeoutError:
        logger.warning(
            "Timed out while processing question for telegram_id=%s after %.1f seconds",
            telegram_id,
            DEFAULT_TIMEOUT_SECONDS,
        )
        reply_text = TIMEOUT_REPLY_TEXT
    except AskAPIUnavailableError:
        logger.exception(
            "API unavailable while processing question for telegram_id=%s",
            telegram_id,
        )
        reply_text = UNAVAILABLE_REPLY_TEXT
    except AskAPIResponseError:
        logger.exception("API returned invalid payload for telegram_id=%s", telegram_id)
        reply_text = INVALID_RESPONSE_REPLY_TEXT

    request = await create_request(
        user_id=user.id,
        content_type="text",
        raw_content=question,
        ai_response=reply_text,
    )
    await send_reply(reply_text)

    return BotReply(message=reply_text, sources=sources, request_id=request.id)

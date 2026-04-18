from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from app_runtime import log_extra, setup_logging

from .api_client import (
    DEFAULT_TIMEOUT_SECONDS,
    AskAPIClient,
    AskAPIResponseError,
    AskAPITimeoutError,
    AskAPIUnauthorizedError,
    AskAPIUnavailableError,
    AskSource,
)
from .core import config
from .core.crud import create_request, get_or_create_user

setup_logging("bot")
logger = logging.getLogger("bot.service")

TIMEOUT_REPLY_TEXT = (
    f"Сервис отвечает дольше {int(DEFAULT_TIMEOUT_SECONDS)} секунд. Попробуйте позже."
)
UNAVAILABLE_REPLY_TEXT = "Сервис ответов сейчас недоступен. Попробуйте позже."
UNAUTHORIZED_REPLY_TEXT = "Сервис ответов отклонил запрос. Проверьте конфигурацию доступа."
INVALID_RESPONSE_REPLY_TEXT = "Не удалось обработать ответ сервиса. Попробуйте позже."
EMPTY_INDEX_REPLY_TEXT = "База знаний пока не подготовлена. \
        Обратитесь к администратору и запустите индексацию документов."
TELEGRAM_MESSAGE_LIMIT = 4096
TELEGRAM_WEB_CONTINUATION_NOTICE = "Далее в веб-интерфейсе."
_DOCUMENT_EXTENSIONS = (".pdf", ".docx", ".txt", ".html", ".htm")

ReplySender = Callable[[str], Awaitable[None]]
SUPPORTED_CONTENT_TYPES = frozenset({"text", "image", "pdf"})


@dataclass(slots=True)
class BotReply:
    message: str
    sources: list[AskSource]
    metadata: dict[str, Any]
    request_id: int


def _normalize_source_field(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _truncate_source_title(value: str, *, max_length: int = 120) -> str:
    if len(value) <= max_length:
        return value
    return f"{value[: max_length - 3].rstrip()}..."


def _humanize_source_label(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        return candidate

    if "/" in candidate or "\\" in candidate:
        candidate = PurePosixPath(candidate.replace("\\", "/")).name

    lowered = candidate.lower()
    for extension in _DOCUMENT_EXTENSIONS:
        if lowered.endswith(extension):
            candidate = candidate[: -len(extension)]
            break

    candidate = re.sub(r"[-_\s]+", " ", candidate).strip(" ._-")
    return candidate or value.strip()


def _resolve_source_title(source: AskSource, index: int) -> str:
    metadata = source.metadata
    for key in ("title", "source", "Class Index"):
        title = _normalize_source_field(metadata.get(key))
        if title is not None:
            if key == "Class Index":
                return f"Class {title}"
            return _truncate_source_title(_humanize_source_label(title))

    fallback = _normalize_source_field(source.content.splitlines()[0] if source.content else None)
    if fallback is not None:
        return _truncate_source_title(_humanize_source_label(fallback), max_length=80)
    return f"Источник {index}"


def _format_sources_list(sources: list[AskSource]) -> str:
    unique_sources: list[str] = []
    seen: set[tuple[str, str | None]] = set()

    for source in sources:
        title = _resolve_source_title(source, len(unique_sources) + 1)
        page = _normalize_source_field(source.metadata.get("page"))
        source_key = (title, page)
        if source_key in seen:
            continue

        seen.add(source_key)
        if page is not None:
            unique_sources.append(f"{len(unique_sources) + 1}. {title}, стр. {page}")
        else:
            unique_sources.append(f"{len(unique_sources) + 1}. {title}")

    if not unique_sources:
        return ""

    return "Источники:\n" + "\n".join(unique_sources)


def _format_answer_metadata(metadata: dict[str, Any]) -> str:
    lines: list[str] = []

    confidence = metadata.get("confidence")
    if isinstance(confidence, (int, float)):
        lines.append(f"Уверенность: {float(confidence):.2f}")

    if metadata.get("fallback_used"):
        lines.append("Режим ответа: fallback по найденным документам.")

    return "\n".join(lines)


def _build_reply_text(answer: str, sources: list[AskSource], metadata: dict[str, Any]) -> str:
    normalized_answer = answer.strip()
    # Временно отключено: не показывать в Telegram «Уверенность» и «Источники».
    # metadata_block = _format_answer_metadata(metadata)
    # sources_block = _format_sources_list(sources) if config.SHOW_SOURCES else ""
    # blocks = [normalized_answer]
    # if metadata_block:
    #     blocks.append(metadata_block)
    # if sources_block:
    #     blocks.append(sources_block)
    # return "\n\n".join(blocks)
    return normalized_answer


def _split_reply_text(text: str, *, max_length: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    normalized_text = text.strip()
    if not normalized_text:
        return [""]

    chunks: list[str] = []
    remaining = normalized_text

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        split_at = -1
        for delimiter in ("\n\n", "\n", " "):
            candidate = remaining.rfind(delimiter, 0, max_length + 1)
            if candidate >= max_length // 2:
                split_at = candidate + len(delimiter)
                break

        if split_at == -1:
            truncated = remaining[: max_length - len(TELEGRAM_WEB_CONTINUATION_NOTICE) - 1].rstrip()
            chunks.append(f"{truncated}\n{TELEGRAM_WEB_CONTINUATION_NOTICE}")
            break

        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()

    return chunks


async def _send_reply_chunks(send_reply: ReplySender, reply_text: str) -> None:
    for chunk in _split_reply_text(reply_text):
        await send_reply(chunk)


def _reply_for_api_unavailable(error: AskAPIUnavailableError) -> str:
    if error.error_code == "vector_index_empty":
        return EMPTY_INDEX_REPLY_TEXT
    return UNAVAILABLE_REPLY_TEXT


async def process_question(
    telegram_id: int,
    username: str | None,
    question: str,
    send_reply: ReplySender,
    api_client: AskAPIClient | None = None,
    *,
    content_type: str = "text",
    raw_content: str | None = None,
) -> BotReply:
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("Question must be a non-empty string.")
    if content_type not in SUPPORTED_CONTENT_TYPES:
        raise ValueError(f"Unsupported content_type: {content_type}")

    client = api_client or AskAPIClient()
    user = await get_or_create_user(telegram_id=telegram_id, username=username)

    reply_text: str
    sources: list[AskSource] = []
    metadata: dict[str, Any] = {}

    try:
        conversation_session_id = f"tg:{telegram_id}"
        logger.info(
            "Calling /ask with session_id=%s question_len=%s content_type=%s",
            conversation_session_id,
            len(normalized_question),
            content_type,
            extra=log_extra(
                telegram_id=str(telegram_id),
                endpoint="/ask",
                stage="request",
            ),
        )
        try:
            result = await client.ask(normalized_question, session_id=conversation_session_id)
        except TypeError:
            # Backward compatibility for custom test doubles that still use ask(question).
            result = await client.ask(normalized_question)
        logger.info(
            "Received /ask response for session_id=%s",
            conversation_session_id,
            extra=log_extra(
                telegram_id=str(telegram_id),
                endpoint="/ask",
                stage="response",
            ),
        )
        sources = result.sources if config.SHOW_SOURCES else []
        reply_text = _build_reply_text(result.answer, sources, result.metadata)
        metadata = result.metadata
    except AskAPITimeoutError:
        logger.warning(
            "Timed out while processing question for telegram_id=%s after %.1f seconds",
            telegram_id,
            DEFAULT_TIMEOUT_SECONDS,
            extra=log_extra(
                telegram_id=str(telegram_id),
                stage="network",
                error_type="AskAPITimeoutError",
            ),
        )
        reply_text = TIMEOUT_REPLY_TEXT
        metadata = {}
    except AskAPIUnauthorizedError:
        logger.error(
            "API rejected bot credentials for telegram_id=%s",
            telegram_id,
            extra=log_extra(
                telegram_id=str(telegram_id),
                stage="auth",
                error_type="AskAPIUnauthorizedError",
            ),
        )
        reply_text = UNAUTHORIZED_REPLY_TEXT
        metadata = {}
    except AskAPIUnavailableError as exc:
        log_message = "API unavailable while processing question for telegram_id=%s"
        log_kwargs = {
            "extra": log_extra(
                telegram_id=str(telegram_id),
                stage="network",
                error_type=exc.error_code or "AskAPIUnavailableError",
            )
        }
        if exc.error_code == "vector_index_empty":
            logger.warning(log_message, telegram_id, **log_kwargs)
        else:
            logger.error(log_message, telegram_id, exc_info=True, **log_kwargs)
        reply_text = _reply_for_api_unavailable(exc)
        metadata = {}
    except AskAPIResponseError:
        logger.error(
            "API returned invalid payload for telegram_id=%s",
            telegram_id,
            extra=log_extra(
                telegram_id=str(telegram_id),
                stage="response",
                error_type="AskAPIResponseError",
            ),
            exc_info=True,
        )
        reply_text = INVALID_RESPONSE_REPLY_TEXT
        metadata = {}

    await _send_reply_chunks(send_reply, reply_text)
    request_id = -1
    try:
        request = await create_request(
            user_id=user.id,
            content_type=content_type,
            raw_content=raw_content if raw_content is not None else normalized_question,
            ai_response=reply_text,
        )
        request_id = request.id
    except Exception:
        logger.error(
            "Failed to persist request history for telegram_id=%s",
            telegram_id,
            extra=log_extra(
                telegram_id=str(telegram_id),
                stage="database",
                error_type="database_write_failed",
            ),
            exc_info=True,
        )

    return BotReply(message=reply_text, sources=sources, metadata=metadata, request_id=request_id)


async def process_text_question(
    telegram_id: int,
    username: str | None,
    question: str,
    send_reply: ReplySender,
    api_client: AskAPIClient | None = None,
) -> BotReply:
    return await process_question(
        telegram_id=telegram_id,
        username=username,
        question=question,
        send_reply=send_reply,
        api_client=api_client,
        content_type="text",
        raw_content=question.strip(),
    )

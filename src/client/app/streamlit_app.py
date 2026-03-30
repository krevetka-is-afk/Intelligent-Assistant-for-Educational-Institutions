import logging
import os
import re
from pathlib import PurePosixPath

import requests
import streamlit as st

from app_runtime import log_extra, setup_logging

setup_logging("client")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")
API_URL = f"{API_BASE_URL}/ask"
SHOW_SOURCES = (os.getenv("SHOW_SOURCES", "1") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

logger = logging.getLogger("client")

_DOCUMENT_EXTENSIONS = (".pdf", ".docx", ".txt", ".html", ".htm")


def _normalize_source_field(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _truncate_text(value: str, *, max_length: int = 120) -> str:
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


def _format_source_title(source: dict, index: int) -> str:
    metadata = source.get("metadata", {})
    for key in ("title", "source", "Class Index"):
        title = _normalize_source_field(metadata.get(key))
        if title is not None:
            if key == "Class Index":
                return f"Class {title}"
            return _truncate_text(_humanize_source_label(title))

    content = source.get("content") or ""
    fallback_line = content.splitlines()[0] if content else None
    fallback = _normalize_source_field(fallback_line)
    if fallback is not None:
        return _truncate_text(_humanize_source_label(fallback), max_length=80)
    return f"Источник {index}"


def _format_source_excerpt(source: dict) -> str:
    content = _normalize_source_field(source.get("content"))
    if content is None:
        return "Источник доступен без фрагмента текста."
    return _truncate_text(content, max_length=320)


def _format_source_meta(source: dict, title: str) -> str:
    metadata = source.get("metadata", {})
    parts = []

    page = _normalize_source_field(metadata.get("page"))
    if page is not None:
        parts.append(f"Стр. {page}")

    source_path = _normalize_source_field(metadata.get("source"))
    if source_path is not None:
        humanized_path = _humanize_source_label(source_path)
        if humanized_path and humanized_path != title:
            parts.append(source_path)

    return " | ".join(parts)


def _render_sources(sources: list[dict]) -> None:
    with st.expander("Источники"):
        for index, source in enumerate(sources, start=1):
            title = _format_source_title(source, index)
            st.markdown(f"**{index}. {title}**")
            meta = _format_source_meta(source, title)
            if meta:
                st.caption(meta)
            st.write(_format_source_excerpt(source))


st.title("ask about study process")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        metadata = message.get("metadata", {})
        if metadata:
            meta_parts = []
            if metadata.get("confidence") is not None:
                meta_parts.append(f"confidence: {float(metadata['confidence']):.2f}")
            if metadata.get("fallback_used"):
                meta_parts.append("fallback")
            if SHOW_SOURCES and metadata.get("num_sources") is not None:
                meta_parts.append(f"sources: {metadata['num_sources']}")
            if meta_parts:
                st.caption(" | ".join(meta_parts))

        sources = message.get("sources", [])
        if SHOW_SOURCES and sources:
            _render_sources(sources)

promt = st.chat_input("pass your question here")


def get_response(promt: str):
    logger.info("Sending request to server", extra=log_extra(endpoint="/ask", stage="request"))

    question = promt
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY

    try:
        headers = {"X-API-Key": API_KEY} if API_KEY else {}
        response = requests.post(
            url=API_URL,
            json={"question": question},
            headers=headers,
            timeout=90,
            headers=headers,
        )
        logger.debug(
            "Server responded with status code %s",
            response.status_code,
            extra=log_extra(endpoint="/ask", stage="response"),
        )
        response.raise_for_status()

        payload = response.json()
        return {
            "answer": payload.get("answer", "Пустой ответ от сервера."),
            "metadata": payload.get("metadata", {}),
            "sources": payload.get("sources", []),
        }

    except requests.exceptions.ConnectionError:
        logger.error(
            "Cannot connect to server at %s",
            API_URL,
            extra=log_extra(endpoint="/ask", stage="network", error_type="ConnectionError"),
        )
        return {"answer": "Не удалось подключиться к серверу. Попробуйте позже."}

    except requests.exceptions.Timeout:
        logger.warning(
            "Request to server timed out",
            extra=log_extra(endpoint="/ask", stage="network", error_type="Timeout"),
        )
        return {"answer": "Сервер не ответил вовремя. Попробуйте позже."}

    except requests.exceptions.HTTPError as e:
        logger.error(
            "Server returned HTTP %s: %s",
            e.response.status_code,
            e.response.text,
            extra=log_extra(
                endpoint="/ask",
                stage="response",
                error_type=f"http_{e.response.status_code}",
            ),
        )
        if e.response.status_code == 401:
            return {"answer": "Доступ к API отклонён. Проверьте API_KEY."}
        return {"answer": "Сервер вернул ошибку."}

    except requests.exceptions.RequestException:
        logger.exception(
            "HTTP request to server failed",
            extra=log_extra(endpoint="/ask", stage="network", error_type="RequestException"),
        )
        return {"answer": "Сервер недоступен."}

    except ValueError:
        logger.exception(
            "Failed to decode JSON response",
            extra=log_extra(endpoint="/ask", stage="response", error_type="invalid_json"),
        )
        return {"answer": "Сервер вернул некорректный ответ."}

    except Exception:
        logger.exception(
            "Unexpected client error",
            extra=log_extra(endpoint="/ask", stage="request", error_type="unexpected"),
        )
        return {"answer": "Unexpected error occurred."}


if promt:
    logger.info("User submitted a prompt", extra=log_extra(stage="ui"))
    st.chat_message("user").markdown(promt)
    st.session_state.messages.append({"role": "user", "content": promt})

    response = get_response(promt)
    with st.chat_message("assistant"):
        st.markdown(response["answer"])
        metadata = response.get("metadata", {})
        if metadata:
            meta_parts = []
            if metadata.get("confidence") is not None:
                meta_parts.append(f"confidence: {float(metadata['confidence']):.2f}")
            if metadata.get("fallback_used"):
                meta_parts.append("fallback")
            if SHOW_SOURCES and metadata.get("num_sources") is not None:
                meta_parts.append(f"sources: {metadata['num_sources']}")
            if meta_parts:
                st.caption(" | ".join(meta_parts))
        sources = response.get("sources", [])
        if SHOW_SOURCES and sources:
            _render_sources(sources)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response["answer"],
            "metadata": response.get("metadata", {}),
            "sources": response.get("sources", []),
        }
    )

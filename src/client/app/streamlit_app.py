import logging
import os

import requests
import streamlit as st

from app_runtime import log_extra, setup_logging

setup_logging("client")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")
API_URL = f"{API_BASE_URL}/ask"

logger = logging.getLogger("client")


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
            if metadata.get("num_sources") is not None:
                meta_parts.append(f"sources: {metadata['num_sources']}")
            if meta_parts:
                st.caption(" | ".join(meta_parts))

        sources = message.get("sources", [])
        if sources:
            with st.expander("Sources"):
                for index, source in enumerate(sources, start=1):
                    source_title = source.get("metadata", {}).get("title") or source.get(
                        "metadata", {}
                    ).get("source", f"Source {index}")
                    st.markdown(f"**{index}. {source_title}**")
                    st.write(source.get("content", ""))

promt = st.chat_input("pass your question here")


def get_response(promt: str):
    logger.info("Sending request to server", extra=log_extra(endpoint="/ask", stage="request"))

    question = promt
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY

    try:
        response = requests.post(
            url=API_URL,
            json={"question": question},
            headers=headers,
            timeout=90,
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
            if metadata.get("num_sources") is not None:
                meta_parts.append(f"sources: {metadata['num_sources']}")
            if meta_parts:
                st.caption(" | ".join(meta_parts))
        sources = response.get("sources", [])
        if sources:
            with st.expander("Sources"):
                for index, source in enumerate(sources, start=1):
                    source_title = source.get("metadata", {}).get("title") or source.get(
                        "metadata", {}
                    ).get("source", f"Source {index}")
                    st.markdown(f"**{index}. {source_title}**")
                    st.write(source.get("content", ""))
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response["answer"],
            "metadata": response.get("metadata", {}),
            "sources": response.get("sources", []),
        }
    )

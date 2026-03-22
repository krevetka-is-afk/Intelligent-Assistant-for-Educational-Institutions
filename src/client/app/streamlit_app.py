import logging
import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_URL = f"{API_BASE_URL}/ask"

logger = logging.getLogger("client")
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.propagate = False


st.title("Помощник по учебному процессу ВШЭ")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

promt = st.chat_input("Задайте вопрос об учебном процессе...")


def get_response(promt: str):
    logger.info("Sending request to server")

    try:
        response = requests.post(
            url=API_URL,
            json={"question": promt},
            timeout=90,
        )
        logger.debug("Server responded with status code %s", response.status_code)
        response.raise_for_status()

        payload = response.json()
        return payload.get("response", "Пустой ответ от сервера."), payload.get("sources", [])

    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to server at %s", API_URL)
        return "Не удалось подключиться к серверу. Попробуйте позже.", []

    except requests.exceptions.Timeout:
        logger.warning("Request to server timed out")
        return "Сервер не ответил вовремя. Попробуйте позже.", []

    except requests.exceptions.HTTPError as e:
        logger.error("Server returned HTTP %s: %s", e.response.status_code, e.response.text)
        return "Сервер вернул ошибку.", []

    except requests.exceptions.RequestException:
        logger.exception("HTTP request to server failed")
        return "Сервер недоступен.", []

    except (ValueError, Exception):
        logger.exception("Unexpected client error")
        return "Произошла непредвиденная ошибка.", []


def format_sources(sources: list) -> str:
    if not sources:
        return ""
    lines = ["\n\n---\n📚 **Источники:**"]
    for s in sources:
        meta = s.get("metadata", {})
        title = meta.get("title") or meta.get("source", "Неизвестный источник")
        page = meta.get("page")
        line = f"- {title}" + (f", стр. {page}" if page else "")
        lines.append(line)
    return "\n".join(lines)


if promt:
    logger.info("User submitted a promt")
    st.chat_message("user").markdown(promt)
    st.session_state.messages.append({"role": "user", "content": promt})

    answer, sources = get_response(promt)
    full_response = answer + format_sources(sources)

    st.chat_message("assistant").markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

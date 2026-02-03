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


st.title("ask about study process")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

promt = st.chat_input("pass your question here")


def get_response(promt: str):
    logger.info("Sending request to server")

    question = promt

    try:
        response = requests.post(
            url=API_URL,
            json={"question": question},
            timeout=10,
        )
        logger.debug("Server responded with status code %s", response.status_code)
        response.raise_for_status()

        payload = response.json()
        return payload.get("response", "Empty response")

    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to server at %s", API_URL)
        return "Cannot connect to the server. Please try again later."

    except requests.exceptions.Timeout:
        logger.warning("Request to server timed out")
        return "Server is taking too long to respond."

    except requests.exceptions.HTTPError as e:
        logger.error(
            "Server returned HTTP %s: %s",
            e.response.status_code,
            e.response.text,
        )
        return "Server returned an error."

    except requests.exceptions.RequestException:
        logger.exception("HTTP request to server failed")
        return "Server is unavailable"

    except ValueError:
        logger.exception("Failed to decode JSON response")

    except Exception:
        logger.exception("Unexpected client error")
        return "Unexpected error occurred."


if promt:
    logger.info("User submitted a promt")
    st.chat_message("user").markdown(promt)
    st.session_state.messages.append({"role": "user", "content": promt})

    response = get_response(promt)
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

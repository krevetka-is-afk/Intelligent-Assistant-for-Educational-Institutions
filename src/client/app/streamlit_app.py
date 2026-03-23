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
        return {
            "answer": payload.get("answer", "Empty response"),
            "metadata": payload.get("metadata", {}),
            "sources": payload.get("sources", []),
        }

    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to server at %s", API_URL)
        return {"answer": "Cannot connect to the server. Please try again later."}

    except requests.exceptions.Timeout:
        logger.warning("Request to server timed out")
        return {"answer": "Server is taking too long to respond."}

    except requests.exceptions.HTTPError as e:
        logger.error(
            "Server returned HTTP %s: %s",
            e.response.status_code,
            e.response.text,
        )
        return {"answer": "Server returned an error."}

    except requests.exceptions.RequestException:
        logger.exception("HTTP request to server failed")
        return {"answer": "Server is unavailable"}

    except ValueError:
        logger.exception("Failed to decode JSON response")

    except Exception:
        logger.exception("Unexpected client error")
        return {"answer": "Unexpected error occurred."}


if promt:
    logger.info("User submitted a promt")
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

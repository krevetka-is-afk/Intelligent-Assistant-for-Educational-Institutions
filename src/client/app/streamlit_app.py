import os

import requests
import streamlit as st

# url = "http://localhost:8000/ask"
# API_URL = "http://server:8000/ask"

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_URL = f"{API_BASE_URL}/ask"

st.title("ask about study process")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

promt = st.chat_input("pass your question here")


def get_response(promt):
    question = str(promt)

    response = requests.post(url=API_URL, json={"question": question})
    return response


if promt:
    st.chat_message("user").markdown(promt)
    st.session_state.messages.append({"role": "user", "content": promt})

    response = get_response(promt)
    if response.status_code == 200:
        st.chat_message("assistant").markdown(response.json().get("response"))
    else:
        st.error(f"Error fetching items: Status code {response.status_code}")

    st.session_state.messages.append({"role": "assistant", "content": response})

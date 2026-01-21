# from langchain_classic import create_retrieval_chain
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# from langchain_text_splitters import RecursiveCharacterTextSplitter

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

# from watsonx langchain import LangChainInterface
from langchain_ollama.llms import OllamaLLM
from vector import retriever

model = OllamaLLM(model="gemma2:2b")

template = """
You are a helpful assistant that helps students find specific information in University sources.

Here is the information you have access to: {information}

Given the student's question,
provide a concise and accurate answer based on the information provided.

Student's question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


st.title("ask about study process")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

promt = st.chat_input("pass your question here")


def llm(promt):
    question = str(prompt)

    information = retriever.invoke(question)

    result = chain.invoke({"information": [information], "question": question})

    return result


if promt:
    st.chat_message("user").markdown(promt)
    st.session_state.messages.append({"role": "user", "content": promt})

    response = llm(promt)
    st.chat_message("assistant").markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.indexes import VectorstoreIndexCreator
from langchain_classic.chains import retrieval_qa
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
# from langchain_classic import create_retrieval_chain
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# from langchain_text_splitters import RecursiveCharacterTextSplitter

import streamlit as st

# from watsonx langchain import LangChainInterface


st.title('ask about study process')

if 'message' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

promt = st.chat_input('pass your question here')

if promt:
    st.chat_message('user').markdown(promt)
    st.session_state.messages.append({'role':'user', 'content':promt})
    
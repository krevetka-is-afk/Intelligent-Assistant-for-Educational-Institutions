import os
from importlib.resources import files

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

DATA_PATH = files("server.data").joinpath("ag_news.csv")

df = pd.read_csv(DATA_PATH)
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Description"],
            metadata={"Class Index": row["Class Index"]},
            id=str(i),
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="news", persist_directory=db_location, embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, id=id)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

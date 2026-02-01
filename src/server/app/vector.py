import os
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from . import config

DATA_PATH = Path(__file__).resolve().parent.parent / "data/ag_news.csv"

df = pd.read_csv(DATA_PATH)
embeddings = OllamaEmbeddings(model=config.embed_model)

_default_db_dir = Path(__file__).resolve().parent.parent / "chrome_langchain_db"
db_location = os.getenv("VECTOR_DB_DIR", str(_default_db_dir))

add_documents = not os.path.exists(db_location)

if add_documents:
    documents: list[Document] = []
    ids: list[str] = []

    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Description"],
            metadata={"Class Index": row["Class Index"]},
            id=str(i),
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="news",
    persist_directory=db_location,
    embedding_function=embeddings,
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

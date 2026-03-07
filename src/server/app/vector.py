import os
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import OllamaEmbeddings

from . import config

DATA_PATH = Path(__file__).resolve().parent.parent / "data/ag_news.csv"
_default_db_dir = Path(__file__).resolve().parent.parent / "chrome_langchain_db"
_raw_db_dir = os.getenv("VECTOR_DB_DIR", str(_default_db_dir))
db_location = str(Path(_raw_db_dir).resolve())

_retriever: BaseRetriever | None = None


def _load_documents() -> tuple[list[Document], list[str]]:
    df = pd.read_csv(DATA_PATH)
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

    return documents, ids


def _create_retriever() -> BaseRetriever:
    embeddings = OllamaEmbeddings(model=config.embed_model)
    add_documents = not os.path.exists(db_location)

    vector_store = Chroma(
        collection_name="news",
        persist_directory=db_location,
        embedding_function=embeddings,
    )

    if add_documents:
        documents, ids = _load_documents()
        vector_store.add_documents(documents=documents, ids=ids)

    return vector_store.as_retriever(search_kwargs={"k": 3})


def get_retriever() -> BaseRetriever:
    global _retriever

    if _retriever is None:
        _retriever = _create_retriever()

    return _retriever

import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from . import config
from .vector import retriever

logger = logging.getLogger("server")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.propagate = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://client:8501"],  # Add your client URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ollama_host = os.getenv("OLLAMA_HOST", config.ollama_port)
model = OllamaLLM(model=config.model, base_url=ollama_host)
template = config.template

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/ask")
async def ask(request: Request):
    logger.info("Received /ask request")

    try:
        data = await request.json()
        question = data.get("question")
        if not question:
            logger.warning("Missing 'question' field in request")
            return {"error": "Can not find question"}

        information = retriever.invoke(question)
        logger.debug("Retriever returned data")
        response = chain.invoke({"information": [information], "question": question})
        logger.info("Successfully processed /ask request")
        return {"response": response}

    except Exception:
        logger.exception("Unhandled exception in /ask")
        return {"error": "An internal server error occurred."}

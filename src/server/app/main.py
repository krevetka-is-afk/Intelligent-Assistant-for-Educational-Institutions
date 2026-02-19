import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
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

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://client:8501",
        "http://localhost:8000",
    ],
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


@app.get("/web", response_class=HTMLResponse)
async def web_interface(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask")
async def ask(request: Request):
    logger.info("Received /ask request")

    try:
        data = await request.json()
        question = data.get("question")
        if not question:
            logger.warning("Missing 'question' field in request")
            return {"error": "Can not find question"}

        documents = retriever.invoke(question)
        logger.debug("Retriever returned %d documents", len(documents))

        response = chain.invoke({"information": [documents], "question": question})
        logger.info("Successfully processed /ask request")

        sources = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in documents
        ]

        return {
            "response": response,
            "sources": sources,
            "metadata": {
                "model": config.model,
                "num_sources": len(sources),
            },
        }

    except Exception:
        logger.exception("Unhandled exception in /ask")
        return {"error": "An internal server error occurred."}

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from . import config
from .vector import get_retriever

logger = logging.getLogger("server")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.propagate = False

limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


ollama_host = os.getenv("OLLAMA_HOST", config.ollama_port)
model = OllamaLLM(model=config.model, base_url=ollama_host, timeout=60)
template = config.template

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
_ALLOWED_METADATA_KEYS = {"source", "title", "page", "Class Index"}


def _normalize_metadata_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def _normalize_source_metadata(raw_metadata: Any) -> dict[str, Any]:
    if not isinstance(raw_metadata, dict):
        return {}

    source = _normalize_metadata_value(raw_metadata.get("source"))
    title = _normalize_metadata_value(raw_metadata.get("title"))
    page = _normalize_metadata_value(raw_metadata.get("page"))
    class_index = _normalize_metadata_value(raw_metadata.get("Class Index"))

    normalized = {
        key: _normalize_metadata_value(value)
        for key, value in raw_metadata.items()
        if key in _ALLOWED_METADATA_KEYS and _normalize_metadata_value(value) is not None
    }

    resolved_title = title or source
    if resolved_title is None and class_index is not None:
        resolved_title = f"Class {class_index}"

    if resolved_title is not None:
        normalized["title"] = resolved_title
    if page is not None:
        normalized["page"] = page
    if source is not None:
        normalized["source"] = source
    if class_index is not None:
        normalized["Class Index"] = class_index

    return normalized


def _error_response(message: str, status_code: int) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


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
@limiter.limit("10/minute")
async def ask(request: Request):
    logger.info("Received /ask request")

    try:
        data = await request.json()
    except json.JSONDecodeError:
        logger.warning("Malformed JSON in /ask request")
        return _error_response("Invalid JSON in request body", 400)

    question = data.get("question")

    if not question or not isinstance(question, str):
        logger.warning("Missing or invalid 'question' field in request")
        return _error_response("Question must be a non-empty string", 400)

    question = question.strip()
    if not question:
        return _error_response("Question must be a non-empty string", 400)
    if len(question) > 500:
        logger.warning("Question exceeds maximum length")
        return _error_response("Question must not exceed 500 characters", 400)

    t_start = time.perf_counter()

    try:
        documents = get_retriever().invoke(question)
        t_retriever = time.perf_counter()
        logger.debug("Retriever returned %d documents (%.2fs)", len(documents), t_retriever - t_start)
    except ConnectionError:
        logger.error("Could not connect to vector store")
        return _error_response("Vector store is unavailable. Please try again later.", 503)
    except Exception:
        logger.exception("Retriever failed in /ask")
        return _error_response("Failed to retrieve documents. Please try again later.", 500)

    try:
        response = chain.invoke({"information": [documents], "question": question})
        t_llm = time.perf_counter()
        t_total = t_llm - t_start
        logger.info(
            "Request processed: retriever=%.2fs, llm=%.2fs, total=%.2fs",
            t_retriever - t_start,
            t_llm - t_retriever,
            t_total,
        )
        if t_total > 7:
            logger.warning("Response time %.2fs exceeds target of 7s", t_total)
    except ConnectionError:
        logger.error("Could not connect to LLM service")
        return _error_response("LLM service is unavailable. Please try again later.", 503)
    except Exception:
        logger.exception("LLM chain failed in /ask")
        return _error_response("Failed to generate a response. Please try again later.", 500)

    sources = [
        {
            "content": doc.page_content,
            "metadata": _normalize_source_metadata(doc.metadata),
        }
        for doc in documents
    ]
    answer = str(response)

    return {
        "answer": answer,
        "sources": sources,
        "metadata": {
            "model": config.model,
            "num_sources": len(sources),
        },
    }

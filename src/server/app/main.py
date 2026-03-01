import json
import logging
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from fastapi.templating import Jinja2Templates
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

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

limiter = Limiter(key_func=get_remote_address)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

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
    allow_headers=["Content-Type", "X-API-Key"],
)

async def verify_api_key(api_key: str = Security(api_key_header)):
    expected_key = os.getenv("API_KEY")
    if not expected_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    if api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key


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
@limiter.limit("10/minute")
async def ask(request: Request, api_key: str = Depends(verify_api_key)):
    logger.info("Received /ask request")

    try:
        data = await request.json()
    except json.JSONDecodeError:
        logger.warning("Malformed JSON in /ask request")
        return {"error": "Invalid JSON in request body"}

    question = data.get("question")

    if not question or not isinstance(question, str):
        logger.warning("Missing or invalid 'question' field in request")
        return {"error": "Question must be a non-empty string"}

    question = question.strip()
    if not question:
        return {"error": "Question must be a non-empty string"}
    if len(question) > 500:
        logger.warning("Question exceeds maximum length")
        return {"error": "Question must not exceed 500 characters"}

    try:
        documents = retriever.invoke(question)
        logger.debug("Retriever returned %d documents", len(documents))
    except ConnectionError:
        logger.error("Could not connect to vector store")
        return {"error": "Vector store is unavailable. Please try again later."}
    except Exception:
        logger.exception("Retriever failed in /ask")
        return {"error": "Failed to retrieve documents. Please try again later."}

    try:
        response = chain.invoke({"information": [documents], "question": question})
        logger.info("Successfully processed /ask request")
    except ConnectionError:
        logger.error("Could not connect to LLM service")
        return {"error": "LLM service is unavailable. Please try again later."}
    except Exception:
        logger.exception("LLM chain failed in /ask")
        return {"error": "Failed to generate a response. Please try again later."}

    _ALLOWED_METADATA_KEYS = {"source", "title", "page", "Class Index"}
    sources = [
        {
            "content": doc.page_content,
            "metadata": {
                k: v for k, v in doc.metadata.items() if k in _ALLOWED_METADATA_KEYS
            },
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

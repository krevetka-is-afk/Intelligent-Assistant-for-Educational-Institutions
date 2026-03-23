from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .metrics import (
    rag_errors_total,
    rag_fallback_total,
    rag_generation_seconds,
    rag_requests_total,
    rag_retrieval_seconds,
    rag_total_seconds,
    render_metrics,
)
from .rag import ask_question
from .vector import EmptyVectorStoreError, VectorStoreUnavailableError

logger = logging.getLogger("server")
logger.setLevel(logging.INFO)
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


def _error_response(message: str, status_code: int) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)


@app.get("/web", response_class=HTMLResponse)
async def web_interface(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask")
@limiter.limit("10/minute")
async def ask(request: Request):
    request_id = uuid4().hex[:12]
    rag_requests_total.inc()

    try:
        data = await request.json()
    except json.JSONDecodeError:
        logger.warning("request_id=%s malformed JSON in /ask request", request_id)
        return _error_response("Invalid JSON in request body", 400)

    question = data.get("question")
    if not question or not isinstance(question, str):
        logger.warning("request_id=%s missing or invalid question", request_id)
        return _error_response("Question must be a non-empty string", 400)

    question = question.strip()
    if not question:
        return _error_response("Question must be a non-empty string", 400)
    if len(question) > 500:
        logger.warning("request_id=%s question exceeds maximum length", request_id)
        return _error_response("Question must not exceed 500 characters", 400)

    logger.info(
        "request_id=%s processing question length=%s",
        request_id,
        len(question),
    )

    try:
        result = await ask_question(question)
    except EmptyVectorStoreError as exc:
        rag_errors_total.labels(stage="vector_store").inc()
        logger.warning("request_id=%s vector store is empty: %s", request_id, exc)
        return _error_response(str(exc), 503)
    except VectorStoreUnavailableError:
        rag_errors_total.labels(stage="vector_store").inc()
        logger.exception("request_id=%s vector store failure", request_id)
        return _error_response("Vector store is unavailable. Please try again later.", 503)
    except Exception:
        rag_errors_total.labels(stage="unexpected").inc()
        logger.exception("request_id=%s unexpected /ask failure", request_id)
        return _error_response("Failed to generate a response. Please try again later.", 500)

    metadata = result.metadata
    rag_retrieval_seconds.observe(metadata["retrieval_time_ms"] / 1000)
    rag_generation_seconds.observe(metadata["generation_time_ms"] / 1000)
    rag_total_seconds.observe(metadata["total_time_ms"] / 1000)
    if metadata["fallback_used"]:
        rag_fallback_total.inc()

    logger.info(
        (
            "request_id=%s completed retrieved=%s fallback=%s reason=%s "
            "retrieval_ms=%s generation_ms=%s total_ms=%s"
        ),
        request_id,
        len(result.retrieved_documents),
        metadata["fallback_used"],
        metadata["fallback_reason"],
        metadata["retrieval_time_ms"],
        metadata["generation_time_ms"],
        metadata["total_time_ms"],
    )

    return {
        "answer": result.answer,
        "sources": result.sources,
        "metadata": metadata,
    }

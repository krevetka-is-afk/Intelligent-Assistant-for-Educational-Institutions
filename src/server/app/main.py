from __future__ import annotations

import hmac
import json
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app_runtime import log_extra, setup_logging

from . import config
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

setup_logging("server")
logger = logging.getLogger("server")

try:
    config.validate_runtime_config()
except RuntimeError as exc:
    logger.critical(
        "Server configuration is invalid: %s",
        exc,
        extra=log_extra(stage="startup", error_type="config"),
    )
    raise

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
    allow_headers=["Content-Type", "X-API-Key"],
)


def _error_response(message: str, status_code: int) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


class UnauthorizedAPIKeyError(RuntimeError):
    """Raised when the caller does not provide a valid X-API-Key header."""


@app.exception_handler(UnauthorizedAPIKeyError)
async def handle_unauthorized(_: Request, __: UnauthorizedAPIKeyError) -> JSONResponse:
    return _error_response("Unauthorized", 401)


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


async def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if x_api_key is None or not hmac.compare_digest(x_api_key, config.API_KEY or ""):
        logger.warning(
            "Rejected unauthorized request to protected endpoint",
            extra=log_extra(endpoint="/ask", error_type="unauthorized"),
        )
        raise UnauthorizedAPIKeyError("Unauthorized")


async def _parse_question(
    request: Request, *, request_id: str, endpoint: str
) -> str | JSONResponse:
    try:
        data = await request.json()
    except json.JSONDecodeError:
        logger.warning(
            "Malformed JSON in request body",
            extra=log_extra(
                request_id=request_id,
                endpoint=endpoint,
                stage="validation",
                error_type="invalid_json",
            ),
        )
        return _error_response("Invalid JSON in request body", 400)

    question = data.get("question")
    if not question or not isinstance(question, str):
        logger.warning(
            "Missing or invalid question",
            extra=log_extra(
                request_id=request_id,
                endpoint=endpoint,
                stage="validation",
                error_type="invalid_question",
            ),
        )
        return _error_response("Question must be a non-empty string", 400)

    question = question.strip()
    if not question:
        return _error_response("Question must be a non-empty string", 400)
    if len(question) > 500:
        logger.warning(
            "Question exceeds maximum length",
            extra=log_extra(
                request_id=request_id,
                endpoint=endpoint,
                stage="validation",
                error_type="question_too_long",
            ),
        )
        return _error_response("Question must not exceed 500 characters", 400)
    return question


async def _process_question(
    question: str, *, request_id: str, endpoint: str
) -> dict[str, object] | JSONResponse:
    logger.info(
        "Processing question length=%s",
        len(question),
        extra=log_extra(request_id=request_id, endpoint=endpoint, stage="request"),
    )

    try:
        result = await ask_question(question)
    except EmptyVectorStoreError as exc:
        rag_errors_total.labels(stage="vector_store").inc()
        logger.warning(
            "Vector store is empty: %s",
            exc,
            extra=log_extra(
                request_id=request_id,
                endpoint=endpoint,
                stage="vector_store",
                error_type=type(exc).__name__,
            ),
        )
        return _error_response(str(exc), 503)
    except VectorStoreUnavailableError:
        rag_errors_total.labels(stage="vector_store").inc()
        logger.exception(
            "Vector store failure",
            extra=log_extra(
                request_id=request_id,
                endpoint=endpoint,
                stage="vector_store",
                error_type="VectorStoreUnavailableError",
            ),
        )
        return _error_response("Vector store is unavailable. Please try again later.", 503)
    except Exception:
        rag_errors_total.labels(stage="unexpected").inc()
        logger.exception(
            "Unexpected request failure",
            extra=log_extra(
                request_id=request_id,
                endpoint=endpoint,
                stage="request",
                error_type="unexpected",
            ),
        )
        return _error_response("Failed to generate a response. Please try again later.", 500)

    metadata = result.metadata
    rag_retrieval_seconds.observe(metadata["retrieval_time_ms"] / 1000)
    rag_generation_seconds.observe(metadata["generation_time_ms"] / 1000)
    rag_total_seconds.observe(metadata["total_time_ms"] / 1000)
    if metadata["fallback_used"]:
        rag_fallback_total.inc()

    logger.info(
        (
            "Completed request retrieved=%s fallback=%s reason=%s "
            "retrieval_ms=%s generation_ms=%s total_ms=%s"
        ),
        len(result.retrieved_documents),
        metadata["fallback_used"],
        metadata["fallback_reason"],
        metadata["retrieval_time_ms"],
        metadata["generation_time_ms"],
        metadata["total_time_ms"],
        extra=log_extra(request_id=request_id, endpoint=endpoint, stage="response"),
    )

    return {
        "answer": result.answer,
        "sources": result.sources,
        "metadata": metadata,
    }


@app.post("/ask")
@limiter.limit("10/minute")
async def ask(request: Request, _: None = Depends(verify_api_key)):
    request_id = uuid4().hex[:12]
    rag_requests_total.inc()

    question = await _parse_question(request, request_id=request_id, endpoint="/ask")
    if isinstance(question, JSONResponse):
        return question
    return await _process_question(question, request_id=request_id, endpoint="/ask")


@app.post("/web/ask")
@limiter.limit("10/minute")
async def ask_from_web(request: Request):
    request_id = uuid4().hex[:12]
    rag_requests_total.inc()

    question = await _parse_question(request, request_id=request_id, endpoint="/web/ask")
    if isinstance(question, JSONResponse):
        return question
    return await _process_question(question, request_id=request_id, endpoint="/web/ask")

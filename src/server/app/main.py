from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, Form, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app_runtime import log_extra, setup_logging

from . import config
from .document_ingestion import IndexingSummary, index_directory
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
from .vector import (
    EmptyVectorStoreError,
    VectorStoreUnavailableError,
    clear_vector_cache,
    ensure_vector_store_ready,
)

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


def _prepare_rag_runtime() -> None:
    if not config.PREPARE_RAG_ON_STARTUP:
        logger.info("Skipping RAG startup preparation", extra=log_extra(stage="startup"))
        return

    logger.info("Preparing RAG runtime", extra=log_extra(stage="startup"))
    try:
        chunk_count = ensure_vector_store_ready()
    except EmptyVectorStoreError:
        if not config.AUTO_INDEX_ON_STARTUP:
            raise

        logger.warning(
            "Vector store is empty on startup; indexing source documents",
            extra=log_extra(stage="startup", error_type="vector_index_empty"),
        )
        clear_vector_cache()
        summary = index_directory(
            config.DOCUMENTS_DIR,
            config.VECTOR_DB_DIR,
            rebuild=False,
        )
        _log_startup_indexing_summary(summary)
        clear_vector_cache()
        chunk_count = ensure_vector_store_ready()

    logger.info(
        "RAG runtime ready with %s indexed chunks",
        chunk_count,
        extra=log_extra(stage="startup"),
    )


def _log_startup_indexing_summary(summary: IndexingSummary) -> None:
    logger.info(
        (
            "Startup indexing finished: files_seen=%s indexed_files=%s "
            "skipped_files=%s failed_files=%s chunks_written=%s"
        ),
        summary.files_seen,
        summary.indexed_files,
        summary.skipped_files,
        summary.failed_files,
        summary.chunks_written,
        extra=log_extra(stage="startup"),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(_prepare_rag_runtime)
    yield


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
WEB_SESSION_COOKIE_NAME = "web_session"
WEB_SESSION_MAX_AGE_SECONDS = 8 * 60 * 60

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


def _error_response(message: str, status_code: int, *, code: str | None = None) -> JSONResponse:
    content: dict[str, str] = {"error": message}
    if code is not None:
        content["code"] = code
    return JSONResponse(status_code=status_code, content=content)


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
async def metrics(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> Response:
    if not _is_valid_api_key(x_api_key):
        logger.warning(
            "Rejected unauthorized request to protected endpoint",
            extra=log_extra(endpoint="/metrics", error_type="unauthorized"),
        )
        raise UnauthorizedAPIKeyError("Unauthorized")

    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)


@app.get("/web", response_class=HTMLResponse)
async def web_interface(
    request: Request, x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    authenticated = _has_valid_web_session(request)
    response = templates.TemplateResponse(
        request,
        "index.html",
        {
            "authenticated": authenticated,
            "error_message": None,
            "web_login_enabled": config.WEB_UI_PASSWORD is not None,
        },
    )
    return response


async def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if not _is_valid_api_key(x_api_key):
        logger.warning(
            "Rejected unauthorized request to protected endpoint",
            extra=log_extra(endpoint="/ask", error_type="unauthorized"),
        )
        raise UnauthorizedAPIKeyError("Unauthorized")


def _is_valid_api_key(candidate: str | None) -> bool:
    return candidate is not None and hmac.compare_digest(candidate, config.API_KEY or "")


def _is_valid_web_password(candidate: str | None) -> bool:
    return (
        candidate is not None
        and config.WEB_UI_PASSWORD is not None
        and hmac.compare_digest(candidate, config.WEB_UI_PASSWORD)
    )


def _build_web_session_signature(expires_at: int) -> str:
    digest = hmac.new(
        (config.WEB_UI_PASSWORD or "").encode("utf-8"),
        str(expires_at).encode("utf-8"),
        digestmod="sha256",
    ).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _build_web_session_cookie() -> str:
    if config.WEB_UI_PASSWORD is None:
        raise RuntimeError("WEB_UI_PASSWORD is not set")
    expires_at = int(time.time()) + WEB_SESSION_MAX_AGE_SECONDS
    signature = _build_web_session_signature(expires_at)
    return f"{expires_at}.{signature}"


def _has_valid_web_session(request: Request) -> bool:
    if config.WEB_UI_PASSWORD is None:
        return False

    token = request.cookies.get(WEB_SESSION_COOKIE_NAME)
    if not token:
        return False

    expires_at_text, _, signature = token.partition(".")
    if not expires_at_text or not signature:
        return False

    try:
        expires_at = int(expires_at_text)
    except ValueError:
        return False

    if expires_at <= int(time.time()):
        return False

    expected_signature = _build_web_session_signature(expires_at)
    return hmac.compare_digest(signature, expected_signature)


def _set_web_session_cookie(response: Response, request: Request) -> None:
    response.set_cookie(
        key=WEB_SESSION_COOKIE_NAME,
        value=_build_web_session_cookie(),
        max_age=WEB_SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )


def _clear_web_session_cookie(response: Response) -> None:
    response.delete_cookie(WEB_SESSION_COOKIE_NAME)


async def verify_web_access(
    request: Request, x_api_key: str | None = Header(default=None, alias="X-API-Key")
) -> None:
    if _is_valid_api_key(x_api_key) or _has_valid_web_session(request):
        return

    logger.warning(
        "Rejected unauthorized request to web endpoint",
        extra=log_extra(endpoint="/web/ask", error_type="unauthorized"),
    )
    raise UnauthorizedAPIKeyError("Unauthorized")


@app.post("/web/login")
@limiter.limit("5/minute")
async def web_login(request: Request, web_password: str = Form(...)) -> Response:
    if config.WEB_UI_PASSWORD is None:
        logger.warning(
            "Rejected web login because WEB_UI_PASSWORD is not configured",
            extra=log_extra(endpoint="/web/login", error_type="web_auth_disabled"),
        )
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "authenticated": False,
                "error_message": "Веб-вход отключен. Установите WEB_UI_PASSWORD на сервере.",
                "web_login_enabled": False,
            },
            status_code=503,
        )

    if not _is_valid_web_password(web_password):
        logger.warning(
            "Rejected unauthorized web login",
            extra=log_extra(endpoint="/web/login", error_type="unauthorized"),
        )
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "authenticated": False,
                "error_message": "Неверный web-пароль.",
                "web_login_enabled": True,
            },
            status_code=401,
        )

    response = RedirectResponse(url="/web", status_code=303)
    _set_web_session_cookie(response, request)
    return response


@app.post("/web/logout")
async def web_logout() -> RedirectResponse:
    response = RedirectResponse(url="/web", status_code=303)
    _clear_web_session_cookie(response)
    return response


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
        return _error_response(str(exc), 503, code="vector_index_empty")
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
        return _error_response(
            "Vector store is unavailable. Please try again later.",
            503,
            code="vector_store_unavailable",
        )
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
async def ask_from_web(request: Request, _: None = Depends(verify_web_access)):
    request_id = uuid4().hex[:12]
    rag_requests_total.inc()

    question = await _parse_question(request, request_id=request_id, endpoint="/web/ask")
    if isinstance(question, JSONResponse):
        return question
    return await _process_question(question, request_id=request_id, endpoint="/web/ask")

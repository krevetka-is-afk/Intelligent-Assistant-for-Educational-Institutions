from __future__ import annotations

import asyncio
import hmac
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
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
from .ask_service import AskService
from .auth_crud import (
    BootstrapAlreadyConfiguredError,
    ExpiredInviteError,
    InvalidCredentialsError,
    InvalidInviteError,
    UsernameAlreadyExistsError,
    accept_invite,
    authenticate_user,
    create_bootstrap_admin,
    create_invite,
    create_web_session,
    get_user_by_session_token,
    has_admin_user,
    revoke_session,
)
from .auth_database import dispose_auth_db, init_auth_db
from .auth_models import WebUser
from .conversation_memory import ConversationMemoryStore
from .document_ingestion import IndexingSummary, index_directory
from .metrics import rag_requests_total, render_metrics
from .principal_context import PrincipalContext
from .question_validation import QuestionValidationError, normalize_question
from .rag import ask_question
from .vector import EmptyVectorStoreError, clear_vector_cache, ensure_vector_store_ready

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
WEB_CREDENTIALS_VALIDATION_MESSAGE = (
    "Проверьте имя пользователя и пароль. Имя пользователя должно быть непустым, "
    "не длиннее 128 символов, пароль — от 8 до 255 символов."
)
WEB_INVITE_EXPIRY_VALIDATION_MESSAGE = "Срок действия инвайта должен\
     быть положительным числом часов."
SESSION_ID_MAX_LENGTH = 128
SESSION_ID_VALIDATION_MESSAGE = (
    f"session_id must be a non-empty string up to {SESSION_ID_MAX_LENGTH} characters"
)
conversation_memory_store = ConversationMemoryStore(
    max_messages=config.CONVERSATION_MEMORY_WINDOW,
    ttl_seconds=config.CONVERSATION_MEMORY_TTL_SECONDS,
    max_sessions=config.CONVERSATION_MEMORY_MAX_SESSIONS,
)


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
    await init_auth_db()

    async def _rag_startup_worker() -> None:
        try:
            await asyncio.to_thread(_prepare_rag_runtime)
        except Exception:
            logger.exception(
                "RAG startup preparation failed",
                extra=log_extra(stage="startup", error_type="rag_startup"),
            )

    rag_task = asyncio.create_task(_rag_startup_worker())
    try:
        yield
    finally:
        rag_task.cancel()
        try:
            await rag_task
        except asyncio.CancelledError:
            pass
        await dispose_auth_db()


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
    allow_headers=["Content-Type", "X-API-Key", "X-Telegram-Service-Key"],
)


def _error_response(message: str, status_code: int, *, code: str | None = None) -> JSONResponse:
    content: dict[str, str] = {"error": message}
    if code is not None:
        content["code"] = code
    return JSONResponse(status_code=status_code, content=content)


async def _call_ask_question(question: str, conversation_history: list[str] | None = None):
    return await ask_question(question, conversation_history=conversation_history)


ask_service = AskService(
    memory_store=conversation_memory_store,
    ask_question=_call_ask_question,
    logger=logger,
)


class UnauthorizedAPIKeyError(RuntimeError):
    """Raised when the caller does not provide a valid X-API-Key header."""


@app.exception_handler(UnauthorizedAPIKeyError)
async def handle_unauthorized(_: Request, __: UnauthorizedAPIKeyError) -> JSONResponse:
    return _error_response("Unauthorized", 401)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return await _render_web_page(request)


@app.get("/web", response_class=HTMLResponse)
async def web_interface_redirect(_request: Request):
    return RedirectResponse(url="/", status_code=307)


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


async def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if not _is_valid_api_key(x_api_key):
        logger.warning(
            "Rejected unauthorized request to protected endpoint",
            extra=log_extra(endpoint="/ask", error_type="unauthorized"),
        )
        raise UnauthorizedAPIKeyError("Unauthorized")


def _is_valid_api_key(candidate: str | None) -> bool:
    return candidate is not None and hmac.compare_digest(candidate, config.API_KEY or "")


def _is_valid_telegram_service_key(candidate: str | None) -> bool:
    return candidate is not None and hmac.compare_digest(
        candidate,
        config.TELEGRAM_SERVICE_KEY or "",
    )


def _is_valid_bootstrap_token(candidate: str | None) -> bool:
    return (
        candidate is not None
        and config.WEB_BOOTSTRAP_ADMIN_TOKEN is not None
        and hmac.compare_digest(candidate, config.WEB_BOOTSTRAP_ADMIN_TOKEN)
    )


def _request_is_secure(request: Request) -> bool:
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    if forwarded_proto:
        scheme = forwarded_proto.split(",", maxsplit=1)[0].strip()
        return scheme == "https"
    return request.url.scheme == "https"


def _get_web_session_token(request: Request) -> str | None:
    token = request.cookies.get(WEB_SESSION_COOKIE_NAME)
    if token:
        return token
    return None


async def _get_current_web_user(request: Request) -> WebUser | None:
    if hasattr(request.state, "web_user_resolved"):
        return request.state.web_user

    token = _get_web_session_token(request)
    if not token:
        request.state.web_user = None
        request.state.web_user_resolved = True
        return None

    user = await get_user_by_session_token(token)
    request.state.web_user = user
    request.state.web_user_resolved = True
    if user is None:
        request.state.clear_web_session_cookie = True
    return user


def _set_web_session_cookie(response: Response, request: Request, token: str) -> None:
    response.set_cookie(
        key=WEB_SESSION_COOKIE_NAME,
        value=token,
        max_age=WEB_SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=_request_is_secure(request),
    )


def _clear_web_session_cookie(response: Response) -> None:
    response.delete_cookie(WEB_SESSION_COOKIE_NAME)


async def _render_web_page(
    request: Request,
    *,
    error_message: str | None = None,
    success_message: str | None = None,
    invite_code: str | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    current_user = await _get_current_web_user(request)
    admin_exists = await has_admin_user()
    response = templates.TemplateResponse(
        request,
        "index.html",
        {
            "authenticated": current_user is not None,
            "current_user": current_user,
            "is_admin": bool(current_user and current_user.is_admin),
            "admin_exists": admin_exists,
            "show_sources": config.SHOW_SOURCES,
            "bootstrap_enabled": config.WEB_BOOTSTRAP_ADMIN_TOKEN is not None,
            "error_message": error_message,
            "success_message": success_message,
            "invite_code": invite_code,
        },
        status_code=status_code,
    )
    if getattr(request.state, "clear_web_session_cookie", False):
        _clear_web_session_cookie(response)
    return response


async def verify_web_access(
    request: Request, x_api_key: str | None = Header(default=None, alias="X-API-Key")
) -> None:
    if _is_valid_api_key(x_api_key):
        return

    if await _get_current_web_user(request) is not None:
        return

    logger.warning(
        "Rejected unauthorized request to web endpoint",
        extra=log_extra(endpoint="/web/ask", error_type="unauthorized"),
    )
    raise UnauthorizedAPIKeyError("Unauthorized")


async def verify_telegram_service_key(
    x_telegram_service_key: str | None = Header(default=None, alias="X-Telegram-Service-Key"),
) -> None:
    if not _is_valid_telegram_service_key(x_telegram_service_key):
        logger.warning(
            "Rejected unauthorized Telegram service request",
            extra=log_extra(endpoint="/telegram/ask", error_type="unauthorized"),
        )
        raise UnauthorizedAPIKeyError("Unauthorized")


@app.post("/web/login")
@limiter.limit("5/minute")
async def web_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
) -> Response:
    if not await has_admin_user():
        logger.warning(
            "Rejected web login because bootstrap is not completed",
            extra=log_extra(endpoint="/web/login", error_type="bootstrap_required"),
        )
        return await _render_web_page(
            request,
            error_message="Сначала нужно создать bootstrap-admin.",
            status_code=503,
        )

    try:
        user = await authenticate_user(username, password)
    except ValueError:
        logger.warning(
            "Rejected invalid web login payload",
            extra=log_extra(endpoint="/web/login", error_type="invalid_payload"),
        )
        return await _render_web_page(
            request,
            error_message=WEB_CREDENTIALS_VALIDATION_MESSAGE,
            status_code=400,
        )
    except InvalidCredentialsError:
        logger.warning(
            "Rejected unauthorized web login",
            extra=log_extra(endpoint="/web/login", error_type="unauthorized"),
        )
        return await _render_web_page(
            request,
            error_message="Неверное имя пользователя или пароль.",
            status_code=401,
        )

    session_token = await create_web_session(
        user_id=user.id,
        user_agent=request.headers.get("user-agent"),
    )
    response = RedirectResponse(url="/", status_code=303)
    _set_web_session_cookie(response, request, session_token)
    return response


@app.post("/web/bootstrap")
@limiter.limit("3/minute")
async def web_bootstrap(
    request: Request,
    bootstrap_token: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
) -> Response:
    if config.WEB_BOOTSTRAP_ADMIN_TOKEN is None:
        logger.warning(
            "Rejected bootstrap because bootstrap token is not configured",
            extra=log_extra(endpoint="/web/bootstrap", error_type="bootstrap_disabled"),
        )
        return await _render_web_page(
            request,
            error_message="Bootstrap отключен. Установите WEB_BOOTSTRAP_ADMIN_TOKEN на сервере.",
            status_code=503,
        )

    if not _is_valid_bootstrap_token(bootstrap_token):
        logger.warning(
            "Rejected bootstrap with invalid bootstrap token",
            extra=log_extra(endpoint="/web/bootstrap", error_type="unauthorized"),
        )
        return await _render_web_page(
            request,
            error_message="Неверный bootstrap token.",
            status_code=401,
        )

    try:
        user = await create_bootstrap_admin(username, password)
    except BootstrapAlreadyConfiguredError:
        return await _render_web_page(
            request,
            error_message="Bootstrap уже завершен. Используйте обычный вход.",
            status_code=409,
        )
    except UsernameAlreadyExistsError:
        return await _render_web_page(
            request,
            error_message="Такое имя пользователя уже занято.",
            status_code=409,
        )
    except ValueError:
        return await _render_web_page(
            request,
            error_message=WEB_CREDENTIALS_VALIDATION_MESSAGE,
            status_code=400,
        )

    session_token = await create_web_session(
        user_id=user.id,
        user_agent=request.headers.get("user-agent"),
    )
    response = RedirectResponse(url="/", status_code=303)
    _set_web_session_cookie(response, request, session_token)
    return response


@app.post("/web/invite/accept")
@limiter.limit("5/minute")
async def web_accept_invite(
    request: Request,
    invite_code: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
) -> Response:
    try:
        user = await accept_invite(invite_code, username, password)
    except InvalidInviteError:
        return await _render_web_page(
            request,
            error_message="Инвайт-код не найден.",
            status_code=404,
        )
    except ExpiredInviteError:
        return await _render_web_page(
            request,
            error_message="Инвайт-код просрочен или уже использован.",
            status_code=410,
        )
    except UsernameAlreadyExistsError:
        return await _render_web_page(
            request,
            error_message="Такое имя пользователя уже занято.",
            status_code=409,
        )
    except ValueError:
        return await _render_web_page(
            request,
            error_message=WEB_CREDENTIALS_VALIDATION_MESSAGE,
            status_code=400,
        )

    session_token = await create_web_session(
        user_id=user.id,
        user_agent=request.headers.get("user-agent"),
    )
    response = RedirectResponse(url="/", status_code=303)
    _set_web_session_cookie(response, request, session_token)
    return response


@app.post("/web/admin/invites")
@limiter.limit("10/minute")
async def web_create_invite(
    request: Request,
    recipient_label: str = Form(default=""),
    expires_in_hours: int = Form(default=72),
) -> Response:
    current_user = await _get_current_web_user(request)
    if current_user is None:
        logger.warning(
            "Rejected unauthorized invite creation",
            extra=log_extra(endpoint="/web/admin/invites", error_type="unauthorized"),
        )
        return await _render_web_page(
            request,
            error_message="Сначала войдите в web-интерфейс.",
            status_code=401,
        )
    if not current_user.is_admin:
        logger.warning(
            "Rejected non-admin invite creation",
            extra=log_extra(endpoint="/web/admin/invites", error_type="forbidden"),
        )
        return await _render_web_page(
            request,
            error_message="Только администратор может создавать инвайты.",
            status_code=403,
        )

    try:
        invite_code, _ = await create_invite(
            created_by_user_id=current_user.id,
            recipient_label=recipient_label,
            expires_in_hours=expires_in_hours,
        )
    except ValueError:
        return await _render_web_page(
            request,
            error_message=WEB_INVITE_EXPIRY_VALIDATION_MESSAGE,
            status_code=400,
        )

    return await _render_web_page(
        request,
        success_message=("Инвайт создан. Скопируйте код и передайте его пользователю."),
        invite_code=invite_code,
    )


@app.post("/web/logout")
async def web_logout(request: Request) -> RedirectResponse:
    token = _get_web_session_token(request)
    if token:
        await revoke_session(token)
    response = RedirectResponse(url="/", status_code=303)
    _clear_web_session_cookie(response)
    return response


@dataclass(slots=True)
class ParsedQuestion:
    question: str


def _normalize_session_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("session_id must be a string")

    session_id = value.strip()
    if not session_id:
        return None
    if len(session_id) > SESSION_ID_MAX_LENGTH:
        raise ValueError(f"session_id must not exceed {SESSION_ID_MAX_LENGTH} characters")
    return session_id


@dataclass(slots=True)
class ParsedTelegramQuestion:
    question: str
    telegram_user_id: int


async def _parse_question(
    request: Request, *, request_id: str, endpoint: str
) -> ParsedQuestion | JSONResponse:
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
    if not isinstance(data, dict):
        return _error_response("Invalid JSON in request body", 400)

    try:
        question = normalize_question(data.get("question"))
    except QuestionValidationError as exc:
        logger.warning(
            "Question validation failed",
            extra=log_extra(
                request_id=request_id,
                endpoint=endpoint,
                stage="validation",
                error_type=exc.code,
            ),
        )
        return _error_response(str(exc), 400)

    try:
        _normalize_session_id(data.get("session_id"))
    except ValueError:
        logger.warning(
            "Invalid session_id",
            extra=log_extra(
                request_id=request_id,
                endpoint=endpoint,
                stage="validation",
                error_type="invalid_session_id",
            ),
        )
        return _error_response(SESSION_ID_VALIDATION_MESSAGE, 400)

    return ParsedQuestion(question=question)


async def _parse_telegram_question(
    request: Request, *, request_id: str
) -> ParsedTelegramQuestion | JSONResponse:
    endpoint = "/telegram/ask"
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
    if not isinstance(data, dict):
        return _error_response("Invalid JSON in request body", 400)

    telegram_user_id = data.get("telegram_user_id")
    if isinstance(telegram_user_id, bool) or not isinstance(telegram_user_id, int):
        return _error_response("telegram_user_id must be a positive integer", 400)
    if telegram_user_id <= 0:
        return _error_response("telegram_user_id must be a positive integer", 400)

    try:
        question = normalize_question(data.get("question"))
    except QuestionValidationError as exc:
        logger.warning(
            "Question validation failed",
            extra=log_extra(
                request_id=request_id,
                endpoint=endpoint,
                stage="validation",
                error_type=exc.code,
            ),
        )
        return _error_response(str(exc), 400)

    return ParsedTelegramQuestion(question=question, telegram_user_id=telegram_user_id)


@app.post("/ask")
@limiter.limit("10/minute")
async def ask(request: Request, _: None = Depends(verify_api_key)):
    request_id = uuid4().hex[:12]
    rag_requests_total.inc()

    parsed = await _parse_question(request, request_id=request_id, endpoint="/ask")
    if isinstance(parsed, JSONResponse):
        return parsed

    return await ask_service.ask(
        parsed.question,
        request_id=request_id,
        endpoint="/ask",
        principal=PrincipalContext.generic_api(),
    )


@app.post("/web/ask")
@limiter.limit("10/minute")
async def ask_from_web(request: Request, _: None = Depends(verify_web_access)):
    request_id = uuid4().hex[:12]
    rag_requests_total.inc()

    parsed = await _parse_question(request, request_id=request_id, endpoint="/web/ask")
    if isinstance(parsed, JSONResponse):
        return parsed

    current_user = await _get_current_web_user(request)
    principal = (
        PrincipalContext.web_user(current_user.id)
        if current_user is not None
        else PrincipalContext.generic_api()
    )

    return await ask_service.ask(
        parsed.question,
        request_id=request_id,
        endpoint="/web/ask",
        principal=principal,
    )


@app.post("/telegram/ask")
@limiter.limit("10/minute")
async def ask_from_telegram(request: Request, _: None = Depends(verify_telegram_service_key)):
    request_id = uuid4().hex[:12]
    rag_requests_total.inc()

    parsed = await _parse_telegram_question(request, request_id=request_id)
    if isinstance(parsed, JSONResponse):
        return parsed

    return await ask_service.ask(
        parsed.question,
        request_id=request_id,
        endpoint="/telegram/ask",
        principal=PrincipalContext.telegram_user(parsed.telegram_user_id),
    )

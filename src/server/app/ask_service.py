from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from fastapi.responses import JSONResponse

from app_runtime import log_extra

from .conversation_memory import ConversationMemoryStore
from .metrics import (
    rag_errors_total,
    rag_fallback_total,
    rag_generation_seconds,
    rag_retrieval_seconds,
    rag_total_seconds,
)
from .principal_context import PrincipalContext
from .question_validation import normalize_question
from .rag import RAGResponse
from .vector import EmptyVectorStoreError, VectorStoreUnavailableError

VECTOR_INDEX_EMPTY_MESSAGE = "Vector index is empty. Run indexing first."
AskQuestion = Callable[[str, list[str] | None], Awaitable[RAGResponse]]
POLICY_REJECTION_FALLBACK_REASONS = frozenset(
    {
        "policy_forbidden_control_or_secret_request",
        "policy_forbidden_instruction_override",
        "policy_output_violation",
    }
)


def _conversation_memory_key_for_success(
    memory_key: str | None,
    result: RAGResponse | None,
) -> str | None:
    """Return a verified memory key only for successful RAG results."""
    if memory_key is None or not isinstance(result, RAGResponse):
        return None
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if metadata.get("fallback_reason") in POLICY_REJECTION_FALLBACK_REASONS:
        return None
    return memory_key


class AskService:
    def __init__(
        self,
        *,
        memory_store: ConversationMemoryStore,
        ask_question: AskQuestion,
        logger: logging.Logger,
    ) -> None:
        self._memory_store = memory_store
        self._ask_question = ask_question
        self._logger = logger

    async def ask(
        self,
        question: str,
        *,
        principal: PrincipalContext,
        request_id: str,
        endpoint: str,
    ) -> dict[str, object] | JSONResponse:
        normalized_question = normalize_question(question)
        memory_key = principal.memory_key
        web_user_id = principal.web_user_id
        web_user_id_value = str(web_user_id) if web_user_id is not None else None

        self._logger.info(
            "Processing question length=%s principal_authority=%s",
            len(normalized_question),
            principal.authority,
            extra=log_extra(
                request_id=request_id,
                endpoint=endpoint,
                stage="request",
                web_user_id=web_user_id_value,
            ),
        )

        conversation_history: list[str] = []
        if memory_key is not None:
            try:
                conversation_history = await self._memory_store.get_recent_user_messages(memory_key)
                self._logger.info(
                    "Loaded conversation history scope=%s messages=%s",
                    principal.authority,
                    len(conversation_history),
                    extra=log_extra(
                        request_id=request_id,
                        endpoint=endpoint,
                        stage="conversation_memory",
                        web_user_id=web_user_id_value,
                    ),
                )
            except Exception:
                self._logger.exception(
                    "Failed to load conversation history",
                    extra=log_extra(
                        request_id=request_id,
                        endpoint=endpoint,
                        stage="conversation_memory",
                        error_type="conversation_read_failed",
                        web_user_id=web_user_id_value,
                    ),
                )
                conversation_history = []

        try:
            result = await self._ask_question(normalized_question, conversation_history)
        except EmptyVectorStoreError as exc:
            rag_errors_total.labels(stage="vector_store").inc()
            self._logger.warning(
                "Vector store is empty: %s",
                exc,
                extra=log_extra(
                    request_id=request_id,
                    endpoint=endpoint,
                    stage="vector_store",
                    error_type=type(exc).__name__,
                ),
            )
            return self._error_response(
                VECTOR_INDEX_EMPTY_MESSAGE,
                503,
                code="vector_index_empty",
            )
        except VectorStoreUnavailableError:
            rag_errors_total.labels(stage="vector_store").inc()
            self._logger.exception(
                "Vector store failure",
                extra=log_extra(
                    request_id=request_id,
                    endpoint=endpoint,
                    stage="vector_store",
                    error_type="VectorStoreUnavailableError",
                ),
            )
            return self._error_response(
                "Vector store is unavailable. Please try again later.",
                503,
                code="vector_store_unavailable",
            )
        except Exception:
            rag_errors_total.labels(stage="unexpected").inc()
            self._logger.exception(
                "Unexpected request failure",
                extra=log_extra(
                    request_id=request_id,
                    endpoint=endpoint,
                    stage="request",
                    error_type="unexpected",
                ),
            )
            return self._error_response(
                "Failed to generate a response. Please try again later.", 500
            )

        if memory_key_to_store := _conversation_memory_key_for_success(memory_key, result):
            try:
                await self._memory_store.append_user_message(
                    memory_key_to_store,
                    normalized_question,
                )
                self._logger.info(
                    "Stored conversation message scope=%s",
                    principal.authority,
                    extra=log_extra(
                        request_id=request_id,
                        endpoint=endpoint,
                        stage="conversation_memory",
                        web_user_id=web_user_id_value,
                    ),
                )
            except Exception:
                self._logger.exception(
                    "Failed to persist conversation memory",
                    extra=log_extra(
                        request_id=request_id,
                        endpoint=endpoint,
                        stage="conversation_memory",
                        error_type="conversation_write_failed",
                        web_user_id=web_user_id_value,
                    ),
                )

        metadata = result.metadata
        rag_retrieval_seconds.observe(metadata["retrieval_time_ms"] / 1000)
        rag_generation_seconds.observe(metadata["generation_time_ms"] / 1000)
        rag_total_seconds.observe(metadata["total_time_ms"] / 1000)
        if metadata["fallback_used"]:
            rag_fallback_total.inc()

        self._logger.info(
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
            extra=log_extra(
                request_id=request_id,
                endpoint=endpoint,
                stage="response",
                web_user_id=web_user_id_value,
            ),
        )

        return {
            "answer": result.answer,
            "sources": result.sources,
            "metadata": metadata,
        }

    @staticmethod
    def _error_response(message: str, status_code: int, *, code: str | None = None) -> JSONResponse:
        content: dict[str, str] = {"error": message}
        if code is not None:
            content["code"] = code
        return JSONResponse(status_code=status_code, content=content)

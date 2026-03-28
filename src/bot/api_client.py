from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

from app_runtime import log_extra, setup_logging

setup_logging("bot")
logger = logging.getLogger("bot.api")

DEFAULT_API_BASE_URL = (
    os.getenv("API_BASE_URL") or os.getenv("RAG_API_URL", "http://localhost:8000/ask")
).rstrip("/")
if DEFAULT_API_BASE_URL.endswith("/ask"):
    DEFAULT_API_BASE_URL = DEFAULT_API_BASE_URL[: -len("/ask")]
DEFAULT_API_KEY = os.getenv("API_KEY")
DEFAULT_TIMEOUT_SECONDS = 25.0


@dataclass(slots=True)
class AskSource:
    content: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class AskResult:
    answer: str
    sources: list[AskSource]
    metadata: dict[str, Any]


class AskAPIError(Exception):
    """Base exception for bot <-> API integration errors."""


class AskAPITimeoutError(AskAPIError):
    """Raised when /ask does not respond within the timeout."""


class AskAPIUnavailableError(AskAPIError):
    """Raised when /ask is unreachable or returns 5xx."""


class AskAPIUnauthorizedError(AskAPIError):
    """Raised when /ask rejects the configured API key."""


class AskAPIResponseError(AskAPIError):
    """Raised when /ask returns an unexpected payload."""


class AskAPIClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = DEFAULT_API_KEY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_base_url = (base_url or DEFAULT_API_BASE_URL).rstrip("/")
        self.api_url = f"{resolved_base_url}/ask"
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._client = client

    async def ask(self, question: str) -> AskResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question must be a non-empty string")

        if self._client is not None:
            return await self._ask_with_client(self._client, normalized_question)

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            return await self._ask_with_client(client, normalized_question)

    async def _ask_with_client(self, client: httpx.AsyncClient, question: str) -> AskResult:
        try:
            response = await client.post(
                self.api_url,
                json={"question": question},
                headers=self._build_headers(),
                timeout=self.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            logger.warning(
                "Timed out after %.1f seconds while calling %s",
                self.timeout_seconds,
                self.api_url,
                extra=log_extra(endpoint="/ask", stage="network", error_type="TimeoutException"),
            )
            raise AskAPITimeoutError("Timed out while calling /ask") from exc
        except httpx.RequestError as exc:
            logger.error(
                "API %s is unavailable: %s",
                self.api_url,
                exc,
                extra=log_extra(endpoint="/ask", stage="network", error_type=type(exc).__name__),
            )
            raise AskAPIUnavailableError("API is unavailable") from exc

        if response.status_code == 401:
            logger.error(
                "API %s rejected the configured API key",
                self.api_url,
                extra=log_extra(endpoint="/ask", stage="auth", error_type="unauthorized"),
            )
            raise AskAPIUnauthorizedError("API rejected the configured API key")
        if response.status_code >= 500:
            logger.error(
                "API %s returned %s: %s",
                self.api_url,
                response.status_code,
                response.text,
                extra=log_extra(
                    endpoint="/ask",
                    stage="response",
                    error_type=f"http_{response.status_code}",
                ),
            )
            raise AskAPIUnavailableError("API is unavailable")
        if response.status_code >= 400:
            logger.error(
                "API %s returned %s: %s",
                self.api_url,
                response.status_code,
                response.text,
                extra=log_extra(
                    endpoint="/ask",
                    stage="response",
                    error_type=f"http_{response.status_code}",
                ),
            )
            raise AskAPIResponseError(f"API returned HTTP {response.status_code}")

        try:
            payload = response.json()
        except ValueError as exc:
            logger.exception(
                "API %s returned invalid JSON",
                self.api_url,
                extra=log_extra(endpoint="/ask", stage="response", error_type="invalid_json"),
            )
            raise AskAPIResponseError("API returned invalid JSON") from exc

        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            logger.error(
                "API %s returned payload without answer: %s",
                self.api_url,
                payload,
                extra=log_extra(endpoint="/ask", stage="response", error_type="invalid_payload"),
            )
            raise AskAPIResponseError("API payload does not contain answer text")

        raw_sources = payload.get("sources", [])
        if not isinstance(raw_sources, list):
            logger.error(
                "API %s returned invalid sources payload: %s",
                self.api_url,
                payload,
                extra=log_extra(endpoint="/ask", stage="response", error_type="invalid_payload"),
            )
            raise AskAPIResponseError("API payload contains invalid sources")

        sources: list[AskSource] = []
        for source in raw_sources:
            if not isinstance(source, dict):
                logger.warning(
                    "Skipping malformed source item from %s: %s",
                    self.api_url,
                    source,
                    extra=log_extra(endpoint="/ask", stage="response", error_type="invalid_source"),
                )
                continue

            content = source.get("content", "")
            metadata = source.get("metadata", {})
            sources.append(
                AskSource(
                    content=content if isinstance(content, str) else str(content),
                    metadata=metadata if isinstance(metadata, dict) else {},
                )
            )

        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        return AskResult(answer=answer.strip(), sources=sources, metadata=metadata)

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

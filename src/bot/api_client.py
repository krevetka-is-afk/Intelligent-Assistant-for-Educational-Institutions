from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger("bot.api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.propagate = False

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
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


class AskAPIResponseError(AskAPIError):
    """Raised when /ask returns an unexpected payload."""


class AskAPIClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_base_url = (base_url or DEFAULT_API_BASE_URL).rstrip("/")
        self.api_url = f"{resolved_base_url}/ask"
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
                timeout=self.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            logger.warning(
                "Timed out after %.1f seconds while calling %s",
                self.timeout_seconds,
                self.api_url,
            )
            raise AskAPITimeoutError("Timed out while calling /ask") from exc
        except httpx.RequestError as exc:
            logger.error("API %s is unavailable: %s", self.api_url, exc)
            raise AskAPIUnavailableError("API is unavailable") from exc

        if response.status_code >= 500:
            logger.error(
                "API %s returned %s: %s",
                self.api_url,
                response.status_code,
                response.text,
            )
            raise AskAPIUnavailableError("API is unavailable")
        if response.status_code >= 400:
            logger.error(
                "API %s returned %s: %s",
                self.api_url,
                response.status_code,
                response.text,
            )
            raise AskAPIResponseError(f"API returned HTTP {response.status_code}")

        try:
            payload = response.json()
        except ValueError as exc:
            logger.exception("API %s returned invalid JSON", self.api_url)
            raise AskAPIResponseError("API returned invalid JSON") from exc

        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            logger.error(
                "API %s returned payload without answer: %s",
                self.api_url,
                payload,
            )
            raise AskAPIResponseError("API payload does not contain answer text")

        raw_sources = payload.get("sources", [])
        if not isinstance(raw_sources, list):
            logger.error("API %s returned invalid sources payload: %s", self.api_url, payload)
            raise AskAPIResponseError("API payload contains invalid sources")

        sources: list[AskSource] = []
        for source in raw_sources:
            if not isinstance(source, dict):
                logger.warning("Skipping malformed source item from %s: %s", self.api_url, source)
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

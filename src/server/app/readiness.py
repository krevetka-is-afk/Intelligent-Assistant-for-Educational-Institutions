from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Literal

from .rag import RAGResponse

ReadinessStatus = Literal["initializing", "ready", "degraded", "failed"]
ReadinessCheck = Literal["rag", "web_auth"]

DEGRADED_FALLBACK_REASONS = frozenset(
    {"llm_unavailable", "llm_timeout", "rag_timeout_budget_exhausted"}
)
CHECK_NAMES: tuple[ReadinessCheck, ...] = ("rag", "web_auth")


@dataclass(slots=True)
class CheckState:
    status: ReadinessStatus = "initializing"
    details: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None


class ReadinessState:
    def __init__(self) -> None:
        self._lock = Lock()
        self._generation = 0
        self._checks = self._initial_checks()
        self._updated_at = self._now()

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _initial_checks() -> dict[ReadinessCheck, CheckState]:
        return {name: CheckState() for name in CHECK_NAMES}

    def reset(self) -> int:
        with self._lock:
            self._generation += 1
            self._checks = self._initial_checks()
            self._updated_at = self._now()
            return self._generation

    def generation(self) -> int:
        with self._lock:
            return self._generation

    def mark_rag_initializing(self, *, generation: int | None = None, **details: Any) -> bool:
        return self._set_check(
            "rag",
            "initializing",
            generation=generation,
            details=details,
        )

    def mark_rag_ready(self, *, generation: int | None = None, **details: Any) -> bool:
        return self._set_check("rag", "ready", generation=generation, details=details)

    def mark_rag_failed(self, reason: str, *, generation: int | None = None) -> bool:
        return self._set_check("rag", "failed", generation=generation, reason=reason)

    def mark_web_auth_ready(self, *, generation: int | None = None) -> bool:
        return self._set_check("web_auth", "ready", generation=generation)

    def mark_web_auth_failed(self, reason: str, *, generation: int | None = None) -> bool:
        return self._set_check("web_auth", "failed", generation=generation, reason=reason)

    def observe_rag_response(self, response: RAGResponse, *, generation: int | None = None) -> bool:
        metadata = response.metadata if isinstance(response.metadata, dict) else {}
        fallback_reason = metadata.get("fallback_reason")
        fallback_used = metadata.get("fallback_used") is True

        with self._lock:
            if self._is_stale(generation):
                return False

            rag = self._checks["rag"]
            if rag.status not in {"ready", "degraded"}:
                return False
            if (
                fallback_used
                and fallback_reason in DEGRADED_FALLBACK_REASONS
                and response.retrieved_documents
            ):
                rag.status = "degraded"
                rag.reason = str(fallback_reason)
            elif not fallback_used:
                rag.status = "ready"
                rag.reason = None
            else:
                return False

            self._updated_at = self._now()
            return True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            status = self._aggregate_status()
            checks = {name: check.status for name, check in self._checks.items()}
            details = {
                name: dict(check.details) for name, check in self._checks.items() if check.details
            }
            reasons = {
                name: check.reason
                for name, check in self._checks.items()
                if check.reason is not None
            }
            payload: dict[str, Any] = {
                "status": status,
                "checks": checks,
                "updated_at": self._updated_at,
            }
            if details:
                payload["details"] = details
            if reasons:
                payload["reasons"] = reasons
                payload["reason"] = next(iter(reasons.values()))
            return payload

    def _aggregate_status(self) -> ReadinessStatus:
        statuses = [check.status for check in self._checks.values()]
        if "failed" in statuses:
            return "failed"
        if "initializing" in statuses:
            return "initializing"
        if "degraded" in statuses:
            return "degraded"
        return "ready"

    def _set_check(
        self,
        check_name: ReadinessCheck,
        status: ReadinessStatus,
        *,
        generation: int | None = None,
        details: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> bool:
        with self._lock:
            if self._is_stale(generation):
                return False

            check = self._checks[check_name]
            check.status = status
            check.details = dict(details or {})
            check.reason = reason
            self._updated_at = self._now()
            return True

    def _is_stale(self, generation: int | None) -> bool:
        return generation is not None and generation != self._generation


readiness_state = ReadinessState()

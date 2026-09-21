from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PrincipalSource(StrEnum):
    GENERIC_API = "generic_api"
    WEB_USER = "web_user"
    TELEGRAM_SERVICE = "telegram_service"


@dataclass(frozen=True, slots=True)
class PrincipalContext:
    """Verified authority whose memory key is derived exclusively on the server."""

    source: PrincipalSource
    subject_id: int | None = None

    @classmethod
    def generic_api(cls) -> PrincipalContext:
        return cls(source=PrincipalSource.GENERIC_API)

    @classmethod
    def web_user(cls, user_id: int) -> PrincipalContext:
        return cls(source=PrincipalSource.WEB_USER, subject_id=cls._positive_id(user_id))

    @classmethod
    def telegram_user(cls, telegram_user_id: int) -> PrincipalContext:
        return cls(
            source=PrincipalSource.TELEGRAM_SERVICE,
            subject_id=cls._positive_id(telegram_user_id),
        )

    @property
    def memory_key(self) -> str | None:
        if self.source is PrincipalSource.WEB_USER and self.subject_id is not None:
            return f"web:{self.subject_id}"
        if self.source is PrincipalSource.TELEGRAM_SERVICE and self.subject_id is not None:
            return f"tg:{self.subject_id}"
        return None

    @property
    def authority(self) -> str:
        if self.subject_id is None:
            return self.source.value
        return f"{self.source.value}:{self.subject_id}"

    @property
    def web_user_id(self) -> int | None:
        if self.source is PrincipalSource.WEB_USER:
            return self.subject_id
        return None

    @staticmethod
    def _positive_id(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("principal subject id must be a positive integer")
        return value

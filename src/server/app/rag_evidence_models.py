from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .vector import RetrievedDocument

EvidenceStatus = Literal["sufficient", "partial", "none", "unknown"]
EvidenceJudgeVerdict = Literal["complete", "incomplete", "unsupported", "unknown"]


@dataclass(slots=True)
class EvidenceRerankResult:
    selected_documents: list[RetrievedDocument]
    candidate_documents: list[RetrievedDocument]
    diagnostics: dict[str, Any]


@dataclass(slots=True)
class EvidenceWindowResult:
    prompt_documents: list[RetrievedDocument]
    diagnostics: dict[str, Any]


@dataclass(slots=True)
class EvidenceSufficiencyResult:
    status: EvidenceStatus
    supported_ids: list[str] = field(default_factory=list)
    missing_aspects: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass(slots=True)
class EvidenceAnswerJudgeResult:
    verdict: EvidenceJudgeVerdict
    evidence_ids: list[str] = field(default_factory=list)
    missing_aspects: list[str] = field(default_factory=list)
    reason: str = ""

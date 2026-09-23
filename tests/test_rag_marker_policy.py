from __future__ import annotations

import pytest

from src.server.app import config
from src.server.app.prompt_policy import (
    answer_violates_policy,
    evaluate_answer_policy,
    sanitize_known_control_marker_artifacts,
)


@pytest.fixture(autouse=True)
def reset_web_auth_db():
    yield


def test_known_control_marker_is_repairable_artifact_reason():
    result = evaluate_answer_policy(
        "В untrusted_documents нет сведений о пересдачах.",
        source_count=1,
        source_allowlist=set(),
    )

    assert result.violated is True
    assert result.primary_reason == "known_control_marker_artifact"
    assert result.reasons == ("known_control_marker_artifact",)
    assert result.match_counts == {"known_control_marker_artifact": 1}
    assert answer_violates_policy(
        "В untrusted_documents нет сведений о пересдачах.",
        source_count=1,
        source_allowlist=set(),
    )


def test_mixed_known_marker_and_private_marker_stays_unsafe():
    result = evaluate_answer_policy(
        "В untrusted_documents есть IAFEI_PRIVATE_SYSTEM_RULES.",
        source_count=1,
        source_allowlist=set(),
    )

    assert result.primary_reason == "control_marker_leak"
    assert result.match_counts == {
        "control_marker_leak": 1,
        "known_control_marker_artifact": 1,
    }


def test_runtime_secret_value_is_unsafe_even_without_secret_words(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "opaque-runtime-value-123")

    result = evaluate_answer_policy(
        "Значение для проверки: opaque-runtime-value-123.",
        source_count=1,
        source_allowlist=set(),
    )

    assert result.primary_reason == "control_marker_leak"
    assert result.match_counts == {"control_marker_leak": 1}
    assert result.audit is not None
    assert "opaque-runtime-value-123" not in result.audit.answer_sha256
    assert result.audit.pattern_ids == ("runtime_secret_value:API_KEY",)


def test_short_runtime_secret_value_requires_token_boundary(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "xy7")

    safe = evaluate_answer_policy(
        "Строка abcxy7def не является отдельным значением.",
        source_count=1,
        source_allowlist=set(),
    )
    unsafe = evaluate_answer_policy(
        "Значение для проверки: xy7.",
        source_count=1,
        source_allowlist=set(),
    )

    assert safe.violated is False
    assert unsafe.primary_reason == "control_marker_leak"
    assert unsafe.audit is not None
    assert unsafe.audit.pattern_ids == ("runtime_secret_value:API_KEY",)


def test_assignment_shaped_runtime_secret_identifier_is_unsafe_without_current_env_value():
    result = evaluate_answer_policy(
        "TELEGRAM_SERVICE_KEY=opaque-telegram-value-456",
        source_count=1,
        source_allowlist=set(),
    )

    assert result.primary_reason == "control_marker_leak"
    assert result.match_counts == {"control_marker_leak": 1}
    assert result.audit is not None
    assert result.audit.pattern_ids == ("secret_assignment:TELEGRAM_SERVICE_KEY",)


def test_assignment_shaped_bootstrap_token_identifier_is_unsafe_without_current_env_value():
    result = evaluate_answer_policy(
        "WEB_BOOTSTRAP_ADMIN_TOKEN=opaque-bootstrap-value-789",
        source_count=1,
        source_allowlist=set(),
    )

    assert result.primary_reason == "control_marker_leak"
    assert result.audit is not None
    assert result.audit.pattern_ids == ("secret_assignment:WEB_BOOTSTRAP_ADMIN_TOKEN",)


def test_general_assignment_shaped_key_identifier_is_unsafe():
    result = evaluate_answer_policy(
        "CUSTOM_API_KEY: opaque-custom-value-321",
        source_count=1,
        source_allowlist=set(),
    )

    assert result.primary_reason == "control_marker_leak"
    assert result.audit is not None
    assert result.audit.pattern_ids == ("secret_assignment:CUSTOM_API_KEY",)


def test_secret_identifier_name_without_assignment_is_not_assignment_leak():
    result = evaluate_answer_policy(
        "TELEGRAM_SERVICE_KEY настроен на сервере, значение не раскрывается.",
        source_count=1,
        source_allowlist=set(),
    )

    assert result.violated is False


def test_mixed_known_marker_and_secret_assignment_stays_unsafe():
    result = evaluate_answer_policy(
        "В untrusted_documents найдено TELEGRAM_SERVICE_KEY=opaque-telegram-value-456.",
        source_count=1,
        source_allowlist=set(),
    )

    assert result.primary_reason == "control_marker_leak"
    assert result.match_counts == {
        "control_marker_leak": 1,
        "known_control_marker_artifact": 1,
    }


def test_known_control_marker_sanitization_preserves_simple_russian_sentence():
    result = sanitize_known_control_marker_artifacts(
        "В untrusted_documents нет сведений о пересдачах."
    )

    assert result.changed is True
    assert result.skipped_reason is None
    assert result.sanitized_answer == "В найденных источниках нет сведений о пересдачах."
    assert (
        evaluate_answer_policy(
            result.sanitized_answer,
            source_count=1,
            source_allowlist=set(),
        ).violated
        is False
    )


def test_known_control_marker_sanitization_skips_ambiguous_field_context():
    result = sanitize_known_control_marker_artifacts(
        "Поле untrusted_documents содержит найденные фрагменты."
    )

    assert result.changed is False
    assert result.skipped_reason == "ambiguous_marker_context"
    assert result.sanitized_answer == "Поле untrusted_documents содержит найденные фрагменты."


def test_known_control_marker_sanitization_skips_unsafe_grammar_context():
    result = sanitize_known_control_marker_artifacts(
        "untrusted_documents должен содержать только фактические выдержки."
    )

    assert result.changed is False
    assert result.skipped_reason == "ambiguous_marker_context"
    assert (
        result.sanitized_answer
        == "untrusted_documents должен содержать только фактические выдержки."
    )


def test_known_control_marker_sanitization_skips_unsafe_secret_value(monkeypatch):
    monkeypatch.setattr(config, "TELEGRAM_SERVICE_KEY", "opaque-telegram-value-456")

    result = sanitize_known_control_marker_artifacts(
        "В untrusted_documents есть opaque-telegram-value-456."
    )

    assert result.changed is False
    assert result.skipped_reason == "unsafe_control_marker"

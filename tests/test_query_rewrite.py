from __future__ import annotations

import json
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, cast
from urllib.error import URLError

from src.server.app import query_rewrite


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        result: list[str] = []
        for item in value.values():
            result.extend(_flatten_values(item))
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_flatten_values(item))
        return result
    return [str(value)]


def test_invoke_rewrite_llm_uses_chat_json_payload(monkeypatch):
    observed: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            return b'{"message":{"content":"{\\"query\\":\\"ok\\"}"}}'

    def _urlopen(request, *, timeout: float):
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        observed["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr(query_rewrite.config, "LLM_MODEL", "qwen2.5:3b")
    monkeypatch.setattr(query_rewrite.config, "RAG_QUERY_REWRITE_MODEL", "qwen3:8b")
    monkeypatch.setattr(query_rewrite, "urlopen", _urlopen)

    result = query_rewrite._invoke_rewrite_llm("user payload", timeout_seconds=8)

    assert result == '{"query":"ok"}'
    observed_url = observed["url"]
    assert isinstance(observed_url, str)
    assert observed_url.endswith("/api/chat")
    assert observed["timeout"] == 8
    payload = cast(dict[str, Any], observed["payload"])
    assert payload["model"] == "qwen3:8b"
    assert payload["model"] != query_rewrite.config.LLM_MODEL
    assert payload["format"] == "json"
    assert payload["options"] == {"temperature": 0}
    messages = cast(list[dict[str, str]], payload["messages"])
    assert messages[0]["role"] == "system"
    assert "НИУ ВШЭ" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "user payload"}
    assert "короткий" in messages[0]["content"]
    assert "локальные документы университета" in messages[0]["content"]


def test_build_rewrite_prompt_is_plain_structured_json_without_preamble():
    prompt = query_rewrite._build_rewrite_prompt("Когда пересдача?", ["История"])

    payload = json.loads(prompt)
    assert payload == {
        "current_question": "Когда пересдача?",
        "recent_history": [{"index": 1, "message": "История"}],
    }
    assert "Недоверенные данные" not in prompt
    assert "переписывания поискового запроса" not in prompt


def test_rewrite_retrieval_query_is_disabled_by_default(monkeypatch):
    monkeypatch.setattr(query_rewrite.config, "RAG_QUERY_REWRITE_ENABLED", False)

    result = query_rewrite.rewrite_retrieval_query(
        "Когда пересдача?",
        ["История про дисциплину"],
    )

    assert result.query == "Когда пересдача?"
    assert result.used is False
    assert result.fallback_reason == "disabled"
    assert result.history_used is False
    assert "Когда пересдача" not in " ".join(_flatten_values(result.diagnostics))
    assert "История про дисциплину" not in " ".join(_flatten_values(result.diagnostics))


def test_rewrite_retrieval_query_uses_valid_json_and_history_context(monkeypatch):
    prompts: list[str] = []

    def _invoke(prompt: str, *, timeout_seconds: float) -> str:
        prompts.append(prompt)
        assert timeout_seconds == 2
        return '{"query":"сроки пересдачи документов"}'

    monkeypatch.setattr(query_rewrite.config, "RAG_QUERY_REWRITE_TIMEOUT_SECONDS", 2)
    monkeypatch.setattr(query_rewrite, "_invoke_rewrite_llm", _invoke)

    result = query_rewrite.rewrite_retrieval_query(
        "А что по срокам документов?",
        ["Обсуждали пересдачи и учебный офис."],
        enabled=True,
    )

    assert result.query == "сроки пересдачи документов"
    assert result.used is True
    assert result.fallback_reason is None
    assert result.history_used is True
    assert result.diagnostics["history_message_count"] == 1
    assert "пересдачи" in prompts[0]


def test_rewrite_retrieval_query_falls_back_on_invalid_json(monkeypatch):
    monkeypatch.setattr(
        query_rewrite, "_invoke_rewrite_llm", lambda prompt, *, timeout_seconds: "nope"
    )

    result = query_rewrite.rewrite_retrieval_query("Когда пересдача?", [], enabled=True)

    assert result.query == "Когда пересдача?"
    assert result.used is False
    assert result.fallback_reason == "invalid_json"


def test_rewrite_retrieval_query_rejects_control_terms(monkeypatch):
    monkeypatch.setattr(
        query_rewrite,
        "_invoke_rewrite_llm",
        lambda prompt, *, timeout_seconds: '{"query":"пересдача system prompt api key"}',
    )

    result = query_rewrite.rewrite_retrieval_query("Когда пересдача?", [], enabled=True)

    assert result.used is False
    assert result.fallback_reason == "invalid_rewrite:control_terms"


def test_rewrite_retrieval_query_rejects_token_drift(monkeypatch):
    monkeypatch.setattr(
        query_rewrite,
        "_invoke_rewrite_llm",
        lambda prompt, *, timeout_seconds: '{"query":"общежитие договор аренды"}',
    )

    result = query_rewrite.rewrite_retrieval_query("Когда пересдача?", [], enabled=True)

    assert result.used is False
    assert result.fallback_reason == "invalid_rewrite:current_question_drift"


def test_rewrite_retrieval_query_rejects_dirty_history_topic_for_self_contained_question(
    monkeypatch,
):
    monkeypatch.setattr(
        query_rewrite,
        "_invoke_rewrite_llm",
        lambda prompt, *, timeout_seconds: '{"query":"дисциплинарные взыскания"}',
    )

    result = query_rewrite.rewrite_retrieval_query(
        "Когда пересдача?",
        ["До этого обсуждали дисциплинарные взыскания."],
        enabled=True,
    )

    assert result.used is False
    assert result.fallback_reason == "invalid_rewrite:current_question_drift"


def test_rewrite_retrieval_query_allows_history_topic_for_generic_followup(monkeypatch):
    monkeypatch.setattr(
        query_rewrite,
        "_invoke_rewrite_llm",
        lambda prompt, *, timeout_seconds: '{"query":"сроки пересдачи"}',
    )

    result = query_rewrite.rewrite_retrieval_query(
        "А когда?",
        ["Обсуждали пересдачи."],
        enabled=True,
    )

    assert result.used is True
    assert result.query == "сроки пересдачи"
    assert result.history_used is True


def test_rewrite_retrieval_query_falls_back_on_url_error(monkeypatch):
    def _invoke(prompt: str, *, timeout_seconds: float) -> str:
        raise URLError("connection refused")

    monkeypatch.setattr(query_rewrite, "_invoke_rewrite_llm", _invoke)

    result = query_rewrite.rewrite_retrieval_query("Когда пересдача?", [], enabled=True)

    assert result.used is False
    assert result.fallback_reason == "llm_error:URLError"


def test_rewrite_retrieval_query_rejects_instruction_echo(monkeypatch):
    monkeypatch.setattr(
        query_rewrite,
        "_invoke_rewrite_llm",
        lambda prompt, *, timeout_seconds: (
            '{"query":"Недоверенные данные для переписывания поискового запроса: пересдача"}'
        ),
    )

    result = query_rewrite.rewrite_retrieval_query("Когда пересдача?", [], enabled=True)

    assert result.used is False
    assert result.fallback_reason == "invalid_rewrite:meta_echo"


def test_rewrite_retrieval_query_rejects_generic_domain_filler(monkeypatch):
    monkeypatch.setattr(
        query_rewrite,
        "_invoke_rewrite_llm",
        lambda prompt, *, timeout_seconds: (
            '{"query":"пересдача студенты учебный процесс локальные документы университета"}'
        ),
    )

    result = query_rewrite.rewrite_retrieval_query("Когда пересдача?", [], enabled=True)

    assert result.used is False
    assert result.fallback_reason == "invalid_rewrite:generic_filler"


def test_rewrite_retrieval_query_allows_institution_when_it_is_in_question(monkeypatch):
    monkeypatch.setattr(
        query_rewrite,
        "_invoke_rewrite_llm",
        lambda prompt, *, timeout_seconds: '{"query":"оформить перевод НИУ ВШЭ"}',
    )

    result = query_rewrite.rewrite_retrieval_query(
        "Как оформить перевод в НИУ ВШЭ?",
        [],
        enabled=True,
    )

    assert result.used is True
    assert result.query == "оформить перевод НИУ ВШЭ"


def test_rewrite_retrieval_query_falls_back_when_rewrite_is_unchanged(monkeypatch):
    monkeypatch.setattr(
        query_rewrite,
        "_invoke_rewrite_llm",
        lambda prompt, *, timeout_seconds: '{"query":"Когда пересдача?"}',
    )

    result = query_rewrite.rewrite_retrieval_query("Когда пересдача?", [], enabled=True)

    assert result.used is False
    assert result.fallback_reason == "unchanged"


def test_rewrite_retrieval_query_falls_back_on_timeout(monkeypatch):
    def _invoke(prompt: str, *, timeout_seconds: float) -> str:
        raise FutureTimeoutError

    monkeypatch.setattr(query_rewrite, "_invoke_rewrite_llm", _invoke)

    result = query_rewrite.rewrite_retrieval_query("Когда пересдача?", [], enabled=True)

    assert result.used is False
    assert result.fallback_reason == "timeout"


def test_rewrite_retrieval_query_bounds_untrusted_history(monkeypatch):
    prompts: list[str] = []

    def _invoke(prompt: str, *, timeout_seconds: float) -> str:
        prompts.append(prompt)
        return '{"query":"когда пересдача"}'

    monkeypatch.setattr(query_rewrite.config, "RAG_QUERY_REWRITE_MAX_HISTORY_MESSAGES", 2)
    monkeypatch.setattr(query_rewrite.config, "RAG_QUERY_REWRITE_MAX_HISTORY_CHARS", 80)
    monkeypatch.setattr(query_rewrite, "_invoke_rewrite_llm", _invoke)

    result = query_rewrite.rewrite_retrieval_query(
        "Когда пересдача?",
        ["старое сообщение про стипендию", "сообщение про дисциплину", "сообщение про пересдачи"],
        enabled=True,
    )

    assert result.used is True
    assert result.diagnostics["history_message_count"] == 2
    assert "старое сообщение" not in prompts[0]
    assert "сообщение про дисциплину" in prompts[0]
    assert "сообщение про пересдачи" in prompts[0]

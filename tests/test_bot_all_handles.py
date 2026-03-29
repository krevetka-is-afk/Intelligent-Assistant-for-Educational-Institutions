from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path


def _load_all_handlers_module(monkeypatch):
    class _Filter:
        def __getattr__(self, name):
            return self

        def __eq__(self, other):
            return self

    class _Router:
        def message(self, *args, **kwargs):
            return lambda func: func

        def callback_query(self, *args, **kwargs):
            return lambda func: func

    class _State:
        pass

    class _StatesGroup:
        pass

    class _InlineKeyboardButton:
        def __init__(self, *, text, callback_data):
            self.text = text
            self.callback_data = callback_data

    class _InlineKeyboardMarkup:
        def __init__(self, *, inline_keyboard):
            self.inline_keyboard = inline_keyboard

    aiohttp_module = types.ModuleType("aiohttp")

    class _ClientTimeout:
        def __init__(self, *, total):
            self.total = total

    class _ClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    aiohttp_module.ClientTimeout = _ClientTimeout
    aiohttp_module.ClientSession = _ClientSession
    monkeypatch.setitem(sys.modules, "aiohttp", aiohttp_module)

    core_package = types.ModuleType("core")
    core_package.__path__ = []
    core_config = types.ModuleType("core.config")
    core_config.RAG_API_URL = "http://rag.test/ask"
    core_config.API_KEY = None
    core_crud = types.ModuleType("core.crud")

    async def _create_query(*args, **kwargs):
        return None

    async def _get_or_create_user(*args, **kwargs):
        return types.SimpleNamespace(id=1)

    core_crud.create_request = _create_query
    core_crud.get_or_create_user = _get_or_create_user
    core_package.config = core_config
    core_package.crud = core_crud
    monkeypatch.setitem(sys.modules, "core", core_package)
    monkeypatch.setitem(sys.modules, "core.config", core_config)
    monkeypatch.setitem(sys.modules, "core.crud", core_crud)

    handlers_package = types.ModuleType("handlers")
    handlers_package.__path__ = []
    handlers_common = types.ModuleType("handlers.common")
    handlers_common.read_image = lambda payload: "image text"
    handlers_common.read_PDF = lambda payload: "pdf text"
    handlers_package.common = handlers_common
    monkeypatch.setitem(sys.modules, "handlers", handlers_package)
    monkeypatch.setitem(sys.modules, "handlers.common", handlers_common)

    aiogram_module = types.ModuleType("aiogram")
    aiogram_module.F = _Filter()
    aiogram_module.Router = _Router
    aiogram_types = types.ModuleType("aiogram.types")
    aiogram_types.Message = object
    aiogram_types.CallbackQuery = object
    aiogram_types.InlineKeyboardButton = _InlineKeyboardButton
    aiogram_types.InlineKeyboardMarkup = _InlineKeyboardMarkup
    aiogram_module.types = aiogram_types
    monkeypatch.setitem(sys.modules, "aiogram", aiogram_module)
    monkeypatch.setitem(sys.modules, "aiogram.types", aiogram_types)

    aiogram_filters = types.ModuleType("aiogram.filters")
    aiogram_filters.Command = lambda value: value
    monkeypatch.setitem(sys.modules, "aiogram.filters", aiogram_filters)

    aiogram_fsm_context = types.ModuleType("aiogram.fsm.context")
    aiogram_fsm_context.FSMContext = object
    monkeypatch.setitem(sys.modules, "aiogram.fsm.context", aiogram_fsm_context)

    aiogram_fsm_state = types.ModuleType("aiogram.fsm.state")
    aiogram_fsm_state.State = _State
    aiogram_fsm_state.StatesGroup = _StatesGroup
    monkeypatch.setitem(sys.modules, "aiogram.fsm.state", aiogram_fsm_state)

    module_path = (
        Path(__file__).resolve().parents[1] / "src" / "bot" / "handlers" / "all_handlers.py"
    )
    spec = importlib.util.spec_from_file_location("test_all_handlers_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_confirmation_preview_escapes_html(monkeypatch):
    module = _load_all_handlers_module(monkeypatch)

    preview = module.build_confirmation_preview("Распознанный текст:", "a < b & c > d")

    assert "<b>Распознанный текст:</b>" in preview
    assert "a &lt; b &amp; c &gt; d" in preview


def test_call_ask_api_uses_15_second_timeout_and_handles_error_payload(monkeypatch):
    module = _load_all_handlers_module(monkeypatch)
    captured = {}

    class _Response:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self):
            return {"error": "Сервис временно недоступен"}

    class _ClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, *, json, timeout, headers=None):
            captured["url"] = url
            captured["json"] = json
            captured["timeout_total"] = timeout.total
            return _Response()

    monkeypatch.setattr(module.aiohttp, "ClientSession", _ClientSession)

    response, sources = asyncio.run(module.call_ask_api("Когда сессия?"))

    assert captured == {
        "url": "http://rag.test/ask",
        "json": {"question": "Когда сессия?"},
        "timeout_total": 15,
    }
    assert response == "Ошибка сервера: Сервис временно недоступен"
    assert sources == []

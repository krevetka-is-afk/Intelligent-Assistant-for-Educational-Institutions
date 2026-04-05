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

    src_package = types.ModuleType("src")
    src_package.__path__ = []
    bot_package = types.ModuleType("src.bot")
    bot_package.__path__ = []
    core_package = types.ModuleType("src.bot.core")
    core_package.__path__ = []
    handlers_package = types.ModuleType("src.bot.handlers")
    handlers_package.__path__ = []
    core_crud = types.ModuleType("src.bot.core.crud")

    async def _get_or_create_user(*args, **kwargs):
        return types.SimpleNamespace(id=1)

    core_crud.get_or_create_user = _get_or_create_user
    core_package.crud = core_crud
    monkeypatch.setitem(sys.modules, "src", src_package)
    monkeypatch.setitem(sys.modules, "src.bot", bot_package)
    monkeypatch.setitem(sys.modules, "src.bot.core", core_package)
    monkeypatch.setitem(sys.modules, "src.bot.core.crud", core_crud)

    handlers_common = types.ModuleType("src.bot.handlers.common")
    handlers_common.EmptyExtractedTextError = RuntimeError
    handlers_common.MediaProcessingError = RuntimeError
    handlers_common.PDFTooLargeError = RuntimeError
    handlers_common.prepare_text_for_api = lambda payload: payload
    handlers_common.read_image = lambda payload: "image text"
    handlers_common.read_PDF = lambda payload: "pdf text"
    handlers_common.validate_pdf_size = lambda size: None
    handlers_package.common = handlers_common
    monkeypatch.setitem(sys.modules, "src.bot.handlers", handlers_package)
    monkeypatch.setitem(sys.modules, "src.bot.handlers.common", handlers_common)

    service_module = types.ModuleType("src.bot.service")

    async def _process_question(*args, **kwargs):
        raise NotImplementedError

    service_module.process_question = _process_question
    monkeypatch.setitem(sys.modules, "src.bot.service", service_module)

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
    spec = importlib.util.spec_from_file_location("src.bot.handlers.all_handlers", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_build_confirmation_preview_escapes_html(monkeypatch):
    module = _load_all_handlers_module(monkeypatch)

    preview = module.build_confirmation_preview("Распознанный текст:", "a < b & c > d")

    assert "<b>Распознанный текст:</b>" in preview
    assert "a &lt; b &amp; c &gt; d" in preview


def test_send_answer_delegates_to_process_question(monkeypatch):
    module = _load_all_handlers_module(monkeypatch)
    captured = {}
    sent_messages = []

    class _ThinkingMessage:
        async def delete(self):
            captured["thinking_deleted"] = True

    class _Message:
        from_user = types.SimpleNamespace(id=42, username="student")

        async def answer(self, text, **kwargs):
            sent_messages.append((text, kwargs))
            if text == "⏳ Обрабатываю вопрос...":
                return _ThinkingMessage()
            return None

    async def _fake_process_question(**kwargs):
        captured["question"] = kwargs["question"]
        captured["content_type"] = kwargs["content_type"]
        captured["raw_content"] = kwargs["raw_content"]
        await kwargs["send_reply"]("Ответ для пользователя")

    monkeypatch.setattr(module, "process_question", _fake_process_question)

    asyncio.run(
        module.send_answer(
            _Message(),
            "Когда дедлайн?",
            "text",
            raw_content="Когда дедлайн?",
        )
    )

    assert captured == {
        "question": "Когда дедлайн?",
        "content_type": "text",
        "raw_content": "Когда дедлайн?",
        "thinking_deleted": True,
    }
    assert sent_messages[0][0] == "⏳ Обрабатываю вопрос..."
    assert sent_messages[1][0] == "Ответ для пользователя"

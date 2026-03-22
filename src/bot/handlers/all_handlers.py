import html
import logging

import aiohttp
import core.config as config
from aiogram import F, Router, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from core.crud import create_query, get_or_create_user
from handlers.common import read_image, read_PDF

logger = logging.getLogger(__name__)

router = Router()

TELEGRAM_LIMIT = 4096
MAX_PDF_SIZE = 20 * 1024 * 1024  # 20 MB


class QuestionStates(StatesGroup):
    waiting_for_content = State()
    awaiting_confirmation = State()


# ---------------------------------------------------------------------------
# Keyboards
# ---------------------------------------------------------------------------


def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Задать вопрос", callback_data="ask_question")],
            [InlineKeyboardButton(text="О боте", callback_data="about")],
        ]
    )


def back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Назад в меню", callback_data="back_to_menu")],
        ]
    )


def confirm_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Всё верно", callback_data="confirm_yes"),
                InlineKeyboardButton(text="❌ Нет", callback_data="confirm_no"),
            ]
        ]
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_answer(response: str, sources: list) -> list[str]:
    sources_lines = ["", "📚 <b>Источники:</b>"]
    for s in sources:
        meta = s.get("metadata", {})
        title = meta.get("title") or meta.get("source", "Неизвестно")
        page = meta.get("page")
        line = f"• {title}" + (f", стр. {page}" if page else "")
        sources_lines.append(line)
    sources_text = "\n".join(sources_lines)

    full = response + sources_text
    if len(full) <= TELEGRAM_LIMIT:
        return [full]

    parts = []
    while len(response) > TELEGRAM_LIMIT:
        parts.append(response[:TELEGRAM_LIMIT])
        response = response[TELEGRAM_LIMIT:]
    last = response + "\n" + sources_text
    if len(last) <= TELEGRAM_LIMIT:
        parts.append(last)
    else:
        parts.append(response)
        parts.append(sources_text[:TELEGRAM_LIMIT])
    return parts


async def call_ask_api(question: str) -> tuple[str, list]:
    if not config.RAG_API_URL:
        return "RAG_API_URL не задан в конфигурации.", []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                config.RAG_API_URL,
                json={"question": question},
                timeout=aiohttp.ClientTimeout(total=90),
            ) as resp:
                data = await resp.json()
        return data.get("response", "Нет ответа от сервера."), data.get("sources", [])
    except Exception as e:
        logger.error("Error calling RAG API: %s", e)
        return "Сервер недоступен. Попробуйте позже.", []


async def send_answer(
    message: types.Message, question: str, content_type: str, user_id: int
) -> None:
    thinking = await message.answer("⏳ Обрабатываю вопрос...")
    response, sources = await call_ask_api(question)
    await thinking.delete()

    parts = format_answer(response, sources)
    for part in parts:
        await message.answer(part, parse_mode="HTML", reply_markup=back_keyboard())

    await create_query(
        user_id=user_id, content_type=content_type, question=question, answer=response
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@router.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext) -> None:
    await state.clear()
    user = await get_or_create_user(
        telegram_id=message.from_user.id,
        username=message.from_user.username,
    )
    await message.answer(
        "Привет! Я помощник по учебным материалам ВШЭ.\n"
        "Задавай вопросы — отвечу с источниками.\n\n"
        "Используй меню ниже:",
        reply_markup=main_menu_keyboard(),
    )
    logger.info("User %s started the bot (db id=%s)", message.from_user.id, user.id)


@router.message(Command("help"))
async def cmd_help(message: types.Message) -> None:
    await message.answer(
        "<b>Что умеет бот:</b>\n"
        "• Отвечать на текстовые вопросы по учебным материалам\n"
        "• Распознавать текст с фотографий (OCR)\n"
        "• Извлекать текст из PDF-файлов (до 20 МБ)\n\n"
        "Нажми <b>«Задать вопрос»</b> и отправь текст, фото или PDF.",
        parse_mode="HTML",
        reply_markup=main_menu_keyboard(),
    )


# ---------------------------------------------------------------------------
# Callbacks — menu navigation
# ---------------------------------------------------------------------------


@router.callback_query(F.data == "ask_question")
async def cb_ask_question(callback: types.CallbackQuery, state: FSMContext) -> None:
    await state.set_state(QuestionStates.waiting_for_content)
    await callback.message.answer(
        "Отправь текстовый вопрос, фотографию или PDF-файл:",
        reply_markup=back_keyboard(),
    )
    await callback.answer()


@router.callback_query(F.data == "about")
async def cb_about(callback: types.CallbackQuery) -> None:
    await callback.message.answer(
        "Этот бот помогает студентам и преподавателям ВШЭ находить ответы "
        "на вопросы по учебным процессам, используя базу знаний с источниками.",
        reply_markup=back_keyboard(),
    )
    await callback.answer()


@router.callback_query(F.data == "back_to_menu")
async def cb_back_to_menu(callback: types.CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await callback.message.answer("Главное меню:", reply_markup=main_menu_keyboard())
    await callback.answer()


# ---------------------------------------------------------------------------
# Content handlers — waiting_for_content state
# ---------------------------------------------------------------------------


@router.message(QuestionStates.waiting_for_content, F.text)
async def handle_text(message: types.Message, state: FSMContext) -> None:
    question = message.text.strip()
    if not question:
        await message.answer("Вопрос не может быть пустым. Попробуй снова.")
        return
    user = await get_or_create_user(message.from_user.id, message.from_user.username)
    await state.clear()
    await send_answer(message, question, "text", user.id)


@router.message(QuestionStates.waiting_for_content, F.photo)
async def handle_photo(message: types.Message, state: FSMContext) -> None:
    photo = message.photo[-1]
    file = await message.bot.download(photo.file_id)
    image_bytes = file.read() if hasattr(file, "read") else file

    ocr_text = read_image(image_bytes)
    if not ocr_text:
        await message.answer("Не удалось распознать текст на изображении. Попробуй другое фото.")
        return

    await state.update_data(pending_question=ocr_text, content_type="image")
    await state.set_state(QuestionStates.awaiting_confirmation)

    preview = html.escape(ocr_text[:1000] + ("..." if len(ocr_text) > 1000 else ""))
    await message.answer(
        f"📄 <b>Распознанный текст:</b>\n\n{preview}\n\nВсё верно?",
        parse_mode="HTML",
        reply_markup=confirm_keyboard(),
    )


@router.message(QuestionStates.waiting_for_content, F.document)
async def handle_document(message: types.Message, state: FSMContext) -> None:
    doc = message.document
    if not doc.file_name or not doc.file_name.lower().endswith(".pdf"):
        await message.answer("Пожалуйста, отправь файл в формате PDF.")
        return
    if doc.file_size and doc.file_size > MAX_PDF_SIZE:
        await message.answer("Файл слишком большой. Максимальный размер — 20 МБ.")
        return

    file = await message.bot.download(doc.file_id)
    pdf_bytes = file.read() if hasattr(file, "read") else file

    pdf_text = read_PDF(pdf_bytes)
    if not pdf_text:
        await message.answer(
            "Не удалось извлечь текст из PDF. Возможно, файл содержит только изображения."
        )
        return

    await state.update_data(pending_question=pdf_text, content_type="pdf")
    await state.set_state(QuestionStates.awaiting_confirmation)

    preview = html.escape(pdf_text[:1000] + ("..." if len(pdf_text) > 1000 else ""))
    await message.answer(
        f"📄 <b>Извлечённый текст из PDF:</b>\n\n{preview}\n\nВсё верно?",
        parse_mode="HTML",
        reply_markup=confirm_keyboard(),
    )


# ---------------------------------------------------------------------------
# Confirmation callbacks
# ---------------------------------------------------------------------------


@router.callback_query(QuestionStates.awaiting_confirmation, F.data == "confirm_yes")
async def cb_confirm_yes(callback: types.CallbackQuery, state: FSMContext) -> None:
    data = await state.get_data()
    question = data.get("pending_question", "")
    content_type = data.get("content_type", "text")
    await state.clear()

    user = await get_or_create_user(callback.from_user.id, callback.from_user.username)
    await callback.answer()
    await send_answer(callback.message, question, content_type, user.id)


@router.callback_query(QuestionStates.awaiting_confirmation, F.data == "confirm_no")
async def cb_confirm_no(callback: types.CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await callback.message.answer(
        "Хорошо, отправь материал ещё раз или задай вопрос текстом.",
        reply_markup=back_keyboard(),
    )
    await callback.answer()

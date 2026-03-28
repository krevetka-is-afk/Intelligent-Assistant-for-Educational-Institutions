import asyncio
import logging

import core.config as config
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from core.database import init_db
from handlers.all_handlers import router

from app_runtime import log_extra, setup_logging

setup_logging("bot")
logger = logging.getLogger(__name__)


async def main() -> None:
    try:
        config.validate_runtime_config()
    except RuntimeError as exc:
        logger.critical(
            "Bot configuration is invalid: %s",
            exc,
            extra=log_extra(stage="startup", error_type="config"),
        )
        raise

    try:
        await init_db()
    except Exception:
        logger.critical(
            "Database initialization failed",
            extra=log_extra(stage="startup", error_type="database_init_failed"),
            exc_info=True,
        )
        raise
    logger.info("Database initialized", extra=log_extra(stage="startup"))

    bot = Bot(
        token=config.BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    logger.info("Bot started", extra=log_extra(stage="startup"))
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())

from __future__ import annotations

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties

from bot.handlers import register_basic_handlers
from bot.inline_handlers import register_inline_handlers
from infrastructure.database import Database


def create_bot(token: str, webapp_url: str, db: Database | None = None) -> tuple[Bot, Dispatcher]:
    """Создать экземпляры бота и диспетчера с подключёнными хендлерами."""
    bot = Bot(token=token, default=DefaultBotProperties(parse_mode="HTML"))
    dp = Dispatcher()
    register_basic_handlers(dp, webapp_url, db=db)
    register_inline_handlers(dp, webapp_url, db=db)
    return bot, dp


"""Запускает только веб-сервер без Telegram-бота (для локальной разработки)."""
import asyncio
import logging

from aiohttp import web

from infrastructure.config import get_settings
from infrastructure.database import Database
from infrastructure.extraid_database import ExtraIDDatabase
from web.server import create_web_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    settings = get_settings()

    db = Database(settings.database)
    await db.connect()
    await db.init_schema()

    extraid_db = ExtraIDDatabase(settings.extraid_database.dsn)
    await extraid_db.connect()

    web_app = create_web_app(
        db=db,
        bot_token=settings.bot_token,
        extraid_db=extraid_db,
        webapp_url=settings.webapp_url,
        stars_rate_rub=settings.stars_rate_rub,
        stars_markup=settings.stars_markup,
        stars_test_mode=settings.stars_test_mode,
    )

    runner = web.AppRunner(web_app)
    await runner.setup()
    site = web.TCPSite(runner, host=settings.web_host, port=settings.web_port)
    await site.start()

    logger.info("WebApp сервер запущен на http://%s:%s", settings.web_host, settings.web_port)

    try:
        await asyncio.Event().wait()  # hold forever
    finally:
        await runner.cleanup()
        await db.close()
        await extraid_db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

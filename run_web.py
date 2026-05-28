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

    rustore_payment_service = None
    if settings.rustore:
        from infrastructure.rustore_payments import RuStoreInvoiceVerifier
        rustore_payment_service = RuStoreInvoiceVerifier(settings.rustore)

    web_app = create_web_app(
        db=db,
        bot_token=settings.bot_token,
        extraid_db=extraid_db,
        rustore_payment_service=rustore_payment_service,
        rustore_console_app_id=settings.rustore.console_app_id if settings.rustore else "",
        rustore_app_url=settings.rustore_app_url,
        payment_provider_order=settings.payment_provider_order,
        webapp_url=settings.webapp_url,
        stars_rate_rub=settings.stars_rate_rub,
        stars_markup=settings.stars_markup,
        stars_test_mode=settings.stars_test_mode,
        android_latest_version_code=settings.android_latest_version_code,
        android_latest_version_name=settings.android_latest_version_name,
        android_min_supported_version_code=settings.android_min_supported_version_code,
        android_update_channel_url=settings.android_update_channel_url,
        android_apk_url=settings.android_apk_url,
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

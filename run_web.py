"""Запускает только веб-сервер без Telegram-бота (для локальной разработки)."""
import asyncio
import logging

from aiohttp import web

from infrastructure.config import get_settings
from infrastructure.database import Database
from infrastructure.extraid_database import ExtraIDDatabase
from infrastructure.support_database import SupportDatabase
from infrastructure.support_json_database import SupportJsonDatabase
from support.channels.max import MaxSupportClient
from support.delivery import SupportDeliveryDispatcher
from support.notifier import SupportAdminNotifier
from support.service import SupportService
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
    if settings.auto_migrate_on_start:
        await db.init_schema()
    else:
        await db.verify_schema_ready()

    extraid_db = ExtraIDDatabase(settings.extraid_database.dsn)
    await extraid_db.connect()

    support_db = None
    support_service = None
    support_max_client = MaxSupportClient(settings.support_max_bot_token) if settings.support_max_bot_token else None
    support_delivery_dispatcher = None
    if settings.support_enabled:
        try:
            support_db = SupportDatabase(settings.support_database)
            await support_db.connect()
            await support_db.init_schema()
            logger.info("Support storage initialized with PostgreSQL database %s", settings.support_database.database)
        except Exception as exc:
            logger.warning(
                "Support PostgreSQL storage is unavailable (%s). Falling back to JSON storage.",
                exc,
            )
            try:
                if support_db:
                    await support_db.disconnect()
            except Exception:
                pass
            support_db = SupportJsonDatabase()
            await support_db.connect()
            await support_db.init_schema()
        support_service = SupportService(support_db=support_db, game_db=db)
        support_delivery_dispatcher = SupportDeliveryDispatcher(
            support_db=support_db,
            max_client=support_max_client,
        )

    payment_service = None
    if settings.yookassa:
        from infrastructure.payments import PaymentService
        payment_service = PaymentService(settings.yookassa)

    rustore_payment_service = None
    if settings.rustore:
        from infrastructure.rustore_payments import RuStoreInvoiceVerifier
        rustore_payment_service = RuStoreInvoiceVerifier(settings.rustore)

    robokassa_payment_service = None
    if settings.robokassa:
        from infrastructure.robokassa_payments import RobokassaPaymentService
        robokassa_payment_service = RobokassaPaymentService(settings.robokassa)

    web_app = create_web_app(
        db=db,
        bot_token=settings.bot_token,
        extraid_db=extraid_db,
        payment_service=payment_service,
        robokassa_payment_service=robokassa_payment_service,
        rustore_payment_service=rustore_payment_service,
        rustore_console_app_id=settings.rustore.console_app_id if settings.rustore else "",
        rustore_app_url=settings.rustore_app_url,
        payment_provider_order=settings.payment_provider_order,
        payment_primary_provider=settings.payment_primary_provider,
        payment_fallback_provider=settings.payment_fallback_provider,
        payments_required=settings.payments_required,
        webapp_url=settings.webapp_url,
        extra_shop_url=settings.extra_shop_url,
        stars_rate_rub=settings.stars_rate_rub,
        stars_markup=settings.stars_markup,
        stars_test_mode=settings.stars_test_mode,
        android_latest_version_code=settings.android_latest_version_code,
        android_latest_version_name=settings.android_latest_version_name,
        android_min_supported_version_code=settings.android_min_supported_version_code,
        android_update_channel_url=settings.android_update_channel_url,
        android_apk_url=settings.android_apk_url,
        shop_allow_max_level_particles=settings.shop_allow_max_level_particles,
        support_service=support_service,
        support_max_client=support_max_client,
        support_admin_notifier=SupportAdminNotifier(
            max_client=support_max_client,
            max_admin_id=settings.support_max_admin_id,
        ),
    )

    runner = web.AppRunner(web_app)
    await runner.setup()
    site = web.TCPSite(runner, host=settings.web_host, port=settings.web_port)
    await site.start()

    logger.info("WebApp сервер запущен на http://%s:%s", settings.web_host, settings.web_port)
    support_delivery_task = (
        asyncio.create_task(_support_delivery_task(support_delivery_dispatcher))
        if support_delivery_dispatcher is not None
        else None
    )

    try:
        await asyncio.Event().wait()  # hold forever
    finally:
        if support_delivery_task:
            support_delivery_task.cancel()
            await asyncio.gather(support_delivery_task, return_exceptions=True)
        await runner.cleanup()
        await db.close()
        await extraid_db.disconnect()
        if support_db:
            await support_db.disconnect()


async def _support_delivery_task(dispatcher: SupportDeliveryDispatcher) -> None:
    while True:
        try:
            delivered = await dispatcher.dispatch_once()
            if delivered:
                logger.info("Support delivery outbox sent %d message(s)", delivered)
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Support delivery outbox failed")
            await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())

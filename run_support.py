"""Run only the separate support bots and support web endpoints."""
from __future__ import annotations

import asyncio
import logging

from aiohttp import web
from aiogram.exceptions import TelegramConflictError, TelegramNetworkError, TelegramServerError

from infrastructure.config import get_settings
from infrastructure.support_database import SupportDatabase
from infrastructure.support_json_database import SupportJsonDatabase
from support.channels.max import MaxSupportClient
from support.channels.telegram import create_support_bot
from support.delivery import SupportDeliveryDispatcher
from support.notifier import SupportAdminNotifier
from support.service import SupportService
from support.web import register_support_routes


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    settings = get_settings()
    if not settings.support_enabled:
        raise RuntimeError("Support is disabled. Set SUPPORT_ENABLED=true or configure support bot tokens.")

    support_db = await _open_support_storage(settings)
    support_service = SupportService(support_db=support_db, game_db=None)
    support_max_client = MaxSupportClient(settings.support_max_bot_token) if settings.support_max_bot_token else None
    support_telegram_bot = None
    support_telegram_dp = None
    if settings.support_telegram_bot_token:
        support_telegram_bot, support_telegram_dp = create_support_bot(
            settings.support_telegram_bot_token,
            support_service=support_service,
        )

    app = web.Application()
    app["support_max_client"] = support_max_client
    app["support_admin_notifier"] = SupportAdminNotifier(
        telegram_bot=support_telegram_bot,
        telegram_admin_id=settings.support_telegram_admin_id,
        max_client=support_max_client,
        max_admin_id=settings.support_max_admin_id,
    )
    support_service.notifier = app["support_admin_notifier"]
    app.router.add_get("/", lambda request: web.HTTPFound("/support/admin"))
    app.router.add_get("/health", lambda request: web.json_response({"status": "ok", "service": "support"}))
    register_support_routes(
        app,
        support_service,
        admin_secret=settings.admin_session_secret,
        admin_channel_id=str(settings.support_max_admin_id or settings.support_telegram_admin_id or "support-admin"),
        max_webhook_secret=settings.support_max_webhook_secret,
    )

    runner = web.AppRunner(app)
    await runner.setup()
    bound_port = await _start_support_site(runner, host=settings.web_host, port=settings.web_port)
    storage_kind = "postgresql" if isinstance(support_db, SupportDatabase) else "json"
    polling_kind = "active" if (support_telegram_bot and support_telegram_dp) else "inactive"
    logger.info(
        "Support web server started on http://%s:%s (storage=%s telegram_polling=%s admin_channel=%s)",
        settings.web_host,
        bound_port,
        storage_kind,
        polling_kind,
        settings.support_telegram_admin_id or settings.support_max_admin_id or "none",
    )

    dispatcher = SupportDeliveryDispatcher(
        support_db=support_db,
        telegram_bot=support_telegram_bot,
        max_client=support_max_client,
    )
    tasks: list[asyncio.Task] = [asyncio.create_task(_support_delivery_task(dispatcher))]
    if support_telegram_bot and support_telegram_dp:
        tasks.append(asyncio.create_task(_support_bot_polling_task(support_telegram_bot, support_telegram_dp)))

    try:
        await asyncio.Event().wait()
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if support_telegram_bot:
            await support_telegram_bot.session.close()
        await support_db.disconnect()
        await runner.cleanup()


async def _open_support_storage(settings):
    support_db = None
    try:
        support_db = SupportDatabase(settings.support_database)
        await support_db.connect()
        await support_db.init_schema()
        logger.info("Support storage initialized with PostgreSQL database %s", settings.support_database.database)
        return support_db
    except Exception as exc:
        logger.warning("Support PostgreSQL storage is unavailable (%s). Falling back to JSON storage.", exc)
        try:
            if support_db:
                await support_db.disconnect()
        except Exception:
            pass
        support_json = SupportJsonDatabase()
        await support_json.connect()
        await support_json.init_schema()
        return support_json


async def _start_support_site(runner: web.AppRunner, *, host: str, port: int) -> int:
    last_error: OSError | None = None
    for candidate in [int(port), *range(int(port) + 1, int(port) + 10)]:
        try:
            site = web.TCPSite(runner, host=host, port=candidate)
            await site.start()
            if candidate != int(port):
                logger.warning("Support web port %s is busy; bound to %s instead.", port, candidate)
            return candidate
        except OSError as exc:
            last_error = exc
            if getattr(exc, "errno", None) not in {48, 98}:
                raise
    raise RuntimeError(f"Unable to bind support web server near port {port}: {last_error}")


async def _support_bot_polling_task(bot, dp) -> None:
    conflict_logged = False
    while True:
        try:
            await bot.delete_webhook(drop_pending_updates=True, request_timeout=10)
            logger.info("Support Telegram webhook removed before polling")
            conflict_logged = False
            await dp.start_polling(bot)
        except TelegramConflictError as exc:
            if not conflict_logged:
                logger.error(
                    "Support Telegram polling conflict (409): another process already polls this bot token. "
                    "Only one polling process is allowed per token. Message: %s",
                    exc,
                )
                conflict_logged = True
            await asyncio.sleep(30)
        except (TelegramNetworkError, TelegramServerError) as exc:
            logger.warning("Support Telegram polling temporarily unavailable: %s. Retry in 10 seconds.", exc)
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Support Telegram polling crashed. Retry in 10 seconds.")
            await asyncio.sleep(10)


async def _support_delivery_task(dispatcher: SupportDeliveryDispatcher) -> None:
    while True:
        try:
            delivered = await dispatcher.dispatch_once(limit=25)
            if delivered:
                logger.info("Support delivery outbox sent %d message(s)", delivered)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Support delivery outbox failed")
        await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())

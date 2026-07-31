import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiogram import Bot
from aiogram.exceptions import TelegramConflictError, TelegramNetworkError, TelegramServerError
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from aiohttp import web

from bot import create_bot
from bot.constants import ADMIN_ID
from bot.max_client import MaxGameBotClient
from infrastructure.config import get_settings
from infrastructure.database import Database, SCHEMA_VERSION
from infrastructure.extraid_database import ExtraIDDatabase
from infrastructure.support_database import SupportDatabase
from infrastructure.support_json_database import SupportJsonDatabase
from infrastructure.notifications import (
    NOTIFICATION_QUIET_END_HOUR,
    NOTIFICATION_QUIET_START_HOUR,
    build_webapp_url,
    format_telegram_notification_message,
    is_discretionary_notification,
    notification_section,
)
from infrastructure.push_notifications import FcmPushSender, build_android_push_payload
from support.channels.max import MaxSupportClient
from support.channels.telegram import create_support_bot
from support.delivery import SupportDeliveryDispatcher
from support.notifier import SupportAdminNotifier
from support.service import SupportService
from web.server import create_web_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

db: Database | None = None
web_runner: web.AppRunner | None = None
web_site: web.BaseSite | None = None

REMINDER_PUSH_QUIET_START_HOUR = NOTIFICATION_QUIET_START_HOUR
REMINDER_PUSH_QUIET_END_HOUR = NOTIFICATION_QUIET_END_HOUR


def _notification_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _device_local_datetime(device: dict, *, now_utc: datetime | None = None) -> datetime | None:
    now = now_utc or _notification_utc_now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    timezone_name = str(device.get("timezone") or "").strip()
    if timezone_name:
        try:
            return now.astimezone(ZoneInfo(timezone_name))
        except (ZoneInfoNotFoundError, ValueError):
            logger.debug("Unknown push-device timezone: %s", timezone_name)

    offset = device.get("utc_offset_minutes")
    try:
        offset_minutes = int(offset)
    except (TypeError, ValueError):
        return None
    if offset_minutes < -14 * 60 or offset_minutes > 14 * 60:
        return None
    return now.astimezone(timezone(timedelta(minutes=offset_minutes)))


def _is_quiet_android_reminder_push(
    device: dict,
    *,
    category: str,
    event_type: str,
    now_utc: datetime | None = None,
) -> bool:
    if not is_discretionary_notification(category):
        return False

    local_now = _device_local_datetime(device, now_utc=now_utc)
    if local_now is None:
        return False

    hour = local_now.hour
    return hour >= REMINDER_PUSH_QUIET_START_HOUR or hour < REMINDER_PUSH_QUIET_END_HOUR


def _next_android_reminder_push_time(device: dict, *, now_utc: datetime | None = None) -> datetime | None:
    local_now = _device_local_datetime(device, now_utc=now_utc)
    if local_now is None:
        return None
    if local_now.hour < REMINDER_PUSH_QUIET_END_HOUR:
        next_local = local_now.replace(
            hour=REMINDER_PUSH_QUIET_END_HOUR,
            minute=0,
            second=0,
            microsecond=0,
        )
    else:
        next_local = (local_now + timedelta(days=1)).replace(
            hour=REMINDER_PUSH_QUIET_END_HOUR,
            minute=0,
            second=0,
            microsecond=0,
        )
    return next_local.astimezone(timezone.utc)


async def _postpone_quiet_telegram_notification(
    db: Database,
    notif: dict,
    *,
    category: str,
) -> bool:
    if not is_discretionary_notification(category):
        return False
    if not hasattr(db, "get_notification_timezone"):
        return False
    try:
        timezone_context = await db.get_notification_timezone(int(notif["user_id"]))
    except Exception:
        logger.debug(
            "Не удалось определить локальное время для Telegram user_id=%s",
            notif.get("user_id"),
            exc_info=True,
        )
        return False
    if not timezone_context:
        return False
    if not _is_quiet_android_reminder_push(
        timezone_context,
        category=category,
        event_type=str(notif.get("event_type") or ""),
    ):
        return False
    defer_until = _next_android_reminder_push_time(timezone_context)
    if defer_until is None or not hasattr(db, "postpone_notification"):
        return False
    await db.postpone_notification(int(notif["id"]), defer_until)
    logger.info(
        "Telegram notification postponed until %s during local quiet hours for user_id=%s",
        defer_until.isoformat(),
        notif["user_id"],
    )
    return True


async def main() -> None:
    global db, web_runner, web_site
    settings = get_settings()

    db = Database(settings.database)
    await db.connect()
    if settings.auto_migrate_on_start:
        schema_changed = await db.init_schema()
    else:
        await db.verify_schema_ready()
        schema_changed = False

    extraid_db = ExtraIDDatabase(settings.extraid_database.dsn)
    await extraid_db.connect()

    support_db = None
    support_service = None
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

    # Инициализация платежного сервиса YooKassa
    payment_service = None
    if settings.yookassa:
        try:
            from infrastructure.payments import PaymentService
            payment_service = PaymentService(settings.yookassa)
            logger.info("YooKassa платежный сервис инициализирован")
            logger.info(f"YooKassa shop_id: {settings.yookassa.shop_id}, test_mode: {settings.yookassa.test_mode}")
        except ImportError:
            logger.warning("Модуль YooKassa-платежей недоступен. Платежи будут недоступны.")
            logger.warning("Проверьте зависимость HTTP-клиента: requests.")
        except Exception as e:
            logger.error(f"Ошибка инициализации YooKassa: {e}", exc_info=True)
    else:
        logger.warning("YooKassa настройки не найдены. Платежи будут недоступны.")
        logger.warning("Добавьте в .env файл:")
        logger.warning("  YOOKASSA_SHOP_ID=ваш_shop_id")
        logger.warning("  YOOKASSA_SECRET_KEY=ваш_secret_key")
        logger.warning("  YOOKASSA_TEST_MODE=%s", "true" if settings.environment == "development" else "false")

    robokassa_payment_service = None
    if settings.robokassa:
        try:
            from infrastructure.robokassa_payments import RobokassaPaymentService
            robokassa_payment_service = RobokassaPaymentService(settings.robokassa)
            logger.info(
                "Robokassa платежный сервис инициализирован: merchant=%s test_mode=%s",
                settings.robokassa.merchant_login,
                settings.robokassa.test_mode,
            )
        except Exception as e:
            logger.error("Ошибка инициализации Robokassa: %s", e, exc_info=True)
    else:
        logger.warning("Robokassa настройки не найдены. Telegram checkout будет использовать fallback.")

    rustore_payment_service = None
    if settings.rustore:
        try:
            from infrastructure.rustore_payments import RuStoreInvoiceVerifier
            rustore_payment_service = RuStoreInvoiceVerifier(settings.rustore)
            logger.info(
                "RuStore payment verifier initialized: console_app_id=%s sandbox=%s",
                settings.rustore.console_app_id or "not-set",
                settings.rustore.sandbox,
            )
        except Exception as e:
            logger.error("Ошибка инициализации RuStore verifier: %s", e, exc_info=True)

    configured_payment_providers = {
        "robokassa": bool(robokassa_payment_service),
        "yookassa": bool(payment_service),
        "rustore": bool(rustore_payment_service),
        "stars": True,
    }
    logger.info(
        "Payment providers: primary=%s fallback=%s order=%s configured=%s payments_required=%s",
        settings.payment_primary_provider,
        settings.payment_fallback_provider,
        settings.payment_provider_order,
        configured_payment_providers,
        settings.payments_required,
    )
    if settings.payments_required and not configured_payment_providers.get(settings.payment_primary_provider, False):
        logger.error(
            "Primary payment provider %s is not configured while PAYMENTS_REQUIRED=true.",
            settings.payment_primary_provider,
        )

    push_sender = FcmPushSender()
    support_max_client = MaxSupportClient(settings.support_max_bot_token) if settings.support_max_bot_token else None

    web_app = create_web_app(
        db,
        settings.bot_token,
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
        android_releases_enabled=settings.android_releases_enabled,
        android_release_storage_dir=settings.android_release_storage_dir,
        android_release_public_base_url=settings.android_release_public_base_url,
        android_release_package_name=settings.android_release_package_name,
        android_direct_signing_cert_sha256=settings.android_direct_signing_cert_sha256,
        android_rustore_signing_cert_sha256=settings.android_rustore_signing_cert_sha256,
        android_apksigner_command=settings.android_apksigner_command,
        android_aapt_command=settings.android_aapt_command,
        android_release_max_bytes=settings.android_release_max_bytes,
        android_release_chunk_bytes=settings.android_release_chunk_bytes,
        android_upload_token_ttl_seconds=settings.android_upload_token_ttl_seconds,
        support_service=support_service,
        support_max_client=support_max_client,
    )
    web_app["push_sender"] = push_sender
    web_runner = web.AppRunner(web_app)
    await web_runner.setup()
    
    try:
        web_site = web.TCPSite(web_runner, host=settings.web_host, port=settings.web_port)
        await web_site.start()
        logging.getLogger(__name__).info(
            "WebApp server запущен на %s:%s", settings.web_host, settings.web_port
        )
        logging.getLogger(__name__).info(
            "Healthcheck доступен на http://%s:%s/health", settings.web_host, settings.web_port
        )
    except Exception as e:
        logging.getLogger(__name__).error(
            "Не удалось запустить WebApp сервер: %s", e, exc_info=True
        )
        raise
    
    webapp_url = settings.webapp_url
    if not webapp_url.startswith("https://"):
        logging.getLogger(__name__).warning(
            "WebApp URL должен быть HTTPS! Текущий URL: %s. "
            "Используйте ngrok или другой туннель для локальной разработки. "
            "Установите WEBAPP_URL в .env с HTTPS адресом.",
            webapp_url
        )
        if webapp_url == "https://example.com/chibiarena":
            logging.getLogger(__name__).error(
                "WebApp URL не настроен! Установите WEBAPP_URL в .env с HTTPS адресом. "
                "Для локальной разработки используйте ngrok: ngrok http %s",
                settings.web_port
            )

    bot, dp = create_bot(settings.bot_token, webapp_url, db=db)
    web_app["telegram_bot"] = bot

    support_telegram_bot = None
    support_telegram_dp = None
    if support_service is not None and settings.support_telegram_bot_token:
        support_telegram_bot, support_telegram_dp = create_support_bot(
            settings.support_telegram_bot_token,
            support_service=support_service,
        )
        web_app["support_telegram_bot"] = support_telegram_bot

    support_delivery_dispatcher = (
        SupportDeliveryDispatcher(
            support_db=support_db,
            telegram_bot=support_telegram_bot,
            max_client=support_max_client,
        )
        if support_db is not None
        else None
    )
    web_app["support_admin_notifier"] = SupportAdminNotifier(
        telegram_bot=support_telegram_bot,
        telegram_admin_id=settings.support_telegram_admin_id,
        max_client=support_max_client,
        max_admin_id=settings.support_max_admin_id,
    )
    if support_service is not None:
        support_service.notifier = web_app["support_admin_notifier"]

    logging.getLogger(__name__).info(
        "Бот запущен в окружении %s. WebApp: %s",
        settings.environment,
        settings.webapp_url,
    )

    if support_service is not None:
        storage_kind = "postgresql" if isinstance(support_db, SupportDatabase) else "json"
        polling_kind = "active" if (support_telegram_bot and support_telegram_dp) else "inactive"
        logger.info(
            "Support subsystem ready: storage=%s telegram_polling=%s admin_channel=%s",
            storage_kind,
            polling_kind,
            settings.support_telegram_admin_id or settings.support_max_admin_id or "none",
        )
    else:
        logger.info("Support subsystem disabled")

    await _notify_admin(bot, settings, schema_changed)

    # Запускаем мониторинг TPS
    # from tps_monitor import get_tps_monitor
    #tps_monitor = get_tps_monitor()
    #tps_monitor.start()
    #logger.info("TPS мониторинг запущен")

    background_tasks: list[asyncio.Task[None]] = []

    # Запускаем фоновые задачи для новой очереди уведомлений
    background_tasks.append(asyncio.create_task(_generator_notifications_task(db)))
    background_tasks.append(asyncio.create_task(_scheduled_notifications_task(db)))
    max_game_bot = (
        MaxGameBotClient(settings.max_bot_token)
        if settings.max_bot_token
        else None
    )
    background_tasks.append(asyncio.create_task(
        _notification_outbox_task(
            bot,
            db,
            settings.webapp_url,
            push_sender=push_sender,
            max_bot=max_game_bot,
        )
    ))
    if support_telegram_bot is not None and support_telegram_dp is not None:
        background_tasks.append(asyncio.create_task(_support_bot_polling_task(support_telegram_bot, support_telegram_dp)))
    if support_delivery_dispatcher is not None:
        background_tasks.append(asyncio.create_task(_support_delivery_task(support_delivery_dispatcher)))

    # Запускаем фоновую задачу для экспирации инвайтов в дружеские игры
    background_tasks.append(asyncio.create_task(_expire_friend_invites_task(db)))

    # Запускаем фоновую задачу для очистки сгенерированных inline-карточек
    background_tasks.append(asyncio.create_task(_generated_inline_cleanup_task()))

    # Запускаем фоновую задачу для очистки устаревших строк daily_quests_progress
    background_tasks.append(asyncio.create_task(_daily_quests_cleanup_task(db)))

    try:
        while True:
            try:
                await bot.delete_webhook(drop_pending_updates=True, request_timeout=10)
                logger.info("Telegram webhook удалён перед запуском polling")
                await dp.start_polling(bot)
            except TelegramNetworkError as e:
                logger.warning("Telegram polling временно недоступен: %s. Повтор через 10 секунд.", e)
                await asyncio.sleep(10)
            except Exception:
                logger.exception("Telegram polling упал. WebApp остаётся запущенным, повтор через 10 секунд.")
                await asyncio.sleep(10)
    finally:
        # Останавливаем мониторинг TPS
        #try:
            #from tps_monitor import get_tps_monitor
            #tps_monitor = get_tps_monitor()
            #tps_monitor.stop()
            #logger.info("TPS мониторинг остановлен")
        #except Exception as e:
            # logger.warning(f"Ошибка при остановке TPS мониторинга: {e}")
        for task in background_tasks:
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
            logger.info("Фоновые задачи остановлены")

        if db:
            await db.close()
            logging.getLogger(__name__).info("Подключение к БД закрыто")
        if extraid_db:
            await extraid_db.disconnect()
            logging.getLogger(__name__).info("Подключение к ExtraID БД закрыто")
        if support_db:
            await support_db.disconnect()
            logging.getLogger(__name__).info("Подключение к Support БД закрыто")
        if support_telegram_bot:
            await support_telegram_bot.session.close()
        if web_runner:
            await web_runner.cleanup()
            logging.getLogger(__name__).info("WebApp сервер остановлен")


async def _support_bot_polling_task(bot: Bot, dp) -> None:
    """Run the separate Telegram support bot without touching the game bot dispatcher."""
    conflict_logged = False
    while True:
        try:
            await bot.delete_webhook(drop_pending_updates=True, request_timeout=10)
            logger.info("Support Telegram webhook removed before polling")
            conflict_logged = False
            await dp.start_polling(bot)
        except TelegramConflictError as e:
            if not conflict_logged:
                logger.error(
                    "Support Telegram polling conflict (409): another process already polls this bot token. "
                    "Only one polling process is allowed per token. Message: %s",
                    e,
                )
                conflict_logged = True
            await asyncio.sleep(30)
        except (TelegramNetworkError, TelegramServerError) as e:
            logger.warning("Support Telegram polling temporarily unavailable: %s. Retry in 10 seconds.", e)
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Support Telegram polling crashed. Retry in 10 seconds.")
            await asyncio.sleep(10)


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


async def _generator_notifications_task(db: Database | None) -> None:
    """Фоновая задача для постановки событий генератора в очередь."""
    if not db:
        return

    while True:
        try:
            await asyncio.sleep(30)
            queued = await db.check_generator_notifications()
            if queued:
                logger.info("Generator notifications queued: %d", len(queued))
        except Exception as e:
            logger.error(f"Ошибка в задаче проверки уведомлений о генераторе: {e}", exc_info=True)
            await asyncio.sleep(10)


async def _scheduled_notifications_task(db: Database | None) -> None:
    """Фоновая задача для персональных магазинных и дневных уведомлений."""
    if not db:
        return

    while True:
        try:
            await asyncio.sleep(60)
            created = await db.enqueue_due_scheduled_notifications(limit=100)
            if created:
                logger.info("Scheduled notifications queued: %d", created)
        except Exception as e:
            logger.error("Ошибка в задаче расписания уведомлений: %s", e, exc_info=True)
            await asyncio.sleep(10)


async def _notification_outbox_task(
    bot: Bot,
    db: Database | None,
    webapp_url: str,
    *,
    push_sender: FcmPushSender | None = None,
    max_bot: MaxGameBotClient | None = None,
) -> None:
    """Единый dispatcher личных уведомлений из notification_outbox."""
    if not db:
        return

    while True:
        try:
            await asyncio.sleep(5)
            notifications = await db.fetch_pending_notifications(limit=50)
            for notif in notifications:
                await _deliver_notification(
                    bot,
                    db,
                    webapp_url,
                    notif,
                    push_sender=push_sender,
                    max_bot=max_bot,
                )
        except Exception as e:
            logger.error("Ошибка dispatcher уведомлений: %s", e, exc_info=True)
            await asyncio.sleep(10)


async def _record_returnclock_delivery_event(
    db: Database,
    notif: dict,
    payload: dict,
    *,
    event_type: str,
    channel: str | None = None,
    provider_message_id: str | None = None,
    event_key: str = "default",
    metadata: dict | None = None,
) -> None:
    """Best-effort append-only attribution for ReturnClock-linked notifications."""
    if not hasattr(db, "record_returnclock_delivery_event"):
        return
    decision_id = (
        notif.get("returnclock_decision_id")
        or payload.get("rc_decision_id")
        or payload.get("decision_id")
    )
    if not decision_id:
        return
    notification_id = int(notif["id"])
    attempt = int(notif.get("attempts") or 0)
    delivery_id = (
        notif.get("returnclock_delivery_id")
        or payload.get("delivery_id")
        or str(notification_id)
    )
    safe_key = str(event_key or "default").replace(":", "_")[:96]
    event_id = (
        f"outbox:{notification_id}:attempt:{attempt}:"
        f"{channel or 'none'}:{safe_key}:{event_type}"
    )
    try:
        await db.record_returnclock_delivery_event(
            int(notif["user_id"]),
            str(decision_id),
            event_id=event_id,
            event_type=event_type,
            outbox_id=notification_id,
            delivery_id=str(delivery_id),
            provider_message_id=(
                str(provider_message_id) if provider_message_id else None
            ),
            channel=(str(channel) if channel else None),
            metadata=metadata or {},
        )
    except Exception:
        logger.warning(
            "ReturnClock delivery telemetry failed: notification_id=%s event=%s",
            notification_id,
            event_type,
            exc_info=True,
        )


async def _update_returnclock_delivery_status(
    db: Database,
    notif: dict,
    payload: dict,
    status: str,
) -> None:
    if not hasattr(db, "update_returnclock_decision"):
        return
    decision_id = (
        notif.get("returnclock_decision_id")
        or payload.get("rc_decision_id")
        or payload.get("decision_id")
    )
    if not decision_id:
        return
    try:
        await db.update_returnclock_decision(
            int(notif["user_id"]),
            str(decision_id),
            status=status,
            outbox_id=int(notif["id"]),
        )
    except Exception:
        logger.warning(
            "ReturnClock decision status failed: notification_id=%s status=%s",
            notif.get("id"),
            status,
            exc_info=True,
        )


async def _deliver_notification(
    bot: Bot,
    db: Database,
    webapp_url: str,
    notif: dict,
    *,
    push_sender: FcmPushSender | None = None,
    max_bot: MaxGameBotClient | None = None,
) -> None:
    from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError

    notif_id = int(notif["id"])
    payload = notif.get("payload") or {}
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}
    payload = dict(payload)
    payload["notification_id"] = str(notif_id)
    decision_id = (
        notif.get("returnclock_decision_id")
        or payload.get("rc_decision_id")
        or payload.get("decision_id")
    )
    delivery_id = notif.get("returnclock_delivery_id") or payload.get("delivery_id")
    if decision_id:
        payload["rc_decision_id"] = str(decision_id)
        payload["entrypoint"] = "notification"
    if delivery_id:
        payload["delivery_id"] = str(delivery_id)

    category = str(notif["category"])
    event_type = str(notif["event_type"])
    section = notification_section(category, payload)
    body = format_telegram_notification_message(event_type, payload)
    user_id = int(notif["user_id"])
    if hasattr(db, "notification_cancellation_reason"):
        try:
            cancellation_reason = await db.notification_cancellation_reason(
                {**notif, "payload": payload}
            )
        except Exception:
            cancellation_reason = None
            logger.warning(
                "Не удалось перепроверить актуальность уведомления id=%s",
                notif_id,
                exc_info=True,
            )
        if cancellation_reason:
            await db.cancel_notification(notif_id, cancellation_reason)
            return

    mobile_only = bool(payload.get("mobile_only"))
    delivery_mode = "app_then_telegram"
    if hasattr(db, "get_notification_delivery_mode"):
        try:
            delivery_mode = await db.get_notification_delivery_mode(user_id)
        except Exception:
            logger.debug("Не удалось получить режим доставки уведомлений user_id=%s", user_id, exc_info=True)
    if mobile_only:
        delivery_mode = "app_only"
    if delivery_mode not in {"app_then_telegram", "app_only", "telegram_only"}:
        delivery_mode = "app_then_telegram"

    telegram_sent = False
    telegram_blocked = False
    push_attempted = False
    push_sent = False

    if delivery_mode in {"app_then_telegram", "app_only"}:
        push_attempted, push_sent, push_deferred = await _send_android_pushes(
            db,
            notif,
            category=category,
            event_type=event_type,
            payload=payload,
            push_sender=push_sender,
        )
        if push_deferred:
            return
        if push_sent:
            await db.mark_notification_sent(notif_id)
            await _update_returnclock_delivery_status(db, notif, payload, "sent")
            return

    if delivery_mode == "app_only":
        await db.mark_notification_failed(notif_id)
        await _record_returnclock_delivery_event(
            db,
            notif,
            payload,
            event_type="provider_unavailable",
            channel="android",
            metadata={"reason": "app_only_without_successful_push"},
        )
        await _update_returnclock_delivery_status(db, notif, payload, "failed")
        return

    max_identity = None
    if max_bot is not None and hasattr(db, "get_platform_identity_for_user"):
        try:
            max_identity = await db.get_platform_identity_for_user(user_id, "max")
        except Exception:
            logger.debug(
                "Не удалось получить MAX identity user_id=%s",
                user_id,
                exc_info=True,
            )
    if max_identity:
        try:
            result = await max_bot.send_message(
                str(max_identity["subject"]),
                body,
                open_app=True,
                text_format="html",
            )
        except Exception as exc:
            logger.debug(
                "Не удалось отправить MAX-уведомление %s: %s",
                notif_id,
                exc,
            )
            result = {
                "ok": False,
                "status": None,
                "error_class": type(exc).__name__,
            }
        if result.get("ok"):
            await _record_returnclock_delivery_event(
                db,
                notif,
                payload,
                event_type="provider_accepted",
                channel="max",
                provider_message_id=(
                    (result.get("data") or {}).get("body", {}).get("mid")
                    if isinstance(result.get("data"), dict)
                    else None
                ),
            )
            await db.mark_notification_sent(notif_id)
            await _update_returnclock_delivery_status(db, notif, payload, "sent")
            return
        await _record_returnclock_delivery_event(
            db,
            notif,
            payload,
            event_type="provider_failed",
            channel="max",
            metadata={
                "status": result.get("status"),
                "error_class": result.get("error_class"),
            },
        )
        await db.mark_notification_failed(notif_id)
        await _update_returnclock_delivery_status(db, notif, payload, "failed")
        return

    if await _postpone_quiet_telegram_notification(db, notif, category=category):
        return

    try:
        # ВАЖНО: используем web_app=WebAppInfo, а не url=. Иначе Telegram
        # открывает URL во внешнем браузере без Telegram.WebApp.initData,
        # и аутентификация в игре проваливается. Через web_app= Telegram
        # поднимает Mini App контекст и WebApp SDK передаёт initData,
        # который фронт использует для /api/profile.
        webapp_url_with_section = build_webapp_url(webapp_url, section=section, payload=payload)
        keyboard = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(
                text="Открыть ExtraArena",
                web_app=WebAppInfo(url=webapp_url_with_section),
            )
        ]])
        telegram_message = await bot.send_message(
            chat_id=user_id,
            text=body,
            parse_mode="HTML",
            reply_markup=keyboard,
        )
        telegram_sent = True
        await _record_returnclock_delivery_event(
            db,
            notif,
            payload,
            event_type="provider_accepted",
            channel="telegram",
            provider_message_id=getattr(telegram_message, "message_id", None),
        )
    except (TelegramForbiddenError, TelegramBadRequest):
        telegram_blocked = True
        await _record_returnclock_delivery_event(
            db,
            notif,
            payload,
            event_type="provider_blocked",
            channel="telegram",
        )
    except Exception as e:
        logger.debug("Не удалось отправить Telegram-уведомление %s: %s", notif_id, e)
        await _record_returnclock_delivery_event(
            db,
            notif,
            payload,
            event_type="provider_failed",
            channel="telegram",
            metadata={"error_class": type(e).__name__},
        )

    if telegram_sent:
        await db.mark_notification_sent(notif_id)
        await _update_returnclock_delivery_status(db, notif, payload, "sent")
    elif telegram_blocked and not push_attempted:
        await db.mark_notification_blocked(notif_id)
        await _update_returnclock_delivery_status(db, notif, payload, "blocked")
    else:
        await db.mark_notification_failed(notif_id)
        await _update_returnclock_delivery_status(db, notif, payload, "failed")


async def _send_android_pushes(
    db: Database,
    notif: dict,
    *,
    category: str,
    event_type: str,
    payload: dict,
    push_sender: FcmPushSender | None,
) -> tuple[bool, bool, bool]:
    if push_sender is None or not hasattr(db, "get_push_devices"):
        return False, False, False

    user_id = int(notif["user_id"])
    try:
        devices = await db.get_push_devices(user_id, platform="android")
    except Exception:
        logger.debug("Не удалось получить Android push-устройства user_id=%s", user_id, exc_info=True)
        return False, False, False

    if not devices:
        return False, False, False

    payload_data = build_android_push_payload(category, event_type, payload)
    any_sent = False
    push_attempted = False
    quiet_skipped = False
    defer_until_candidates = []
    now_utc = _notification_utc_now()
    for device_index, device in enumerate(devices):
        token = device.get("token")
        if not token:
            continue
        device_key = str(device.get("id") or device_index)
        if _is_quiet_android_reminder_push(
            device,
            category=category,
            event_type=event_type,
            now_utc=now_utc,
        ):
            quiet_skipped = True
            defer_until = _next_android_reminder_push_time(device, now_utc=now_utc)
            if defer_until is not None:
                defer_until_candidates.append(defer_until)
            continue
        push_attempted = True
        result = await push_sender.send(
            token=token,
            title=payload_data.title,
            body=payload_data.body,
            data=payload_data.data,
        )
        if result.ok:
            any_sent = True
            await _record_returnclock_delivery_event(
                db,
                notif,
                payload,
                event_type="provider_accepted",
                channel="android",
                provider_message_id=getattr(result, "message_id", None),
                event_key=device_key,
                metadata={"device_id": device.get("id")},
            )
            continue
        await _record_returnclock_delivery_event(
            db,
            notif,
            payload,
            event_type="provider_failed",
            channel="android",
            event_key=device_key,
            metadata={
                "device_id": device.get("id"),
                "error": result.error,
                "permanent": bool(result.permanent),
            },
        )
        if hasattr(db, "mark_push_device_error"):
            await db.mark_push_device_error(
                token,
                result.error or "push_delivery_failed",
                permanent=result.permanent,
            )
    if quiet_skipped and not push_attempted:
        defer_until = min(defer_until_candidates) if defer_until_candidates else None
        if defer_until is not None and hasattr(db, "postpone_notification"):
            await db.postpone_notification(int(notif["id"]), defer_until)
            await _record_returnclock_delivery_event(
                db,
                notif,
                payload,
                event_type="deferred_quiet_hours",
                channel="android",
                metadata={"defer_until": defer_until.isoformat()},
            )
            logger.info(
                "Android notification postponed until %s during local quiet hours for user_id=%s",
                defer_until.isoformat(),
                user_id,
            )
            return True, False, True
        logger.info("Android notification skipped during local quiet hours for user_id=%s", user_id)
        await _record_returnclock_delivery_event(
            db,
            notif,
            payload,
            event_type="skipped_quiet_hours",
            channel="android",
        )
        return True, True, False
    return push_attempted or quiet_skipped, any_sent, False


async def _expire_friend_invites_task(db: Database | None) -> None:
    if not db:
        return

    while True:
        try:
            await asyncio.sleep(30)
            expired = await db.expire_old_invites()
            if expired:
                logger.info(f"Expired {expired} friend invite(s)")
        except Exception as e:
            logger.error(f"Ошибка в задаче экспирации инвайтов: {e}", exc_info=True)
            await asyncio.sleep(10)


async def _generated_inline_cleanup_task() -> None:
    from pathlib import Path

    GENERATED_DIR = Path(__file__).parent / "generated" / "inline" / "profile"
    TTL_SECONDS = 900  # 15 minutes
    MAX_FILES = 200

    while True:
        await asyncio.sleep(300)  # run every 5 minutes
        try:
            if not GENERATED_DIR.exists():
                continue
            files = sorted(
                GENERATED_DIR.glob("*.png"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            now = time.time()
            removed = 0
            for fp in files:
                age = now - fp.stat().st_mtime
                if age > TTL_SECONDS:
                    fp.unlink(missing_ok=True)
                    removed += 1
            # cap total files
            remaining = [f for f in files if f.exists()]
            for fp in remaining[MAX_FILES:]:
                fp.unlink(missing_ok=True)
                removed += 1
            if removed:
                logger.info(f"Inline card cleanup: removed {removed} old PNG(s)")
        except Exception:
            pass


async def _daily_quests_cleanup_task(db) -> None:
    """Периодически удаляет строки daily_quests_progress старше 30 дней.

    Таблица растёт на 5 строк на активного пользователя в день без встроенной экспирации.
    Очистка использует индекс reset_date/id и ограниченные батчи, чтобы не создавать
    длинную транзакцию. Запускается раз в 6 часов; первый прогон через 60с после старта.
    """
    RETENTION_DAYS = 30
    INTERVAL_SECONDS = 6 * 3600  # 4 раза в сутки
    await asyncio.sleep(60)
    while True:
        try:
            removed = await db.cleanup_old_daily_quests_progress(RETENTION_DAYS)
            if removed:
                logger.info(f"Daily-quests cleanup: removed {removed} rows older than {RETENTION_DAYS}d")
        except Exception:
            logger.warning("Daily-quests cleanup task failed", exc_info=True)
        await asyncio.sleep(INTERVAL_SECONDS)


async def _notify_admin(bot: Bot, settings, schema_changed: bool) -> None:
    if not db:
        return

    stats = await db.get_statistics()
    last_updated: datetime | None = db.schema_last_updated
    last_updated_str = last_updated.isoformat() if last_updated else "неизвестно"
    schema_state = "обновлена" if schema_changed else "актуальна"
    db_info = db.settings

    webapp_url = settings.webapp_url
    if not webapp_url.startswith("https://"):
        webapp_url = f"{webapp_url} (⚠️ требуется HTTPS)"
    
    text = (
        "<b>ExtraCards запущен</b>\n"
        f"WebApp: {webapp_url}\n"
        f"Окружение: {settings.environment}\n"
        f"БД: {db_info.host}:{db_info.port}/{db_info.database}\n"
        f"Web сервер: {settings.web_host}:{settings.web_port}\n"
        f"Схема БД: v{SCHEMA_VERSION} ({schema_state})\n"
        f"Последнее обновление схемы: {last_updated_str}\n"
        f"Игроков: {stats['players']}\n"
        f"Активных ExtraPass: {stats['extra_pass_active']}\n"
        f"Макс. трофеев: {stats['max_trophies_global']}"
    )

    try:
        await bot.send_message(ADMIN_ID, text)
    except Exception as e:
        logger.warning(f"Не удалось отправить сообщение админу: {e}")


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiogram import Bot
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiohttp import web

from bot import create_bot
from bot.constants import ADMIN_ID
from infrastructure.config import get_settings
from infrastructure.database import Database, SCHEMA_VERSION
from infrastructure.extraid_database import ExtraIDDatabase
from infrastructure.notifications import (
    build_webapp_url,
    format_notification_message,
    notification_section,
)
from infrastructure.push_notifications import FcmPushSender, build_android_push_payload
from web.server import create_web_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

db: Database | None = None
web_runner: web.AppRunner | None = None
web_site: web.BaseSite | None = None

REMINDER_PUSH_QUIET_START_HOUR = 22
REMINDER_PUSH_QUIET_END_HOUR = 9


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
    if category != "reminders" or event_type != "daily_reminder":
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


async def main() -> None:
    global db, web_runner, web_site
    settings = get_settings()

    db = Database(settings.database)
    await db.connect()
    schema_changed = await db.init_schema()

    extraid_db = ExtraIDDatabase(settings.extraid_database.dsn)
    await extraid_db.connect()

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
            logger.warning("Проверьте зависимости HTTP-клиента: requests и curl для обхода VPN при необходимости.")
        except Exception as e:
            logger.error(f"Ошибка инициализации YooKassa: {e}", exc_info=True)
    else:
        logger.warning("YooKassa настройки не найдены. Платежи будут недоступны.")
        logger.warning("Добавьте в .env файл:")
        logger.warning("  YOOKASSA_SHOP_ID=ваш_shop_id")
        logger.warning("  YOOKASSA_SECRET_KEY=ваш_secret_key")
        logger.warning("  YOOKASSA_TEST_MODE=true")

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

    push_sender = FcmPushSender()
    web_app = create_web_app(
        db,
        settings.bot_token,
        extraid_db=extraid_db,
        payment_service=payment_service,
        rustore_payment_service=rustore_payment_service,
        rustore_console_app_id=settings.rustore.console_app_id if settings.rustore else "",
        rustore_app_url=settings.rustore_app_url,
        payment_provider_order=settings.payment_provider_order,
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

    logging.getLogger(__name__).info(
        "Бот запущен в окружении %s. WebApp: %s",
        settings.environment,
        settings.webapp_url,
    )

    await _notify_admin(bot, settings, schema_changed)

    # Запускаем мониторинг TPS
    # from tps_monitor import get_tps_monitor
    #tps_monitor = get_tps_monitor()
    #tps_monitor.start()
    #logger.info("TPS мониторинг запущен")

    background_tasks: list[asyncio.Task[None]] = []

    # Запускаем фоновую задачу для проверки уведомлений о кубике
    background_tasks.append(asyncio.create_task(_dice_notifications_task(bot, db, push_sender=push_sender)))

    # Запускаем фоновые задачи для новой очереди уведомлений
    background_tasks.append(asyncio.create_task(_generator_notifications_task(db)))
    background_tasks.append(asyncio.create_task(_scheduled_notifications_task(db)))
    background_tasks.append(asyncio.create_task(_notification_outbox_task(bot, db, settings.webapp_url, push_sender=push_sender)))

    # Запускаем фоновую задачу для экспирации инвайтов в дружеские игры
    background_tasks.append(asyncio.create_task(_expire_friend_invites_task(db)))

    # Запускаем фоновую задачу для очистки сгенерированных inline-карточек
    background_tasks.append(asyncio.create_task(_generated_inline_cleanup_task()))

    try:
        await bot.delete_webhook(drop_pending_updates=True)
        logger.info("Telegram webhook удалён перед запуском polling")
        await dp.start_polling(bot)
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
        if web_runner:
            await web_runner.cleanup()
            logging.getLogger(__name__).info("WebApp сервер остановлен")


async def _dice_notifications_task(
    bot: Bot,
    db: Database | None,
    *,
    push_sender: FcmPushSender | None = None,
) -> None:
    """Фоновая задача для проверки и отправки уведомлений о готовности кубика."""
    if not db:
        return
    
    from aiogram.exceptions import TelegramForbiddenError, TelegramBadRequest
    
    while True:
        try:
            # Проверяем каждые 5 секунд для тестирования (было 300 секунд)
            await asyncio.sleep(5)
            
            notifications = await db.check_dice_ready_notifications()
            for notif in notifications:
                _push_attempted, push_sent, _push_deferred = await _send_android_pushes(
                    db,
                    {"user_id": notif["user_id"]},
                    category="reminders",
                    event_type="dice_ready",
                    payload={"section": "arena"},
                    push_sender=push_sender,
                )
                if push_sent:
                    await db.mark_dice_notification_sent(notif["user_id"])
                    continue

                telegram_done = False
                try:
                    await bot.send_message(
                        chat_id=notif["user_id"],
                        text="🎲 Эй! Самое время бросить кости!",
                    )
                    telegram_done = True
                except (TelegramForbiddenError, TelegramBadRequest):
                    # Игнорируем ошибки отправки (заблокированные пользователи и т.д.)
                    telegram_done = True
                except Exception as e:
                    logger.debug(f"Не удалось отправить уведомление о кубике пользователю {notif['user_id']}: {e}")
                if telegram_done:
                    await db.mark_dice_notification_sent(notif["user_id"])
        except Exception as e:
            logger.error(f"Ошибка в задаче проверки уведомлений о кубике: {e}", exc_info=True)
            await asyncio.sleep(10)  # Ждем 10 секунд перед повторной попыткой


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
) -> None:
    """Единый dispatcher личных уведомлений из notification_outbox."""
    if not db:
        return

    while True:
        try:
            await asyncio.sleep(5)
            notifications = await db.fetch_pending_notifications(limit=50)
            for notif in notifications:
                await _deliver_notification(bot, db, webapp_url, notif, push_sender=push_sender)
        except Exception as e:
            logger.error("Ошибка dispatcher уведомлений: %s", e, exc_info=True)
            await asyncio.sleep(10)


async def _deliver_notification(
    bot: Bot,
    db: Database,
    webapp_url: str,
    notif: dict,
    *,
    push_sender: FcmPushSender | None = None,
) -> None:
    from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError

    notif_id = int(notif["id"])
    payload = notif.get("payload") or {}
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}

    category = str(notif["category"])
    event_type = str(notif["event_type"])
    section = notification_section(category, payload)
    body = format_notification_message(event_type, payload)
    user_id = int(notif["user_id"])
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
            return

    if delivery_mode == "app_only":
        await db.mark_notification_failed(notif_id)
        return

    try:
        keyboard = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(
                text="Открыть ExtraArena",
                url=build_webapp_url(webapp_url, section=section),
            )
        ]])
        await bot.send_message(
            chat_id=user_id,
            text=body,
            reply_markup=keyboard,
        )
        telegram_sent = True
    except (TelegramForbiddenError, TelegramBadRequest):
        telegram_blocked = True
    except Exception as e:
        logger.debug("Не удалось отправить Telegram-уведомление %s: %s", notif_id, e)

    if telegram_sent:
        await db.mark_notification_sent(notif_id)
    elif telegram_blocked and not push_attempted:
        await db.mark_notification_blocked(notif_id)
    else:
        await db.mark_notification_failed(notif_id)


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
    for device in devices:
        token = device.get("token")
        if not token:
            continue
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
            continue
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
            logger.info(
                "Android reminder push postponed until %s during local quiet hours for user_id=%s",
                defer_until.isoformat(),
                user_id,
            )
            return True, False, True
        logger.info("Android reminder push skipped during local quiet hours for user_id=%s", user_id)
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

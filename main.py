import asyncio
import logging
import time
from datetime import datetime

from aiogram import Bot
from aiohttp import web

from bot import create_bot
from bot.constants import ADMIN_ID
from infrastructure.config import get_settings
from infrastructure.database import Database, SCHEMA_VERSION
from infrastructure.extraid_database import ExtraIDDatabase
from web.server import create_web_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

db: Database | None = None
web_runner: web.AppRunner | None = None
web_site: web.BaseSite | None = None


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

    web_app = create_web_app(
        db,
        settings.bot_token,
        extraid_db=extraid_db,
        payment_service=payment_service,
        webapp_url=settings.webapp_url,
        extra_shop_url=settings.extra_shop_url,
        stars_rate_rub=settings.stars_rate_rub,
        stars_markup=settings.stars_markup,
        stars_test_mode=settings.stars_test_mode,
    )
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

    # Запускаем фоновую задачу для проверки уведомлений о кубике
    asyncio.create_task(_dice_notifications_task(bot, db))

    # Запускаем фоновую задачу для проверки уведомлений о генераторе ключей
    asyncio.create_task(_generator_notifications_task(bot, db))

    # Запускаем фоновую задачу для экспирации инвайтов в дружеские игры
    asyncio.create_task(_expire_friend_invites_task(db))

    # Запускаем фоновую задачу для очистки сгенерированных inline-карточек
    asyncio.create_task(_generated_inline_cleanup_task())

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
        
        if db:
            await db.close()
            logging.getLogger(__name__).info("Подключение к БД закрыто")
        if extraid_db:
            await extraid_db.disconnect()
            logging.getLogger(__name__).info("Подключение к ExtraID БД закрыто")
        if web_runner:
            await web_runner.cleanup()
            logging.getLogger(__name__).info("WebApp сервер остановлен")


async def _dice_notifications_task(bot: Bot, db: Database | None) -> None:
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
                try:
                    await bot.send_message(
                        chat_id=notif["user_id"],
                        text="🎲 Эй! Самое время бросить кости!",
                    )
                    await db.mark_dice_notification_sent(notif["user_id"])
                except (TelegramForbiddenError, TelegramBadRequest):
                    # Игнорируем ошибки отправки (заблокированные пользователи и т.д.)
                    await db.mark_dice_notification_sent(notif["user_id"])  # Помечаем как отправленное, чтобы не спамить
                except Exception as e:
                    logger.debug(f"Не удалось отправить уведомление о кубике пользователю {notif['user_id']}: {e}")
        except Exception as e:
            logger.error(f"Ошибка в задаче проверки уведомлений о кубике: {e}", exc_info=True)
            await asyncio.sleep(10)  # Ждем 10 секунд перед повторной попыткой


async def _generator_notifications_task(bot: Bot, db: Database | None) -> None:
    """Фоновая задача для проверки и отправки уведомлений о готовых ключах генератора."""
    if not db:
        return

    from aiogram.exceptions import TelegramForbiddenError, TelegramBadRequest

    while True:
        try:
            await asyncio.sleep(30)

            notifications = await db.check_generator_notifications()
            for notif in notifications:
                user_id = notif["user_id"]
                try:
                    status = await db.get_generator_status(user_id)
                    key_count = status.get("accumulated_keys", 0)
                    await bot.send_message(
                        chat_id=user_id,
                        text=f"🔑 Генератор накопил {key_count} ключ(ей)! Забери их в разделе «Генератор».",
                    )
                    await db.mark_generator_notification_sent(user_id)
                except (TelegramForbiddenError, TelegramBadRequest):
                    await db.mark_generator_notification_sent(user_id)
                except Exception as e:
                    logger.debug(f"Не удалось отправить уведомление о генераторе пользователю {user_id}: {e}")
        except Exception as e:
            logger.error(f"Ошибка в задаче проверки уведомлений о генераторе: {e}", exc_info=True)
            await asyncio.sleep(10)


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

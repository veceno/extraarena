from __future__ import annotations

import re
from datetime import datetime
from html import escape
from typing import List, Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    PreCheckoutQuery,
    ReplyKeyboardMarkup,
    SuccessfulPayment,
    WebAppInfo,
)

from bot.constants import ADMIN_ID, CATEGORY_ALIASES, CATEGORY_LABELS, DEFAULT_CATEGORY
from infrastructure.database import Database
from infrastructure.storage import add_user_id, get_all_user_ids
from infrastructure.payments_logic import process_successful_payment

router = Router(name="basic")


NEWS_FETCH_LIMIT = 20
NEWS_MAX_CHARS = 1500
OPTION_PATTERN = re.compile(
    r'(?P<prefix>\s*)(?:--|-)(?P<key>[a-zA-Z_]+)\s*=\s*(?P<value>"[^"]*"|\'[^\']*\'|[^\s]+)'
)


def register_basic_handlers(dp: Dispatcher, webapp_url: str, db: Database | None = None) -> None:
    webapp_info = WebAppInfo(url=webapp_url)

    async def _remember_user(message: Message) -> bool:
        if not message.from_user:
            return False
        add_user_id(message.from_user.id)
        created = False
        if db:
            created = await db.ensure_user(
                user_id=message.from_user.id,
                username=message.from_user.username,
                first_name=message.from_user.first_name,
                last_name=message.from_user.last_name,
            )
        return created

    @router.message(CommandStart())
    async def handle_start(message: Message, bot: Bot) -> None:
        created = await _remember_user(message)
        keyboard = ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="🔥 ExtraCards Season 0 🏟️", web_app=webapp_info)],
                [
                    KeyboardButton(text="🙋‍♂️ Профиль"),
                    KeyboardButton(text="📊 Аналитика"),
                ],
                [
                    KeyboardButton(text="📰 Новости"),
                    KeyboardButton(text="🤝 Поддержка"),
                ],
            ],
            resize_keyboard=True,
        )
        await message.answer(
            "Привет! Жми кнопку, чтобы открыть ExtraCards WebApp.",
            reply_markup=keyboard,
        )
        if created:
            await message.answer("✨ Учётная запись создана. Добро пожаловать в арену!")
        
        # Проверяем готовность кубика
        if db and message.from_user:
            await _check_and_notify_dice_ready(message.from_user.id, bot)

    @router.message(Command("id"))
    async def handle_id(message: Message) -> None:
        await _remember_user(message)
        user_id = message.from_user.id if message.from_user else "unknown"
        await message.answer(f"Твой Telegram ID: <code>{user_id}</code>")

    @router.message(Command("tps"))
    async def handle_tps(message: Message) -> None:
        """Команда для отображения TPS (Ticks Per Second) сервера."""
        if not message.from_user:
            await message.answer("Не удалось определить пользователя.")
            return

        admin_id = message.from_user.id
        if admin_id != ADMIN_ID:
            await message.answer("Команда доступна только администратору.")
            return

        await _remember_user(message)

        try:
            from tps_monitor import get_tps_monitor
            
            monitor = get_tps_monitor()
            stats = monitor.get_statistics()
            
            # Форматируем сообщение
            text = (
                f"<b>📊 Производительность сервера</b>\n\n"
                f"{stats['status_emoji']} <b>Статус:</b> {stats['status']}\n\n"
                f"<b>Текущий TPS:</b> {stats['current_tps']}\n"
                f"<b>Средний TPS (1 мин):</b> {stats['average_tps_1m']}\n"
                f"<b>Средний TPS (5 мин):</b> {stats['average_tps_5m']}\n"
                f"<b>Мин. TPS (1 мин):</b> {stats['min_tps_1m']}\n"
                f"<b>Макс. TPS (1 мин):</b> {stats['max_tps_1m']}\n\n"
                f"<b>Всего тиков:</b> {stats['total_ticks']:,}\n"
                f"<b>Время работы:</b> {stats['uptime_formatted']}\n\n"
                f"<i>Идеальное значение: 20.0 TPS</i>"
            )
            
            await message.answer(text, parse_mode="HTML")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Ошибка получения TPS: {e}", exc_info=True)
            await message.answer(f"❌ Ошибка получения статистики TPS: {e}")

    @router.message(Command("broadcast"))
    async def handle_broadcast(message: Message, bot: Bot) -> None:
        if not message.from_user:
            await message.answer("Не удалось определить пользователя.")
            return

        admin_id = message.from_user.id
        if admin_id != ADMIN_ID:
            await message.answer("Команда доступна только администратору.")
            return

        await _remember_user(message)

        payload = _extract_payload(
            message.text or message.caption or "", commands=("/broadcast",)
        )
        text, button_text, button_url = _parse_broadcast_payload(payload)
        photo_id = message.photo[-1].file_id if message.photo else None

        if not text and not photo_id:
            await message.answer("Добавьте текст сообщения или изображение для рассылки.")
            return

        recipients = [uid for uid in get_all_user_ids() if uid != admin_id]
        if not recipients:
            await message.answer("Некому отправлять сообщение - база пользователей пуста.")
            return

        keyboard = _build_broadcast_keyboard(button_text, button_url)

        sent, failed = 0, 0
        for user_id in recipients:
            try:
                if photo_id:
                    await bot.send_photo(
                        chat_id=user_id,
                        photo=photo_id,
                        caption=text or None,
                        reply_markup=keyboard,
                    )
                else:
                    await bot.send_message(
                        chat_id=user_id,
                        text=text,
                        reply_markup=keyboard,
                    )
                sent += 1
            except (TelegramForbiddenError, TelegramBadRequest):
                failed += 1
            except Exception:
                failed += 1

        await message.answer(
            f"Рассылка завершена.\nУспешно: {sent}\nОшибок: {failed}\nПолучателей: {len(recipients)}"
        )

    @router.message(Command("stat"))
    async def handle_stat(message: Message) -> None:
        if not message.from_user or message.from_user.id != ADMIN_ID:
            await message.answer("Команда доступна только администратору.")
            return

        if not db:
            await message.answer("База данных недоступна.")
            return

        stats = await db.get_statistics()
        text = (
            "<b>Статистика бота</b>\n"
            f"Игроков: <b>{stats['players']}</b>\n"
            f"Активных ExtraPass: {stats['extra_pass_active']}\n"
            f"Суммарные трофеи: {stats['total_trophies']}\n"
            f"Максимум трофеев у игрока: {stats['max_trophies_global']}"
        )
        await message.answer(text)

    @router.message(Command("del"))
    async def handle_delete(message: Message) -> None:
        if not message.from_user or message.from_user.id != ADMIN_ID:
            await message.answer("Команда доступна только администратору.")
            return

        if not db:
            await message.answer("База данных недоступна.")
            return

        parts = (message.text or "").split()
        if len(parts) < 2:
            await message.answer("Использование: /del <user_id>")
            return

        try:
            user_id = int(parts[1])
        except ValueError:
            await message.answer("Некорректный user_id. Используйте число.")
            return

        deleted = await db.delete_user(user_id)
        if deleted:
            await message.answer(
                f"✅ Пользователь {user_id} полностью удалён из базы данных.\n\n"
                f"<b>Удалены:</b>\n"
                f"• Карты пользователя (user_cards)\n"
                f"• Колоды (deck_presets)\n"
                f"• Почта (user_mail)\n"
                f"• Кейсы (user_cases)\n"
                f"• Кулдауны (cooldowns)\n"
                f"• Платежи (payments)\n"
                f"• Уведомления (notifications)\n"
                f"• Настройки (user_settings)\n"
                f"• Профиль (profiles)\n"
                f"• Посты в сообществе (кроме админских)\n"
                f"• Сообщения в глобальном чате (кроме админских)\n"
                f"• Лайки постов (кроме лайков админских постов)\n"
                f"• Пользователь (users)\n\n"
                f"<b>Сохранены:</b>\n"
                f"• Админские посты в сообществе\n"
                f"• Админские сообщения в глобальном чате\n"
                f"• Лайки админских постов",
                parse_mode="HTML"
            )
        else:
            await message.answer(f"❌ Пользователь {user_id} не найден или произошла ошибка при удалении.")

    @router.message(Command("news_post"))
    async def handle_news_post(message: Message) -> None:
        if not message.from_user or message.from_user.id != ADMIN_ID:
            await message.answer("Команда доступна только администратору.")
            return

        if not db:
            await message.answer("База данных недоступна.")
            return

        payload = _extract_payload(
            message.text or message.caption or "", commands=("/news_post",)
        )
        news_payload = _parse_news_payload(payload)

        if not news_payload["text"]:
            await message.answer("Добавьте текст новости.")
            return

        if news_payload["category"] not in CATEGORY_LABELS:
            await message.answer(
                "Укажите категорию через `--category=` (обновление/событие/важное/интересное).",
                parse_mode="Markdown",
            )
            return

        photo_id = message.photo[-1].file_id if message.photo else None

        await db.create_news_entry(
            author_id=message.from_user.id,
            text=news_payload["text"],
            category=news_payload["category"],
            button_text=news_payload["button_text"],
            button_url=news_payload["button_url"],
            photo_file_id=photo_id,
        )

        preview = CATEGORY_LABELS[news_payload["category"]]
        await message.answer(
            f"Новость сохранена в разделе «Новости».\nКатегория: {preview}",
        )

    @router.message(Command("community_post"))
    async def handle_community_post(message: Message) -> None:
        if not message.from_user or message.from_user.id != ADMIN_ID:
            await message.answer("Команда доступна только администратору.")
            return

        if not db:
            await message.answer("База данных недоступна.")
            return

        payload = _extract_payload(
            message.text or message.caption or "", commands=("/community_post",)
        )
        
        # Парсим опции: --title= и --content=
        text, options = _extract_options(
            payload, allowed={"title", "content"}
        )
        
        title = options.get("title", "").strip()
        content = text.strip() if text else options.get("content", "").strip()
        
        if not title and not content:
            # Если нет опций, используем весь текст как контент, а заголовок - первые 50 символов
            full_text = text.strip()
            if not full_text:
                await message.answer(
                    "Использование: /community_post --title=\"Заголовок\" Текст поста\n"
                    "Или: /community_post Текст поста (заголовок будет автоматически)"
                )
                return
            content = full_text
            title = full_text[:50] + ("..." if len(full_text) > 50 else "")
        elif not title:
            title = content[:50] + ("..." if len(content) > 50 else "")
        elif not content:
            content = text.strip() if text else title

        photo_id = message.photo[-1].file_id if message.photo else None

        result = await db.create_community_post(
            author_id=message.from_user.id,
            title=title,
            content=content,
            photo_file_id=photo_id,
        )

        if result.get("success"):
            await message.answer(
                f"✅ Пост создан в разделе «Коммьюнити».\n"
                f"Заголовок: {title}\n"
                f"Контент: {content[:100]}{'...' if len(content) > 100 else ''}"
            )
        else:
            await message.answer(
                f"❌ Ошибка создания поста: {result.get('error', 'неизвестная ошибка')}"
            )

    async def _check_and_notify_dice_ready(user_id: int, bot: Bot) -> None:
        """Проверить, готов ли кубик, и отправить уведомление."""
        if not db:
            return
        
        try:
            status = await db.get_dice_status(user_id)
            # Проверяем, можно ли бросать (значит, прошёл час)
            if status.get("can_roll") and status.get("last_roll"):
                # Проверяем, не отправляли ли уже уведомление
                notifications = await db.check_dice_ready_notifications()
                for notif in notifications:
                    if notif["user_id"] == user_id:
                        try:
                            await bot.send_message(
                                chat_id=user_id,
                                text="🎲 Счастливый кубик снова доступен! Брось его в разделе «Арена» и получи награду!",
                            )
                            await db.mark_dice_notification_sent(user_id)
                        except (TelegramForbiddenError, TelegramBadRequest):
                            pass  # Пользователь заблокировал бота или не найден
                        break
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка проверки кубика: {e}", exc_info=True)

    @router.message(lambda msg: msg.text == "🙋‍♂️ Профиль")
    async def handle_profile(message: Message, bot: Bot) -> None:
        await _remember_user(message)
        if not db:
            await message.answer("Профили недоступны без подключения к базе.")
            return

        if not message.from_user:
            return
        
        # Проверяем готовность кубика
        await _check_and_notify_dice_ready(message.from_user.id, bot)

        profile = await db.get_user_profile(message.from_user.id)
        if not profile:
            await message.answer("Профиль пока не создан. Попробуйте ещё раз.")
            return

        reg_date = profile["reg_date"]
        reg_str = reg_date.strftime("%d.%m.%Y") if isinstance(reg_date, datetime) else str(reg_date)
        extra_pass = "⚡ Активирован" if profile["extra_pass"] == "active" else "⛔ Не активирован"
        squad = profile["squad_id"] if profile["squad_id"] else "-"
        status_badge = {
            "active": "🟢",
            "warn": "🟡",
            "banned": "🔴",
        }.get(profile["status"], "⚪")

        text = (
            "<b>🏟️ ExtraCards | Профиль</b>\n"
            f"<i>{profile['title'] or 'Новичок'}</i>\n\n"
            f"🆔 ID: <code>{profile['user_id']}</code>\n"
            f"{extra_pass}\n"
            f"🏆 Трофеи: <b>{profile['trophies']}</b> (макс. {profile['max_trophies']})\n"
            f"🧪 Частицы: {profile['particles']} • 💎 {profile['gems']} • 💰 {profile['coins']}\n"
            f"👥 Сквад: {squad}\n"
            f"{status_badge} Статус: {profile['status']}\n"
            f"📅 С нами с: {reg_str}"
        )
        await message.answer(text)

    @router.message(lambda msg: msg.text == "📰 Новости")
    async def handle_news(message: Message) -> None:
        await _remember_user(message)
        if not db:
            await message.answer("Новости временно недоступны.")
            return

        news_items = await db.get_recent_news(limit=NEWS_FETCH_LIMIT)
        if not news_items:
            await message.answer("Пока нет новостей. Возвращайся позже!")
            return

        pages = _build_news_pages(news_items)
        text = pages[0]
        keyboard = _build_news_keyboard(current=0, total=len(pages))
        await message.answer(
            text,
            reply_markup=keyboard,
            parse_mode="HTML",
        )

    @router.callback_query(F.data.startswith("news_page:"))
    async def handle_news_page(callback: CallbackQuery) -> None:
        if not db:
            await callback.answer("Новости недоступны", show_alert=True)
            return

        try:
            page_index = int(callback.data.split(":", 1)[1])
        except (ValueError, AttributeError):
            await callback.answer("Некорректная страница", show_alert=True)
            return

        news_items = await db.get_recent_news(limit=NEWS_FETCH_LIMIT)
        pages = _build_news_pages(news_items)
        if not pages:
            await callback.message.edit_text(
                "Пока нет новостей. Возвращайся позже!",
                reply_markup=None,
            )
            await callback.answer()
            return

        total = len(pages)
        page_index = max(0, min(page_index, total - 1))
        keyboard = _build_news_keyboard(page_index, total)
        await callback.message.edit_text(
            pages[page_index],
            reply_markup=keyboard,
            parse_mode="HTML",
        )
        await callback.answer()

    @router.callback_query(F.data == "news_close")
    async def handle_news_close(callback: CallbackQuery) -> None:
        if callback.message:
            await callback.message.delete()
        await callback.answer()

    @router.callback_query(F.data == "broadcast_close")
    async def handle_broadcast_close(callback: CallbackQuery) -> None:
        if callback.message:
            await callback.message.delete()
        await callback.answer()

    @router.message(lambda msg: msg.text == "🤝 Поддержка")
    async def handle_support(message: Message) -> None:
        await _remember_user(message)
        support_keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="Перейти в поддержку", url="https://t.me/lqsup")]
            ]
        )
        await message.answer(
            "Нажмите на кнопку ниже, чтобы перейти в канал поддержки:",
            reply_markup=support_keyboard,
        )

    @router.message(lambda msg: msg.web_app_data is not None)
    async def handle_webapp_data(message: Message) -> None:
        await _remember_user(message)
        payload = message.web_app_data.data if message.web_app_data else "{}"
        await message.answer(f"Получены данные WebApp:\n<code>{payload}</code>")

    @router.message(F.photo & ~F.text & ~F.caption)
    async def handle_admin_photo_id(message: Message) -> None:
        """Обработчик для получения file_id изображения от админа."""
        if not message.from_user or message.from_user.id != ADMIN_ID:
            return  # Пропускаем, если не админ
        
        if not message.photo:
            return
        
        # Получаем file_id самого большого размера изображения
        photo_id = message.photo[-1].file_id
        
        await message.answer(
            f"📷 Telegram Image ID:\n<code>{photo_id}</code>",
            parse_mode="HTML"
        )

    @router.pre_checkout_query()
    async def handle_pre_checkout_query(query: PreCheckoutQuery, bot: Bot) -> None:
        """Обработчик предварительной проверки платежа через Telegram Stars."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Всегда подтверждаем запрос для Stars
            await bot.answer_pre_checkout_query(
                pre_checkout_query_id=query.id,
                ok=True
            )
            logger.info(f"Pre-checkout query подтвержден для invoice_payload={query.invoice_payload}")
        except Exception as e:
            logger.error(f"Ошибка обработки pre-checkout query: {e}", exc_info=True)
            try:
                await bot.answer_pre_checkout_query(
                    pre_checkout_query_id=query.id,
                    ok=False,
                    error_message="Ошибка обработки платежа. Попробуйте позже."
                )
            except:
                pass

    @router.message(F.successful_payment)
    async def handle_successful_payment(message: Message, bot: Bot) -> None:
        """Обработчик успешного платежа через Telegram Stars."""
        import logging
        logger = logging.getLogger(__name__)
        
        if not message.from_user or not message.successful_payment:
            return
        
        user_id = message.from_user.id
        payment: SuccessfulPayment = message.successful_payment
        
        try:
            # Извлекаем invoice_payload
            invoice_payload = payment.invoice_payload
            payment_id = f"stars_{invoice_payload}"
            
            logger.info(
                f"✅ Платеж Stars успешно получен. "
                f"User: {user_id}, Payment ID: {payment_id}, "
                f"Amount: {payment.total_amount} {payment.currency}, "
                f"Invoice payload: {invoice_payload}"
            )
            
            if not db:
                logger.error("База данных недоступна для обработки платежа Stars")
                await message.answer("❌ Ошибка: база данных недоступна. Обратитесь в поддержку.")
                return
            
            # Получаем запись о платеже из БД
            payment_record = await db.get_payment_by_id(payment_id)

            logger.info(
                "STARS LOOKUP: payment_id=%s found=%s record_keys=%s",
                payment_id, bool(payment_record),
                list(payment_record.keys()) if payment_record else 'NONE',
            )
            
            if not payment_record:
                logger.warning(f"Платеж {payment_id} не найден в БД, создаем запись")
                # Создаем запись о платеже
                await db.create_payment(
                    user_id=user_id,
                    payment_id=payment_id,
                    amount=float(payment.total_amount),
                    currency=payment.currency,
                    description=payment.invoice_payload,
                    metadata={"invoice_payload": invoice_payload},
                    status="succeeded"
                )
                payment_record = await db.get_payment_by_id(payment_id)
            
            # Фиксируем статус как succeeded, чтобы запись была актуальна
            await db.update_payment_status(
                payment_id=payment_id,
                status="succeeded"
            )
            payment_record = await db.get_payment_by_id(payment_id)
            if not payment_record:
                logger.error("Не удалось получить запись о платеже %s после обновления статуса", payment_id)
                await message.answer("❌ Ошибка обработки платежа. Обратитесь в поддержку.")
                return
            
            # Универсально выдаём награды и создаём письмо
            processing_result = await process_successful_payment(
                db=db,
                payment_id=payment_id,
                payment_record=payment_record,
                source="telegram_stars",
                logger=logger,
            )

            if processing_result["status"] == "missing_payment":
                await message.answer("❌ Платеж не найден. Обратитесь в поддержку.")
                return

            if processing_result["status"] == "already_processed":
                await message.answer("✅ Ваш платеж уже был обработан ранее. Товары выданы.")
                return

            if processing_result["rewards_text"]:
                rewards_message = "✅ Платеж успешно обработан!\n\nПолучено:\n" + "\n".join(
                    f"• {reward}" for reward in processing_result["rewards_text"]
                )
            else:
                rewards_message = "✅ Платеж успешно обработан!"

            await message.answer(rewards_message)
                
        except Exception as e:
            logger.error(f"Ошибка обработки платежа Stars для user_id {user_id}: {e}", exc_info=True)
            try:
                await message.answer("❌ Произошла ошибка при обработке платежа. Обратитесь в поддержку.")
            except:
                pass

    @router.message()
    async def remember_user(message: Message) -> None:
        await _remember_user(message)

    dp.include_router(router)


def _extract_payload(text: str, commands: tuple[str, ...]) -> str:
    if not text:
        return ""
    raw = text.strip()
    if not raw:
        return ""
    first_word, *rest = raw.split(maxsplit=1)
    first_lower = first_word.lower()
    for cmd in commands:
        cmd_lower = cmd.lower()
        if first_lower == cmd_lower or first_lower.startswith(f"{cmd_lower}@"):
            return rest[0].lstrip() if rest else ""
    return raw


def _parse_broadcast_payload(raw: str) -> tuple[str, Optional[str], Optional[str]]:
    text, options = _extract_options(raw, allowed={"button_text", "button_url"})
    return text, options.get("button_text"), options.get("button_url")


def _parse_news_payload(raw: str) -> dict[str, Optional[str]]:
    text, options = _extract_options(
        raw, allowed={"button_text", "button_url", "category"}
    )
    category_value = options.get("category", DEFAULT_CATEGORY)
    category = CATEGORY_ALIASES.get(category_value.lower(), category_value.lower())
    return {
        "text": text,
        "button_text": options.get("button_text"),
        "button_url": options.get("button_url"),
        "category": category,
    }


def _extract_options(raw: str, allowed: set[str]) -> tuple[str, dict[str, str]]:
    matches = []
    options: dict[str, str] = {}

    for match in OPTION_PATTERN.finditer(raw):
        key = match.group("key").lower()
        if key not in allowed:
            continue
        value = match.group("value").strip()
        if value and value[0] in {'"', "'"} and value[-1] == value[0]:
            value = value[1:-1]
        options[key] = value
        matches.append(match.span())

    if not matches:
        return raw.strip(), options

    cleaned_parts = []
    last = 0
    for start, end in matches:
        cleaned_parts.append(raw[last:start])
        last = end
    cleaned_parts.append(raw[last:])
    cleaned = "".join(cleaned_parts)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    return cleaned, options


def _build_news_pages(news_items) -> List[str]:
    pages: List[str] = []
    current_entries: List[str] = []
    current_len = 0

    for item in news_items:
        entry = _render_news_entry(item)
        entry_len = len(entry)
        if current_entries and (
            current_len + entry_len > NEWS_MAX_CHARS or len(current_entries) >= 2
        ):
            pages.append("\n\n".join(current_entries))
            current_entries = []
            current_len = 0
        current_entries.append(entry)
        current_len += entry_len

    if current_entries:
        pages.append("\n\n".join(current_entries))

    return pages


def _render_news_entry(item) -> str:
    category_label = CATEGORY_LABELS.get(item["category"], item["category"])
    created_at = item["created_at"]
    if isinstance(created_at, datetime):
        date_str = created_at.strftime("%d.%m.%Y")
    else:
        date_str = str(created_at)

    text = escape(item["text"] or "")
    entry = f"<b>{category_label}</b> • {date_str}\n{text}"

    if item.get("photo_file_id"):
        entry += "\n📷 <i>В новости есть изображение</i>"

    if item.get("button_url"):
        label = escape(item["button_text"] or "Подробнее")
        entry += f'\n<a href="{item["button_url"]}">{label}</a>'

    return entry


def _build_news_keyboard(current: int, total: int) -> InlineKeyboardMarkup | None:
    if total <= 1:
        return InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="Закрыть", callback_data="news_close")]]
        )

    nav_row = []
    if current > 0:
        nav_row.append(InlineKeyboardButton(text="⟵ Назад", callback_data=f"news_page:{current - 1}"))

    nav_row.append(InlineKeyboardButton(text=f"{current + 1}/{total}", callback_data=f"news_page:{current}"))

    if current < total - 1:
        nav_row.append(InlineKeyboardButton(text="Вперёд ⟶", callback_data=f"news_page:{current + 1}"))

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            nav_row,
            [InlineKeyboardButton(text="Закрыть", callback_data="news_close")],
        ]
    )
    return keyboard


def _build_broadcast_keyboard(
    button_text: Optional[str], button_url: Optional[str]
) -> InlineKeyboardMarkup:
    rows = []
    if button_text and button_url:
        rows.append([InlineKeyboardButton(text=button_text, url=button_url)])
    rows.append([InlineKeyboardButton(text="Закрыть", callback_data="broadcast_close")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


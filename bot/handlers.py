from __future__ import annotations

import logging
import re
from html import escape
from typing import Any, Optional
from urllib.parse import urlparse

from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    PreCheckoutQuery,
    SuccessfulPayment,
    WebAppInfo,
)

from bot.constants import ADMIN_ID
from infrastructure.database import Database
from infrastructure.payments_logic import process_successful_payment
from infrastructure.storage import add_user_id, get_all_user_ids

router = Router(name="basic")

BROADCAST_PENDING: dict[int, dict[str, Any]] = {}
BROADCAST_CONFIRM_CALLBACK = "broadcast_confirm"
BROADCAST_CANCEL_CALLBACK = "broadcast_cancel"
BROADCAST_CLOSE_CALLBACK = "broadcast_close"
BROADCAST_CAPTION_LIMIT = 1000

OPTION_PATTERN = re.compile(
    r'(?P<prefix>\s*)(?:--|-)(?P<key>[a-zA-Z_]+)\s*=\s*(?P<value>"[^"]*"|\'[^\']*\'|[^\s]+)'
)


def register_basic_handlers(dp: Dispatcher, webapp_url: str, db: Database | None = None) -> None:
    webapp_info = WebAppInfo(url=webapp_url)

    async def _remember_user_id(message: Message) -> None:
        if message.from_user:
            add_user_id(message.from_user.id)

    @router.message(CommandStart())
    async def handle_start(message: Message) -> None:
        await _remember_user_id(message)
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    _game_button(
                        text="Открыть ExtraArena",
                        webapp_url=webapp_url,
                        webapp_info=webapp_info,
                    )
                ]
            ]
        )
        await message.answer(
            "Нажми кнопку ниже, чтобы открыть игру.",
            reply_markup=keyboard,
        )

    @router.message(Command("id"))
    async def handle_id(message: Message) -> None:
        await _remember_user_id(message)
        user_id = message.from_user.id if message.from_user else "unknown"
        await message.answer(f"Твой Telegram ID: <code>{user_id}</code>")

    @router.message(Command("broadcast"))
    async def handle_broadcast(message: Message, bot: Bot) -> None:
        if not message.from_user:
            await message.answer("Не удалось определить пользователя.")
            return

        admin_id = message.from_user.id
        if admin_id != ADMIN_ID:
            await message.answer("Команда доступна только администратору.")
            return

        await _remember_user_id(message)

        raw_payload = _extract_payload(
            message.text or message.caption or "", commands=("/broadcast",)
        )
        text, button_text, button_url = _parse_broadcast_payload(raw_payload)
        photo_id = message.photo[-1].file_id if message.photo else None

        if not text and not photo_id:
            await message.answer(
                "Добавьте текст сообщения или изображение для рассылки.\n\n"
                "HTML поддерживается: <b>жирный</b>, <i>курсив</i>, "
                '<a href="https://example.com">ссылка</a>.'
            )
            return

        broadcast_payload = {
            "text": text,
            "photo_id": photo_id,
            "button_text": button_text,
            "button_url": button_url,
        }
        recipient_keyboard = _build_broadcast_keyboard(button_text, button_url)

        try:
            await message.answer("Предпросмотр рассылки:")
            await _send_broadcast_payload(
                bot,
                admin_id,
                broadcast_payload,
                reply_markup=recipient_keyboard,
            )
        except TelegramBadRequest as exc:
            await message.answer(
                "Telegram не принял сообщение. Проверьте HTML-разметку, URL кнопки "
                f"и длину текста.\n\n<code>{escape(str(exc))}</code>"
            )
            return

        BROADCAST_PENDING[admin_id] = broadcast_payload
        recipients = len([uid for uid in get_all_user_ids() if uid != admin_id])
        await message.answer(
            f"Отправить эту рассылку? Получателей: {recipients}",
            reply_markup=_build_broadcast_confirm_keyboard(),
        )

    @router.callback_query(F.data == BROADCAST_CONFIRM_CALLBACK)
    async def handle_broadcast_confirm(callback: CallbackQuery, bot: Bot) -> None:
        if callback.from_user.id != ADMIN_ID:
            await callback.answer("Недоступно", show_alert=True)
            return

        payload = BROADCAST_PENDING.pop(callback.from_user.id, None)
        if not payload:
            await callback.answer("Нет рассылки для отправки", show_alert=True)
            return

        recipients = [uid for uid in get_all_user_ids() if uid != callback.from_user.id]
        if not recipients:
            if callback.message:
                await callback.message.edit_text("Некому отправлять сообщение - база пользователей пуста.")
            await callback.answer()
            return

        if callback.message:
            await callback.message.edit_text("Рассылка отправляется...")

        keyboard = _build_broadcast_keyboard(payload.get("button_text"), payload.get("button_url"))
        sent, failed = 0, 0
        for user_id in recipients:
            try:
                await _send_broadcast_payload(bot, user_id, payload, reply_markup=keyboard)
                sent += 1
            except (TelegramForbiddenError, TelegramBadRequest):
                failed += 1
            except Exception:
                logging.getLogger(__name__).exception(
                    "Unexpected broadcast delivery failure: user_id=%s", user_id
                )
                failed += 1

        if callback.message:
            await callback.message.edit_text(
                f"Рассылка завершена.\nУспешно: {sent}\nОшибок: {failed}\nПолучателей: {len(recipients)}"
            )
        await callback.answer()

    @router.callback_query(F.data == BROADCAST_CANCEL_CALLBACK)
    async def handle_broadcast_cancel(callback: CallbackQuery) -> None:
        if callback.from_user.id != ADMIN_ID:
            await callback.answer("Недоступно", show_alert=True)
            return
        BROADCAST_PENDING.pop(callback.from_user.id, None)
        if callback.message:
            await callback.message.edit_text("Рассылка отменена.")
        await callback.answer()

    @router.callback_query(F.data == BROADCAST_CLOSE_CALLBACK)
    async def handle_broadcast_close(callback: CallbackQuery) -> None:
        if callback.message:
            await callback.message.delete()
        await callback.answer()

    @router.pre_checkout_query()
    async def handle_pre_checkout_query(query: PreCheckoutQuery, bot: Bot) -> None:
        """Обработчик предварительной проверки платежа через Telegram Stars."""
        logger = logging.getLogger(__name__)

        try:
            await bot.answer_pre_checkout_query(
                pre_checkout_query_id=query.id,
                ok=True,
            )
            logger.info("Pre-checkout query подтвержден для invoice_payload=%s", query.invoice_payload)
        except Exception as exc:
            logger.error("Ошибка обработки pre-checkout query: %s", exc, exc_info=True)
            try:
                await bot.answer_pre_checkout_query(
                    pre_checkout_query_id=query.id,
                    ok=False,
                    error_message="Ошибка обработки платежа. Попробуйте позже.",
                )
            except Exception:
                pass

    @router.message(F.successful_payment)
    async def handle_successful_payment(message: Message, bot: Bot) -> None:
        """Обработчик успешного платежа через Telegram Stars."""
        logger = logging.getLogger(__name__)

        if not message.from_user or not message.successful_payment:
            return

        user_id = message.from_user.id
        payment: SuccessfulPayment = message.successful_payment

        try:
            invoice_payload = payment.invoice_payload
            payment_id = f"stars_{invoice_payload}"

            logger.info(
                "✅ Платеж Stars успешно получен. User: %s, Payment ID: %s, Amount: %s %s, Invoice payload: %s",
                user_id,
                payment_id,
                payment.total_amount,
                payment.currency,
                invoice_payload,
            )

            if not db:
                logger.error("База данных недоступна для обработки платежа Stars")
                await message.answer("❌ Ошибка: база данных недоступна. Обратитесь в поддержку.")
                return

            payment_record = await db.get_payment_by_id(payment_id)
            logger.info(
                "STARS LOOKUP: payment_id=%s found=%s record_keys=%s",
                payment_id,
                bool(payment_record),
                list(payment_record.keys()) if payment_record else "NONE",
            )

            if not payment_record:
                logger.warning("Платеж %s не найден в БД, создаем запись", payment_id)
                await db.create_payment(
                    user_id=user_id,
                    payment_id=payment_id,
                    amount=float(payment.total_amount),
                    currency=payment.currency,
                    description=payment.invoice_payload,
                    metadata={"invoice_payload": invoice_payload},
                    status="succeeded",
                )
                payment_record = await db.get_payment_by_id(payment_id)

            await db.update_payment_status(
                payment_id=payment_id,
                status="succeeded",
            )
            payment_record = await db.get_payment_by_id(payment_id)
            if not payment_record:
                logger.error("Не удалось получить запись о платеже %s после обновления статуса", payment_id)
                await message.answer("❌ Ошибка обработки платежа. Обратитесь в поддержку.")
                return

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

        except Exception as exc:
            logger.error("Ошибка обработки платежа Stars для user_id %s: %s", user_id, exc, exc_info=True)
            try:
                await message.answer("❌ Произошла ошибка при обработке платежа. Обратитесь в поддержку.")
            except Exception:
                pass

    dp.include_router(router)


def _game_button(text: str, webapp_url: str, webapp_info: WebAppInfo) -> InlineKeyboardButton:
    if webapp_url.startswith("https://"):
        return InlineKeyboardButton(text=text, web_app=webapp_info)
    return InlineKeyboardButton(text=text, url=webapp_url)


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
    return "".join(cleaned_parts).strip(), options


def _safe_button_url(url: Optional[str]) -> Optional[str]:
    raw = (url or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https", "tg"}:
        return None
    return raw


def _build_broadcast_keyboard(
    button_text: Optional[str], button_url: Optional[str]
) -> InlineKeyboardMarkup:
    rows = []
    safe_url = _safe_button_url(button_url)
    if button_text and safe_url:
        rows.append([InlineKeyboardButton(text=button_text, url=safe_url)])
    rows.append([InlineKeyboardButton(text="Закрыть", callback_data=BROADCAST_CLOSE_CALLBACK)])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _build_broadcast_confirm_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Отправить", callback_data=BROADCAST_CONFIRM_CALLBACK),
                InlineKeyboardButton(text="Отменить", callback_data=BROADCAST_CANCEL_CALLBACK),
            ]
        ]
    )


async def _send_broadcast_payload(
    bot: Bot,
    chat_id: int,
    payload: dict[str, Any],
    *,
    reply_markup: InlineKeyboardMarkup | None,
) -> None:
    text = str(payload.get("text") or "")
    photo_id = payload.get("photo_id")

    if photo_id and text and len(text) <= BROADCAST_CAPTION_LIMIT:
        await bot.send_photo(
            chat_id=chat_id,
            photo=photo_id,
            caption=text,
            parse_mode="HTML",
            reply_markup=reply_markup,
        )
        return

    if photo_id and not text:
        await bot.send_photo(
            chat_id=chat_id,
            photo=photo_id,
            reply_markup=reply_markup,
        )
        return

    if photo_id:
        await bot.send_photo(chat_id=chat_id, photo=photo_id)

    if text:
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
            reply_markup=reply_markup,
        )
        return

from __future__ import annotations

from typing import Any

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from infrastructure.telegram_proxy import create_telegram_bot
from support.attachments import compress_support_attachment
from support.bot_client import BotReply, SupportBotConversationManager
from support.constants import SupportChannel


def create_support_bot(token: str, support_service: Any | None = None) -> tuple[Bot, Dispatcher]:
    bot = create_telegram_bot(token, default=DefaultBotProperties(parse_mode="HTML"))
    dp = Dispatcher(support_bot=True)
    router = Router(name="support")
    conversation = SupportBotConversationManager(support_service) if support_service is not None else None

    def _user_context(message: Message) -> dict[str, str]:
        user = message.from_user
        channel_id = str(user.id) if user else str(message.chat.id)
        display_name = (
            user.full_name
            if user and user.full_name
            else user.username
            if user and user.username
            else "Telegram user"
        )
        return {
            "channel_id": channel_id,
            "external_user_id": f"tg:{channel_id}",
            "display_name": display_name,
        }

    def _reply_markup(reply: BotReply) -> InlineKeyboardMarkup | None:
        if not reply.choices:
            return None
        rows = [
            [InlineKeyboardButton(text=choice.label, callback_data=f"support:{choice.id}")]
            for choice in reply.choices
        ]
        return InlineKeyboardMarkup(inline_keyboard=rows)

    async def _send_reply(message: Message, reply: BotReply) -> None:
        await message.answer(reply.text, reply_markup=_reply_markup(reply))

    async def _download_image_attachment(message: Message):
        if message.photo:
            photo = message.photo[-1]
            file_obj = await bot.download(photo.file_id)
            if file_obj is None:
                return None
            return compress_support_attachment(
                file_obj.getvalue(),
                original_filename=f"telegram-photo-{message.message_id}.jpg",
            )
        document = message.document
        if document and document.mime_type in {"image/png", "image/jpeg", "image/webp"}:
            file_obj = await bot.download(document.file_id)
            if file_obj is None:
                return None
            return compress_support_attachment(
                file_obj.getvalue(),
                original_filename=document.file_name or f"telegram-document-{message.message_id}",
            )
        return None

    async def _record_reply_attachment(reply: BotReply, attachment_metadata) -> bool:
        if support_service is None or attachment_metadata is None or not reply.ticket or not reply.message:
            return False
        identity_id = None
        if reply.identity:
            identity_id = reply.identity.get("id")
        if identity_id is None and reply.ticket:
            identity_id = reply.ticket.get("requester_identity_id")
        await support_service.record_attachment(
            ticket_id=reply.ticket["id"],
            message_id=reply.message["id"],
            uploader_identity_id=identity_id,
            metadata=attachment_metadata.as_record(),
        )
        return True

    @router.message(CommandStart())
    async def support_start(message: Message) -> None:
        if conversation is None:
            await message.answer("ExtraArena Support\n\nСообщение получено. Поддержка скоро ответит.")
            return
        context = _user_context(message)
        reply = await conversation.begin(channel=SupportChannel.TELEGRAM, **context)
        await _send_reply(message, reply)

    @router.message(Command("new"))
    async def support_new(message: Message) -> None:
        if conversation is None:
            await message.answer("Сообщение получено. Поддержка скоро ответит.")
            return
        context = _user_context(message)
        conversation.clear_session(channel=SupportChannel.TELEGRAM, channel_id=context["channel_id"])
        reply = await conversation.begin(channel=SupportChannel.TELEGRAM, **context)
        await _send_reply(message, reply)

    @router.callback_query(F.data.startswith("support:topic:"))
    async def support_topic(callback: CallbackQuery) -> None:
        if conversation is None or not callback.message:
            await callback.answer()
            return
        user = callback.from_user
        channel_id = str(user.id)
        topic = str(callback.data or "").removeprefix("support:topic:")
        reply = await conversation.select_topic(
            channel=SupportChannel.TELEGRAM,
            channel_id=channel_id,
            topic=topic,
        )
        await callback.message.answer(reply.text, reply_markup=_reply_markup(reply))
        await callback.answer()

    @router.callback_query(F.data.startswith("support:scope:"))
    async def support_account_scope(callback: CallbackQuery) -> None:
        if conversation is None or not callback.message:
            await callback.answer()
            return
        user = callback.from_user
        channel_id = str(user.id)
        scope = str(callback.data or "").removeprefix("support:scope:")
        reply = await conversation.select_account_scope(
            channel=SupportChannel.TELEGRAM,
            channel_id=channel_id,
            scope=scope,
        )
        await callback.message.answer(reply.text, reply_markup=_reply_markup(reply))
        await callback.answer()

    @router.message()
    async def support_message(message: Message) -> None:
        if conversation is None:
            await message.answer("Сообщение получено. Поддержка скоро посмотрит обращение.")
            return
        context = _user_context(message)
        attachment_metadata = None
        try:
            attachment_metadata = await _download_image_attachment(message)
        except ValueError as exc:
            await message.answer(f"Не получилось принять вложение: {exc}. Поддерживаются PNG, JPEG и WebP до 10 МБ.")
            return
        body = message.text or message.caption or ("Вложение" if attachment_metadata is not None else "")
        reply = await conversation.receive_text(
            channel=SupportChannel.TELEGRAM,
            **context,
            body=body,
            metadata={
                "telegram_message_id": message.message_id,
                "has_attachment": attachment_metadata is not None,
            },
        )
        attachment_saved = await _record_reply_attachment(reply, attachment_metadata)
        if attachment_metadata is not None and not attachment_saved:
            reply = BotReply(
                reply.text + "\n\nВложение пока не сохранено: сначала завершите создание обращения.",
                reply.choices,
                ticket=reply.ticket,
                message=reply.message,
                identity=reply.identity,
            )
        await _send_reply(message, reply)

    dp.include_router(router)
    return bot, dp

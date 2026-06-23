from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)


class SupportAdminNotifier:
    def __init__(
        self,
        *,
        telegram_bot: Any | None = None,
        telegram_admin_id: int | str | None = None,
        max_client: Any | None = None,
        max_admin_id: int | str | None = None,
    ) -> None:
        self.telegram_bot = telegram_bot
        self.telegram_admin_id = str(telegram_admin_id or "")
        self.max_client = max_client
        self.max_admin_id = str(max_admin_id or "")

    async def send_admin_code(self, code: str) -> None:
        text = f"ExtraArena Support admin login code:\n<code>{code}</code>"
        if self.telegram_bot and self.telegram_admin_id:
            try:
                await self.telegram_bot.send_message(self.telegram_admin_id, text)
            except Exception:
                logger.warning("Failed to send support admin code via Telegram", exc_info=True)
        if self.max_client and self.max_admin_id:
            try:
                await self.max_client.send_message(self.max_admin_id, f"ExtraArena Support admin login code:\n{code}")
            except Exception:
                logger.warning("Failed to send support admin code via MAX", exc_info=True)

    async def notify_new_ticket(self, ticket: dict[str, Any]) -> None:
        text = (
            "🆕 Новое обращение в поддержку\n"
            f"Тема: {ticket.get('topic') or '—'}\n"
            f"Канал: {ticket.get('channel') or '—'}\n"
            f"ID: {ticket.get('id') or '—'}\n"
            f"Статус: {ticket.get('status') or '—'}"
        )
        await self._send_admin_notification(text)

    async def notify_new_inbound_message(self, ticket: dict[str, Any], message: dict[str, Any]) -> None:
        body_preview = str(message.get("body") or "")[:200]
        text = (
            "💬 Новое сообщение в поддержку\n"
            f"Обращение: {ticket.get('id') or '—'}\n"
            f"Канал: {ticket.get('channel') or '—'}\n"
            f"Сообщение: {body_preview}"
        )
        await self._send_admin_notification(text)

    async def _send_admin_notification(self, text: str) -> None:
        if self.telegram_bot and self.telegram_admin_id:
            try:
                await self.telegram_bot.send_message(self.telegram_admin_id, text)
            except Exception:
                logger.warning("Failed to send support admin notification via Telegram", exc_info=True)
        if self.max_client and self.max_admin_id:
            try:
                await self.max_client.send_message(self.max_admin_id, text)
            except Exception:
                logger.warning("Failed to send support admin notification via MAX", exc_info=True)

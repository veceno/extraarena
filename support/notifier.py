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

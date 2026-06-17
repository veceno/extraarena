from __future__ import annotations

import logging
from typing import Any

from support.constants import SupportChannel


logger = logging.getLogger(__name__)


class SupportDeliveryDispatcher:
    def __init__(self, *, support_db: Any, telegram_bot: Any | None = None, max_client: Any | None = None) -> None:
        self.support_db = support_db
        self.telegram_bot = telegram_bot
        self.max_client = max_client

    async def dispatch_once(self, *, limit: int = 25) -> int:
        rows = await self.support_db.fetch(
            """
            SELECT *
            FROM support_delivery_outbox
            WHERE status = 'pending' AND next_attempt_at <= now()
            ORDER BY created_at ASC
            LIMIT $1
            """,
            int(limit),
        )
        delivered = 0
        for row in rows:
            row = dict(row)
            try:
                await self._send(row)
            except Exception as exc:
                await self.support_db.execute(
                    """
                    UPDATE support_delivery_outbox
                    SET attempts = attempts + 1,
                        last_error = $2,
                        next_attempt_at = now() + interval '60 seconds',
                        updated_at = now()
                    WHERE id = $1
                    """,
                    row["id"],
                    str(exc)[:1000],
                )
                continue
            await self.support_db.execute(
                """
                UPDATE support_delivery_outbox
                SET status = 'sent', attempts = attempts + 1, updated_at = now()
                WHERE id = $1
                """,
                row["id"],
            )
            delivered += 1
        return delivered

    async def _send(self, row: dict[str, Any]) -> None:
        channel = str(row.get("channel") or "")
        channel_id = str(row.get("channel_id") or "")
        payload = row.get("payload") or {}
        text = str(payload.get("text") or "")
        if channel == SupportChannel.TELEGRAM:
            if self.telegram_bot is None:
                raise RuntimeError("telegram_support_client_unavailable")
            await self.telegram_bot.send_message(channel_id, text)
            return
        if channel == SupportChannel.MAX:
            if self.max_client is None:
                raise RuntimeError("max_support_client_unavailable")
            result = await self.max_client.send_message(channel_id, text)
            if isinstance(result, dict) and not result.get("ok", True):
                raise RuntimeError(str(result.get("error") or "max_delivery_failed"))
            return
        raise RuntimeError(f"unsupported_support_channel:{channel}")

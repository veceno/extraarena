from __future__ import annotations

import hmac
from typing import Any

import aiohttp


MAX_API_BASE = "https://platform-api2.max.ru"


def verify_max_webhook_secret(expected: str, received: str | None) -> bool:
    if not expected or not received:
        return False
    return hmac.compare_digest(str(expected), str(received))


def normalize_max_update(update: dict[str, Any]) -> dict[str, Any]:
    """Normalize the MAX update variants needed by the game bot."""
    update_type = str(update.get("update_type") or update.get("type") or "")
    message = update.get("message") if isinstance(update.get("message"), dict) else {}
    body = message.get("body") if isinstance(message.get("body"), dict) else {}
    sender = (
        message.get("sender")
        if isinstance(message.get("sender"), dict)
        else update.get("user")
        if isinstance(update.get("user"), dict)
        else {}
    )
    user_id = (
        sender.get("user_id")
        or sender.get("id")
        or message.get("user_id")
        or update.get("user_id")
    )
    text = body.get("text") if isinstance(body, dict) else None
    if text is None:
        text = message.get("text") or ""
    return {
        "update_type": update_type,
        "user_id": str(user_id or ""),
        "text": str(text or "").strip(),
        "display_name": str(
            sender.get("name")
            or sender.get("first_name")
            or sender.get("username")
            or "Игрок"
        ),
    }


class MaxGameBotClient:
    def __init__(
        self,
        token: str,
        *,
        api_base: str = MAX_API_BASE,
        timeout_seconds: float = 8.0,
    ) -> None:
        self.token = str(token or "").strip()
        self.api_base = str(api_base or MAX_API_BASE).rstrip("/")
        self.timeout_seconds = max(1.0, float(timeout_seconds))

    async def send_message(
        self,
        user_id: str | int,
        text: str,
        *,
        open_app: bool = False,
        text_format: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"text": str(text or "")[:4000]}
        if text_format in {"html", "markdown"}:
            payload["format"] = text_format
        if open_app:
            payload["attachments"] = [
                {
                    "type": "inline_keyboard",
                    "payload": {
                        "buttons": [
                            [
                                {
                                    "type": "open_app",
                                    "text": "Играть в ExtraArena",
                                }
                            ]
                        ]
                    },
                }
            ]
        headers = {
            "Authorization": self.token,
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.api_base}/messages",
                params={"user_id": str(user_id)},
                headers=headers,
                json=payload,
            ) as response:
                response_text = await response.text()
                if response.status >= 400:
                    return {
                        "ok": False,
                        "status": response.status,
                        "error": response_text[:1000],
                    }
                try:
                    data = await response.json()
                except Exception:
                    data = {"text": response_text}
                return {"ok": True, "status": response.status, "data": data}

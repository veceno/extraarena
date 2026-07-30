from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qsl

import jwt as pyjwt
from aiohttp import web

from bot.max_client import (
    MaxGameBotClient,
    normalize_max_update,
    verify_max_webhook_secret,
)
from infrastructure.config import get_settings


logger = logging.getLogger(__name__)

MAX_INIT_DATA_MAX_BYTES = 16 * 1024
MAX_INIT_DATA_MAX_AGE_SECONDS = 60 * 60
MAX_INIT_DATA_CLOCK_SKEW_SECONDS = 5 * 60


def verify_max_init_data(
    init_data: str,
    bot_token: str,
    *,
    now: int | None = None,
) -> dict[str, str] | None:
    """Validate MAX WebAppData without trusting any client-decoded object."""
    if not init_data or not bot_token:
        return None
    try:
        if len(init_data.encode("utf-8")) > MAX_INIT_DATA_MAX_BYTES:
            return None
        pairs = parse_qsl(
            init_data,
            keep_blank_values=True,
            strict_parsing=True,
        )
        keys = [key for key, _value in pairs]
        if (
            not keys
            or any(not key for key in keys)
            or len(keys) != len(set(keys))
            or keys.count("hash") != 1
        ):
            return None
        received_hash = next(value for key, value in pairs if key == "hash")
        if len(received_hash) != 64:
            return None
        values = {key: value for key, value in pairs if key != "hash"}
        launch_params = "\n".join(
            f"{key}={value}" for key, value in sorted(values.items())
        )
        secret_key = hmac.new(
            key=b"WebAppData",
            msg=bot_token.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        calculated_hash = hmac.new(
            key=secret_key,
            msg=launch_params.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(calculated_hash, received_hash):
            return None

        auth_date = int(values.get("auth_date") or 0)
        current_time = int(time.time() if now is None else now)
        if auth_date <= 0:
            return None
        if auth_date < current_time - MAX_INIT_DATA_MAX_AGE_SECONDS - MAX_INIT_DATA_CLOCK_SKEW_SECONDS:
            return None
        if auth_date > current_time + MAX_INIT_DATA_CLOCK_SKEW_SECONDS:
            return None
        return values
    except (TypeError, ValueError, StopIteration, UnicodeError):
        return None


def extract_max_user(data: dict[str, str]) -> dict[str, Any] | None:
    try:
        raw_user = json.loads(data.get("user") or "")
        if not isinstance(raw_user, dict):
            return None
        user_id = int(raw_user.get("id"))
        if user_id <= 0:
            return None
        return {
            "id": user_id,
            "first_name": str(raw_user.get("first_name") or "")[:256] or None,
            "last_name": str(raw_user.get("last_name") or "")[:256] or None,
            "username": str(raw_user.get("username") or "")[:256] or None,
            "language_code": str(raw_user.get("language_code") or "")[:32] or None,
            "photo_url": str(raw_user.get("photo_url") or "")[:2048] or None,
        }
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _make_game_session_token(
    user_id: int,
    session_id: uuid.UUID,
    *,
    jwt_secret: str,
    expiry_days: int,
) -> tuple[str, datetime]:
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=max(1, int(expiry_days)))
    token = pyjwt.encode(
        {
            "user_id": int(user_id),
            "session_id": str(session_id),
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
        },
        jwt_secret,
        algorithm="HS256",
    )
    return token, expires_at


async def max_auth_exchange_handler(request: web.Request) -> web.Response:
    settings = get_settings()
    if not settings.max_bot_token:
        return web.json_response({"error": "max_auth_unavailable"}, status=503)
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)
    if not isinstance(payload, dict):
        return web.json_response({"error": "invalid_json"}, status=400)

    init_data = str(payload.get("init_data") or "").strip()
    verified = verify_max_init_data(init_data, settings.max_bot_token)
    if not verified:
        return web.json_response({"error": "invalid_max_init_data"}, status=401)
    max_user = extract_max_user(verified)
    if not max_user:
        return web.json_response({"error": "invalid_max_user"}, status=401)

    db = request.app["db"]
    extraid_db = request.app.get("extraid_db")
    if extraid_db is None:
        return web.json_response({"error": "auth_storage_unavailable"}, status=503)

    try:
        internal_user_id, created = await db.resolve_or_create_platform_user(
            provider="max",
            subject=str(max_user["id"]),
            username=max_user.get("username"),
            first_name=max_user.get("first_name"),
            last_name=max_user.get("last_name"),
            profile=max_user,
        )
        session_id = uuid.uuid4()
        token, expires_at = _make_game_session_token(
            internal_user_id,
            session_id,
            jwt_secret=settings.jwt_secret,
            expiry_days=settings.jwt_expiry_days,
        )
        # A MAX launch is a non-detachable platform session. Keep one active
        # launch session per player while preserving email/Android sessions.
        await extraid_db.execute(
            """
            UPDATE auth_sessions
            SET revoked = TRUE, revoked_at = NOW()
            WHERE user_id = $1 AND auth_method = 'max' AND revoked = FALSE
            """,
            int(internal_user_id),
        )
        await extraid_db.create_auth_session(
            int(internal_user_id),
            "max",
            hashlib.sha256(token.encode("utf-8")).hexdigest(),
            expires_at,
            device_label=str(payload.get("device_label") or "MAX Mini App")[:256],
            session_id=session_id,
        )
        extra = await extraid_db.get_any_extra_account_by_user_id(internal_user_id)
        return web.json_response(
            {
                "ok": True,
                "token": token,
                "user_id": internal_user_id,
                "provider": "max",
                "created": bool(created),
                "extra_id_bound": bool(extra and not extra.get("deleted_at")),
                "expires_at": expires_at.isoformat(),
            },
            headers={"Cache-Control": "no-store"},
        )
    except Exception:
        logger.exception("MAX auth exchange failed")
        return web.json_response({"error": "max_auth_failed"}, status=500)


async def max_bot_webhook_handler(request: web.Request) -> web.Response:
    settings = get_settings()
    if not settings.max_bot_token or not settings.max_bot_webhook_secret:
        return web.json_response({"error": "max_bot_unavailable"}, status=503)
    if not verify_max_webhook_secret(
        settings.max_bot_webhook_secret,
        request.headers.get("X-Max-Bot-Api-Secret"),
    ):
        return web.json_response({"error": "invalid_max_webhook_secret"}, status=401)
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)
    if not isinstance(payload, dict):
        return web.json_response({"error": "invalid_json"}, status=400)

    update = normalize_max_update(payload)
    user_id = update["user_id"]
    if not user_id:
        return web.json_response({"ok": True, "status": "ignored"})

    text = update["text"].split(maxsplit=1)[0].lower() if update["text"] else ""
    update_type = update["update_type"]
    bot = request.app.get("max_bot_client")
    if bot is None:
        bot = MaxGameBotClient(settings.max_bot_token)

    if update_type == "bot_started" or text in {"/start", "start"}:
        reply = (
            "Добро пожаловать в ExtraArena!\n\n"
            "Открой игру кнопкой ниже. MAX-аккаунт будет подтверждён автоматически, "
            "а ExtraID можно создать и навсегда привязать внутри игры."
        )
        open_app = True
    elif text in {"/id", "id"}:
        reply = f"Твой MAX ID: {user_id}"
        open_app = False
    else:
        reply = "Команды: /start — открыть ExtraArena, /id — показать MAX ID."
        open_app = True

    try:
        result = await bot.send_message(user_id, reply, open_app=open_app)
    except Exception:
        logger.exception("MAX bot reply raised user=%s", user_id)
        return web.json_response({"error": "max_delivery_failed"}, status=502)
    if not result.get("ok"):
        logger.warning(
            "MAX bot reply failed user=%s status=%s",
            user_id,
            result.get("status"),
        )
        return web.json_response({"error": "max_delivery_failed"}, status=502)
    return web.json_response({"ok": True})


def register_max_routes(app: web.Application) -> None:
    settings = get_settings()
    app["max_bot_client"] = (
        MaxGameBotClient(settings.max_bot_token)
        if settings.max_bot_token
        else None
    )
    app.router.add_post("/api/auth/max", max_auth_exchange_handler)
    app.router.add_post("/api/max/webhook", max_bot_webhook_handler)

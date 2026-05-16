from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import string as _stdstring
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt as pyjwt
from aiohttp import web, ClientTimeout, ClientSession

from infrastructure.config import get_settings
from infrastructure.database import Database
from infrastructure.extraid_database import ExtraIDDatabase

logger = logging.getLogger(__name__)

# Rate limiting store: {key: [timestamp, ...]}
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(key: str, max_requests: int, window_seconds: int) -> bool:
    now = time.time()
    timestamps = _rate_limit_store[key]
    _rate_limit_store[key] = [t for t in timestamps if now - t < window_seconds]
    if len(_rate_limit_store[key]) >= max_requests:
        return False
    _rate_limit_store[key].append(now)
    return True

# Import server auth utilities from the parent module
# These will be set at registration time

_verify_init_data_fn = None
_validate_auth_date_fn = None
_extract_user_id_from_init_data_fn = None
_verify_jwt_token_async_fn = None
_display_id_generator_fn = None
_mask_email_fn = None
_nickname_valid_fn = None
_get_ip_country_fn = None
require_user_id_fn = None

_extraid_db: ExtraIDDatabase | None = None


def _get_db(request) -> Database:
    return request.app["db"]


def _get_extraid_db(request) -> ExtraIDDatabase:
    return request.app["extraid_db"]


def _make_jwt_session(user_id: int, session_id, settings) -> tuple[str, datetime]:
    now = datetime.now(timezone.utc)
    session_exp = now + timedelta(days=settings.jwt_expiry_days)
    payload = {
        "user_id": user_id,
        "session_id": str(session_id),
        "iat": int(now.timestamp()),
        "exp": int(session_exp.timestamp()),
    }
    return pyjwt.encode(payload, settings.jwt_secret, algorithm="HS256"), session_exp


def _hash_jwt(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# --- ExtraID: Register ---

async def extraid_register_handler(request: web.Request) -> web.Response:
    db = _get_db(request)
    extraid_db = _get_extraid_db(request)
    settings = get_settings()
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)

    try:
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""
        nickname = (data.get("nickname") or "").strip() or None

        if "@" not in email or "." not in email:
            return web.json_response({"error": "invalid_email"}, status=400)
        if len(password) < 8:
            return web.json_response({"error": "password_too_short", "message": "Пароль должен быть не менее 8 символов"}, status=400)
        if nickname and not _nickname_valid_fn(nickname):
            return web.json_response({"error": "invalid_nickname", "message": "Никнейм: 3-20 символов, только a-z, A-Z, 0-9, _, -"}, status=400)

        existing_email = await extraid_db.get_extra_account_by_email(email)
        if existing_email:
            return web.json_response({"error": "email_taken"}, status=409)
        if nickname:
            existing_nick = await extraid_db.fetchrow(
                "SELECT id FROM extra_accounts WHERE LOWER(nickname) = LOWER($1) AND deleted_at IS NULL", nickname
            )
            if existing_nick:
                return web.json_response({"error": "nickname_taken"}, status=409)

        tg_auth = request.rel_url.query.get("_auth")
        user_id = None
        auth_source = "email_registration"
        if tg_auth:
            verified = _verify_init_data_fn(tg_auth, request.app["bot_token"])
            if verified and _validate_auth_date_fn(verified):
                user_id = _extract_user_id_from_init_data_fn(verified)
                if user_id:
                    auth_source = "telegram"

        if user_id is None:
            user_id = await extraid_db.get_synthetic_user_id()
            await db.ensure_user(user_id=user_id, username=None, first_name=nickname or email, last_name=None)
            await db.execute(
                "UPDATE users SET auth_source = 'email_registration' WHERE user_id = $1",
                user_id
            )

        display_id = _display_id_generator_fn(lambda did: False)
        while await extraid_db.fetchval("SELECT 1 FROM extra_accounts WHERE display_id = $1", display_id):
            display_id = _display_id_generator_fn(lambda did: False)

        password_hash = (await asyncio.to_thread(
            lambda: bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()
        ))
        extra = await extraid_db.create_extra_account(user_id, display_id, email, password_hash, nickname)
        await db.execute(
            "UPDATE users SET extra_account_id = $1, auth_source = $2 WHERE user_id = $3",
            extra["id"], auth_source, user_id
        )

        return web.json_response({
            "ok": True,
            "display_id": display_id,
            "email_sent": False,
        })
    except Exception:
        logger.exception("ExtraID register handler error")
        return web.json_response({"error": "server_error", "message": "Внутренняя ошибка сервера"}, status=500)


# --- ExtraID: Login ---

async def extraid_login_handler(request: web.Request) -> web.Response:
    db = _get_db(request)
    extraid_db = _get_extraid_db(request)
    settings = get_settings()
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)

    try:
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""

        extra = await extraid_db.get_extra_account_by_email(email)
        if not extra:
            return web.json_response({"error": "invalid_credentials"}, status=401)

        match = await asyncio.to_thread(
            lambda: bcrypt.checkpw(password.encode(), extra["password_hash"].encode())
        )
        if not match:
            return web.json_response({"error": "invalid_credentials"}, status=401)

        user_id = extra["user_id"]

        now = datetime.now(timezone.utc)
        session_exp = now + timedelta(days=settings.jwt_expiry_days)
        session_uuid = str(uuid.uuid4())
        token, _ = _make_jwt_session(user_id, session_uuid, settings)
        token_hash = _hash_jwt(token)

        await extraid_db.execute(
            """
            INSERT INTO auth_sessions (session_id, user_id, auth_method, token_hash, expires_at, device_label)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            session_uuid, user_id, "email_password", token_hash, session_exp, data.get("device_label")
        )

        reg_bonus = False
        if not extra["reg_bonus_claimed"]:
            reg_bonus = True
            await extraid_db.mark_reg_bonus_claimed(extra["id"])
            await db.execute("UPDATE users SET keys = COALESCE(keys, 0) + 3 WHERE user_id = $1", user_id)

        return web.json_response({
            "ok": True,
            "token": token,
            "user_id": user_id,
            "display_id": extra["display_id"],
            "reg_bonus": reg_bonus,
        })
    except Exception:
        logger.exception("ExtraID login handler error")
        return web.json_response({"error": "server_error", "message": "Внутренняя ошибка сервера"}, status=500)


# --- ExtraID: Link ---

async def extraid_link_handler(request: web.Request) -> web.Response:
    db = _get_db(request)
    extraid_db = _get_extraid_db(request)
    settings = get_settings()
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)

    try:
        auth_token = request.rel_url.query.get("_auth") or ""
        tg_init_data = data.get("tg_init_data") or ""

        jw_result = await _verify_jwt_token_async_fn(auth_token, extraid_db, settings)
        if not jw_result:
            return web.json_response({"error": "invalid_jwt_session"}, status=401)
        jwt_user_id, _session_id = jw_result

        verified = _verify_init_data_fn(tg_init_data, request.app["bot_token"])
        if not verified:
            return web.json_response({"error": "invalid_tg_init_data"}, status=400)
        tg_user_id = _extract_user_id_from_init_data_fn(verified)
        if not tg_user_id:
            return web.json_response({"error": "invalid_tg_init_data"}, status=400)

        existing_link = await extraid_db.fetchrow(
            "SELECT id FROM extra_accounts WHERE user_id = $1 AND deleted_at IS NULL", tg_user_id
        )
        if existing_link:
            return web.json_response({"error": "telegram_already_linked"}, status=409)

        extra = await extraid_db.get_extra_account_by_user_id(jwt_user_id)
        if not extra:
            return web.json_response({"error": "extra_account_not_found"}, status=404)

        tg_exists = await db.fetchval("SELECT 1 FROM users WHERE user_id = $1", tg_user_id)
        if not tg_exists:
            return web.json_response({"error": "tg_user_not_found"}, status=404)

        await extraid_db.execute("UPDATE extra_accounts SET user_id = $1 WHERE id = $2", tg_user_id, extra["id"])
        await db.execute("UPDATE users SET extra_account_id = $1, auth_source = 'telegram' WHERE user_id = $2", extra["id"], tg_user_id)

        return web.json_response({"ok": True, "merged": False, "trophies_transferred": 0})
    except Exception:
        logger.exception("extraid_link_handler error")
        return web.json_response({"error": "server_error"}, status=500)


# --- ExtraID: Profile ---

async def extraid_profile_handler(request: web.Request) -> web.Response:
    extraid_db = _get_extraid_db(request)
    user_id = await require_user_id_fn(request)

    extra = await extraid_db.get_extra_account_by_user_id(user_id)
    if not extra:
        return web.json_response({"extra_id_bound": False, "user_id": user_id})

    return web.json_response({
        "extra_id_bound": True,
        "user_id": user_id,
        "display_id": extra["display_id"],
        "nickname": extra.get("nickname"),
        "email": _mask_email_fn(extra["email"]),
        "is_email_verified": extra["is_email_verified"],
        "reg_date": extra["created_at"].isoformat() if extra.get("created_at") else None,
        "linked_telegram": (extra["user_id"] is not None and extra["user_id"] < 9000000000000),
    })


# --- ExtraID: Patch Profile ---

async def extraid_patch_profile_handler(request: web.Request) -> web.Response:
    extraid_db = _get_extraid_db(request)
    user_id = await require_user_id_fn(request)
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)

    try:
        nickname = (data.get("nickname") or "").strip()
        if not _nickname_valid_fn(nickname):
            return web.json_response({"error": "invalid_nickname", "message": "3-20 символов, a-zA-Z0-9_-"}, status=400)

        extra = await extraid_db.get_extra_account_by_user_id(user_id)
        if not extra:
            return web.json_response({"error": "extra_account_not_found"}, status=404)

        existing = await extraid_db.fetchrow(
            "SELECT id FROM extra_accounts WHERE LOWER(nickname) = LOWER($1) AND id != $2 AND deleted_at IS NULL",
            nickname, extra["id"]
        )
        if existing:
            return web.json_response({"error": "nickname_taken"}, status=409)

        await extraid_db.execute("UPDATE extra_accounts SET nickname = $1, updated_at = NOW() WHERE id = $2", nickname, extra["id"])
        return web.json_response({"ok": True, "nickname": nickname})
    except Exception:
        logger.exception("extraid_patch_profile_handler error")
        return web.json_response({"error": "server_error"}, status=500)


# --- ExtraID: Delete ---

async def extraid_delete_account_handler(request: web.Request) -> web.Response:
    db = _get_db(request)
    extraid_db = _get_extraid_db(request)
    user_id = await require_user_id_fn(request)
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)

    try:
        password = data.get("password") or ""
        confirm = data.get("confirm") or ""

        if confirm != "DELETE":
            return web.json_response({"error": "confirm_required", "message": 'Введите confirm: "DELETE"'}, status=400)

        extra = await extraid_db.get_extra_account_by_user_id(user_id)
        if not extra:
            return web.json_response({"error": "extra_account_not_found"}, status=404)

        match = await asyncio.to_thread(
            lambda: bcrypt.checkpw(password.encode(), extra["password_hash"].encode())
        )
        if not match:
            return web.json_response({"error": "invalid_password"}, status=401)

        await extraid_db.revoke_all_user_sessions(user_id)
        await extraid_db.soft_delete_extra_account(extra["id"])
        await db.execute(
            "UPDATE users SET extra_account_id = NULL WHERE user_id = $1",
            user_id
        )
        return web.json_response({"ok": True})
    except Exception:
        logger.exception("ExtraID delete handler error")
        return web.json_response({"error": "server_error", "message": "Внутренняя ошибка сервера"}, status=500)


# --- Auth Sessions: List ---

async def auth_sessions_list_handler(request: web.Request) -> web.Response:
    extraid_db = _get_extraid_db(request)
    settings = get_settings()
    user_id = await require_user_id_fn(request)

    current_session_id = None
    auth_param = request.rel_url.query.get("_auth")
    if auth_param:
        jw = await _verify_jwt_token_async_fn(auth_param, extraid_db, settings)
        if jw:
            current_session_id = jw[1]

    sessions = await extraid_db.get_user_sessions(user_id)
    result = []
    for s in sessions:
        result.append({
            "session_id": str(s["session_id"]),
            "auth_method": s["auth_method"],
            "created_at": s["created_at"].isoformat() if s.get("created_at") else None,
            "last_used_at": s["last_used_at"].isoformat() if s.get("last_used_at") else None,
            "device_label": s.get("device_label"),
            "is_current": str(s["session_id"]) == current_session_id,
        })
    return web.json_response({"sessions": result})


# --- Auth Sessions: Revoke ---

async def auth_session_revoke_handler(request: web.Request) -> web.Response:
    extraid_db = _get_extraid_db(request)
    settings = get_settings()
    user_id = await require_user_id_fn(request)

    target_session_id = request.match_info.get("session_id")
    if not target_session_id:
        return web.json_response({"error": "session_id_required"}, status=400)

    auth_param = request.rel_url.query.get("_auth")
    if auth_param:
        jw = await _verify_jwt_token_async_fn(auth_param, extraid_db, settings)
        if jw and jw[1] == target_session_id:
            return web.json_response({"error": "cannot_revoke_current_session"}, status=400)

    session = await extraid_db.fetchrow(
        "SELECT user_id FROM auth_sessions WHERE session_id = $1 AND revoked = FALSE", target_session_id
    )
    if not session or session["user_id"] != user_id:
        return web.json_response({"error": "session_not_found"}, status=404)

    await extraid_db.revoke_session(target_session_id)
    return web.json_response({"ok": True})


# --- Auth: Logout ---

async def auth_logout_handler(request: web.Request) -> web.Response:
    extraid_db = _get_extraid_db(request)
    settings = get_settings()
    auth_param = request.rel_url.query.get("_auth")
    if not auth_param:
        return web.json_response({"error": "auth_required"}, status=401)

    jw = await _verify_jwt_token_async_fn(auth_param, extraid_db, settings)
    if not jw:
        return web.json_response({"error": "invalid_jwt_session"}, status=401)

    await extraid_db.revoke_session(jw[1])
    return web.json_response({"ok": True})




# --- Device Analytics ---

async def analytics_device_handler(request: web.Request) -> web.Response:
    extraid_db = _get_extraid_db(request)
    settings = get_settings()
    user_id = await require_user_id_fn(request)

    try:
        data = await request.json()
    except Exception:
        data = {}

    platform = data.get("platform") or "unknown"
    device_model = data.get("device_model")
    os_name = data.get("os_name")
    os_version = data.get("os_version")
    browser_name = data.get("browser_name")
    browser_version = data.get("browser_version")
    app_version = data.get("app_version")
    screen_width = data.get("screen_width")
    screen_height = data.get("screen_height")
    device_pixel_ratio = data.get("device_pixel_ratio")
    locale_language = data.get("locale_language")
    locale_region = data.get("locale_region")
    timezone_str = data.get("timezone")
    tg_platform = data.get("tg_platform")
    tg_version = data.get("tg_version")
    raw_user_agent = request.headers.get("User-Agent")

    client_ip = request.remote or "0.0.0.0"
    ip_country = await _get_ip_country_fn(client_ip, settings) if client_ip != "127.0.0.1" else None
    ip_hash = hashlib.sha256(client_ip.encode()).hexdigest()

    session_id = None
    auth_param = request.rel_url.query.get("_auth")
    if auth_param:
        jw = await _verify_jwt_token_async_fn(auth_param, extraid_db, settings)
        if jw:
            session_id = jw[1]

    await extraid_db.log_device_analytics(
        user_id, platform, session_id=session_id,
        device_model=device_model, os_name=os_name, os_version=os_version,
        browser_name=browser_name, browser_version=browser_version, app_version=app_version,
        screen_width=screen_width, screen_height=screen_height, device_pixel_ratio=device_pixel_ratio,
        locale_language=locale_language, locale_region=locale_region, timezone=timezone_str,
        tg_platform=tg_platform, tg_version=tg_version,
        ip_country=ip_country, ip_hash=ip_hash, raw_user_agent=raw_user_agent
    )

    return web.json_response({"ok": True})


def register_handlers(
    app: web.Application,
    extraid_db: ExtraIDDatabase,
    verify_init_data,
    validate_auth_date,
    extract_user_id_from_init_data,
    verify_jwt_token_async,
    display_id_generator,
    mask_email,
    nickname_valid,
    get_ip_country,
    require_user_id,
):
    """Register ExtraID route handlers with resolver functions injected."""
    global _verify_init_data_fn, _validate_auth_date_fn, _extract_user_id_from_init_data_fn
    global _verify_jwt_token_async_fn, _display_id_generator_fn, _mask_email_fn
    global _nickname_valid_fn, _get_ip_country_fn, require_user_id_fn
    global _extraid_db

    _extraid_db = extraid_db
    _verify_init_data_fn = verify_init_data
    _validate_auth_date_fn = validate_auth_date
    _extract_user_id_from_init_data_fn = extract_user_id_from_init_data
    _verify_jwt_token_async_fn = verify_jwt_token_async
    _display_id_generator_fn = display_id_generator
    _mask_email_fn = mask_email
    _nickname_valid_fn = nickname_valid
    _get_ip_country_fn = get_ip_country
    require_user_id_fn = require_user_id

    # ExtraID
    app.router.add_post("/api/extraid/register", extraid_register_handler)
    app.router.add_post("/api/extraid/login", extraid_login_handler)
    app.router.add_post("/api/extraid/link", extraid_link_handler)
    app.router.add_get("/api/extraid/profile", extraid_profile_handler)
    app.router.add_patch("/api/extraid/profile", extraid_patch_profile_handler)
    app.router.add_post("/api/extraid/delete", extraid_delete_account_handler)

    # Auth sessions
    app.router.add_get("/api/auth/sessions", auth_sessions_list_handler)
    app.router.add_delete("/api/auth/sessions/{session_id}", auth_session_revoke_handler)
    app.router.add_post("/api/auth/logout", auth_logout_handler)

    # Device analytics
    app.router.add_post("/api/analytics/device", analytics_device_handler)

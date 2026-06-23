from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import secrets
import string as _stdstring
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt as pyjwt
from aiohttp import web, ClientTimeout

from infrastructure.config import get_settings
from infrastructure.database import Database
from infrastructure.extraid_database import ExtraIDDatabase, SYNTHETIC_USER_ID_MIN
from infrastructure.push_notifications import build_android_push_payload
from infrastructure.telegram_proxy import create_telegram_aiohttp_session

logger = logging.getLogger(__name__)

# Rate limiting store: {key: [timestamp, ...]}
_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_EMAIL_RE = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}$"
)
_NICKNAME_RE = re.compile(r"^[A-Za-z0-9_-]{3,20}$")


def _prune_rate_limit_store(now: float, window_seconds: int) -> None:
    for store_key, timestamps in list(_rate_limit_store.items()):
        kept = [t for t in timestamps if now - t < window_seconds]
        if kept:
            _rate_limit_store[store_key] = kept
        else:
            _rate_limit_store.pop(store_key, None)


def _check_rate_limit(key: str, max_requests: int, window_seconds: int) -> bool:
    now = time.time()
    _prune_rate_limit_store(now, window_seconds)
    timestamps = _rate_limit_store[key]
    _rate_limit_store[key] = [t for t in timestamps if now - t < window_seconds]
    if len(_rate_limit_store[key]) >= max_requests:
        return False
    _rate_limit_store[key].append(now)
    return True


async def _check_rate_limit_for_request(
    request: web.Request,
    key: str,
    max_requests: int,
    window_seconds: int,
) -> bool:
    extraid_db = _get_extraid_db(request)
    shared_limiter = getattr(extraid_db, "check_rate_limit", None)
    if shared_limiter is not None:
        return bool(await shared_limiter(key, max_requests, window_seconds))
    if get_settings().environment != "development":
        raise RuntimeError("ExtraID shared rate limit backend is required outside development.")
    return _check_rate_limit(key, max_requests, window_seconds)


def _bearer_token_from_request(request: web.Request) -> str:
    authorization = str((getattr(request, "headers", {}) or {}).get("Authorization") or "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def _looks_like_jwt_bearer(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", str(value or "").strip()))

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


def _parse_telegram_id(value: Any) -> int | None:
    try:
        telegram_id = int(str(value or "").strip())
    except (TypeError, ValueError):
        return None
    return telegram_id if telegram_id > 0 else None


def _valid_email(email: str) -> bool:
    email = (email or "").strip()
    if len(email) > 254 or email.count("@") != 1:
        return False
    local, domain = email.split("@", 1)
    if not local or not domain or len(local) > 64:
        return False
    if local.startswith(".") or local.endswith(".") or ".." in local or ".." in domain:
        return False
    return bool(_EMAIL_RE.fullmatch(email))


def _valid_nickname(nickname: str) -> bool:
    return bool(_NICKNAME_RE.fullmatch((nickname or "").strip()))


def _make_bot_auth_code() -> str:
    return "".join(secrets.choice(_stdstring.digits) for _ in range(6))


async def _get_any_extra_account_by_email(extraid_db: ExtraIDDatabase, email: str) -> dict | None:
    if hasattr(extraid_db, "get_any_extra_account_by_email"):
        return await extraid_db.get_any_extra_account_by_email(email)
    row = await extraid_db.fetchrow(
        "SELECT * FROM extra_accounts WHERE LOWER(email) = LOWER($1) LIMIT 1",
        email,
    )
    return dict(row) if row else None


async def _require_jwt_user_id_from_request(request: web.Request, data: dict[str, Any] | None = None) -> int | None:
    extraid_db = _get_extraid_db(request)
    settings = get_settings()
    data = data or {}
    bearer = _bearer_token_from_request(request)
    # Do not accept JWT via _auth query param: JWTs in URLs get logged by proxies/CDNs.
    # Only Bearer header or JSON body `auth` field are accepted.
    token = (
        bearer
        or str(data.get("auth") or "").strip()
    )
    if not token:
        return None
    result = await _verify_jwt_token_async_fn(token, extraid_db, settings)
    if not result:
        return None
    return int(result[0])


async def _send_telegram_transfer_code(request: web.Request, user_id: int, code: str) -> bool:
    text = (
        "🔐 Код для переноса ExtraArena в приложение: "
        f"{code}\n\nОн действует 5 минут. Никому его не показывай."
    )

    bot = request.app.get("telegram_bot") or request.app.get("bot")
    if bot is not None:
        try:
            await bot.send_message(user_id, text)
            return True
        except Exception:
            logger.warning("Failed to send Telegram transfer code via bot object", exc_info=True)

    bot_token = request.app.get("bot_token") or ""
    if not bot_token:
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        async with create_telegram_aiohttp_session(timeout=ClientTimeout(total=8)) as session:
            async with session.post(url, json={"chat_id": user_id, "text": text}) as resp:
                body = await resp.json(content_type=None)
                if resp.status >= 400 or not body.get("ok"):
                    logger.warning("Telegram transfer code delivery failed: status=%s body=%s", resp.status, body)
                    return False
                return True
    except Exception:
        logger.warning("Failed to send Telegram transfer code via HTTP API", exc_info=True)
        return False


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
        client = str(data.get("client") or data.get("platform") or request.headers.get("X-ExtraArena-Client", "")).strip().lower()

        if not _valid_email(email):
            return web.json_response({"error": "invalid_email"}, status=400)
        if len(password) < 8:
            return web.json_response({"error": "password_too_short", "message": "Пароль должен быть не менее 8 символов"}, status=400)
        if nickname and not _nickname_valid_fn(nickname):
            return web.json_response({"error": "invalid_nickname", "message": "Никнейм: 3-20 символов, только a-z, A-Z, 0-9, _, -"}, status=400)

        existing_email = await _get_any_extra_account_by_email(extraid_db, email)
        if existing_email:
            return web.json_response({"error": "email_taken"}, status=409)
        if nickname:
            existing_nick = await extraid_db.fetchrow(
                "SELECT id FROM extra_accounts WHERE LOWER(nickname) = LOWER($1) AND deleted_at IS NULL", nickname
            )
            if existing_nick:
                return web.json_response({"error": "nickname_taken"}, status=409)

        bearer_auth = _bearer_token_from_request(request)
        tg_auth = request.rel_url.query.get("_auth")
        user_id = None
        auth_source = "email_registration"
        created_synthetic_user = False
        if bearer_auth:
            jwt_result = await _verify_jwt_token_async_fn(bearer_auth, extraid_db, settings)
            if jwt_result:
                user_id = jwt_result[0]
                auth_source = "extraid_mobile" if client in {"android", "android_app", "mobile", "mobile_app"} else "email_registration"
                existing_extra = await extraid_db.get_any_extra_account_by_user_id(user_id)
                if existing_extra:
                    return web.json_response({
                        "error": "extraid_already_exists",
                        "display_id": existing_extra["display_id"],
                        "message": "ExtraID уже создавался для этого аккаунта.",
                    }, status=409)
            else:
                return web.json_response({"error": "invalid_auth"}, status=401)
        elif tg_auth:
            verified = _verify_init_data_fn(tg_auth, request.app["bot_token"])
            if verified and _validate_auth_date_fn(verified):
                user_id = _extract_user_id_from_init_data_fn(verified)
                if user_id:
                    auth_source = "telegram"
                    existing_extra = await extraid_db.get_any_extra_account_by_user_id(user_id)
                    if existing_extra:
                        return web.json_response({
                            "error": "extraid_already_exists",
                            "display_id": existing_extra["display_id"],
                            "message": "ExtraID уже создавался для этого аккаунта.",
                        }, status=409)
            elif _looks_like_jwt_bearer(tg_auth):
                return web.json_response({"error": "invalid_auth"}, status=401)
            else:
                jwt_result = await _verify_jwt_token_async_fn(tg_auth, extraid_db, settings)
                if jwt_result:
                    user_id = jwt_result[0]
                    auth_source = "extraid_mobile" if client in {"android", "android_app", "mobile", "mobile_app"} else "email_registration"
                    existing_extra = await extraid_db.get_any_extra_account_by_user_id(user_id)
                    if existing_extra:
                        return web.json_response({
                            "error": "extraid_already_exists",
                            "display_id": existing_extra["display_id"],
                            "message": "ExtraID уже создавался для этого аккаунта.",
                        }, status=409)
                else:
                    return web.json_response({"error": "invalid_auth"}, status=401)

        if user_id is None:
            user_id = await extraid_db.get_synthetic_user_id()
            await db.ensure_user(user_id=user_id, username=None, first_name=nickname or email, last_name=None)
            created_synthetic_user = True
            await db.execute(
                "UPDATE users SET auth_source = 'email_registration' WHERE user_id = $1",
                user_id
            )
        else:
            existing_for_user = await extraid_db.get_any_extra_account_by_user_id(user_id)
            if existing_for_user:
                return web.json_response({
                    "error": "extraid_already_exists",
                    "display_id": existing_for_user["display_id"],
                    "message": "ExtraID уже создавался для этого аккаунта.",
                }, status=409)
            await db.ensure_user(user_id=user_id, username=None, first_name=nickname or email, last_name=None)

        display_id = _display_id_generator_fn(lambda did: False)
        while await extraid_db.fetchval("SELECT 1 FROM extra_accounts WHERE display_id = $1", display_id):
            display_id = _display_id_generator_fn(lambda did: False)

        password_hash = (await asyncio.to_thread(
            lambda: bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()
        ))
        try:
            extra = await extraid_db.create_extra_account(user_id, display_id, email, password_hash, nickname)
        except Exception as e:
            constraint = str(getattr(e, "constraint_name", "") or "").lower()
            if e.__class__.__name__ == "UniqueViolationError" and ("email" in constraint or "active_email" in constraint):
                await _compensate_synthetic_user_cleanup(db, user_id, created_synthetic_user)
                return web.json_response({"error": "email_taken"}, status=409)
            if e.__class__.__name__ == "UniqueViolationError" and "nickname" in constraint:
                await _compensate_synthetic_user_cleanup(db, user_id, created_synthetic_user)
                return web.json_response({"error": "nickname_taken"}, status=409)
            if e.__class__.__name__ == "UniqueViolationError" and "display_id" in constraint:
                await _compensate_synthetic_user_cleanup(db, user_id, created_synthetic_user)
                logger.warning("display_id collision during register: %s", display_id)
                return web.json_response({"error": "server_error", "message": "Внутренняя ошибка сервера"}, status=500)
            logger.exception("create_extra_account failed during register; cleaning up synthetic user=%s", user_id)
            await _compensate_synthetic_user_cleanup(db, user_id, created_synthetic_user)
            raise
        try:
            await db.execute(
                "UPDATE users SET extra_account_id = $1, auth_source = $2 WHERE user_id = $3",
                extra["id"], auth_source, user_id
            )
        except Exception:
            logger.exception("Failed to link users.extra_account_id after create_extra_account; rolling back extra_accounts row for user=%s", user_id)
            await extraid_db.execute("DELETE FROM extra_accounts WHERE id = $1", extra["id"])
            await _compensate_synthetic_user_cleanup(db, user_id, created_synthetic_user)
            raise

        reg_bonus = False
        if auth_source == "telegram" and not await extraid_db.has_user_claimed_reg_bonus(user_id):
            try:
                await db.execute("UPDATE users SET keys = COALESCE(keys, 0) + 3 WHERE user_id = $1", user_id)
            except Exception:
                logger.warning("Failed to credit reg bonus keys for user=%s; bonus not marked claimed", user_id, exc_info=True)
                raise
            reg_bonus_claimed = await extraid_db.mark_reg_bonus_claimed(extra["id"])
            if reg_bonus_claimed is not False:
                reg_bonus = True

        return web.json_response({
            "ok": True,
            "display_id": display_id,
            "email_sent": False,
            "linked_telegram": auth_source == "telegram",
            "reg_bonus": reg_bonus,
        })
    except Exception:
        logger.exception("ExtraID register handler error")
        return web.json_response({"error": "server_error", "message": "Внутренняя ошибка сервера"}, status=500)


async def _compensate_synthetic_user_cleanup(db: Database, user_id: int, created_synthetic: bool) -> None:
    """Best-effort cleanup of an orphaned synthetic users row created mid-registration."""
    if not created_synthetic:
        return
    if user_id >= SYNTHETIC_USER_ID_MIN:
        try:
            await db.delete_user(user_id)
        except Exception:
            logger.warning("Compensating delete_user failed for synthetic user=%s", user_id, exc_info=True)


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

        if not email or not password:
            return web.json_response({"error": "invalid_credentials"}, status=401)

        client_key_email = hashlib.sha256(email.encode()).hexdigest()[:16] if email else "empty"
        client_key = f"extraid_login:{getattr(request, 'remote', None) or 'unknown'}:{client_key_email}"
        if not await _check_rate_limit_for_request(request, client_key, 10, 60):
            return web.json_response({"error": "rate_limited", "message": "Слишком много попыток входа. Попробуйте позже."}, status=429)

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
        session_uuid = uuid.uuid4()
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
        if not extra["reg_bonus_claimed"] and not await extraid_db.has_user_claimed_reg_bonus(user_id):
            try:
                await db.execute("UPDATE users SET keys = COALESCE(keys, 0) + 3 WHERE user_id = $1", user_id)
            except Exception:
                logger.warning("Failed to credit reg bonus keys for user=%s; bonus not marked claimed", user_id, exc_info=True)
                raise
            reg_bonus_claimed = await extraid_db.mark_reg_bonus_claimed(extra["id"])
            if reg_bonus_claimed is not False:
                reg_bonus = True

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


# --- App Anonymous Auth ---

async def anonymous_auth_handler(request: web.Request) -> web.Response:
    db = _get_db(request)
    extraid_db = _get_extraid_db(request)
    settings = get_settings()

    client_key = f"anonymous:{getattr(request, 'remote', None) or 'unknown'}"
    if not await _check_rate_limit_for_request(request, client_key, 8, 60):
        return web.json_response({"error": "rate_limited"}, status=429)

    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)

    try:
        nickname = (data.get("nickname") or "").strip()
        if not _valid_nickname(nickname):
            return web.json_response({
                "error": "invalid_nickname",
                "message": "Никнейм: 3-20 символов, только a-z, A-Z, 0-9, _, -",
            }, status=400)

        user_id = await extraid_db.get_synthetic_user_id()
        await db.ensure_user(user_id=user_id, username=None, first_name=nickname, last_name=None)

        await db.execute(
            "UPDATE users SET auth_source = 'android_anonymous' WHERE user_id = $1",
            user_id,
        )

        session_uuid = uuid.uuid4()
        token, session_exp = _make_jwt_session(user_id, session_uuid, settings)
        await extraid_db.execute(
            """
            INSERT INTO auth_sessions (session_id, user_id, auth_method, token_hash, expires_at, device_label)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            session_uuid,
            user_id,
            "android_anonymous",
            _hash_jwt(token),
            session_exp,
            data.get("device_label"),
        )

        return web.json_response({
            "ok": True,
            "token": token,
            "user_id": user_id,
            "anonymous": True,
            "nickname": nickname,
        })
    except Exception:
        logger.exception("Anonymous auth handler error")
        return web.json_response({"error": "server_error", "message": "Внутренняя ошибка сервера"}, status=500)


# --- Telegram -> ExtraID transfer ---

async def telegram_transfer_request_code_handler(request: web.Request) -> web.Response:
    db = _get_db(request)
    extraid_db = _get_extraid_db(request)

    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)

    telegram_id = _parse_telegram_id(data.get("telegram_id"))
    if telegram_id is None:
        return web.json_response({"error": "invalid_telegram_id"}, status=400)

    client_key = f"telegram_transfer_code:{getattr(request, 'remote', None) or 'unknown'}:{telegram_id}"
    if not await _check_rate_limit_for_request(request, client_key, 4, 60):
        return web.json_response({"error": "rate_limited"}, status=429)

    try:
        user_exists = await db.fetchval("SELECT 1 FROM users WHERE user_id = $1", telegram_id)
        if not user_exists:
            return web.json_response({"error": "telegram_user_not_found"}, status=404)

        existing_extra = await extraid_db.get_any_extra_account_by_user_id(telegram_id)
        if existing_extra:
            return web.json_response({
                "error": "extraid_already_exists",
                "display_id": existing_extra["display_id"],
                "message": "ExtraID уже создавался для этого аккаунта.",
            }, status=409)

        await extraid_db.cleanup_old_bot_codes(telegram_id)
        for _ in range(12):
            code = _make_bot_auth_code()
            code_exists = await extraid_db.fetchval("SELECT 1 FROM bot_auth_codes WHERE code = $1", code)
            if not code_exists:
                break
        else:
            logger.error("Could not allocate Telegram transfer code")
            return web.json_response({"error": "server_error"}, status=500)

        await extraid_db.create_bot_auth_code(code, telegram_id)
        if not await _send_telegram_transfer_code(request, telegram_id, code):
            await extraid_db.cleanup_old_bot_codes(telegram_id)
            return web.json_response({"error": "telegram_delivery_failed"}, status=502)

        return web.json_response({"ok": True, "ttl_seconds": 300})
    except Exception:
        logger.exception("telegram_transfer_request_code_handler error")
        return web.json_response({"error": "server_error", "message": "Внутренняя ошибка сервера"}, status=500)


async def telegram_transfer_complete_handler(request: web.Request) -> web.Response:
    db = _get_db(request)
    extraid_db = _get_extraid_db(request)
    settings = get_settings()

    client_key = f"telegram_transfer_complete:{getattr(request, 'remote', None) or 'unknown'}"
    if not await _check_rate_limit_for_request(request, client_key, 8, 60):
        return web.json_response({"error": "rate_limited"}, status=429)

    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)

    try:
        telegram_id = _parse_telegram_id(data.get("telegram_id"))
        code = str(data.get("code") or "").strip()
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""

        if telegram_id is None:
            return web.json_response({"error": "invalid_telegram_id"}, status=400)
        if len(code) != 6 or not code.isdigit():
            return web.json_response({"error": "invalid_code"}, status=400)
        if not _valid_email(email):
            return web.json_response({"error": "invalid_email"}, status=400)
        if len(password) < 8:
            return web.json_response({"error": "password_too_short"}, status=400)

        user_exists = await db.fetchval("SELECT 1 FROM users WHERE user_id = $1", telegram_id)
        if not user_exists:
            return web.json_response({"error": "telegram_user_not_found"}, status=404)

        auth_code = await extraid_db.verify_bot_auth_code(code)
        if not auth_code:
            return web.json_response({"error": "invalid_code"}, status=400)
        if int(auth_code["user_id"]) != telegram_id:
            return web.json_response({"error": "code_user_mismatch"}, status=400)

        existing_for_user = await extraid_db.get_any_extra_account_by_user_id(telegram_id)
        if existing_for_user:
            return web.json_response({
                "error": "extraid_already_exists",
                "display_id": existing_for_user["display_id"],
                "message": "ExtraID уже создавался для этого аккаунта.",
            }, status=409)

        existing_email = await _get_any_extra_account_by_email(extraid_db, email)
        if existing_email:
            return web.json_response({"error": "email_taken"}, status=409)

        display_id = _display_id_generator_fn(lambda did: False)
        while await extraid_db.fetchval("SELECT 1 FROM extra_accounts WHERE display_id = $1", display_id):
            display_id = _display_id_generator_fn(lambda did: False)

        password_hash = (await asyncio.to_thread(
            lambda: bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()
        ))
        try:
            extra = await extraid_db.create_extra_account(telegram_id, display_id, email, password_hash, None)
        except Exception as e:
            constraint = str(getattr(e, "constraint_name", "") or "").lower()
            if e.__class__.__name__ == "UniqueViolationError" and ("email" in constraint or "active_email" in constraint):
                return web.json_response({"error": "email_taken"}, status=409)
            logger.exception("create_extra_account failed during telegram transfer for user=%s", telegram_id)
            raise
        try:
            await db.execute(
                "UPDATE users SET extra_account_id = $1, auth_source = 'telegram_transfer' WHERE user_id = $2",
                extra["id"], telegram_id
            )
        except Exception:
            logger.exception("Failed to link users.extra_account_id after transfer create for user=%s; rolling back", telegram_id)
            await extraid_db.execute("DELETE FROM extra_accounts WHERE id = $1", extra["id"])
            raise

        session_uuid = uuid.uuid4()
        token, session_exp = _make_jwt_session(telegram_id, session_uuid, settings)
        await extraid_db.execute(
            """
            INSERT INTO auth_sessions (session_id, user_id, auth_method, token_hash, expires_at, device_label)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            session_uuid,
            telegram_id,
            "telegram_transfer",
            _hash_jwt(token),
            session_exp,
            data.get("device_label"),
        )
        consumed = await extraid_db.consume_bot_auth_code(code)
        if not consumed or int(consumed["user_id"]) != telegram_id:
            logger.warning("transfer complete: bot code consume failed after account creation for telegram_id=%s", telegram_id)
            await extraid_db.soft_delete_extra_account(extra["id"])
            await db.execute("UPDATE users SET extra_account_id = NULL, auth_source = 'telegram' WHERE user_id = $1", telegram_id)
            return web.json_response({"error": "invalid_code"}, status=400)
        await extraid_db.mark_bot_code_used(code, session_uuid)

        reg_bonus = False
        if not await extraid_db.has_user_claimed_reg_bonus(telegram_id):
            try:
                await db.execute("UPDATE users SET keys = COALESCE(keys, 0) + 3 WHERE user_id = $1", telegram_id)
            except Exception:
                logger.warning("Failed to credit reg bonus keys for user=%s; bonus not marked claimed", telegram_id, exc_info=True)
                raise
            reg_bonus_claimed = await extraid_db.mark_reg_bonus_claimed(extra["id"])
            if reg_bonus_claimed is not False:
                reg_bonus = True

        return web.json_response({
            "ok": True,
            "token": token,
            "user_id": telegram_id,
            "display_id": display_id,
            "reg_bonus": reg_bonus,
        })
    except Exception:
        logger.exception("telegram_transfer_complete_handler error")
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
        auth_token = _bearer_token_from_request(request)
        tg_init_data = data.get("tg_init_data") or ""

        if not auth_token:
            return web.json_response({"error": "auth_required"}, status=401)
        jw_result = await _verify_jwt_token_async_fn(auth_token, extraid_db, settings)
        if not jw_result:
            return web.json_response({"error": "invalid_jwt_session"}, status=401)
        jwt_user_id, _session_id = jw_result

        verified = _verify_init_data_fn(tg_init_data, request.app["bot_token"])
        if not verified:
            return web.json_response({"error": "invalid_tg_init_data"}, status=400)
        if not _validate_auth_date_fn(verified):
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
        old_user_id = int(extra.get("user_id") or jwt_user_id)

        # Prevent re-binding an ExtraID that is already linked to a real Telegram account.
        # Only synthetic (email-only/anonymous) accounts may be linked to Telegram.
        if old_user_id is not None and old_user_id < SYNTHETIC_USER_ID_MIN and old_user_id != tg_user_id:
            return web.json_response({
                "error": "already_linked_to_telegram",
                "message": "Этот ExtraID уже привязан к Telegram-аккаунту.",
            }, status=409)

        tg_exists = await db.fetchval("SELECT 1 FROM users WHERE user_id = $1", tg_user_id)
        if not tg_exists:
            return web.json_response({"error": "tg_user_not_found"}, status=404)

        # Block if this Telegram ID already had an ExtraID created (even soft-deleted) — prevents duplication.
        prior_binding = await extraid_db.get_any_extra_account_by_user_id(tg_user_id)
        if prior_binding:
            return web.json_response({
                "error": "telegram_already_linked",
                "message": "К этому Telegram-аккаунту уже привязан ExtraID.",
            }, status=409)

        try:
            await db.execute("UPDATE users SET extra_account_id = $1, auth_source = 'telegram' WHERE user_id = $2", extra["id"], tg_user_id)
            await extraid_db.execute("UPDATE extra_accounts SET user_id = $1 WHERE id = $2", tg_user_id, extra["id"])
        except Exception:
            logger.exception("link: failed to re-point ExtraID ownership for tg_user_id=%s; rolling back users update", tg_user_id)
            await db.execute("UPDATE users SET extra_account_id = NULL WHERE user_id = $1", tg_user_id)
            raise
        if old_user_id != tg_user_id:
            try:
                await db.execute("UPDATE users SET extra_account_id = NULL WHERE user_id = $1", old_user_id)
                await extraid_db.revoke_all_user_sessions(old_user_id)
            except Exception:
                logger.warning("link: failed to clear old owner user_id=%s (best-effort)", old_user_id, exc_info=True)

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
        "linked_telegram": (extra["user_id"] is not None and extra["user_id"] < SYNTHETIC_USER_ID_MIN),
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

        # Telegram-bound accounts cannot be deleted: prevents the duplication exploit
        # (pump Telegram-only account -> create ExtraID -> delete -> create new ExtraID).
        is_telegram_bound = extra["user_id"] is not None and extra["user_id"] < SYNTHETIC_USER_ID_MIN
        if is_telegram_bound:
            return web.json_response({
                "error": "cannot_delete_telegram_bound",
                "message": "Невозможно удалить ExtraID, привязанный к Telegram. Обратитесь в поддержку.",
            }, status=403)

        if not password:
            return web.json_response({"error": "invalid_password"}, status=401)

        match = await asyncio.to_thread(
            lambda: bcrypt.checkpw(password.encode(), extra["password_hash"].encode())
        )
        if not match:
            return web.json_response({"error": "invalid_password"}, status=401)

        await extraid_db.revoke_all_user_sessions(user_id)
        await extraid_db.soft_delete_extra_account(extra["id"])
        # Reset auth_source and clear the link on the users row.
        # For synthetic (email-only/anonymous) accounts, fully remove the orphaned users row.
        if user_id >= SYNTHETIC_USER_ID_MIN:
            try:
                await db.delete_user(user_id)
            except Exception:
                logger.warning("delete_user failed for synthetic user=%s during ExtraID delete", user_id, exc_info=True)
                await db.execute(
                    "UPDATE users SET extra_account_id = NULL, auth_source = 'telegram' WHERE user_id = $1",
                    user_id,
                )
        else:
            await db.execute(
                "UPDATE users SET extra_account_id = NULL, auth_source = 'telegram' WHERE user_id = $1",
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
    auth_param = _bearer_token_from_request(request)
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
    try:
        target_session_uuid = uuid.UUID(str(target_session_id))
    except (TypeError, ValueError):
        return web.json_response({"error": "invalid_session_id"}, status=400)

    auth_param = _bearer_token_from_request(request)
    if auth_param:
        jw = await _verify_jwt_token_async_fn(auth_param, extraid_db, settings)
        if jw and jw[1] == target_session_id:
            return web.json_response({"error": "cannot_revoke_current_session"}, status=400)

    session = await extraid_db.fetchrow(
        "SELECT user_id FROM auth_sessions WHERE session_id = $1 AND revoked = FALSE", target_session_uuid
    )
    if not session or session["user_id"] != user_id:
        return web.json_response({"error": "session_not_found"}, status=404)

    await extraid_db.revoke_session(target_session_uuid)
    return web.json_response({"ok": True})


# --- Auth: Logout ---

async def auth_logout_handler(request: web.Request) -> web.Response:
    extraid_db = _get_extraid_db(request)
    settings = get_settings()
    auth_param = _bearer_token_from_request(request)
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
    auth_param = _bearer_token_from_request(request)
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


# --- Android push devices ---

async def push_register_handler(request: web.Request) -> web.Response:
    db = _get_db(request)
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)

    user_id = await _require_jwt_user_id_from_request(request, data)
    if not user_id:
        return web.json_response({"error": "auth_required"}, status=401)

    token = str(data.get("token") or "").strip()
    if not token:
        return web.json_response({"error": "token_required"}, status=400)

    device_timezone = str(data.get("timezone") or "").strip()[:96] or None
    utc_offset_minutes = None
    try:
        utc_offset_minutes = int(data.get("utc_offset_minutes"))
    except (TypeError, ValueError):
        utc_offset_minutes = None
    if utc_offset_minutes is not None and not (-14 * 60 <= utc_offset_minutes <= 14 * 60):
        utc_offset_minutes = None

    try:
        device = await db.register_push_device(
            user_id,
            token=token,
            platform=str(data.get("platform") or "android").strip().lower() or "android",
            app_version=(data.get("app_version") or None),
            device_label=(data.get("device_label") or None),
            os_name=(data.get("os_name") or None),
            os_version=(data.get("os_version") or None),
            timezone=device_timezone,
            utc_offset_minutes=utc_offset_minutes,
        )
        return web.json_response({"ok": True, "device_id": device.get("id")})
    except Exception:
        logger.exception("push_register_handler error")
        return web.json_response({"error": "server_error"}, status=500)


async def push_unregister_handler(request: web.Request) -> web.Response:
    db = _get_db(request)
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)

    user_id = await _require_jwt_user_id_from_request(request, data)
    if not user_id:
        return web.json_response({"error": "auth_required"}, status=401)

    token = str(data.get("token") or "").strip()
    if not token:
        return web.json_response({"error": "token_required"}, status=400)

    try:
        removed = await db.unregister_push_device(user_id, token=token)
        return web.json_response({"ok": True, "removed": removed})
    except Exception:
        logger.exception("push_unregister_handler error")
        return web.json_response({"error": "server_error"}, status=500)


async def push_test_handler(request: web.Request) -> web.Response:
    db = _get_db(request)
    try:
        data = await request.json()
    except Exception:
        data = {}

    user_id = await _require_jwt_user_id_from_request(request, data)
    if not user_id:
        return web.json_response({"error": "auth_required"}, status=401)

    sender = request.app.get("push_sender")
    if sender is None:
        return web.json_response({"error": "push_sender_unavailable"}, status=503)

    devices = await db.get_push_devices(user_id, platform="android")
    payload = build_android_push_payload(
        "reminders",
        "daily_reminder",
        {"text": data.get("body") or "Тестовый пуш ExtraArena готов.", "section": "arena"},
    )
    sent = 0
    failed = 0
    for device in devices:
        result = await sender.send(
            token=device["token"],
            title=payload.title,
            body=payload.body,
            data=payload.data,
        )
        if result.ok:
            sent += 1
        else:
            failed += 1
            await db.mark_push_device_error(
                device["token"],
                result.error or "push_test_failed",
                permanent=result.permanent,
            )
    return web.json_response({"ok": sent > 0, "sent": sent, "failed": failed, "devices": len(devices)})


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
    app.router.add_post("/api/auth/anonymous", anonymous_auth_handler)
    app.router.add_post("/api/telegram-transfer/request-code", telegram_transfer_request_code_handler)
    app.router.add_post("/api/telegram-transfer/complete", telegram_transfer_complete_handler)
    app.router.add_get("/api/auth/sessions", auth_sessions_list_handler)
    app.router.add_delete("/api/auth/sessions/{session_id}", auth_session_revoke_handler)
    app.router.add_post("/api/auth/logout", auth_logout_handler)

    # Device analytics
    app.router.add_post("/api/analytics/device", analytics_device_handler)

    # Android push devices
    app.router.add_post("/api/push/register", push_register_handler)
    app.router.add_post("/api/push/unregister", push_unregister_handler)
    app.router.add_post("/api/push/test", push_test_handler)

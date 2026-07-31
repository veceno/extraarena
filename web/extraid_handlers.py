from __future__ import annotations

import asyncio
import hashlib
import hmac
import html as _html
import ipaddress
import logging
import os
import re
import secrets
import string as _stdstring
import time
import uuid
from collections import defaultdict
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote, urlsplit

import bcrypt
import jwt as pyjwt
from aiohttp import web, ClientSession, ClientTimeout

from infrastructure.config import get_settings
from infrastructure.database import Database
from infrastructure.extraid_database import (
    ExtraIDDatabase,
    SYNTHETIC_USER_ID_MIN,
    _account_email_code_hash,
)
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
_PASSWORD_MIN_CHARS = 8
_BCRYPT_PASSWORD_MAX_BYTES = 72
_EMAIL_VERIFY_TTL_SECONDS = 15 * 60
_PASSWORD_RESET_TTL_SECONDS = 30 * 60
_EMAIL_VERIFY_CODE_RE = re.compile(r"^\d{6}$")
_GENERIC_EMAIL_STATUS = "if_account_exists_email_sent"
_TELEGRAM_TRANSFER_PURPOSE = "telegram_transfer"
_EXTRAID_RECONCILE_BATCH_SIZE = 100
_EXTRAID_RECONCILE_INTERVAL_SECONDS = 5 * 60
_EXTRAID_EMAIL_OUTBOX_BATCH_SIZE = 20
_EXTRAID_EMAIL_OUTBOX_INTERVAL_SECONDS = 5
_DUMMY_PASSWORD_HASH = (
    b"$2b$12$d0KfrAUirzHWG0KoQFKareq4ncBrSfhxsQ0iYydLEGDIzTeXhthFC"
)
_NO_STORE_PATH_PREFIXES = (
    "/api/extraid/",
    "/api/telegram-transfer/",
)
_NO_STORE_EXACT_PATHS = {
    "/api/auth/anonymous",
    "/api/auth/logout",
    "/api/auth/sessions",
}


@web.middleware
async def extraid_no_store_middleware(
    request: web.Request,
    handler,
) -> web.StreamResponse:
    response = await handler(request)
    path = request.path
    if (
        path in _NO_STORE_EXACT_PATHS
        or path.startswith("/api/auth/sessions/")
        or any(path.startswith(prefix) for prefix in _NO_STORE_PATH_PREFIXES)
    ):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
    return response


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


def _parse_ip(value: Any) -> str | None:
    try:
        return str(ipaddress.ip_address(str(value or "").strip()))
    except ValueError:
        return None


def _trusted_client_ip(request: web.Request) -> str:
    """Resolve the client across only the explicitly trusted proxy suffix."""
    peer = _parse_ip(getattr(request, "remote", None))
    peer_address = ipaddress.ip_address(peer) if peer else None
    trusted_networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    configured_cidrs = os.getenv("TRUSTED_PROXY_CIDRS", "").split(",")
    for raw_cidr in configured_cidrs:
        raw_cidr = raw_cidr.strip()
        if not raw_cidr:
            continue
        try:
            trusted_networks.append(ipaddress.ip_network(raw_cidr, strict=False))
        except ValueError:
            logger.warning("Ignoring invalid TRUSTED_PROXY_CIDRS entry: %s", raw_cidr)

    def is_trusted_proxy(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        return address.is_loopback or any(address in network for network in trusted_networks)

    if peer_address and is_trusted_proxy(peer_address):
        headers = getattr(request, "headers", {}) or {}
        cf_ip = _parse_ip(headers.get("CF-Connecting-IP"))
        if cf_ip:
            return cf_ip
        forwarded_chain = str(headers.get("X-Forwarded-For") or "").split(",")
        for raw_address in reversed(forwarded_chain):
            forwarded_ip = _parse_ip(raw_address)
            if not forwarded_ip:
                continue
            forwarded_address = ipaddress.ip_address(forwarded_ip)
            if not is_trusted_proxy(forwarded_address):
                return forwarded_ip
    return peer or "unknown"


def _client_rate_key(request: web.Request) -> str:
    return _rate_limit_subject("ip", _trusted_client_ip(request))


def _rate_limit_subject(namespace: str, value: Any) -> str:
    secret = get_settings().jwt_secret.encode("utf-8")
    message = f"extraid-rate:{namespace}:{str(value).strip().lower()}".encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()[:32]


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


def _password_error(password: Any) -> str | None:
    if not isinstance(password, str) or len(password) < _PASSWORD_MIN_CHARS:
        return "password_too_short"
    if len(password.encode("utf-8")) > _BCRYPT_PASSWORD_MAX_BYTES:
        return "password_too_long"
    return None


def _make_bot_auth_code() -> str:
    return "".join(secrets.choice(_stdstring.digits) for _ in range(6))


def _make_account_action_token() -> tuple[uuid.UUID, str, str]:
    token_id = uuid.uuid4()
    secret = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    return token_id, f"{token_id}.{secret}", token_hash


def _make_email_verification_code(email: str) -> tuple[uuid.UUID, str, str]:
    token_id = uuid.uuid4()
    code = f"{secrets.randbelow(1_000_000):06d}"
    return token_id, code, _account_email_code_hash(code, email, token_id)


def _parse_account_action_token(value: Any) -> tuple[uuid.UUID, str] | None:
    raw = str(value or "").strip()
    if not raw or len(raw) > 256:
        return None
    token_id_raw, separator, secret = raw.partition(".")
    if not separator or not 32 <= len(secret) <= 128:
        return None
    try:
        token_id = uuid.UUID(token_id_raw)
    except (TypeError, ValueError):
        return None
    return token_id, hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _render_extraid_email(
    *,
    title: str,
    intro: str,
    ttl_seconds: int,
    verification_code: str | None = None,
    action_label: str | None = None,
    action_url: str | None = None,
    cancel_url: str | None = None,
) -> tuple[str, str]:
    ttl_minutes = max(1, int(ttl_seconds) // 60)
    expiry_text = (
        f"Код действует {ttl_minutes} минут и используется один раз."
        if verification_code
        else f"Ссылка действует {ttl_minutes} минут и используется один раз."
    )
    text_parts = [
        "ExtraArena · ExtraID",
        "",
        title,
        intro,
    ]
    if verification_code:
        text_parts.extend(["", f"Код подтверждения: {verification_code}"])
    if action_url and action_label:
        text_parts.extend(["", f"{action_label}: {action_url}"])
    text_parts.extend(
        [
            "",
            expiry_text,
            "Никому не сообщайте код, пароль или данные игровой сессии.",
        ]
    )
    if cancel_url:
        text_parts.extend(
            [
                "",
                "Если вы не создавали ExtraID, отмените регистрацию:",
                cancel_url,
            ]
        )
    text = "\n".join(text_parts)

    safe_title = _html.escape(title)
    safe_intro = _html.escape(intro)
    safe_expiry = _html.escape(expiry_text)
    safe_code = _html.escape(str(verification_code or ""))
    action_html = ""
    if action_url and action_label:
        safe_action_url = _html.escape(action_url, quote=True)
        safe_action_label = _html.escape(action_label)
        action_html = f"""
          <tr>
            <td style="padding:8px 36px 28px;text-align:center;">
              <a href="{safe_action_url}" style="display:inline-block;padding:14px 28px;border-radius:12px;background:#f5921e;color:#ffffff;text-decoration:none;font-size:15px;font-weight:800;box-shadow:0 8px 24px rgba(245,146,30,.24);">{safe_action_label}</a>
            </td>
          </tr>"""
    code_html = ""
    if verification_code:
        code_html = f"""
          <tr>
            <td style="padding:8px 36px 24px;">
              <div style="border:1px solid #5b3fa0;border-radius:16px;background:#130d26;padding:20px 16px;text-align:center;font-family:Arial,sans-serif;">
                <div style="margin-bottom:8px;color:#9c7de0;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;">Код подтверждения</div>
                <div style="color:#ffb347;font-family:Menlo,Consolas,'Courier New',monospace;font-size:34px;line-height:1.2;font-weight:800;letter-spacing:9px;">{safe_code}</div>
              </div>
            </td>
          </tr>"""
    cancel_html = ""
    if cancel_url:
        safe_cancel_url = _html.escape(cancel_url, quote=True)
        cancel_html = f"""
          <tr>
            <td style="padding:0 36px 28px;">
              <div style="border-radius:12px;background:#26162f;padding:14px 16px;color:#c4b8e8;font-family:Arial,sans-serif;font-size:12px;line-height:1.55;">
                <strong style="color:#f0ecff;">Не создавали ExtraID?</strong><br>
                Не вводите код. <a href="{safe_cancel_url}" style="color:#ffb347;text-decoration:underline;">Отмените регистрацию</a>, чтобы освободить ваш email.
              </div>
            </td>
          </tr>"""
    preheader = _html.escape(
        f"{title}. {expiry_text}",
        quote=False,
    )
    year = datetime.now(timezone.utc).year
    html = f"""<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>{safe_title}</title>
  </head>
  <body style="margin:0;padding:0;background:#0f0a1a;color:#f0ecff;font-family:Arial,sans-serif;">
    <div style="display:none;max-height:0;overflow:hidden;opacity:0;color:transparent;">{preheader}</div>
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="width:100%;background:#0f0a1a;">
      <tr>
        <td align="center" style="padding:28px 12px;">
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="width:100%;max-width:560px;border-collapse:separate;background:#1a1030;border:1px solid #3d2a70;border-radius:22px;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,.35);">
            <tr>
              <td style="height:5px;background:#f5921e;font-size:0;line-height:0;">&nbsp;</td>
            </tr>
            <tr>
              <td style="padding:30px 36px 12px;text-align:center;">
                <div style="display:inline-block;width:48px;height:48px;line-height:48px;border-radius:15px;background:#f5921e;color:#ffffff;font-family:Arial,sans-serif;font-size:18px;font-weight:900;">EA</div>
                <div style="margin-top:13px;color:#9c7de0;font-family:Arial,sans-serif;font-size:11px;font-weight:800;letter-spacing:1.8px;text-transform:uppercase;">ExtraArena · ExtraID Security</div>
              </td>
            </tr>
            <tr>
              <td style="padding:8px 36px 12px;text-align:center;font-family:Arial,sans-serif;">
                <h1 style="margin:0 0 12px;color:#f0ecff;font-size:26px;line-height:1.2;font-weight:900;">{safe_title}</h1>
                <p style="margin:0;color:#c4b8e8;font-size:15px;line-height:1.6;">{safe_intro}</p>
              </td>
            </tr>
            {code_html}
            {action_html}
            <tr>
              <td style="padding:0 36px 24px;text-align:center;font-family:Arial,sans-serif;">
                <p style="margin:0 0 8px;color:#9c7de0;font-size:12px;line-height:1.5;">{safe_expiry}</p>
                <p style="margin:0;color:#7a6fa0;font-size:11px;line-height:1.5;">Никому не сообщайте код, пароль или данные игровой сессии. Команда ExtraArena никогда не попросит их в сообщениях.</p>
              </td>
            </tr>
            {cancel_html}
            <tr>
              <td style="padding:18px 36px;border-top:1px solid #2d1f52;text-align:center;color:#7a6fa0;font-family:Arial,sans-serif;font-size:10px;line-height:1.5;">
                © {year} ExtraArena · Это автоматическое письмо, отвечать на него не нужно.
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>"""
    return text, html


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


def _email_delivery_available(request: web.Request) -> bool:
    if callable(request.app.get("extraid_email_sender")):
        return True
    return _email_delivery_configured()


def _email_delivery_configured() -> bool:
    return bool(
        os.getenv("BREVO_API_KEY", "").strip()
        and _valid_email(os.getenv("EMAIL_FROM_ADDRESS", "").strip())
        and _public_base_url()
    )


def _public_base_url() -> str:
    configured = (
        os.getenv("PUBLIC_BASE_URL", "").strip()
        or os.getenv("WEBAPP_URL", "").strip()
    )
    settings = get_settings()
    if not configured and settings.environment == "development":
        configured = str(settings.webapp_url or "").strip()
    if not configured:
        return ""
    try:
        parsed = urlsplit(configured)
    except ValueError:
        return ""
    is_local_dev_http = (
        settings.environment == "development"
        and parsed.scheme == "http"
        and (parsed.hostname or "").lower() in {"localhost", "127.0.0.1", "::1"}
    )
    if (
        (parsed.scheme != "https" and not is_local_dev_http)
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


async def _send_extraid_email(
    request: web.Request,
    *,
    to_email: str,
    subject: str,
    text: str,
    html: str,
) -> bool:
    injected_sender = request.app.get("extraid_email_sender")
    if callable(injected_sender):
        try:
            result = injected_sender(
                to_email=to_email,
                subject=subject,
                text=text,
                html=html,
            )
            if asyncio.iscoroutine(result):
                result = await result
            return bool(result)
        except Exception:
            logger.warning("Injected ExtraID email sender failed", exc_info=True)
            return False

    api_key = os.getenv("BREVO_API_KEY", "").strip()
    from_address = os.getenv("EMAIL_FROM_ADDRESS", "").strip()
    from_name = os.getenv("EMAIL_FROM_NAME", "ExtraArena").strip() or "ExtraArena"
    if not api_key or not from_address:
        return False
    payload = {
        "sender": {"name": from_name, "email": from_address},
        "to": [{"email": to_email}],
        "subject": subject,
        "textContent": text,
        "htmlContent": html,
    }
    try:
        async with ClientSession(timeout=ClientTimeout(total=10)) as session:
            async with session.post(
                "https://api.brevo.com/v3/smtp/email",
                headers={
                    "accept": "application/json",
                    "api-key": api_key,
                    "content-type": "application/json",
                },
                json=payload,
            ) as response:
                if 200 <= response.status < 300:
                    return True
                body = await response.text()
                logger.warning(
                    "Brevo ExtraID email failed status=%s body=%s",
                    response.status,
                    body[:500],
                )
    except Exception:
        logger.warning("Brevo ExtraID email request failed", exc_info=True)
    return False


async def _issue_account_email(
    request: web.Request,
    extra: dict,
    *,
    purpose: str,
) -> bool:
    base_url = _public_base_url()
    if not base_url and not callable(request.app.get("extraid_email_sender")):
        return False
    verification_code = None
    token = None
    if purpose == "verify_email":
        token_id, verification_code, token_hash = _make_email_verification_code(
            str(extra["email"])
        )
        ttl_seconds = _EMAIL_VERIFY_TTL_SECONDS
        subject = "Код подтверждения ExtraID"
        title = "Подтвердите email"
        intro = "Введите шестизначный код в окне ExtraID, чтобы защитить аккаунт и включить вход по паролю."
        action_label = None
    elif purpose == "password_reset":
        token_id, token, token_hash = _make_account_action_token()
        ttl_seconds = _PASSWORD_RESET_TTL_SECONDS
        fragment_key = "extraid_reset_token"
        subject = "Сброс пароля ExtraID"
        title = "Сбросьте пароль"
        intro = "Мы получили запрос на смену пароля ExtraID. Нажмите кнопку ниже, чтобы продолжить."
        action_label = "Сбросить пароль"
    else:
        raise ValueError("invalid_account_email_purpose")
    issued = await _get_extraid_db(request).create_account_action_token(
        token_id=token_id,
        extra_account_id=extra["id"],
        purpose=purpose,
        token_hash=token_hash,
        email_snapshot=str(extra["email"]),
        ttl_seconds=ttl_seconds,
        cooldown_seconds=60,
    )
    if not issued:
        return False
    action_url = None
    if token is not None:
        # Password-reset tokens stay in the fragment and are never sent to the
        # reverse proxy. Email verification uses a short code instead.
        action_url = (
            f"{base_url}/#{fragment_key}={quote(token, safe='')}"
            if base_url
            else token
        )
    cancel_token_id = None
    cancel_url = None
    if purpose == "verify_email":
        cancel_token_id, cancel_token, cancel_token_hash = _make_account_action_token()
        cancel_issued = await _get_extraid_db(request).create_account_action_token(
            token_id=cancel_token_id,
            extra_account_id=extra["id"],
            purpose="cancel_registration",
            token_hash=cancel_token_hash,
            email_snapshot=str(extra["email"]),
            ttl_seconds=ttl_seconds,
            cooldown_seconds=0,
        )
        if not cancel_issued:
            await _get_extraid_db(request).revoke_account_action_token(token_id)
            return False
        cancel_url = (
            f"{base_url}/#extraid_cancel_token={quote(cancel_token, safe='')}"
            if base_url
            else cancel_token
        )
    text, html = _render_extraid_email(
        title=title,
        intro=intro,
        ttl_seconds=ttl_seconds,
        verification_code=verification_code,
        action_label=action_label,
        action_url=action_url,
        cancel_url=cancel_url,
    )
    delivered = await _send_extraid_email(
        request,
        to_email=str(extra["email"]),
        subject=subject,
        text=text,
        html=html,
    )
    if not delivered:
        await _get_extraid_db(request).revoke_account_action_token(token_id)
        if cancel_token_id is not None:
            await _get_extraid_db(request).revoke_account_action_token(
                cancel_token_id
            )
    return delivered


async def _allocate_synthetic_user(
    db: Database,
    extraid_db: ExtraIDDatabase,
    *,
    first_name: str,
) -> int:
    """Allocate in ExtraID DB, then claim the ID atomically in the primary DB."""
    for _ in range(100):
        user_id = await extraid_db.get_synthetic_user_id()
        created = await db.ensure_user(
            user_id=user_id,
            username=None,
            first_name=first_name,
            last_name=None,
            update_existing=False,
        )
        # Simple test doubles historically return None after recording a create.
        if created is not False:
            return int(user_id)
    raise RuntimeError("failed_to_allocate_synthetic_user_id")


async def _claim_registration_bonus(
    db: Database,
    extraid_db: ExtraIDDatabase,
    extra: dict,
) -> bool:
    user_id = int(extra["user_id"])
    if bool(extra.get("reg_bonus_claimed")):
        return False
    has_platform_identity = (
        await extraid_db.account_has_identity_provider(extra["id"], "telegram")
        or await extraid_db.account_has_identity_provider(extra["id"], "max")
    )
    if not has_platform_identity:
        return False
    try:
        claimed = await db.claim_extraid_registration_bonus(user_id, keys=3)
    except Exception:
        logger.error(
            "Primary registration bonus claim failed user=%s; verified state remains valid",
            user_id,
            exc_info=True,
        )
        return False
    # The primary-DB ledger is authoritative. This flag remains a compatibility
    # marker that prevents a one-off regrant when rolling out the new ledger.
    try:
        await extraid_db.mark_reg_bonus_claimed(extra["id"])
    except Exception:
        logger.warning(
            "Failed to mirror ExtraID registration bonus marker user=%s",
            user_id,
            exc_info=True,
        )
    return bool(claimed)


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
        password_error = _password_error(password)
        if password_error:
            message = (
                "Пароль должен быть не менее 8 символов"
                if password_error == "password_too_short"
                else "Пароль не должен превышать 72 байта в UTF-8"
            )
            return web.json_response({"error": password_error, "message": message}, status=400)
        if nickname and not _nickname_valid_fn(nickname):
            return web.json_response({"error": "invalid_nickname", "message": "Никнейм: 3-20 символов, только a-z, A-Z, 0-9, _, -"}, status=400)

        client_rate_key = _client_rate_key(request)
        email_rate_key = _rate_limit_subject("email", email)
        if not await _check_rate_limit_for_request(
            request,
            f"extraid_register:client:{client_rate_key}",
            5,
            600,
        ):
            return web.json_response({"error": "rate_limited"}, status=429)
        if not await _check_rate_limit_for_request(
            request,
            f"extraid_register:email:{email_rate_key}",
            3,
            3600,
        ):
            return web.json_response({"error": "rate_limited"}, status=429)

        bearer_auth = _bearer_token_from_request(request)
        if request.rel_url.query.get("_auth"):
            return web.json_response({"error": "auth_in_url_not_allowed"}, status=400)
        tg_auth = str(
            data.get("tg_init_data")
            or request.headers.get("X-Telegram-Init-Data", "")
        ).strip()
        user_id = None
        auth_source = "email_registration"
        identity_provider = "synthetic_user"
        identity_subject = ""
        created_synthetic_user = False
        if bearer_auth:
            jwt_result = await _verify_jwt_token_async_fn(bearer_auth, extraid_db, settings)
            if jwt_result:
                user_id = jwt_result[0]
                max_identity = None
                identity_lookup = getattr(db, "get_platform_identity_for_user", None)
                if identity_lookup is not None:
                    max_identity = await identity_lookup(user_id, "max")
                if max_identity:
                    auth_source = "max"
                    identity_provider = "max"
                    identity_subject = str(max_identity["subject"])
                else:
                    auth_source = "extraid_mobile" if client in {"android", "android_app", "mobile", "mobile_app"} else "email_registration"
                    identity_provider = "synthetic_user"
                    identity_subject = str(user_id)
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
                    identity_provider = "telegram"
                    identity_subject = str(user_id)
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
            if settings.environment != "development":
                return web.json_response({"error": "game_session_required"}, status=401)
        if (
            settings.environment != "development"
            and not _email_delivery_available(request)
        ):
            return web.json_response(
                {"error": "email_delivery_unavailable"},
                status=503,
            )

        existing_email = await _get_any_extra_account_by_email(extraid_db, email)
        if existing_email:
            return web.json_response({"error": "email_taken"}, status=409)
        if nickname:
            existing_nick = await extraid_db.fetchrow(
                "SELECT id FROM extra_accounts WHERE LOWER(nickname) = LOWER($1) AND deleted_at IS NULL",
                nickname,
            )
            if existing_nick:
                return web.json_response({"error": "nickname_taken"}, status=409)

        if user_id is None:
            user_id = await _allocate_synthetic_user(
                db,
                extraid_db,
                first_name=nickname or "Игрок",
            )
            created_synthetic_user = True
            await db.execute(
                "UPDATE users SET auth_source = 'email_registration' WHERE user_id = $1",
                user_id,
            )
            identity_provider = "synthetic_user"
            identity_subject = str(user_id)

        if user_id is not None:
            existing_for_user = await extraid_db.get_any_extra_account_by_user_id(user_id)
            if existing_for_user:
                return web.json_response({
                    "error": "extraid_already_exists",
                    "display_id": existing_for_user["display_id"],
                    "message": "ExtraID уже создавался для этого аккаунта.",
                }, status=409)

        display_id = _display_id_generator_fn(lambda did: False)
        while await extraid_db.fetchval("SELECT 1 FROM extra_accounts WHERE display_id = $1", display_id):
            display_id = _display_id_generator_fn(lambda did: False)

        password_hash = (await asyncio.to_thread(
            lambda: bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()
        ))
        try:
            extra = await extraid_db.create_extra_account(
                user_id,
                display_id,
                email,
                password_hash,
                nickname,
                identity_provider=identity_provider,
                identity_subject=identity_subject or str(user_id),
                registration_origin=(
                    "standalone"
                    if created_synthetic_user
                    else (
                        auth_source
                        if auth_source in {"telegram", "max"}
                        else "existing_user"
                    )
                ),
            )
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
            linked = await db.ensure_extra_account_link(
                user_id=int(user_id),
                extra_account_id=extra["id"],
                auth_source=auth_source,
            )
            if linked != "linked":
                raise RuntimeError("primary_extra_account_owner_conflict")
        except Exception:
            logger.exception("Failed to link users.extra_account_id after create_extra_account; rolling back extra_accounts row for user=%s", user_id)
            await _rollback_created_extra_account(
                db,
                extraid_db,
                extra_account_id=extra["id"],
                user_id=int(user_id),
                created_synthetic=created_synthetic_user,
            )
            raise

        email_sent = False
        if _email_delivery_available(request):
            email_sent = await _issue_account_email(
                request,
                extra,
                purpose="verify_email",
            )
            if not email_sent:
                # A provider timeout is ambiguous: Brevo may already have
                # accepted the message and the user may concurrently verify it.
                # Keep the pending account and retry through the durable outbox
                # instead of deleting credentials after a successful verify.
                await extraid_db.enqueue_account_email_action(
                    email,
                    "verify_email",
                )

        return web.json_response({
            "ok": True,
            "display_id": display_id,
            "email_sent": email_sent,
            "email_verification_required": True,
            "linked_telegram": auth_source == "telegram",
            "linked_max": auth_source == "max",
            "reg_bonus": False,
        })
    except Exception:
        logger.exception("ExtraID register handler error")
        return web.json_response({"error": "server_error", "message": "Внутренняя ошибка сервера"}, status=500)


async def _compensate_synthetic_user_cleanup(db: Database, user_id: int, created_synthetic: bool) -> None:
    """Best-effort cleanup of an orphaned synthetic users row created mid-registration."""
    if not created_synthetic:
        return
    try:
        deleted = await db.delete_user(user_id)
        if not deleted:
            logger.error("Compensating synthetic user delete returned false user=%s", user_id)
    except Exception:
        logger.warning("Compensating delete_user failed for synthetic user=%s", user_id, exc_info=True)


async def _rollback_created_extra_account(
    db: Database,
    extraid_db: ExtraIDDatabase,
    *,
    extra_account_id: Any,
    user_id: int,
    created_synthetic: bool,
) -> None:
    """Undo a new cross-DB account link without leaving a dangling primary pointer.

    Primary state is rolled back first. If that fails, the credential row stays
    intact and can be reconciled or re-linked instead of becoming an
    unrecoverable users.extra_account_id reference to a deleted account.
    """
    if created_synthetic:
        await db.delete_user(int(user_id))
    else:
        await db.execute(
            """
            UPDATE users
            SET extra_account_id = NULL
            WHERE user_id = $1 AND extra_account_id = $2
            """,
            int(user_id),
            extra_account_id,
        )
    await extraid_db.execute(
        "DELETE FROM extra_accounts WHERE id = $1",
        extra_account_id,
    )


async def _rollback_telegram_transfer_account(
    db: Database,
    extraid_db: ExtraIDDatabase,
    *,
    extra_account_id: Any,
    telegram_id: int,
) -> None:
    """CAS-rollback a failed Telegram transfer before deleting credentials."""
    await db.execute(
        """
        UPDATE users
        SET extra_account_id = NULL, auth_source = 'telegram'
        WHERE user_id = $1 AND extra_account_id = $2
        """,
        int(telegram_id),
        extra_account_id,
    )
    await extraid_db.execute(
        "DELETE FROM extra_accounts WHERE id = $1",
        extra_account_id,
    )


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
        if _password_error(password) == "password_too_long":
            return web.json_response({"error": "invalid_credentials"}, status=401)

        client_key_email = _rate_limit_subject("email", email)
        rate_client = _client_rate_key(request)
        limits = (
            (f"extraid_login:client:{rate_client}", 30, 600),
            (f"extraid_login:email:{client_key_email}", 10, 600),
            (f"extraid_login:pair:{rate_client}:{client_key_email}", 10, 60),
        )
        if not all(
            [
                await _check_rate_limit_for_request(request, key, limit, window)
                for key, limit, window in limits
            ]
        ):
            return web.json_response({"error": "rate_limited", "message": "Слишком много попыток входа. Попробуйте позже."}, status=429)

        extra = await extraid_db.get_extra_account_by_email(email)
        stored_password_hash = (
            str(extra["password_hash"]).encode("utf-8")
            if extra
            else _DUMMY_PASSWORD_HASH
        )
        match = await asyncio.to_thread(
            lambda: bcrypt.checkpw(password.encode("utf-8"), stored_password_hash)
        )
        if not extra or not match:
            return web.json_response({"error": "invalid_credentials"}, status=401)
        if extra.get("link_state"):
            return web.json_response({"error": "account_reconcile_pending"}, status=409)
        if (
            bool(extra.get("email_verification_required"))
            and not bool(extra.get("is_email_verified"))
        ):
            return web.json_response(
                {
                    "error": "email_not_verified",
                    "email_verification_required": True,
                    "resend_available": True,
                },
                status=403,
            )

        user_id = extra["user_id"]
        primary_link = await db.ensure_extra_account_link(
            user_id=int(user_id),
            extra_account_id=extra["id"],
        )
        if primary_link != "linked":
            try:
                await extraid_db.mark_primary_link_reconcile_required(
                    extra["id"],
                    user_id=int(user_id),
                )
            except Exception:
                logger.exception(
                    "Failed to persist primary owner conflict account=%s user=%s",
                    extra["id"],
                    user_id,
                )
            return web.json_response(
                {"error": "account_reconcile_pending"},
                status=409,
            )

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
        reg_bonus = await _claim_registration_bonus(db, extraid_db, extra)

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

    client_identity = _client_rate_key(request)
    client_key = f"anonymous:{client_identity}"
    if not await _check_rate_limit_for_request(request, client_key, 5, 300):
        return web.json_response({"error": "rate_limited"}, status=429)
    if not await _check_rate_limit_for_request(
        request,
        f"anonymous_daily:{client_identity}",
        10,
        86_400,
    ):
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

        user_id = await _allocate_synthetic_user(
            db,
            extraid_db,
            first_name=nickname,
        )

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

    generic_response = {"ok": True, "status": "code_sent", "ttl_seconds": 300}
    client_key = f"telegram_transfer_code:client:{_client_rate_key(request)}"
    if not await _check_rate_limit_for_request(request, client_key, 5, 900):
        return web.json_response({"error": "rate_limited"}, status=429)
    if not await _check_rate_limit_for_request(
        request,
        f"telegram_transfer_code:user:{_rate_limit_subject('telegram', telegram_id)}",
        3,
        3600,
    ):
        return web.json_response({"error": "rate_limited"}, status=429)

    try:
        user_exists = await db.fetchval("SELECT 1 FROM users WHERE user_id = $1", telegram_id)
        if not user_exists:
            return web.json_response(generic_response)

        existing_extra = await extraid_db.get_any_extra_account_by_user_id(telegram_id)
        if existing_extra:
            return web.json_response(generic_response)

        await extraid_db.cleanup_old_bot_codes(
            telegram_id,
            purpose=_TELEGRAM_TRANSFER_PURPOSE,
        )
        code = _make_bot_auth_code()
        created = await extraid_db.create_bot_auth_code(
            code,
            telegram_id,
            purpose=_TELEGRAM_TRANSFER_PURPOSE,
        )
        if not created.get("created", True):
            return web.json_response(generic_response)
        if not await _send_telegram_transfer_code(request, telegram_id, code):
            await extraid_db.invalidate_bot_auth_code(
                code,
                purpose=_TELEGRAM_TRANSFER_PURPOSE,
                user_id=telegram_id,
            )
            logger.warning("Telegram transfer code delivery failed user=%s", telegram_id)
        return web.json_response(generic_response)
    except Exception:
        logger.exception("telegram_transfer_request_code_handler error")
        return web.json_response({"error": "server_error", "message": "Внутренняя ошибка сервера"}, status=500)


async def telegram_transfer_complete_handler(request: web.Request) -> web.Response:
    db = _get_db(request)
    extraid_db = _get_extraid_db(request)
    settings = get_settings()

    client_key = f"telegram_transfer_complete:client:{_client_rate_key(request)}"
    if not await _check_rate_limit_for_request(request, client_key, 10, 600):
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
        password_error = _password_error(password)
        if password_error:
            return web.json_response({"error": password_error}, status=400)
        if settings.environment != "development" and not _email_delivery_available(request):
            return web.json_response({"error": "email_delivery_unavailable"}, status=503)
        if not await _check_rate_limit_for_request(
            request,
            f"telegram_transfer_complete:user:{_rate_limit_subject('telegram', telegram_id)}",
            5,
            900,
        ):
            return web.json_response({"error": "rate_limited"}, status=429)

        user_exists = await db.fetchval("SELECT 1 FROM users WHERE user_id = $1", telegram_id)
        if not user_exists:
            return web.json_response({"error": "invalid_code"}, status=400)
        auth_code = await extraid_db.verify_bot_auth_code(
            code,
            purpose=_TELEGRAM_TRANSFER_PURPOSE,
            user_id=telegram_id,
        )
        if not auth_code:
            return web.json_response({"error": "invalid_code"}, status=400)

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
            extra = await extraid_db.create_extra_account(
                telegram_id,
                display_id,
                email,
                password_hash,
                None,
                identity_provider="telegram",
                identity_subject=str(telegram_id),
                registration_origin="telegram",
            )
        except Exception as e:
            constraint = str(getattr(e, "constraint_name", "") or "").lower()
            if e.__class__.__name__ == "UniqueViolationError" and ("email" in constraint or "active_email" in constraint):
                return web.json_response({"error": "email_taken"}, status=409)
            logger.exception("create_extra_account failed during telegram transfer for user=%s", telegram_id)
            raise
        try:
            linked = await db.fetchrow(
                """
                UPDATE users
                SET extra_account_id = $1, auth_source = 'telegram_transfer'
                WHERE user_id = $2 AND extra_account_id IS NULL
                RETURNING user_id
                """,
                extra["id"],
                telegram_id,
            )
            if not linked:
                raise RuntimeError("telegram_user_link_conflict")
        except Exception:
            logger.exception("Failed to link users.extra_account_id after transfer create for user=%s; rolling back", telegram_id)
            await _rollback_telegram_transfer_account(
                db,
                extraid_db,
                extra_account_id=extra["id"],
                telegram_id=telegram_id,
            )
            raise

        try:
            consumed = await extraid_db.consume_bot_auth_code(
                code,
                purpose=_TELEGRAM_TRANSFER_PURPOSE,
                user_id=telegram_id,
            )
        except Exception:
            await _rollback_telegram_transfer_account(
                db,
                extraid_db,
                extra_account_id=extra["id"],
                telegram_id=telegram_id,
            )
            raise
        if not consumed:
            logger.warning("transfer complete: bot code consume failed after account creation for telegram_id=%s", telegram_id)
            await _rollback_telegram_transfer_account(
                db,
                extraid_db,
                extra_account_id=extra["id"],
                telegram_id=telegram_id,
            )
            return web.json_response({"error": "invalid_code"}, status=400)

        email_sent = False
        if _email_delivery_available(request):
            email_sent = await _issue_account_email(
                request,
                extra,
                purpose="verify_email",
            )
            if not email_sent:
                await extraid_db.enqueue_account_email_action(
                    email,
                    "verify_email",
                )
        return web.json_response({
            "ok": True,
            "user_id": telegram_id,
            "display_id": display_id,
            "reg_bonus": False,
            "email_sent": email_sent,
            "email_verification_required": True,
            "verification_pending": True,
        })
    except Exception:
        logger.exception("telegram_transfer_complete_handler error")
        return web.json_response({"error": "server_error", "message": "Внутренняя ошибка сервера"}, status=500)


# --- ExtraID: Link ---

async def extraid_link_handler(request: web.Request) -> web.Response:
    # Linking two existing identities requires a complete, audited game-state
    # merge. Repointing only credentials can orphan the source progress.
    return web.json_response(
        {
            "error": "support_required",
            "message": "Привязка существующего ExtraID к Telegram выполняется через поддержку.",
        },
        status=410,
    )


async def _legacy_extraid_link_handler(request: web.Request) -> web.Response:
    """Legacy implementation retained only for diagnosing pre-disable saga rows."""
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
        if (
            bool(extra.get("email_verification_required"))
            and not bool(extra.get("is_email_verified"))
        ):
            return web.json_response(
                {
                    "error": "email_not_verified",
                    "email_verification_required": True,
                },
                status=403,
            )

        # Prevent re-binding an ExtraID that is already linked to a real Telegram account.
        # The immutable identity ledger is authoritative; numeric ID ranges are not.
        if (
            await extraid_db.account_has_identity_provider(extra["id"], "telegram")
            and old_user_id != tg_user_id
        ):
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

        link_result = await extraid_db.link_extra_account_to_user(
            extra["id"],
            old_user_id,
            tg_user_id,
        )
        if link_result == "target_already_linked":
            return web.json_response({"error": "telegram_already_linked"}, status=409)
        if link_result != "linked":
            return web.json_response({"error": "link_conflict"}, status=409)

        try:
            primary_result = await db.transfer_extra_account_link(
                extra_account_id=extra["id"],
                old_user_id=old_user_id,
                new_user_id=tg_user_id,
            )
        except Exception:
            logger.exception(
                "link: failed to update primary user for tg_user_id=%s; rolling back identity owner",
                tg_user_id,
            )
            try:
                rolled_back = await extraid_db.rollback_extra_account_link(
                    extra["id"],
                    tg_user_id,
                    old_user_id,
                )
            except Exception:
                rolled_back = False
            if not rolled_back:
                await extraid_db.mark_extra_account_link_reconcile_required(
                    extra["id"],
                    old_user_id=old_user_id,
                    new_user_id=tg_user_id,
                )
            raise
        if primary_result != "linked":
            try:
                rolled_back = await extraid_db.rollback_extra_account_link(
                    extra["id"],
                    tg_user_id,
                    old_user_id,
                )
            except Exception:
                rolled_back = False
            if not rolled_back:
                await extraid_db.mark_extra_account_link_reconcile_required(
                    extra["id"],
                    old_user_id=old_user_id,
                    new_user_id=tg_user_id,
                )
                raise RuntimeError("extraid_link_rollback_failed")
            return web.json_response({"error": "telegram_already_linked"}, status=409)

        try:
            completed = await extraid_db.complete_extra_account_link(
                extra["id"],
                tg_user_id,
            )
        except Exception:
            completed = False
        if not completed:
            await extraid_db.mark_extra_account_link_reconcile_required(
                extra["id"],
                old_user_id=old_user_id,
                new_user_id=tg_user_id,
            )
            raise RuntimeError("extraid_link_completion_failed")

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
        "email_verification_required": bool(
            extra.get("email_verification_required")
            and not extra.get("is_email_verified")
        ),
        "reg_date": extra["created_at"].isoformat() if extra.get("created_at") else None,
        "linked_telegram": await extraid_db.account_has_identity_provider(
            extra["id"],
            "telegram",
        ),
        "linked_max": await extraid_db.account_has_identity_provider(
            extra["id"],
            "max",
        ),
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

        # Platform-bound accounts cannot be deleted: prevents the duplication exploit
        # (pump platform account -> create ExtraID -> delete -> create another ExtraID).
        is_telegram_bound = await extraid_db.account_has_identity_provider(
            extra["id"],
            "telegram",
        )
        is_max_bound = await extraid_db.account_has_identity_provider(
            extra["id"],
            "max",
        )
        if is_telegram_bound:
            return web.json_response({
                "error": "cannot_delete_telegram_bound",
                "message": "Невозможно удалить ExtraID, привязанный к Telegram. Обратитесь в поддержку.",
            }, status=403)
        if is_max_bound:
            return web.json_response({
                "error": "cannot_delete_max_bound",
                "message": "Невозможно удалить ExtraID, привязанный к MAX. Обратитесь в поддержку.",
            }, status=403)

        if not password:
            return web.json_response({"error": "invalid_password"}, status=401)
        if _password_error(password) == "password_too_long":
            return web.json_response({"error": "invalid_password"}, status=401)

        match = await asyncio.to_thread(
            lambda: bcrypt.checkpw(password.encode(), extra["password_hash"].encode())
        )
        if not match:
            return web.json_response({"error": "invalid_password"}, status=401)

        # Reset auth_source and clear the link on the users row.
        # For synthetic (email-only/anonymous) accounts, fully remove the orphaned users row.
        is_synthetic = await extraid_db.account_has_identity_provider(
            extra["id"],
            "synthetic_user",
        )
        if is_synthetic:
            deletion_started = await extraid_db.begin_account_deletion(
                extra["id"],
                user_id=user_id,
            )
            if not deletion_started:
                return web.json_response({"error": "account_delete_conflict"}, status=409)
            deleted = await db.delete_user(user_id)
            if not deleted:
                logger.info(
                    "delete_user found synthetic user=%s already absent; completing tombstone",
                    user_id,
                )
            await extraid_db.complete_account_deletion(extra["id"])
        else:
            await extraid_db.revoke_all_user_sessions(user_id)
            await db.execute(
                "UPDATE users SET extra_account_id = NULL, auth_source = 'telegram' WHERE user_id = $1",
                user_id
            )
            await extraid_db.soft_delete_extra_account(extra["id"])
        return web.json_response({"ok": True})
    except Exception:
        logger.exception("ExtraID delete handler error")
        return web.json_response({"error": "server_error", "message": "Внутренняя ошибка сервера"}, status=500)


# --- ExtraID email verification and password recovery ---

async def extraid_email_resend_handler(request: web.Request) -> web.Response:
    extraid_db = _get_extraid_db(request)
    generic = {"ok": True, "status": _GENERIC_EMAIL_STATUS}
    if not _email_delivery_available(request):
        return web.json_response({"error": "email_delivery_unavailable"}, status=503)
    try:
        data = await request.json()
    except Exception:
        data = {}
    email = str(data.get("email") or "").strip().lower()
    client_key = _client_rate_key(request)
    if not await _check_rate_limit_for_request(
        request,
        f"email_verify_resend:client:{client_key}",
        5,
        900,
    ):
        return web.json_response({"error": "rate_limited"}, status=429)
    if not _valid_email(email):
        return web.json_response(generic)
    email_key = _rate_limit_subject("email", email)
    if not await _check_rate_limit_for_request(
        request,
        f"email_verify_resend:email:{email_key}",
        3,
        3600,
    ):
        return web.json_response(generic)
    # Queue-only request path: both matching and unknown addresses execute the
    # same DB statement and never wait for the mail provider.
    await extraid_db.enqueue_account_email_action(email, "verify_email")
    return web.json_response(generic)


async def extraid_unverified_email_change_handler(request: web.Request) -> web.Response:
    extraid_db = _get_extraid_db(request)
    if not _email_delivery_available(request):
        return web.json_response({"error": "email_delivery_unavailable"}, status=503)
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)
    if request.rel_url.query.get("_auth"):
        return web.json_response({"error": "auth_in_url_not_allowed"}, status=400)
    new_email = str(data.get("email") or "").strip().lower()
    if not _valid_email(new_email):
        return web.json_response({"error": "invalid_email"}, status=400)
    owner_user_id = None
    auth_via_telegram = False
    bearer = _bearer_token_from_request(request)
    if bearer:
        jwt_result = await _verify_jwt_token_async_fn(
            bearer,
            extraid_db,
            get_settings(),
        )
        if not jwt_result:
            return web.json_response({"error": "invalid_auth"}, status=401)
        owner_user_id = int(jwt_result[0])
    else:
        tg_init_data = str(
            data.get("tg_init_data")
            or request.headers.get("X-Telegram-Init-Data", "")
        ).strip()
        verified = _verify_init_data_fn(tg_init_data, request.app["bot_token"])
        if not verified or not _validate_auth_date_fn(verified):
            return web.json_response({"error": "invalid_auth"}, status=401)
        owner_user_id = _extract_user_id_from_init_data_fn(verified)
        auth_via_telegram = True
        if not owner_user_id:
            return web.json_response({"error": "invalid_auth"}, status=401)
    client_key = _client_rate_key(request)
    email_key = _rate_limit_subject("email", new_email)
    for key, limit, window in (
        (f"email_change:client:{client_key}", 5, 900),
        (
            f"email_change:user:{_rate_limit_subject('user', owner_user_id)}",
            3,
            3600,
        ),
        (f"email_change:email:{email_key}", 3, 3600),
    ):
        if not await _check_rate_limit_for_request(
            request,
            key,
            limit,
            window,
        ):
            return web.json_response({"error": "rate_limited"}, status=429)
    extra = await extraid_db.get_extra_account_by_user_id(int(owner_user_id))
    if not extra:
        return web.json_response({"error": "extra_account_not_found"}, status=404)
    if auth_via_telegram and not await extraid_db.account_has_identity_provider(
        extra["id"],
        "telegram",
    ):
        return web.json_response({"error": "invalid_auth"}, status=401)
    old_email = str(extra["email"])
    result = await extraid_db.change_unverified_email(
        extra["id"],
        new_email=new_email,
        expected_old_email=old_email,
    )
    if result == "email_taken":
        return web.json_response({"error": "email_taken"}, status=409)
    if result != "changed":
        return web.json_response({"error": "email_change_not_allowed"}, status=409)
    changed_extra = dict(extra)
    changed_extra["email"] = new_email
    if not await _issue_account_email(
        request,
        changed_extra,
        purpose="verify_email",
    ):
        rollback = await extraid_db.change_unverified_email(
            extra["id"],
            new_email=old_email,
            expected_old_email=new_email,
        )
        if rollback != "changed":
            logger.error(
                "Failed to rollback undelivered ExtraID email change account=%s",
                extra["id"],
            )
        return web.json_response({"error": "email_delivery_failed"}, status=503)
    return web.json_response(
        {
            "ok": True,
            "email_sent": True,
            "email_verification_required": True,
        }
    )


async def extraid_email_verify_handler(request: web.Request) -> web.Response:
    db = _get_db(request)
    extraid_db = _get_extraid_db(request)
    if any(
        request.rel_url.query.get(name)
        for name in ("token", "code", "email")
    ):
        return web.json_response({"error": "code_in_url_not_allowed"}, status=400)
    try:
        data = await request.json()
    except Exception:
        data = {}
    if not await _check_rate_limit_for_request(
        request,
        f"email_verify:client:{_client_rate_key(request)}",
        10,
        900,
    ):
        return web.json_response({"error": "rate_limited"}, status=429)

    email = str(data.get("email") or "").strip().lower()
    code = str(data.get("code") or "").strip()
    if email or code:
        if not _valid_email(email) or not _EMAIL_VERIFY_CODE_RE.fullmatch(code):
            return web.json_response(
                {"error": "invalid_or_expired_code"},
                status=400,
            )
        email_key = _rate_limit_subject("email", email)
        if not await _check_rate_limit_for_request(
            request,
            f"email_verify:email:{email_key}",
            8,
            900,
        ):
            return web.json_response({"error": "rate_limited"}, status=429)
        consumed = await extraid_db.consume_email_verification_code(
            email=email,
            code=code,
        )
        invalid_error = "invalid_or_expired_code"
    else:
        # Compatibility window for links already issued before the code flow
        # shipped. New verification emails never contain this token.
        token = str(data.get("token") or "").strip()
        parsed = _parse_account_action_token(token)
        if not parsed:
            return web.json_response(
                {"error": "invalid_or_expired_code"},
                status=400,
            )
        token_id, token_hash = parsed
        consumed = await extraid_db.consume_email_verification_token(
            token_id=token_id,
            token_hash=token_hash,
        )
        invalid_error = "invalid_or_expired_token"
    if not consumed:
        return web.json_response({"error": invalid_error}, status=400)
    primary_link = await db.ensure_extra_account_link(
        user_id=int(consumed["user_id"]),
        extra_account_id=consumed["extra_account_id"],
    )
    if primary_link != "linked":
        try:
            await extraid_db.mark_primary_link_reconcile_required(
                consumed["extra_account_id"],
                user_id=int(consumed["user_id"]),
            )
        except Exception:
            logger.exception(
                "Failed to persist verified ExtraID owner conflict account=%s user=%s",
                consumed["extra_account_id"],
                consumed["user_id"],
            )
        return web.json_response(
            {"error": "account_reconcile_pending"},
            status=409,
        )
    reg_bonus = await _claim_registration_bonus(
        db,
        extraid_db,
        {
            "id": consumed["extra_account_id"],
            "user_id": consumed["user_id"],
            "reg_bonus_claimed": consumed.get("reg_bonus_claimed", False),
        },
    )
    return web.json_response(
        {"ok": True, "email_verified": True, "reg_bonus": reg_bonus}
    )


async def _cleanup_pending_registration_primary(
    db: Database,
    extraid_db: ExtraIDDatabase,
    row: dict,
) -> bool:
    user_id = int(row["user_id"])
    origin = str(row.get("registration_origin") or "legacy")
    if origin == "standalone":
        # False means the desired state (already absent) is already reached.
        await db.delete_user(user_id)
    else:
        await db.execute(
            """
            UPDATE users
            SET extra_account_id = NULL
            WHERE user_id = $1 AND extra_account_id = $2
            """,
            user_id,
            row["id"],
        )
    return await extraid_db.finalize_pending_registration_cleanup(row["id"])


def _same_extra_account_id(value: Any, expected: Any) -> bool:
    return value is not None and str(value) == str(expected)


async def _run_extraid_reconciliation_once(app: web.Application) -> dict[str, int]:
    """Bounded, idempotent repair pass for interrupted cross-database sagas."""
    db = app.get("db")
    extraid_db = app.get("extraid_db")
    stats = {
        "registration_cleanups": 0,
        "account_deletions": 0,
        "links_completed": 0,
        "links_rolled_back": 0,
        "errors": 0,
    }
    if db is None or extraid_db is None:
        logger.error("ExtraID reconciliation skipped: databases are not registered")
        stats["errors"] += 1
        return stats

    cleanup_rows: dict[str, dict] = {}
    expire_pending = getattr(extraid_db, "expire_pending_registrations", None)
    if expire_pending is not None:
        try:
            for row in await expire_pending(_EXTRAID_RECONCILE_BATCH_SIZE):
                cleanup_rows[str(row["id"])] = dict(row)
        except Exception:
            stats["errors"] += 1
            logger.exception("ExtraID reconciliation failed to expire pending registrations")
    load_registration_cleanups = getattr(
        extraid_db,
        "get_pending_registration_cleanups",
        None,
    )
    if load_registration_cleanups is not None:
        try:
            for row in await load_registration_cleanups(
                _EXTRAID_RECONCILE_BATCH_SIZE
            ):
                cleanup_rows[str(row["id"])] = dict(row)
        except Exception:
            stats["errors"] += 1
            logger.exception("ExtraID reconciliation failed to load registration cleanups")
    for row in cleanup_rows.values():
        try:
            if await _cleanup_pending_registration_primary(db, extraid_db, row):
                stats["registration_cleanups"] += 1
        except Exception:
            stats["errors"] += 1
            logger.exception(
                "ExtraID registration cleanup remains pending account=%s",
                row.get("id"),
            )

    load_deletions = getattr(extraid_db, "get_pending_account_deletions", None)
    try:
        deletion_rows = (
            await load_deletions(_EXTRAID_RECONCILE_BATCH_SIZE)
            if load_deletions is not None
            else []
        )
    except Exception:
        deletion_rows = []
        stats["errors"] += 1
        logger.exception("ExtraID reconciliation failed to load account deletions")
    for row in deletion_rows:
        try:
            # Database.delete_user returning False means the desired primary
            # state (row absent) is already reached.
            await db.delete_user(int(row["user_id"]))
            await extraid_db.complete_account_deletion(row["id"])
            stats["account_deletions"] += 1
        except Exception:
            stats["errors"] += 1
            logger.exception(
                "ExtraID primary account purge remains pending account=%s user=%s",
                row.get("id"),
                row.get("user_id"),
            )

    load_links = getattr(extraid_db, "get_pending_identity_reconciliations", None)
    try:
        link_rows = (
            await load_links(_EXTRAID_RECONCILE_BATCH_SIZE)
            if load_links is not None
            else []
        )
    except Exception:
        link_rows = []
        stats["errors"] += 1
        logger.exception("ExtraID reconciliation failed to load identity links")
    for row in link_rows:
        extra_account_id = row.get("id")
        new_user_id = None
        try:
            new_user_id = int(row["user_id"])
            if row.get("link_state") == "primary_owner_reconcile":
                repaired = await db.ensure_extra_account_link(
                    user_id=new_user_id,
                    extra_account_id=extra_account_id,
                )
                if repaired != "linked":
                    stats["errors"] += 1
                    logger.error(
                        "ExtraID primary owner conflict needs manual reconciliation "
                        "account=%s user=%s",
                        extra_account_id,
                        new_user_id,
                    )
                    await extraid_db.revoke_all_user_sessions(new_user_id)
                    continue
                completed = await extraid_db.complete_primary_link_reconciliation(
                    extra_account_id,
                    user_id=new_user_id,
                )
                if not completed:
                    raise RuntimeError("primary_owner_reconcile_completion_failed")
                stats["links_completed"] += 1
                continue
            old_user_id = int(row["link_previous_user_id"])
            if extra_account_id is None:
                raise ValueError("pending_link_missing_account_id")
            new_primary_link = await db.fetchval(
                "SELECT extra_account_id FROM users WHERE user_id = $1",
                new_user_id,
            )
            old_primary_link = await db.fetchval(
                "SELECT extra_account_id FROM users WHERE user_id = $1",
                old_user_id,
            )
            if (
                _same_extra_account_id(new_primary_link, extra_account_id)
                and old_primary_link is None
            ):
                primary_rollback = await db.rollback_extra_account_link(
                    extra_account_id=extra_account_id,
                    source_user_id=old_user_id,
                    telegram_user_id=new_user_id,
                )
                if primary_rollback != "rolled_back":
                    raise RuntimeError("pending_primary_link_rollback_failed")
                rolled_back = await extraid_db.rollback_extra_account_link(
                    extra_account_id,
                    new_user_id,
                    old_user_id,
                )
                if not rolled_back:
                    raise RuntimeError("pending_identity_link_rollback_cas_failed")
                stats["links_rolled_back"] += 1
            elif (
                new_primary_link is None
                and _same_extra_account_id(old_primary_link, extra_account_id)
            ):
                rolled_back = await extraid_db.rollback_extra_account_link(
                    extra_account_id,
                    new_user_id,
                    old_user_id,
                )
                if not rolled_back:
                    raise RuntimeError("pending_link_rollback_cas_failed")
                stats["links_rolled_back"] += 1
            else:
                if row.get("link_state") != "reconcile_required":
                    await extraid_db.mark_extra_account_link_reconcile_required(
                        extra_account_id,
                        old_user_id=old_user_id,
                        new_user_id=new_user_id,
                    )
                stats["errors"] += 1
                logger.error(
                    "ExtraID link needs manual reconciliation account=%s old=%s new=%s "
                    "old_primary=%s new_primary=%s",
                    extra_account_id,
                    old_user_id,
                    new_user_id,
                    old_primary_link,
                    new_primary_link,
                )
        except Exception:
            stats["errors"] += 1
            if new_user_id is not None:
                try:
                    await extraid_db.revoke_all_user_sessions(new_user_id)
                except Exception:
                    logger.exception(
                        "Failed to revoke sessions for corrupt pending link user=%s",
                        new_user_id,
                    )
            logger.exception(
                "ExtraID identity link remains pending account=%s",
                extra_account_id,
            )

    for cleanup_name in (
        "cleanup_account_action_tokens",
        "cleanup_account_email_outbox",
        "cleanup_expired_sessions",
        "cleanup_old_rate_limits",
        "cleanup_used_bot_codes",
    ):
        cleanup = getattr(extraid_db, cleanup_name, None)
        if cleanup is None:
            continue
        try:
            await cleanup()
        except Exception:
            stats["errors"] += 1
            logger.exception("ExtraID housekeeping failed operation=%s", cleanup_name)
    return stats


async def _extraid_reconciliation_loop(app: web.Application) -> None:
    while True:
        await asyncio.sleep(_EXTRAID_RECONCILE_INTERVAL_SECONDS)
        try:
            await _run_extraid_reconciliation_once(app)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Unexpected ExtraID reconciliation pass failure")


async def _run_extraid_email_outbox_once(app: web.Application) -> int:
    extraid_db = app.get("extraid_db")
    if extraid_db is None:
        return 0
    claim_jobs = getattr(extraid_db, "claim_account_email_outbox", None)
    if claim_jobs is None:
        return 0
    jobs = await claim_jobs(
        _EXTRAID_EMAIL_OUTBOX_BATCH_SIZE
    )
    processed = 0
    context = SimpleNamespace(app=app)
    for job in jobs:
        outbox_id = job["outbox_id"]
        try:
            purpose = str(job.get("purpose") or "")
            current_email = str(job.get("email") or "").strip().lower()
            email_snapshot = str(job.get("email_snapshot") or "").strip().lower()
            eligible = (
                job.get("id") is not None
                and job.get("deleted_at") is None
                and current_email == email_snapshot
                and (
                    (
                        purpose == "verify_email"
                        and not bool(job.get("is_email_verified"))
                    )
                    or (
                        purpose == "password_reset"
                        and bool(job.get("is_email_verified"))
                    )
                )
            )
            if not eligible:
                await extraid_db.complete_account_email_outbox(outbox_id)
                processed += 1
                continue
            delivered = await _issue_account_email(
                context,
                job,
                purpose=purpose,
            )
            if delivered:
                await extraid_db.complete_account_email_outbox(outbox_id)
                processed += 1
            else:
                await extraid_db.retry_account_email_outbox(
                    outbox_id,
                    error="delivery_or_token_issue_failed",
                )
        except Exception as exc:
            logger.exception("ExtraID email outbox job failed id=%s", outbox_id)
            try:
                await extraid_db.retry_account_email_outbox(
                    outbox_id,
                    error=type(exc).__name__,
                )
            except Exception:
                logger.exception(
                    "ExtraID email outbox lease could not be released id=%s",
                    outbox_id,
                )
    return processed


async def _extraid_email_outbox_loop(app: web.Application) -> None:
    while True:
        try:
            await _run_extraid_email_outbox_once(app)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Unexpected ExtraID email outbox pass failure")
        await asyncio.sleep(_EXTRAID_EMAIL_OUTBOX_INTERVAL_SECONDS)


async def _extraid_reconciliation_context(app: web.Application):
    # The initial bounded pass closes work left by a crash before traffic starts.
    try:
        await _run_extraid_reconciliation_once(app)
    except Exception:
        # A corrupt legacy row must remain fail-closed, but should not prevent
        # unrelated gameplay traffic from starting.
        logger.exception("Initial ExtraID reconciliation pass failed")
    reconcile_task = asyncio.create_task(
        _extraid_reconciliation_loop(app),
        name="extraid-lifecycle-reconciler",
    )
    email_task = asyncio.create_task(
        _extraid_email_outbox_loop(app),
        name="extraid-email-outbox",
    )
    app["_extraid_reconciliation_task"] = reconcile_task
    app["_extraid_email_outbox_task"] = email_task
    try:
        yield
    finally:
        reconcile_task.cancel()
        email_task.cancel()
        with suppress(asyncio.CancelledError):
            await reconcile_task
        with suppress(asyncio.CancelledError):
            await email_task
        app.pop("_extraid_reconciliation_task", None)
        app.pop("_extraid_email_outbox_task", None)


async def extraid_registration_cancel_handler(request: web.Request) -> web.Response:
    db = _get_db(request)
    extraid_db = _get_extraid_db(request)
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)
    parsed = _parse_account_action_token(data.get("token"))
    if not parsed:
        return web.json_response({"error": "invalid_or_expired_token"}, status=400)
    if not await _check_rate_limit_for_request(
        request,
        f"registration_cancel:client:{_client_rate_key(request)}",
        10,
        900,
    ):
        return web.json_response({"error": "rate_limited"}, status=429)
    token_id, token_hash = parsed
    cancelled = await extraid_db.consume_registration_cancel_token(
        token_id=token_id,
        token_hash=token_hash,
    )
    if not cancelled:
        return web.json_response({"error": "invalid_or_expired_token"}, status=400)
    row = {
        "id": cancelled["extra_account_id"],
        "user_id": cancelled["user_id"],
        "registration_origin": cancelled.get("registration_origin"),
    }
    try:
        await _cleanup_pending_registration_primary(db, extraid_db, row)
    except Exception:
        logger.error(
            "Registration cancel primary cleanup deferred account=%s",
            row["id"],
            exc_info=True,
        )
    return web.json_response({"ok": True, "registration_cancelled": True})


async def extraid_password_reset_request_handler(request: web.Request) -> web.Response:
    extraid_db = _get_extraid_db(request)
    generic = {"ok": True, "status": _GENERIC_EMAIL_STATUS}
    if not _email_delivery_available(request):
        return web.json_response({"error": "email_delivery_unavailable"}, status=503)
    try:
        data = await request.json()
    except Exception:
        data = {}
    email = str(data.get("email") or "").strip().lower()
    client_key = _client_rate_key(request)
    if not await _check_rate_limit_for_request(
        request,
        f"password_reset_request:client:{client_key}",
        5,
        900,
    ):
        return web.json_response({"error": "rate_limited"}, status=429)
    if not _valid_email(email):
        return web.json_response(generic)
    email_key = _rate_limit_subject("email", email)
    if not await _check_rate_limit_for_request(
        request,
        f"password_reset_request:email:{email_key}",
        3,
        3600,
    ):
        return web.json_response(generic)
    await extraid_db.enqueue_account_email_action(email, "password_reset")
    return web.json_response(generic)


async def extraid_password_reset_confirm_handler(request: web.Request) -> web.Response:
    extraid_db = _get_extraid_db(request)
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)
    password = data.get("password") or ""
    password_error = _password_error(password)
    if password_error:
        return web.json_response({"error": password_error}, status=400)
    parsed = _parse_account_action_token(data.get("token"))
    if not parsed:
        return web.json_response({"error": "invalid_or_expired_token"}, status=400)
    if not await _check_rate_limit_for_request(
        request,
        f"password_reset_confirm:client:{_client_rate_key(request)}",
        10,
        900,
    ):
        return web.json_response({"error": "rate_limited"}, status=429)
    token_id, token_hash = parsed
    password_hash = await asyncio.to_thread(
        lambda: bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt(rounds=12),
        ).decode()
    )
    consumed = await extraid_db.consume_password_reset_token(
        token_id=token_id,
        token_hash=token_hash,
        new_password_hash=password_hash,
    )
    if not consumed:
        return web.json_response({"error": "invalid_or_expired_token"}, status=400)
    return web.json_response({"ok": True})


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

    session = await extraid_db.fetchrow(
        "SELECT auth_method FROM auth_sessions WHERE session_id = $1",
        uuid.UUID(str(jw[1])),
    )
    if session and str(session.get("auth_method") or "") == "max":
        return web.json_response(
            {
                "error": "cannot_logout_max_bound",
                "message": "В MAX игровой профиль подключён автоматически; выход недоступен.",
            },
            status=403,
        )

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
    if not app.get("_extraid_no_store_middleware_registered"):
        app.middlewares.append(extraid_no_store_middleware)
        app["_extraid_no_store_middleware_registered"] = True
    if not app.get("_extraid_reconciliation_context_registered"):
        app.cleanup_ctx.append(_extraid_reconciliation_context)
        app["_extraid_reconciliation_context_registered"] = True

    # ExtraID
    app.router.add_post("/api/extraid/register", extraid_register_handler)
    app.router.add_post("/api/extraid/login", extraid_login_handler)
    app.router.add_post("/api/extraid/link", extraid_link_handler)
    app.router.add_get("/api/extraid/profile", extraid_profile_handler)
    app.router.add_patch("/api/extraid/profile", extraid_patch_profile_handler)
    app.router.add_post("/api/extraid/delete", extraid_delete_account_handler)
    app.router.add_post("/api/extraid/email/resend", extraid_email_resend_handler)
    app.router.add_post(
        "/api/extraid/email/change",
        extraid_unverified_email_change_handler,
    )
    app.router.add_post("/api/extraid/email/verify", extraid_email_verify_handler)
    app.router.add_post(
        "/api/extraid/email/cancel",
        extraid_registration_cancel_handler,
    )
    app.router.add_post(
        "/api/extraid/password-reset/request",
        extraid_password_reset_request_handler,
    )
    app.router.add_post(
        "/api/extraid/password-reset/confirm",
        extraid_password_reset_confirm_handler,
    )

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

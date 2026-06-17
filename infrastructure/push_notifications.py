from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from infrastructure.notifications import (
    format_android_notification_title,
    format_notification_message,
    notification_section,
)

logger = logging.getLogger(__name__)

UPDATE_TITLE = "Хорошие новости!"
UPDATE_BODY = "Вышло обновление, скачай новую версию, чтобы продолжить игру"
UPDATE_CHANNEL_URL = "https://t.me/extraarenamobile"
UPDATE_APK_URL = "https://apk.laveqox.ru"
ANDROID_PUSH_CLICK_ACTION = "ru.extraarena.app.PUSH"
ANDROID_GAME_CHANNEL_ID = "extraarena_game"
ANDROID_UPDATES_CHANNEL_ID = "extraarena_updates"


@dataclass(frozen=True)
class AndroidPushPayload:
    title: str
    body: str
    data: dict[str, str]


@dataclass(frozen=True)
class PushDeliveryResult:
    ok: bool
    message_id: str | None = None
    error: str | None = None
    permanent: bool = False


@dataclass(frozen=True)
class PushBroadcastResult:
    total: int
    sent: int
    failed: int

    @property
    def ok(self) -> bool:
        return self.sent > 0 and self.failed == 0


def build_android_push_payload(
    category: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
) -> AndroidPushPayload:
    """Build the Android data payload from the same notification model as Telegram."""
    payload = payload or {}
    category = str(category or "")
    event_type = str(event_type or "")

    if category == "app_update" or event_type in {"app_update", "app_update_required"}:
        title = str(payload.get("title") or UPDATE_TITLE)
        body = str(payload.get("body") or UPDATE_BODY)
        url = str(payload.get("url") or UPDATE_CHANNEL_URL).strip()
        apk_url = str(payload.get("apk_url") or UPDATE_APK_URL).strip()
        if url not in {UPDATE_CHANNEL_URL, UPDATE_APK_URL}:
            url = UPDATE_CHANNEL_URL
        if apk_url not in {UPDATE_CHANNEL_URL, UPDATE_APK_URL}:
            apk_url = UPDATE_APK_URL
        return AndroidPushPayload(
            title=title,
            body=body,
            data={
                "type": "app_update_required",
                "category": category or "app_update",
                "event_type": event_type or "app_update_required",
                "title": title,
                "body": body,
                "url": url,
                "apk_url": apk_url,
                "android_channel_id": ANDROID_UPDATES_CHANNEL_ID,
            },
        )

    body = format_notification_message(event_type, payload)
    section = notification_section(category, payload)
    title = format_android_notification_title(category, event_type, payload)
    data = {
        "type": "game_notification",
        "category": category,
        "event_type": event_type,
        "section": section,
        "title": title,
        "body": body,
        "android_channel_id": ANDROID_GAME_CHANNEL_ID,
    }
    for key in ("invite_id", "invite_action", "request_id", "from_user_id"):
        if payload.get(key) is not None:
            data[key] = str(payload.get(key))
    return AndroidPushPayload(
        title=title,
        body=body,
        data=data,
    )


def _service_account_from_env() -> tuple[Any | None, str | None]:
    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    service_account_b64 = os.getenv("FIREBASE_SERVICE_ACCOUNT_B64", "").strip()
    service_account_file = (
        os.getenv("FIREBASE_SERVICE_ACCOUNT_FILE", "").strip()
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    )

    if service_account_json:
        return json.loads(service_account_json), None
    if service_account_b64:
        raw = base64.b64decode(service_account_b64).decode("utf-8")
        return json.loads(raw), None
    if service_account_file:
        return None, service_account_file
    return None, None


def _is_permanent_fcm_error(error: str) -> bool:
    lower = (error or "").lower()
    return any(
        marker in lower
        for marker in (
            "registration-token-not-registered",
            "invalid-registration-token",
            "requested entity was not found",
            "unregistered",
            "not-found",
            "sender id mismatch",
            "mismatched-credential",
            "invalid-argument",
        )
    )


class FcmPushSender:
    """Small Firebase Admin wrapper that stays inert until credentials are configured."""

    def __init__(self) -> None:
        self._initialized = False
        self._messaging = None
        self._init_error: str | None = None

    @property
    def configured(self) -> bool:
        return self._ensure_initialized()

    @property
    def init_error(self) -> str | None:
        self._ensure_initialized()
        return self._init_error

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return self._messaging is not None

        self._initialized = True
        try:
            import firebase_admin  # type: ignore
            from firebase_admin import credentials, messaging  # type: ignore
        except ModuleNotFoundError as exc:
            self._init_error = str(exc)
            return False

        try:
            if not firebase_admin._apps:
                service_account_data, service_account_file = _service_account_from_env()
                options = {}
                project_id = os.getenv("FIREBASE_PROJECT_ID", "").strip()
                if project_id:
                    options["projectId"] = project_id
                if service_account_data:
                    firebase_admin.initialize_app(
                        credentials.Certificate(service_account_data),
                        options=options or None,
                    )
                elif service_account_file:
                    firebase_admin.initialize_app(
                        credentials.Certificate(service_account_file),
                        options=options or None,
                    )
                else:
                    firebase_admin.initialize_app(options=options or None)
            self._messaging = messaging
            self._init_error = None
            return True
        except Exception as exc:
            self._init_error = str(exc)
            logger.warning("FCM push sender is not configured: %s", exc)
            return False

    async def send(
        self,
        *,
        token: str,
        title: str,
        body: str,
        data: dict[str, Any] | None = None,
    ) -> PushDeliveryResult:
        if not self._ensure_initialized() or self._messaging is None:
            return PushDeliveryResult(ok=False, error=self._init_error or "fcm_not_configured")

        message_data = {str(key): str(value) for key, value in (data or {}).items() if value is not None}
        channel_id = message_data.get("android_channel_id") or ANDROID_GAME_CHANNEL_ID
        try:
            message = self._messaging.Message(
                token=token,
                notification=self._messaging.Notification(
                    title=title,
                    body=body,
                ),
                data=message_data,
                android=self._messaging.AndroidConfig(
                    priority="high",
                    ttl=timedelta(hours=1),
                    notification=self._messaging.AndroidNotification(
                        channel_id=channel_id,
                        click_action=ANDROID_PUSH_CLICK_ACTION,
                        icon="ic_notification",
                        color="#FF8A3D",
                    ),
                ),
            )
            message_id = await asyncio.to_thread(self._messaging.send, message)
            return PushDeliveryResult(ok=True, message_id=str(message_id))
        except Exception as exc:
            error = str(exc)
            return PushDeliveryResult(
                ok=False,
                error=error,
                permanent=_is_permanent_fcm_error(error),
            )


async def send_android_broadcast(
    *,
    db: Any,
    push_sender: FcmPushSender,
    payload: AndroidPushPayload,
    platform: str = "android",
    limit: int = 10000,
) -> PushBroadcastResult:
    if not hasattr(db, "get_push_devices_for_broadcast"):
        return PushBroadcastResult(total=0, sent=0, failed=0)

    devices = await db.get_push_devices_for_broadcast(platform=platform, limit=limit)
    sent = 0
    failed = 0
    for device in devices:
        token = device.get("token")
        if not token:
            continue
        result = await push_sender.send(
            token=token,
            title=payload.title,
            body=payload.body,
            data=payload.data,
        )
        if result.ok:
            sent += 1
        else:
            failed += 1
            if hasattr(db, "mark_push_device_error"):
                await db.mark_push_device_error(
                    token,
                    result.error or "push_delivery_failed",
                    permanent=result.permanent,
                )
    return PushBroadcastResult(total=len(devices), sent=sent, failed=failed)

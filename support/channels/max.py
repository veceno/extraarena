from __future__ import annotations

import hmac
from urllib.parse import urlparse
from typing import Any

import aiohttp

from support.attachments import MAX_SUPPORT_ATTACHMENT_BYTES
from support.constants import SupportChannel


MAX_API_BASE = "https://platform-api2.max.ru"


def verify_max_webhook_secret(expected: str, received: str | None) -> bool:
    if not expected or not received:
        return False
    return hmac.compare_digest(str(expected), str(received))


def normalize_max_message(update: dict[str, Any]) -> dict[str, Any]:
    message = update.get("message") if isinstance(update, dict) else None
    if not isinstance(message, dict):
        message = update
    sender = message.get("sender") or message.get("user") or {}
    body = message.get("body") or {}
    user_id = sender.get("user_id") or sender.get("id") or message.get("user_id")
    text = body.get("text") if isinstance(body, dict) else None
    if text is None:
        text = message.get("text") or ""
    channel_id = str(user_id or "")
    return {
        "channel": SupportChannel.MAX,
        "channel_id": channel_id,
        "external_user_id": f"max:{channel_id}" if channel_id else "",
        "display_name": str(sender.get("name") or sender.get("username") or "MAX user"),
        "body": str(text or ""),
        "attachments": list(body.get("attachments") or message.get("attachments") or []) if isinstance(body, dict) else [],
        "raw": update,
    }


def extract_max_image_attachments(update: dict[str, Any]) -> list[dict[str, str]]:
    message = update.get("message") if isinstance(update, dict) else None
    if not isinstance(message, dict):
        message = update if isinstance(update, dict) else {}
    body = message.get("body") if isinstance(message, dict) else {}
    attachments = []
    if isinstance(body, dict) and isinstance(body.get("attachments"), list):
        attachments.extend(body["attachments"])
    if isinstance(message.get("attachments"), list):
        attachments.extend(message["attachments"])

    results: list[dict[str, str]] = []
    for index, attachment in enumerate(attachments):
        if not isinstance(attachment, dict):
            continue
        attachment_type = str(attachment.get("type") or "").lower()
        payload = attachment.get("payload") if isinstance(attachment.get("payload"), dict) else {}
        url = _first_url(attachment, payload)
        if not url or not _is_https_url(url):
            continue
        content_type = str(
            attachment.get("content_type")
            or payload.get("content_type")
            or payload.get("mime_type")
            or ""
        ).lower()
        if attachment_type and attachment_type not in {"image", "photo", "picture"}:
            if not content_type.startswith("image/"):
                continue
        results.append(
            {
                "url": url,
                "filename": str(
                    attachment.get("filename")
                    or payload.get("filename")
                    or payload.get("name")
                    or f"max-image-{index + 1}"
                ),
                "content_type": content_type or "image/*",
            }
        )
    return results


def _first_url(*objects: dict[str, Any]) -> str:
    keys = ("url", "download_url", "file_url", "media_url", "src")
    for obj in objects:
        for key in keys:
            value = obj.get(key)
            if isinstance(value, str) and value:
                return value
        for nested_key in ("image", "file", "photo"):
            nested = obj.get(nested_key)
            if isinstance(nested, dict):
                nested_url = _first_url(nested)
                if nested_url:
                    return nested_url
    return ""


def _is_https_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    return parsed.scheme == "https" and bool(parsed.netloc)


class MaxSupportClient:
    def __init__(self, token: str, *, api_base: str = MAX_API_BASE) -> None:
        self.token = str(token or "")
        self.api_base = api_base.rstrip("/")

    async def send_message(self, user_id: str, text: str) -> dict[str, Any]:
        url = f"{self.api_base}/messages"
        headers = {"Authorization": self.token}
        payload = {"text": str(text or "")}
        params = {"user_id": str(user_id)}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, params=params) as response:
                if response.status >= 400:
                    return {"ok": False, "status": response.status, "error": await response.text()}
                try:
                    data = await response.json()
                except Exception:
                    data = {"text": await response.text()}
                return {"ok": True, "status": response.status, "data": data}

    async def download_url(self, url: str) -> bytes:
        if not _is_https_url(url):
            raise ValueError("invalid_attachment_url")
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"Authorization": self.token}) as response:
                if response.status >= 400:
                    raise ValueError("attachment_download_failed")
                declared_length = response.headers.get("Content-Length")
                if declared_length and int(declared_length) > MAX_SUPPORT_ATTACHMENT_BYTES:
                    raise ValueError("attachment_too_large")
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.content.iter_chunked(64 * 1024):
                    total += len(chunk)
                    if total > MAX_SUPPORT_ATTACHMENT_BYTES:
                        raise ValueError("attachment_too_large")
                    chunks.append(bytes(chunk))
                return b"".join(chunks)

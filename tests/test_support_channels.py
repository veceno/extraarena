from __future__ import annotations

import json

import pytest

from support.channels.max import (
    MaxSupportClient,
    extract_max_image_attachments,
    normalize_max_message,
    verify_max_webhook_secret,
)
from support.channels.telegram import create_support_bot


def test_max_webhook_secret_uses_constant_time_comparison():
    assert verify_max_webhook_secret("secret", "secret") is True
    assert verify_max_webhook_secret("secret", "wrong") is False
    assert verify_max_webhook_secret("", "wrong") is False


def test_normalize_max_message_keeps_channel_ids_as_text():
    update = {
        "message": {
            "sender": {"user_id": 12345, "name": "Max Player"},
            "body": {"text": "Help"},
        }
    }

    normalized = normalize_max_message(update)

    assert normalized["channel"] == "max"
    assert normalized["channel_id"] == "12345"
    assert normalized["external_user_id"] == "max:12345"
    assert normalized["body"] == "Help"


def test_extract_max_image_attachments_accepts_https_payload_urls():
    update = {
        "message": {
            "body": {
                "attachments": [
                    {
                        "type": "image",
                        "payload": {
                            "url": "https://cdn.example.test/image.png",
                            "filename": "image.png",
                            "content_type": "image/png",
                        },
                    },
                    {"type": "file", "payload": {"url": "http://bad.example.test/file.txt"}},
                ]
            }
        }
    }

    attachments = extract_max_image_attachments(update)

    assert attachments == [
        {
            "url": "https://cdn.example.test/image.png",
            "filename": "image.png",
            "content_type": "image/png",
        }
    ]


@pytest.mark.asyncio
async def test_max_support_client_sends_authorization_header(monkeypatch):
    calls = []

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            return {"ok": True}

        async def text(self):
            return "ok"

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, url, *, headers=None, json=None, params=None):
            calls.append({"url": url, "headers": headers, "json": json, "params": params})
            return FakeResponse()

    monkeypatch.setattr("support.channels.max.aiohttp.ClientSession", lambda *args, **kwargs: FakeSession())
    client = MaxSupportClient("max-token")

    result = await client.send_message("123", "Hello")

    assert result["ok"] is True
    assert calls[0]["headers"]["Authorization"] == "max-token"
    assert calls[0]["params"] == {"user_id": "123"}
    assert calls[0]["json"]["text"] == "Hello"


def test_create_support_bot_returns_separate_dispatcher_without_game_handlers():
    bot, dispatcher = create_support_bot("123456:ABCDEF_support_token")

    assert bot is not None
    assert dispatcher is not None
    assert dispatcher.workflow_data.get("support_bot") is True

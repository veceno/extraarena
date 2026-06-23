from __future__ import annotations

import json
from io import BytesIO
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer
from aiohttp import FormData, web
from PIL import Image

from support.web import SUPPORT_ADMIN_COOKIE, register_support_routes


class FakeSupportService:
    def __init__(self):
        self.created = []
        self.admin_codes = {}
        self.consumed_codes = []
        self.tickets = [
            {"id": "ticket-ultra", "priority_score": 300, "priority_tier": "ultra"},
            {"id": "ticket-guest", "priority_score": 0, "priority_tier": "guest"},
        ]
        self.replies = []
        self.notes = []
        self.status_updates = []
        self.followups = []
        self.attachments = [
            {
                "id": "attachment-1",
                "ticket_id": "ticket-ultra",
                "message_id": "message-1",
                "storage_path": "/uploads/support/2026/06/a.webp",
                "original_filename": "a.png",
                "content_type": "image/webp",
            }
        ]

    async def create_ticket(self, **kwargs):
        self.created.append(kwargs)
        return {
            "ticket": {
                "id": "ticket-1",
                "status": "queued_guest",
                "priority_tier": "guest",
                "channel": kwargs.get("channel"),
                "channel_id": kwargs.get("channel_id"),
                "requester_identity_id": "identity-1",
            },
            "message": {"id": "message-1"},
            "identity": {"id": "identity-1"},
        }

    async def get_active_ticket_for_channel(self, *, channel, channel_id):
        for created in reversed(self.created):
            if created.get("channel") == channel and created.get("channel_id") == channel_id:
                return {
                    "id": "ticket-1",
                    "channel": channel,
                    "channel_id": channel_id,
                    "requester_identity_id": "identity-1",
                    "status": "queued_guest",
                }
        return None

    async def add_inbound_channel_message(self, *, ticket, body, metadata=None):
        self.followups.append({"ticket": ticket, "body": body, "metadata": metadata or {}})
        return {"id": "message-followup", "body": body}

    async def issue_admin_login_code(self, *, admin_channel_id, code, ttl_seconds=300, metadata=None):
        self.admin_codes[code] = {"admin_channel_id": str(admin_channel_id), "code": code}
        return self.admin_codes[code]

    async def consume_admin_login_code(self, code):
        self.consumed_codes.append(code)
        return self.admin_codes.pop(code, None)

    async def list_admin_tickets(self, **kwargs):
        return self.tickets

    async def list_ticket_messages(self, *, ticket_id, public=False):
        return [{"id": "message-1", "ticket_id": ticket_id, "body": "Hello", "direction": "inbound"}]

    async def list_ticket_attachments(self, *, ticket_id):
        return [attachment for attachment in self.attachments if attachment["ticket_id"] == ticket_id]

    async def get_ticket(self, *, ticket_id):
        for ticket in self.tickets:
            if ticket["id"] == ticket_id:
                return {
                    "id": ticket_id, "channel": "telegram", "channel_id": "123",
                    "requester_display_name": "Test User",
                    "requester_external_user_id": "tg:123",
                    "requester_game_user_id": 42,
                    "requester_channel": "telegram",
                    "requester_channel_id": "123",
                    "status": "open",
                }
        return {
            "id": ticket_id, "channel": "site", "channel_id": "site",
            "requester_display_name": "Site Guest",
            "requester_external_user_id": "",
            "requester_game_user_id": None,
            "requester_channel": "site",
            "requester_channel_id": "site",
            "status": "open",
        }

    async def add_user_message(self, *, ticket, body, metadata=None):
        self.followups.append({"ticket": ticket, "body": body, "metadata": metadata or {}})
        return {"id": "message-user-1", "body": body}

    async def list_ticket_attachments(self, *, ticket_id):
        return [a for a in self.attachments if a["ticket_id"] == ticket_id]

    async def list_public_ticket_attachments(self, *, ticket_id):
        return [a for a in self.attachments if a["ticket_id"] == ticket_id and a.get("message_id") != "message-internal"]

    async def record_attachment(self, **kwargs):
        return {"id": "attachment-1", **kwargs.get("metadata", {})}

    async def create_admin_reply(self, *, ticket, body, admin_channel_id):
        self.replies.append((ticket, body, admin_channel_id))
        return {"message": {"id": "reply-1"}, "outbox": {"id": "outbox-1"}}

    async def create_admin_note(self, *, ticket_id, body, admin_channel_id):
        self.notes.append((ticket_id, body, admin_channel_id))
        return {"id": "note-1"}

    async def update_ticket_status(self, *, ticket_id, status, admin_channel_id):
        self.status_updates.append((ticket_id, status, admin_channel_id))
        return {"id": ticket_id, "status": status}


@pytest.fixture
def app(tmp_path):
    service = FakeSupportService()
    application = web.Application()
    application["support_service"] = service
    application["support_admin_notifier"] = SimpleNamespace(sent=[])

    async def send_admin_code(code):
        application["support_admin_notifier"].sent.append(code)

    application["support_admin_notifier"].send_admin_code = send_admin_code
    register_support_routes(
        application,
        service,
        admin_secret="test-support-admin-secret",
        admin_channel_id="6803854304",
        max_webhook_secret="max-secret",
        upload_root=tmp_path / "uploads" / "support",
    )
    return application


@pytest.mark.asyncio
async def test_support_public_page_and_ticket_api_create_guest_ticket(app):
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        page = await client.get("/support")
        assert page.status == 200
        assert "ExtraArena Support" in await page.text()

        response = await client.post(
            "/api/support/tickets",
            json={
                "topic": "account",
                "body": "Cannot sign in",
                "display_name": "Guest",
                "guest": True,
            },
        )
        payload = await response.json()

        assert response.status == 200
        assert payload["status"] == "ok"
        assert payload["ticket_access_token"]
        assert app["support_service"].created[0]["account_scope"] == "guest"
        assert app["support_service"].created[0]["channel"] == "site"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_public_ticket_api_does_not_trust_raw_game_user_id(app):
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/support/tickets",
            json={
                "topic": "payments",
                "body": "Payment issue",
                "display_name": "Someone",
                "game_user_id": 6803854304,
            },
        )

        assert response.status == 200
        assert app["support_service"].created[-1]["game_user_id"] is None
        assert app["support_service"].created[-1]["account_scope"] == "unverified"
    finally:
        await client.close()



@pytest.mark.asyncio
async def test_support_admin_login_uses_one_time_code_cookie_without_returning_code(app, monkeypatch):
    monkeypatch.setattr("support.web.generate_support_code", lambda length=32: "admin-code")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        request = await client.post("/api/support/admin/login/request")
        body = await request.json()
        assert request.status == 200
        assert body == {"status": "ok"}
        assert app["support_admin_notifier"].sent == ["admin-code"]

        verify = await client.post("/api/support/admin/login/verify", json={"code": "admin-code"})
        assert verify.status == 200
        assert SUPPORT_ADMIN_COOKIE in verify.cookies

        cookie = verify.cookies[SUPPORT_ADMIN_COOKIE].value
        tickets = await client.get("/api/support/admin/tickets", cookies={SUPPORT_ADMIN_COOKIE: cookie})
        payload = await tickets.json()
        assert tickets.status == 200
        assert [ticket["id"] for ticket in payload["tickets"]] == ["ticket-ultra", "ticket-guest"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_support_admin_page_and_ticket_detail_api_return_chat_context(app, monkeypatch):
    monkeypatch.setattr("support.web.generate_support_code", lambda length=32: "admin-code")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        page = await client.get("/support/admin")
        page_text = await page.text()
        assert page.status == 200
        assert "ExtraArena Support" in page_text
        assert "/api/support/admin/tickets" in page_text

        await client.post("/api/support/admin/login/request")
        verify = await client.post("/api/support/admin/login/verify", json={"code": "admin-code"})
        cookie = verify.cookies[SUPPORT_ADMIN_COOKIE].value
        detail = await client.get(
            "/api/support/admin/tickets/ticket-ultra",
            cookies={SUPPORT_ADMIN_COOKIE: cookie},
        )
        payload = await detail.json()

        assert detail.status == 200
        assert payload["ticket"]["id"] == "ticket-ultra"
        assert payload["messages"][0]["body"] == "Hello"
        assert payload["attachments"][0]["storage_path"].endswith(".webp")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_support_admin_tickets_rejects_missing_cookie(app):
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get("/api/support/admin/tickets")
        assert response.status == 401
        assert json.loads(await response.text())["error"] == "support_admin_auth_required"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_support_admin_reply_note_and_status_require_session(app, monkeypatch):
    monkeypatch.setattr("support.web.generate_support_code", lambda length=32: "admin-code")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        await client.post("/api/support/admin/login/request")
        verify = await client.post("/api/support/admin/login/verify", json={"code": "admin-code"})
        cookie = verify.cookies[SUPPORT_ADMIN_COOKIE].value
        cookies = {SUPPORT_ADMIN_COOKIE: cookie}

        reply = await client.post(
            "/api/support/admin/tickets/ticket-ultra/reply",
            cookies=cookies,
            json={"body": "We are checking it"},
        )
        note = await client.post(
            "/api/support/admin/tickets/ticket-ultra/note",
            cookies=cookies,
            json={"body": "Internal note"},
        )
        status = await client.post(
            "/api/support/admin/tickets/ticket-ultra/status",
            cookies=cookies,
            json={"status": "pending_admin"},
        )

        assert reply.status == 200
        assert note.status == 200
        assert status.status == 200
        assert app["support_service"].replies[0][1] == "We are checking it"
        assert app["support_service"].notes[0][1] == "Internal note"
        assert app["support_service"].status_updates[0][1] == "pending_admin"
    finally:
        await client.close()


def _png_bytes():
    image = Image.new("RGB", (16, 12), color=(20, 120, 200))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_support_attachment_upload_compresses_and_static_serves_file(app):
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        form = FormData()
        form.add_field("ticket_id", "ticket-1")
        form.add_field("message_id", "message-1")
        form.add_field("access_token", "invalid")
        form.add_field("file", _png_bytes(), filename="screenshot.png", content_type="image/png")
        denied = await client.post("/api/support/attachments", data=form)
        assert denied.status == 401

        created = await client.post(
            "/api/support/tickets",
            json={"topic": "account", "body": "Cannot sign in", "guest": True},
        )
        token = (await created.json())["ticket_access_token"]
        form = FormData()
        form.add_field("ticket_id", "ticket-1")
        form.add_field("message_id", "message-1")
        form.add_field("access_token", token)
        form.add_field("file", _png_bytes(), filename="screenshot.png", content_type="image/png")
        response = await client.post("/api/support/attachments", data=form)
        payload = await response.json()

        assert response.status == 200
        assert payload["attachment"]["content_type"] == "image/webp"
        static = await client.get(payload["attachment"]["storage_path"])
        assert static.status == 200
        assert (await static.read())[:4] == b"RIFF"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_support_max_webhook_validates_secret_and_creates_ticket(app):
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        denied = await client.post("/api/support/max/webhook", headers={"X-Max-Bot-Api-Secret": "bad"}, json={})
        assert denied.status == 401

        menu = await client.post(
            "/api/support/max/webhook",
            headers={"X-Max-Bot-Api-Secret": "max-secret"},
            json={
                "message": {
                    "sender": {"user_id": "max-123", "name": "Max User"},
                    "body": {"text": "Need help"},
                }
            },
        )
        menu_payload = await menu.json()

        assert menu.status == 200
        assert menu_payload["status"] == "ok"
        assert "1. Аккаунт / вход" in menu_payload["reply"]

        await client.post(
            "/api/support/max/webhook",
            headers={"X-Max-Bot-Api-Secret": "max-secret"},
            json={"message": {"sender": {"user_id": "max-123", "name": "Max User"}, "body": {"text": "1"}}},
        )
        await client.post(
            "/api/support/max/webhook",
            headers={"X-Max-Bot-Api-Secret": "max-secret"},
            json={"message": {"sender": {"user_id": "max-123", "name": "Max User"}, "body": {"text": "3"}}},
        )
        created = await client.post(
            "/api/support/max/webhook",
            headers={"X-Max-Bot-Api-Secret": "max-secret"},
            json={
                "message": {
                    "sender": {"user_id": "max-123", "name": "Max User"},
                    "body": {"text": "Need help"},
                }
            },
        )
        payload = await created.json()

        assert created.status == 200
        assert "Обращение создано" in payload["reply"]
        assert app["support_service"].created[-1]["channel"] == "max"
        assert app["support_service"].created[-1]["channel_id"] == "max-123"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_support_ticket_messages_api_returns_site_chat_history(app):
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        created = await client.post(
            "/api/support/tickets",
            json={"topic": "account", "body": "Cannot sign in", "guest": True},
        )
        created_payload = await created.json()
        response = await client.get(
            "/api/support/tickets/ticket-1/messages",
            params={"access_token": created_payload["ticket_access_token"]},
        )
        payload = await response.json()

        assert response.status == 200
        assert payload["messages"][0]["body"] == "Hello"
        assert all(message.get("direction") != "internal" for message in payload["messages"])

        denied = await client.get("/api/support/tickets/ticket-1/messages")
        assert denied.status == 401
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_user_can_send_followup_message_to_existing_ticket(app):
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        created = await client.post(
            "/api/support/tickets",
            json={"topic": "account", "body": "Initial message", "guest": True},
        )
        token = (await created.json())["ticket_access_token"]

        followup = await client.post(
            "/api/support/tickets/ticket-1/messages",
            params={"access_token": token},
            json={"body": "Adding more details"},
        )
        payload = await followup.json()

        assert followup.status == 200
        assert payload["status"] == "ok"
        assert app["support_service"].followups[-1]["body"] == "Adding more details"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_user_message_rejects_empty_body(app):
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        created = await client.post(
            "/api/support/tickets",
            json={"topic": "account", "body": "Initial", "guest": True},
        )
        token = (await created.json())["ticket_access_token"]

        response = await client.post(
            "/api/support/tickets/ticket-1/messages",
            params={"access_token": token},
            json={"body": ""},
        )
        assert response.status == 400
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_user_message_rejects_invalid_token(app):
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/support/tickets/ticket-1/messages",
            params={"access_token": "invalid"},
            json={"body": "Hello"},
        )
        assert response.status == 401
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_ticket_attachments_endpoint_returns_attachments(app):
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        created = await client.post(
            "/api/support/tickets",
            json={"topic": "account", "body": "Initial", "guest": True},
        )
        token = (await created.json())["ticket_access_token"]

        response = await client.get(
            "/api/support/tickets/ticket-1/attachments",
            params={"access_token": token},
        )
        payload = await response.json()

        assert response.status == 200
        assert payload["status"] == "ok"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_admin_ticket_detail_includes_requester_identity(app, monkeypatch):
    monkeypatch.setattr("support.web.generate_support_code", lambda length=32: "admin-code")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        await client.post("/api/support/admin/login/request")
        verify = await client.post("/api/support/admin/login/verify", json={"code": "admin-code"})
        cookie = verify.cookies[SUPPORT_ADMIN_COOKIE].value

        detail = await client.get(
            "/api/support/admin/tickets/ticket-ultra",
            cookies={SUPPORT_ADMIN_COOKIE: cookie},
        )
        payload = await detail.json()

        assert detail.status == 200
        ticket = payload["ticket"]
        assert "requester_display_name" in ticket
        assert "requester_external_user_id" in ticket
        assert "requester_game_user_id" in ticket
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_user_message_works_with_bearer_header_token(app):
    """F1: SPA must be able to send follow-up with access_token via Bearer header."""
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        created = await client.post(
            "/api/support/tickets",
            json={"topic": "account", "body": "Initial", "guest": True},
        )
        token = (await created.json())["ticket_access_token"]

        response = await client.post(
            "/api/support/tickets/ticket-1/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"body": "Follow-up via Bearer header"},
        )
        payload = await response.json()

        assert response.status == 200
        assert payload["status"] == "ok"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_user_message_works_with_query_access_token(app):
    """F1: SPA must be able to send follow-up with access_token via query param."""
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        created = await client.post(
            "/api/support/tickets",
            json={"topic": "account", "body": "Initial", "guest": True},
        )
        token = (await created.json())["ticket_access_token"]

        response = await client.post(
            f"/api/support/tickets/ticket-1/messages?access_token={token}",
            json={"body": "Follow-up via query"},
        )
        payload = await response.json()

        assert response.status == 200
        assert payload["status"] == "ok"
        assert app["support_service"].followups[-1]["body"] == "Follow-up via query"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_public_attachments_exclude_internal_message_attachments(app):
    """F2: holder of ticket token must NOT see attachments linked to internal notes."""
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        created = await client.post(
            "/api/support/tickets",
            json={"topic": "account", "body": "Initial", "guest": True},
        )
        token = (await created.json())["ticket_access_token"]

        service = app["support_service"]
        public_attachments_before = list(service.attachments)
        service.attachments = [
            *public_attachments_before,
            {
                "id": "attachment-internal",
                "ticket_id": "ticket-1",
                "message_id": "message-internal",
                "storage_path": "/uploads/support/2026/06/secret.webp",
                "original_filename": "internal-screenshot.png",
                "content_type": "image/webp",
            },
        ]
        try:
            response = await client.get(
                "/api/support/tickets/ticket-1/attachments",
                params={"access_token": token},
            )
            payload = await response.json()

            assert response.status == 200
            attachment_ids = [a["id"] for a in payload["attachments"]]
            assert "attachment-internal" not in attachment_ids
        finally:
            service.attachments = public_attachments_before
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_upload_attachment_with_message_id_links_to_message(app):
    """F3: upload with message_id should link attachment to the user message."""
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        created = await client.post(
            "/api/support/tickets",
            json={"topic": "account", "body": "Initial", "guest": True},
        )
        token = (await created.json())["ticket_access_token"]

        form = FormData()
        form.add_field("ticket_id", "ticket-1")
        form.add_field("message_id", "message-user-1")
        form.add_field("access_token", token)
        form.add_field("file", _png_bytes(), filename="screenshot.png", content_type="image/png")
        response = await client.post("/api/support/attachments", data=form)
        payload = await response.json()

        assert response.status == 200
        assert payload["attachment"]["content_type"] == "image/webp"
    finally:
        await client.close()

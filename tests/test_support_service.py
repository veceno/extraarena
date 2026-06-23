from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from support.codes import hash_support_code, verify_support_code_hash
from support.constants import AccountScope, SupportChannel, SupportStatus, SupportTopic
from support.service import SupportService


class FakeSupportDB:
    def __init__(self):
        self.identities: list[dict] = []
        self.tickets: list[dict] = []
        self.messages: list[dict] = []
        self.outbox: list[dict] = []
        self.attachments: list[dict] = []
        self.auth_codes: list[dict] = []
        self.admin_codes: list[dict] = []
        self._next_id = 1

    def _id(self, prefix: str) -> str:
        value = f"{prefix}-{self._next_id}"
        self._next_id += 1
        return value

    @staticmethod
    def _json(value):
        if isinstance(value, str):
            return json.loads(value)
        return value

    async def fetchrow(self, query: str, *args):
        normalized = " ".join(query.lower().split())
        if "insert into support_identities" in normalized:
            assert "on conflict" in normalized
            row = {
                "id": self._id("identity"),
                "scope": args[0],
                "external_user_id": args[1],
                "game_user_id": args[2],
                "display_name": args[3],
                "channel": args[4],
                "channel_id": args[5],
                "is_verified": args[6],
                "metadata": self._json(args[7]),
            }
            self.identities.append(row)
            return row
        if "insert into support_tickets" in normalized:
            row = {
                "id": self._id("ticket"),
                "topic": args[0],
                "status": args[1],
                "account_scope": args[2],
                "requester_identity_id": args[3],
                "game_user_id": args[4],
                "channel": args[5],
                "channel_id": args[6],
                "subject": args[7],
                "priority_score": args[8],
                "priority_tier": args[9],
                "profile_snapshot": self._json(args[10]),
                "metadata": self._json(args[11]),
            }
            self.tickets.append(row)
            return row
        if "insert into support_messages" in normalized:
            row = {
                "id": self._id("message"),
                "ticket_id": args[0],
                "identity_id": args[1],
                "direction": args[2],
                "body": args[3],
                "metadata": self._json(args[4]),
            }
            self.messages.append(row)
            return row
        if "insert into support_delivery_outbox" in normalized:
            row = {
                "id": self._id("outbox"),
                "ticket_id": args[0],
                "message_id": args[1],
                "channel": args[2],
                "channel_id": args[3],
                "payload": self._json(args[4]),
                "status": "pending",
            }
            self.outbox.append(row)
            return row
        if "select * from support_tickets where id" in normalized:
            for ticket in self.tickets:
                if ticket["id"] == args[0]:
                    return ticket
            return None
        if "from support_tickets t" in normalized and "where t.id = $1" in normalized:
            for ticket in self.tickets:
                if ticket["id"] == args[0]:
                    return {**ticket, "requester_display_name": "", "requester_external_user_id": "", "requester_game_user_id": None, "requester_channel": ticket.get("channel", ""), "requester_channel_id": ticket.get("channel_id", "")}
            return None
        if "from support_tickets" in normalized and "status <> 'closed'" in normalized:
            channel, channel_id = args[0], args[1]
            for ticket in reversed(self.tickets):
                if ticket["channel"] == channel and ticket["channel_id"] == channel_id and ticket["status"] != "closed":
                    return ticket
            return None
        if "insert into support_attachments" in normalized:
            row = {
                "id": self._id("attachment"),
                "ticket_id": args[0],
                "message_id": args[1],
                "storage_path": args[3],
                "sha256": args[6],
            }
            self.attachments.append(row)
            return row
        if "insert into support_auth_codes" in normalized:
            row = {
                "id": self._id("auth-code"),
                "identity_id": args[0],
                "game_user_id": args[1],
                "purpose": args[2],
                "code_hash": args[3],
                "expires_at": args[4],
                "used_at": None,
                "metadata": self._json(args[5]),
            }
            self.auth_codes.append(row)
            return row
        if "insert into support_admin_login_codes" in normalized:
            row = {
                "id": self._id("admin-code"),
                "admin_channel_id": args[0],
                "code_hash": args[1],
                "expires_at": args[2],
                "used_at": None,
                "metadata": self._json(args[3]),
            }
            self.admin_codes.append(row)
            return row
        if "update support_auth_codes" in normalized:
            code_hash, purpose = args[0], args[1]
            for row in self.auth_codes:
                if row["code_hash"] == code_hash and row["purpose"] == purpose and row["used_at"] is None:
                    row["used_at"] = datetime.now(timezone.utc)
                    return row
            return None
        if "update support_admin_login_codes" in normalized:
            code_hash = args[0]
            for row in self.admin_codes:
                if row["code_hash"] == code_hash and row["used_at"] is None:
                    row["used_at"] = datetime.now(timezone.utc)
                    return row
            return None
        raise AssertionError(f"Unexpected fetchrow query: {query}")

    async def execute(self, query: str, *args):
        normalized = " ".join(query.lower().split())
        if "update support_tickets set status" in normalized:
            for ticket in self.tickets:
                if ticket["id"] == args[1]:
                    ticket["status"] = args[0]
        return "OK"

    async def fetch(self, query: str, *args):
        normalized = " ".join(query.lower().split())
        if "from support_messages" in normalized and "from support_tickets" not in normalized:
            assert "direction <> 'internal'" in normalized
            return [{"id": "message-1", "direction": "inbound"}]
        if "from support_attachments a" in normalized and "left join support_messages" in normalized:
            return [attachment for attachment in self.attachments if attachment["ticket_id"] == args[0]]
        if "from support_attachments" in normalized:
            return [attachment for attachment in self.attachments if attachment["ticket_id"] == args[0]]
        if "from support_tickets" in normalized:
            assert "latest_message_body" in normalized
            return [{**t, "requester_display_name": "", "requester_external_user_id": "", "requester_game_user_id": None, "requester_channel": t.get("channel", ""), "requester_channel_id": t.get("channel_id", "")} for t in self.tickets]
        raise AssertionError(f"Unexpected fetch query: {query}")


class FakeGameDB:
    def __init__(self, profiles: dict[int, dict]):
        self.profiles = profiles

    async def get_user_profile(self, user_id: int):
        return self.profiles.get(user_id)


@pytest.mark.asyncio
async def test_create_ticket_routes_guest_to_guest_queue_and_stores_text_channel_id():
    support_db = FakeSupportDB()
    service = SupportService(support_db=support_db, game_db=FakeGameDB({}))

    result = await service.create_ticket(
        topic=SupportTopic.ACCOUNT,
        channel=SupportChannel.TELEGRAM,
        channel_id=123456,
        external_user_id="tg:123456",
        display_name="Guest",
        body="I need help signing in",
    )

    assert result["ticket"]["account_scope"] == AccountScope.GUEST
    assert result["ticket"]["status"] == SupportStatus.QUEUED_GUEST
    assert result["ticket"]["priority_tier"] == "guest"
    assert support_db.identities[0]["channel_id"] == "123456"
    assert support_db.tickets[0]["channel_id"] == "123456"
    assert support_db.messages[0]["body"] == "I need help signing in"


@pytest.mark.asyncio
async def test_verified_ticket_requires_profile_with_at_least_100_trophies():
    low_support_db = FakeSupportDB()
    low_service = SupportService(
        support_db=low_support_db,
        game_db=FakeGameDB({77: {"user_id": 77, "trophies": 99, "extra_pass": "ultra"}}),
    )

    low_result = await low_service.create_ticket(
        topic=SupportTopic.TECHNICAL,
        channel=SupportChannel.MAX,
        channel_id="max-77",
        external_user_id="max:77",
        display_name="Low",
        game_user_id=77,
        body="Bug report",
    )

    assert low_result["ticket"]["account_scope"] == AccountScope.UNVERIFIED
    assert low_result["ticket"]["status"] == SupportStatus.QUEUED_UNVERIFIED
    assert low_result["ticket"]["priority_tier"] == "unverified"

    verified_support_db = FakeSupportDB()
    verified_service = SupportService(
        support_db=verified_support_db,
        game_db=FakeGameDB({88: {"user_id": 88, "trophies": 100, "extra_pass": "inactive"}}),
    )

    verified_result = await verified_service.create_ticket(
        topic=SupportTopic.TECHNICAL,
        channel=SupportChannel.MAX,
        channel_id="max-88",
        external_user_id="max:88",
        display_name="Verified",
        game_user_id=88,
        body="Bug report",
    )

    assert verified_result["ticket"]["account_scope"] == AccountScope.VERIFIED
    assert verified_result["ticket"]["status"] == SupportStatus.OPEN
    assert verified_result["ticket"]["priority_score"] > low_result["ticket"]["priority_score"]


@pytest.mark.asyncio
async def test_extra_pass_and_ultra_priority_ordering_uses_profile_snapshots():
    support_db = FakeSupportDB()
    service = SupportService(
        support_db=support_db,
        game_db=FakeGameDB(
            {
                1: {"user_id": 1, "trophies": 200, "extra_pass": "inactive"},
                2: {"user_id": 2, "trophies": 200, "extra_pass": "active"},
                3: {"user_id": 3, "trophies": 200, "extra_pass": "ultra"},
            }
        ),
    )

    regular = await service.create_ticket(
        topic=SupportTopic.PAYMENTS,
        channel=SupportChannel.TELEGRAM,
        channel_id="1",
        external_user_id="tg:1",
        display_name="Regular",
        game_user_id=1,
        body="Payment issue",
    )
    extra = await service.create_ticket(
        topic=SupportTopic.PAYMENTS,
        channel=SupportChannel.TELEGRAM,
        channel_id="2",
        external_user_id="tg:2",
        display_name="Extra",
        game_user_id=2,
        body="Payment issue",
    )
    ultra = await service.create_ticket(
        topic=SupportTopic.PAYMENTS,
        channel=SupportChannel.TELEGRAM,
        channel_id="3",
        external_user_id="tg:3",
        display_name="Ultra",
        game_user_id=3,
        body="Payment issue",
    )

    assert regular["ticket"]["profile_snapshot"]["trophies"] == 200
    assert extra["ticket"]["profile_snapshot"]["extra_pass"] == "active"
    assert ultra["ticket"]["profile_snapshot"]["extra_pass"] == "ultra"
    assert ultra["ticket"]["priority_score"] > extra["ticket"]["priority_score"] > regular["ticket"]["priority_score"]

    ordered = service.sort_tickets_for_inbox([regular["ticket"], ultra["ticket"], extra["ticket"]])

    assert [ticket["game_user_id"] for ticket in ordered] == [3, 2, 1]


@pytest.mark.asyncio
async def test_support_code_hashing_and_single_use_auth_admin_codes():
    support_db = FakeSupportDB()
    service = SupportService(support_db=support_db, game_db=FakeGameDB({}))
    code_hash = hash_support_code("123456")

    assert "123456" not in code_hash
    assert verify_support_code_hash("123456", code_hash) is True
    assert verify_support_code_hash("000000", code_hash) is False

    issued = await service.issue_auth_code(
        identity_id="identity-1",
        game_user_id=101,
        code="123456",
        ttl_seconds=600,
    )
    assert issued["code"] == "123456"
    assert support_db.auth_codes[0]["code_hash"] == code_hash

    consumed = await service.consume_auth_code("123456")
    consumed_again = await service.consume_auth_code("123456")

    assert consumed is not None
    assert consumed_again is None

    admin = await service.issue_admin_login_code(admin_channel_id=987654321, code="654321")
    assert admin["code"] == "654321"
    assert support_db.admin_codes[0]["admin_channel_id"] == "987654321"
    assert await service.consume_admin_login_code("654321") is not None
    assert await service.consume_admin_login_code("654321") is None


@pytest.mark.asyncio
async def test_admin_reply_creates_outbound_message_and_delivery_outbox():
    support_db = FakeSupportDB()
    service = SupportService(support_db=support_db, game_db=FakeGameDB({}))
    created = await service.create_ticket(
        topic=SupportTopic.ACCOUNT,
        channel=SupportChannel.TELEGRAM,
        channel_id="123",
        external_user_id="tg:123",
        display_name="Player",
        body="Help",
    )

    reply = await service.create_admin_reply(
        ticket=created["ticket"],
        body="We are checking it",
        admin_channel_id="6803854304",
    )

    assert reply["message"]["direction"] == "outbound"
    assert support_db.outbox[0]["channel"] == SupportChannel.TELEGRAM
    assert support_db.outbox[0]["channel_id"] == "123"
    assert support_db.outbox[0]["payload"]["text"] == "We are checking it"


@pytest.mark.asyncio
async def test_site_admin_reply_is_stored_without_unsupported_outbox():
    support_db = FakeSupportDB()
    service = SupportService(support_db=support_db, game_db=FakeGameDB({}))
    created = await service.create_ticket(
        topic=SupportTopic.ACCOUNT,
        channel=SupportChannel.SITE,
        channel_id="site-session",
        external_user_id="site:site-session",
        display_name="Site Player",
        body="Help",
    )

    reply = await service.create_admin_reply(
        ticket=created["ticket"],
        body="Open the site chat",
        admin_channel_id="6803854304",
    )

    assert reply["message"]["direction"] == "outbound"
    assert reply["outbox"] is None
    assert support_db.outbox == []


@pytest.mark.asyncio
async def test_admin_note_is_internal_and_status_update_is_audited():
    support_db = FakeSupportDB()
    service = SupportService(support_db=support_db, game_db=FakeGameDB({}))
    created = await service.create_ticket(
        topic=SupportTopic.ACCOUNT,
        channel=SupportChannel.TELEGRAM,
        channel_id="123",
        external_user_id="tg:123",
        display_name="Player",
        body="Help",
    )

    note = await service.create_admin_note(
        ticket_id=created["ticket"]["id"],
        body="Asked for screenshot",
        admin_channel_id="6803854304",
    )
    await service.update_ticket_status(
        ticket_id=created["ticket"]["id"],
        status=SupportStatus.PENDING_ADMIN,
        admin_channel_id="6803854304",
    )

    assert note["direction"] == "internal"
    assert support_db.tickets[0]["status"] == SupportStatus.PENDING_ADMIN


@pytest.mark.asyncio
async def test_get_ticket_loads_persisted_ticket_for_admin_actions():
    support_db = FakeSupportDB()
    service = SupportService(support_db=support_db, game_db=FakeGameDB({}))
    created = await service.create_ticket(
        topic=SupportTopic.ACCOUNT,
        channel=SupportChannel.TELEGRAM,
        channel_id="123",
        external_user_id="tg:123",
        display_name="Player",
        body="Help",
    )

    loaded = await service.get_ticket(ticket_id=created["ticket"]["id"])

    assert loaded["id"] == created["ticket"]["id"]


@pytest.mark.asyncio
async def test_record_attachment_links_file_to_ticket_and_message():
    support_db = FakeSupportDB()
    service = SupportService(support_db=support_db, game_db=FakeGameDB({}))

    attachment = await service.record_attachment(
        ticket_id="ticket-1",
        message_id="message-1",
        uploader_identity_id="identity-1",
        metadata={
            "storage_path": "/uploads/support/2026/06/a.webp",
            "original_filename": "a.png",
            "content_type": "image/webp",
            "sha256": "abc",
            "size_bytes": 100,
            "width": 10,
            "height": 8,
        },
    )

    assert attachment["ticket_id"] == "ticket-1"
    assert support_db.attachments[0]["message_id"] == "message-1"

    attachments = await service.list_ticket_attachments(ticket_id="ticket-1")

    assert attachments[0]["sha256"] == "abc"


@pytest.mark.asyncio
async def test_admin_ticket_list_includes_latest_message_columns_in_query():
    support_db = FakeSupportDB()
    service = SupportService(support_db=support_db, game_db=FakeGameDB({}))

    tickets = await service.list_admin_tickets()

    assert tickets == []


@pytest.mark.asyncio
async def test_public_ticket_messages_exclude_internal_notes():
    support_db = FakeSupportDB()
    service = SupportService(support_db=support_db, game_db=FakeGameDB({}))

    messages = await service.list_ticket_messages(ticket_id="ticket-1", public=True)

    assert messages == [{"id": "message-1", "direction": "inbound"}]


@pytest.mark.asyncio
async def test_active_channel_ticket_receives_followup_message():
    support_db = FakeSupportDB()
    service = SupportService(support_db=support_db, game_db=FakeGameDB({}))
    created = await service.create_ticket(
        topic=SupportTopic.TECHNICAL,
        channel=SupportChannel.TELEGRAM,
        channel_id="123",
        external_user_id="tg:123",
        display_name="Player",
        body="First message",
    )

    active = await service.get_active_ticket_for_channel(channel=SupportChannel.TELEGRAM, channel_id="123")
    message = await service.add_inbound_channel_message(ticket=active, body="More details")

    assert active["id"] == created["ticket"]["id"]
    assert message["ticket_id"] == created["ticket"]["id"]
    assert message["identity_id"] == created["ticket"]["requester_identity_id"]
    assert message["body"] == "More details"

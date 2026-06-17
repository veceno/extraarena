from __future__ import annotations

import pytest

from support.bot_client import SupportBotConversationManager, format_text_menu
from support.constants import AccountScope, SupportChannel, SupportTopic


class FakeBotSupportService:
    def __init__(self):
        self.created = []
        self.followups = []
        self.active_ticket = None

    async def get_active_ticket_for_channel(self, *, channel, channel_id):
        if self.active_ticket and self.active_ticket["channel"] == channel and self.active_ticket["channel_id"] == channel_id:
            return self.active_ticket
        return None

    async def create_ticket(self, **kwargs):
        self.created.append(kwargs)
        self.active_ticket = {
            "id": "ticket-1",
            "channel": kwargs["channel"],
            "channel_id": kwargs["channel_id"],
            "requester_identity_id": "identity-1",
        }
        return {"ticket": self.active_ticket, "message": {"id": "message-1"}}

    async def add_inbound_channel_message(self, *, ticket, body, metadata=None):
        self.followups.append({"ticket": ticket, "body": body, "metadata": metadata or {}})
        return {"id": "message-2", "body": body}


@pytest.mark.asyncio
async def test_bot_conversation_creates_ticket_after_topic_scope_and_body():
    service = FakeBotSupportService()
    manager = SupportBotConversationManager(service)

    start = await manager.begin(
        channel=SupportChannel.TELEGRAM,
        channel_id="123",
        display_name="Player",
        external_user_id="tg:123",
    )
    topic = await manager.select_topic(channel=SupportChannel.TELEGRAM, channel_id="123", topic=SupportTopic.PAYMENTS)
    scope = await manager.select_account_scope(
        channel=SupportChannel.TELEGRAM,
        channel_id="123",
        scope=AccountScope.OWN,
    )
    account = await manager.receive_text(
        channel=SupportChannel.TELEGRAM,
        channel_id="123",
        external_user_id="tg:123",
        display_name="Player",
        body="ExtraID-42",
    )
    created = await manager.receive_text(
        channel=SupportChannel.TELEGRAM,
        channel_id="123",
        external_user_id="tg:123",
        display_name="Player",
        body="Payment did not arrive",
    )

    assert start.choices[0].id == "topic:account"
    assert topic.choices[0].id == "scope:own"
    assert "игровой ID" in scope.text
    assert "опишите проблему" in account.text.lower()
    assert "Обращение создано" in created.text
    assert service.created[0]["topic"] == SupportTopic.PAYMENTS
    assert service.created[0]["account_scope"] == AccountScope.UNVERIFIED
    assert service.created[0]["metadata"]["claimed_account"] == "ExtraID-42"


@pytest.mark.asyncio
async def test_bot_conversation_appends_followups_to_active_ticket():
    service = FakeBotSupportService()
    service.active_ticket = {
        "id": "ticket-1",
        "channel": SupportChannel.MAX,
        "channel_id": "max-1",
        "requester_identity_id": "identity-1",
    }
    manager = SupportBotConversationManager(service)

    reply = await manager.receive_text(
        channel=SupportChannel.MAX,
        channel_id="max-1",
        external_user_id="max:max-1",
        display_name="Max Player",
        body="More details",
        metadata={"raw": {"message": 1}},
    )

    assert "добавлено" in reply.text
    assert service.followups[0]["ticket"]["id"] == "ticket-1"
    assert service.followups[0]["body"] == "More details"
    assert service.followups[0]["metadata"]["raw"] == {"message": 1}


def test_format_text_menu_adds_numbered_choices_for_max():
    service = FakeBotSupportService()
    manager = SupportBotConversationManager(service)

    rendered = format_text_menu(manager.topic_prompt())

    assert "1. Аккаунт / вход" in rendered
    assert "5. Другое" in rendered

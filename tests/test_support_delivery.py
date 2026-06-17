from __future__ import annotations

import pytest

from support.delivery import SupportDeliveryDispatcher


class FakeSupportDB:
    def __init__(self):
        self.rows = [
            {
                "id": "outbox-1",
                "channel": "telegram",
                "channel_id": "123",
                "payload": {"text": "Hello TG"},
            },
            {
                "id": "outbox-2",
                "channel": "max",
                "channel_id": "456",
                "payload": {"text": "Hello MAX"},
            },
        ]
        self.executed = []

    async def fetch(self, query, *args):
        assert "support_delivery_outbox" in query
        return list(self.rows)

    async def execute(self, query, *args):
        self.executed.append((query, args))
        return "OK"


class FakeTelegramBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text):
        self.sent.append((chat_id, text))


class FakeMaxClient:
    def __init__(self):
        self.sent = []

    async def send_message(self, user_id, text):
        self.sent.append((user_id, text))
        return {"ok": True}


@pytest.mark.asyncio
async def test_support_delivery_dispatcher_sends_pending_outbox_to_channel_clients():
    db = FakeSupportDB()
    tg = FakeTelegramBot()
    max_client = FakeMaxClient()
    dispatcher = SupportDeliveryDispatcher(support_db=db, telegram_bot=tg, max_client=max_client)

    delivered = await dispatcher.dispatch_once()

    assert delivered == 2
    assert tg.sent == [("123", "Hello TG")]
    assert max_client.sent == [("456", "Hello MAX")]
    assert "status = 'sent'" in db.executed[0][0]
    assert "status = 'sent'" in db.executed[1][0]

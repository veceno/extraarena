from __future__ import annotations

import pytest

from infrastructure.support_json_database import SupportJsonDatabase
from support.constants import SupportChannel, SupportTopic
from support.service import SupportService


@pytest.mark.asyncio
async def test_support_json_database_persists_ticket_message_and_attachment(tmp_path):
    path = tmp_path / "support.json"
    db = SupportJsonDatabase(path)
    await db.connect()
    service = SupportService(support_db=db)

    created = await service.create_ticket(
        topic=SupportTopic.TECHNICAL,
        channel=SupportChannel.TELEGRAM,
        channel_id="123",
        external_user_id="tg:123",
        display_name="Player",
        body="Bug",
    )
    await service.record_attachment(
        ticket_id=created["ticket"]["id"],
        message_id=created["message"]["id"],
        uploader_identity_id=created["identity"]["id"],
        metadata={
            "storage_path": "/uploads/support/2026/06/a.webp",
            "original_filename": "a.png",
            "content_type": "image/webp",
            "sha256": "abc",
            "size_bytes": 42,
            "width": 10,
            "height": 10,
        },
    )
    await db.disconnect()

    reopened = SupportJsonDatabase(path)
    await reopened.connect()
    reopened_service = SupportService(support_db=reopened)
    active = await reopened_service.get_active_ticket_for_channel(channel=SupportChannel.TELEGRAM, channel_id="123")
    messages = await reopened_service.list_ticket_messages(ticket_id=created["ticket"]["id"], public=True)

    assert active["id"] == created["ticket"]["id"]
    assert messages[0]["body"] == "Bug"
    assert path.exists()

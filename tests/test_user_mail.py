import json
from datetime import datetime, timezone

import pytest

from infrastructure.database import Database


class MailDBHarness(Database):
    def __init__(self):
        self._pool = object()
        self.rows = []
        self.fetchrow_result = None
        self.executed = None
        self.fetch_args = None

    async def fetchrow(self, query, *args):
        self.fetch_args = (query, args)
        return self.fetchrow_result

    async def fetch(self, query, *args):
        self.fetch_args = (query, args)
        return list(self.rows)

    async def execute(self, query, *args):
        self.executed = (query, args)


class LegacyContentMailDBHarness(MailDBHarness):
    async def _get_columns(self, table):
        assert table == "user_mail"
        return {"user_id", "sender", "subject", "text", "content", "category", "icon", "attachments"}


class NotificationMailHarness(MailDBHarness):
    def __init__(self, inserted=True):
        super().__init__()
        self.inserted = inserted
        self.mail_kwargs = None

    async def is_notification_enabled(self, user_id, category):
        return True

    async def fetchrow(self, query, *args):
        self.fetch_args = (query, args)
        return {"id": 1} if self.inserted else None

    async def create_mail(self, **kwargs):
        self.mail_kwargs = kwargs
        return {"success": True}


@pytest.mark.asyncio
async def test_unread_mail_count_reads_asyncpg_mapping_value():
    db = MailDBHarness()
    db.fetchrow_result = {"unread_count": "3"}

    assert await db.get_unread_mail_count(42) == 3


@pytest.mark.asyncio
async def test_user_mail_is_serialized_for_clients():
    db = MailDBHarness()
    created_at = datetime(2026, 5, 24, tzinfo=timezone.utc)
    db.rows = [
        {
            "id": 7,
            "user_id": 42,
            "sender": "Система",
            "subject": "Награда",
            "text": "Твои монеты уже здесь",
            "is_read": False,
            "category": "rewards",
            "icon": "🎁",
            "attachments": '{"coins": 100}',
            "created_at": created_at,
        }
    ]

    mails = await db.get_user_mail(42, category="rewards", unread_only=True, limit=10)

    assert mails == [
        {
            "id": 7,
            "mail_id": 7,
            "user_id": 42,
            "sender": "Система",
            "subject": "Награда",
            "text": "Твои монеты уже здесь",
            "content": "Твои монеты уже здесь",
            "body": "Твои монеты уже здесь",
            "is_read": False,
            "category": "rewards",
            "icon": "🎁",
            "attachments": {"coins": 100},
            "created_at": created_at,
        }
    ]
    query, args = db.fetch_args
    assert "is_read = FALSE" in query
    assert "category = $2" in query
    assert args == (42, "rewards", 10)


@pytest.mark.asyncio
async def test_create_mail_serializes_attachments_as_jsonb():
    db = MailDBHarness()

    result = await db.create_mail(
        user_id=42,
        subject="Покупка",
        content="Спасибо за поддержку",
        attachments={"gems": 25},
    )

    assert result == {"success": True}
    query, args = db.executed
    assert "INSERT INTO user_mail" in query
    assert json.loads(args[-1]) == {"gems": 25}


@pytest.mark.asyncio
async def test_create_mail_fills_legacy_content_column_when_present():
    db = LegacyContentMailDBHarness()

    result = await db.create_mail(user_id=42, subject="Письмо", text="Текст")

    assert result == {"success": True}
    query, args = db.executed
    assert "text, content" in query
    assert args[3] == "Текст"


@pytest.mark.asyncio
async def test_enqueue_notification_creates_in_game_mail():
    db = NotificationMailHarness(inserted=True)

    enqueued = await db.enqueue_notification(
        42,
        category="generator",
        event_type="generator_new_key",
        payload={"keys": 1, "section": "generator"},
        dedupe_key="generator:42:1",
    )

    assert enqueued is True
    assert db.mail_kwargs["user_id"] == 42
    assert db.mail_kwargs["subject"] == "Генератор ключей"
    assert db.mail_kwargs["category"] == "system"
    assert db.mail_kwargs["icon"] == "🔑"
    assert db.mail_kwargs["text"] == "Новый ключ готов! В генераторе уже 1 ключ(ей)."
    assert db.mail_kwargs["attachments"]["event_type"] == "generator_new_key"


@pytest.mark.asyncio
async def test_enqueue_notification_does_not_duplicate_mail_when_deduped():
    db = NotificationMailHarness(inserted=False)

    enqueued = await db.enqueue_notification(
        42,
        category="generator",
        event_type="generator_new_key",
        payload={"keys": 1},
        dedupe_key="generator:42:1",
    )

    assert enqueued is False
    assert db.mail_kwargs is None

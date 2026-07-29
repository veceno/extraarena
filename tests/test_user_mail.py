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
        return {
            "id": 1,
            "user_id": args[0],
            "category": args[1],
            "event_type": args[2],
            "payload": json.loads(args[3]),
            "dedupe_key": args[4],
            "returnclock_decision_id": args[5],
            "returnclock_delivery_id": args[6],
            "is_discretionary": args[7],
            "created": self.inserted,
        }

    async def create_returnclock_decision(self, user_id, **kwargs):
        return {"decision_id": kwargs["decision_id"]}

    async def update_returnclock_decision(
        self,
        user_id,
        decision_id,
        **kwargs,
    ):
        return {"decision_id": decision_id}

    async def create_mail(self, **kwargs):
        self.mail_kwargs = kwargs
        return {"success": True}


class WeeklyTokensNoticeHarness(MailDBHarness):
    def __init__(self, existing_mail=False, *, enqueue_result=True):
        super().__init__()
        self.existing_mail = existing_mail
        self.enqueue_result = enqueue_result
        self.notifications = []
        self.mail_kwargs = None

    async def fetchrow(self, query, *args):
        self.fetch_args = (query, args)
        if "FROM squad_cbrp_events" in query:
            return {
                "cbrp": 7,
                "personal_tokens": 3,
                "treasury_tokens": 2,
                "owner_tax_tokens": 1,
            }
        if "FROM user_mail" in query:
            return {"id": 9} if self.existing_mail else None
        return None

    async def enqueue_notification(self, user_id, *, category, event_type, payload=None, dedupe_key=None):
        self.notifications.append(
            {
                "user_id": user_id,
                "category": category,
                "event_type": event_type,
                "payload": payload or {},
                "dedupe_key": dedupe_key,
            }
        )
        return self.enqueue_result

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
    assert "attachments ? 'event_type'" in query
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
async def test_enqueue_notification_does_not_create_in_game_mail_for_regular_notifications():
    db = NotificationMailHarness(inserted=True)

    enqueued = await db.enqueue_notification(
        42,
        category="generator",
        event_type="generator_new_key",
        payload={"keys": 1, "section": "generator"},
        dedupe_key="generator:42:1",
    )

    assert enqueued is True
    assert db.mail_kwargs is None


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


@pytest.mark.asyncio
async def test_weekly_squad_tokens_notice_creates_visible_rewards_mail_without_event_type():
    db = WeeklyTokensNoticeHarness()

    result = await db._create_squad_weekly_tokens_notice(
        user_id=42,
        clan_id=10,
        source_id="weekly_trophy_delta:2026-W24:10:42",
        period_key="2026-W24",
        delta_trophies=250,
        cbrp=0,
        personal_tokens=0,
        treasury_tokens=0,
    )

    assert result == {"notification_enqueued": True, "mail_created": True}
    assert db.notifications[0]["category"] == "squad_weekly_tokens"
    assert db.notifications[0]["event_type"] == "squad_weekly_tokens"
    assert db.notifications[0]["dedupe_key"] == "squad_weekly_tokens:weekly_trophy_delta:2026-W24:10:42"
    assert db.mail_kwargs["subject"] == "Ты получил токены сквада!"
    assert db.mail_kwargs["category"] == "rewards"
    assert db.mail_kwargs["attachments"]["cbrp"] == 7
    assert db.mail_kwargs["attachments"]["personal_tokens"] == 3
    assert db.mail_kwargs["attachments"]["treasury_tokens"] == 2
    assert "event_type" not in db.mail_kwargs["attachments"]


@pytest.mark.asyncio
async def test_weekly_squad_tokens_notice_does_not_duplicate_existing_mail():
    db = WeeklyTokensNoticeHarness(existing_mail=True)

    result = await db._create_squad_weekly_tokens_notice(
        user_id=42,
        clan_id=10,
        source_id="weekly_trophy_delta:2026-W24:10:42",
        period_key="2026-W24",
        delta_trophies=250,
        cbrp=7,
        personal_tokens=3,
        treasury_tokens=2,
    )

    assert result == {"notification_enqueued": True, "mail_created": False}
    assert db.mail_kwargs is None


@pytest.mark.asyncio
async def test_weekly_squad_tokens_notice_respects_disabled_push_toggle_but_keeps_mail():
    db = WeeklyTokensNoticeHarness(enqueue_result=False)

    result = await db._create_squad_weekly_tokens_notice(
        user_id=42,
        clan_id=10,
        source_id="weekly_trophy_delta:2026-W24:10:42",
        period_key="2026-W24",
        delta_trophies=250,
        cbrp=7,
        personal_tokens=3,
        treasury_tokens=2,
    )

    assert result == {"notification_enqueued": False, "mail_created": True}
    assert db.notifications[0]["category"] == "squad_weekly_tokens"
    assert db.mail_kwargs["category"] == "rewards"

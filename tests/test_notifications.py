import random

import pytest

from infrastructure.config import DatabaseSettings
from infrastructure.database import Database
from infrastructure.notifications import (
    NOTIFICATION_DEFAULTS,
    NOTIFICATION_SETTING_BY_CATEGORY,
    REMINDER_DUSTY_WEIGHT,
    choose_reminder_payload,
    classify_generator_event,
    build_webapp_url,
    format_notification_message,
    format_telegram_notification_message,
    is_activity_suppressed_notification,
    is_discretionary_notification,
    notification_priority,
    notification_section,
    next_league_trophies,
    wins_to_next_case,
)


def test_generator_event_classification():
    assert classify_generator_event(stored_keys=0, new_keys=1, cap=2) == "generator_new_key"
    assert classify_generator_event(stored_keys=1, new_keys=1, cap=2) == "generator_full_on_new_key"
    assert classify_generator_event(stored_keys=0, new_keys=2, cap=2) == "generator_full_on_new_key"
    assert classify_generator_event(stored_keys=0, new_keys=3, cap=2) == "generator_full_blocked_key"
    assert classify_generator_event(stored_keys=2, new_keys=1, cap=2) == "generator_full_blocked_key"
    assert classify_generator_event(stored_keys=1, new_keys=0, cap=2) is None


def test_reminder_helpers_calculate_dynamic_numbers():
    assert next_league_trophies(280) == 20
    assert next_league_trophies(10000) is None
    assert wins_to_next_case(3, "active") == 1
    assert wins_to_next_case(0, "ultra") == 3


def test_reminder_excludes_squad_template_for_solo_player():
    rng = random.Random(7)
    seen = {
        choose_reminder_payload(
            {"trophies": 500, "wins_since_last_case": 1, "extra_pass": "inactive", "squad_id": 0},
            rng=rng,
        )["template"]
        for _ in range(50)
    }
    assert "squad_missed" not in seen


def test_reminder_allows_squad_template_for_squad_member():
    rng = random.Random(1)
    seen = {
        choose_reminder_payload(
            {"trophies": 500, "wins_since_last_case": 1, "extra_pass": "inactive", "squad_id": 10},
            rng=rng,
        )["template"]
        for _ in range(100)
    }
    assert "squad_missed" in seen


def test_dusty_reminder_has_lower_weight():
    assert REMINDER_DUSTY_WEIGHT == 1


def test_discretionary_budget_excludes_social_notifications_and_prioritizes_rewards():
    assert is_discretionary_notification("daily_rewards") is True
    assert is_discretionary_notification("generator") is True
    assert is_discretionary_notification("game_invites") is False
    assert is_discretionary_notification("friend_requests") is False
    assert is_activity_suppressed_notification("game_invites") is True
    assert is_activity_suppressed_notification("friend_requests") is True
    assert notification_priority("daily_rewards") > notification_priority("generator")
    assert notification_priority("generator") > notification_priority("shop")
    assert notification_priority("shop") > notification_priority("reminders")


@pytest.mark.asyncio
async def test_generator_notifications_fire_only_on_ready_and_full_transitions():
    class GeneratorDB(Database):
        def __init__(self):
            super().__init__(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
            self._pool = object()
            self.row = {
                "user_id": 42,
                "level": 1,
                "accumulated_keys": 0,
                "last_tick_at": None,
                "notified": False,
                "full_notified": False,
                "notification_cycle": 7,
                "extra_pass": None,
                "extra_pass_expires_at": None,
                "status": "active",
            }
            self.accumulated = 1
            self.enqueued = []

        async def fetch(self, query, *args):
            return [dict(self.row)]

        async def _compute_generator_accumulated(self, row):
            return self.accumulated, self.accumulated, 2, 3600

        async def enqueue_notification(self, user_id, **kwargs):
            self.enqueued.append((user_id, kwargs))
            return True

        async def execute(self, query, *args):
            if "UPDATE generator_state" in query:
                self.row["notified"] = True
                if args[1]:
                    self.row["full_notified"] = True

    db = GeneratorDB()

    assert [item["event_type"] for item in await db.check_generator_notifications()] == ["generator_new_key"]
    assert await db.check_generator_notifications() == []

    db.accumulated = 2
    assert [item["event_type"] for item in await db.check_generator_notifications()] == ["generator_full_on_new_key"]
    assert await db.check_generator_notifications() == []
    assert [kwargs["dedupe_key"] for _, kwargs in db.enqueued] == [
        "generator:42:7:ready",
        "generator:42:7:full",
    ]


@pytest.mark.asyncio
async def test_discretionary_enqueue_honors_shared_24_hour_cap():
    class BudgetDB(Database):
        def __init__(self):
            super().__init__(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
            self._pool = object()
            self.inserted = False

        async def is_notification_enabled(self, user_id, category):
            return True

        async def fetchrow(self, query, *args):
            if "AS sent_24h" in query:
                return {"sent_24h": 1, "sent_7d": 1, "pending_priority": 0}
            self.inserted = True
            return {"id": 1}

    db = BudgetDB()
    assert await db.enqueue_notification(
        42,
        category="generator",
        event_type="generator_new_key",
        payload={"keys": 1},
    ) is False
    assert db.inserted is False


@pytest.mark.asyncio
async def test_higher_priority_discretionary_notification_supersedes_pending_one():
    class PriorityDB(Database):
        def __init__(self):
            super().__init__(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
            self._pool = object()
            self.executed = []

        async def is_notification_enabled(self, user_id, category):
            return True

        async def fetchrow(self, query, *args):
            if "AS sent_24h" in query:
                return {"sent_24h": 0, "sent_7d": 0, "pending_priority": 100}
            return {
                "id": 9,
                "user_id": args[0],
                "category": args[1],
                "event_type": args[2],
                "payload": args[3],
                "dedupe_key": args[4],
                "returnclock_decision_id": args[5],
                "returnclock_delivery_id": args[6],
                "is_discretionary": args[7],
                "created": True,
            }

        async def execute(self, query, *args):
            self.executed.append((query, args))

        async def create_returnclock_decision(self, user_id, **kwargs):
            return {"decision_id": kwargs["decision_id"]}

        async def update_returnclock_decision(self, user_id, decision_id, **kwargs):
            return {"decision_id": decision_id, **kwargs}

    db = PriorityDB()
    assert await db.enqueue_notification(
        42,
        category="daily_rewards",
        event_type="daily_login_reward",
        payload={"available_at": "2026-05-25T10:00:00+00:00"},
    ) is True
    assert any(
        "superseded_by_higher_priority" in query and args[-1] == 400
        for query, args in db.executed
    )


@pytest.mark.asyncio
async def test_recent_activity_suppresses_discretionary_external_delivery():
    class ActivityDB(Database):
        def __init__(self):
            super().__init__(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
            self._pool = object()

        async def is_notification_enabled(self, user_id, category):
            return True

        async def fetchval(self, query, *args):
            assert "FROM user_sessions" in query
            return True

    db = ActivityDB()
    reason = await db.notification_cancellation_reason(
        {
            "id": 5,
            "user_id": 42,
            "category": "generator",
            "event_type": "generator_new_key",
            "payload": {"generator_notification_cycle": 3},
        }
    )
    assert reason == "recent_activity"


def test_social_notification_categories_have_settings_messages_and_sections():
    assert NOTIFICATION_SETTING_BY_CATEGORY["game_invites"] == "notif_game_invites"
    assert NOTIFICATION_SETTING_BY_CATEGORY["friend_requests"] == "notif_friend_requests"
    assert NOTIFICATION_SETTING_BY_CATEGORY["squad_weekly_tokens"] == "notif_squad_weekly_tokens"
    assert NOTIFICATION_DEFAULTS["notif_game_invites"] is True
    assert NOTIFICATION_DEFAULTS["notif_friend_requests"] is True
    assert NOTIFICATION_DEFAULTS["notif_squad_weekly_tokens"] is True
    assert notification_section("game_invites", {"invite_id": 77}) == "friends"
    assert notification_section("friend_requests", {"request_id": 12}) == "friends"
    assert notification_section("squad_weekly_tokens", {}) == "squads"
    assert "вызывает" in format_notification_message("friendly_battle_invite", {"from_name": "Alice"})
    assert "друз" in format_notification_message("friend_request_received", {"from_name": "Alice"})
    assert "Недельный рассчет" in format_notification_message("squad_weekly_tokens", {})


def test_telegram_notification_message_uses_html_formatting_where_configured():
    assert format_telegram_notification_message("generator_new_key", {"keys": 3}) == (
        "🔑 <b>Новый ключ уже готов!</b> - скорее открой кейс! В генераторе уже 3 ключ(ей)."
    )
    assert format_telegram_notification_message("generator_full_on_new_key", {"cap": 5}) == (
        "⚠️ <b>Генератор уже переполнен</b> - собери ключ и открой кейс, чтобы генератор заработал!"
    )


def test_plain_notification_message_does_not_include_telegram_html():
    assert format_notification_message("generator_new_key", {"keys": 3}) == (
        "🔑 Новый ключ уже готов! - скорее открой кейс! В генераторе уже 3 ключ(ей)."
    )
    assert "<b>" not in format_notification_message("generator_full_on_new_key", {"cap": 5})


def test_webapp_notification_url_preserves_returnclock_attribution():
    url = build_webapp_url(
        "https://example.test/play?lang=ru",
        section="arena",
        payload={
            "decision_id": "decision-1",
            "notification_id": 77,
            "delivery_id": "delivery-1",
        },
    )

    assert "section=arena" in url
    assert "rc_decision_id=decision-1" in url
    assert "notification_id=77" in url
    assert "delivery_id=delivery-1" in url
    assert "entrypoint=notification" in url


@pytest.mark.asyncio
async def test_update_user_settings_accepts_new_notification_flags():
    class FakeDB(Database):
        def __init__(self):
            super().__init__(DatabaseSettings("localhost", 5432, "user", "pass", "db"))
            self._pool = object()
            self.executed = None

        async def execute(self, query, *args):
            self.executed = (query, args)

    db = FakeDB()
    await db.update_user_settings(
        42,
        notif_shop=True,
        notif_reminders=False,
        notif_squad_member_role=False,
        notif_squad_weekly_tokens=False,
        notification_delivery_mode="app_only",
        social_disable_talkies=True,
        unknown_key=True,
    )

    query, args = db.executed
    assert "notif_shop" in query
    assert "notif_reminders" in query
    assert "notif_squad_member_role" in query
    assert "notif_squad_weekly_tokens" in query
    assert "notification_delivery_mode" in query
    assert "social_disable_talkies" in query
    assert "unknown_key" not in query
    assert args == (True, False, False, False, "app_only", True, 42)

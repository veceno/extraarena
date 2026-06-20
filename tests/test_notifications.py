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
    format_notification_message,
    format_telegram_notification_message,
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
        "<b>Новый ключ уже готов!</b> - скорее открой кейс! В генераторе уже 3 ключ(ей)."
    )
    assert format_telegram_notification_message("generator_full_on_new_key", {"cap": 5}) == (
        "<b>Генератор уже переполнен</b> - собери ключ и открой кейс, чтобы генератор заработал!"
    )


def test_plain_notification_message_does_not_include_telegram_html():
    assert format_notification_message("generator_new_key", {"keys": 3}) == (
        "Новый ключ уже готов! - скорее открой кейс! В генераторе уже 3 ключ(ей)."
    )
    assert "<b>" not in format_notification_message("generator_full_on_new_key", {"cap": 5})


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

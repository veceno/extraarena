from infrastructure.notifications import (
    NOTIFICATION_DEFAULTS,
    NOTIFICATION_SETTING_BY_CATEGORY,
    format_android_notification_title,
    format_notification_message,
    format_telegram_notification_message,
    notification_section,
)


def test_notification_daily_login_reward_message_android():
    msg = format_notification_message("daily_login_reward", {})
    assert "Забери свою награду за вход" in msg
    assert "уже доступна" in msg


def test_notification_daily_login_reward_message_telegram():
    msg = format_telegram_notification_message("daily_login_reward", {})
    assert "Забери свою награду за вход" in msg


def test_notification_daily_login_reward_title_android():
    title = format_android_notification_title("daily_rewards", "daily_login_reward", {})
    assert "Забери награду за вход" in title


def test_notification_daily_login_reward_custom_payload_title_wins():
    title = format_android_notification_title("daily_rewards", "daily_login_reward", {"title": "Моя награда"})
    assert title == "Моя награда"


def test_notification_daily_rewards_category_mapped_to_setting():
    assert NOTIFICATION_SETTING_BY_CATEGORY["daily_rewards"] == "notif_daily_rewards"
    assert NOTIFICATION_DEFAULTS["notif_daily_rewards"] is False
    assert notification_section("daily_rewards", {}) == "arena"


def test_notification_reminders_default_is_false():
    assert NOTIFICATION_DEFAULTS["notif_reminders"] is False

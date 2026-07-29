import base64
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from infrastructure.notifications import format_notification_message
from infrastructure.push_notifications import (
    build_android_push_payload,
    send_android_broadcast,
    _service_account_from_env,
)


def test_android_push_payload_reuses_telegram_notification_text():
    payload = build_android_push_payload("generator", "generator_new_key", {"keys": 2})

    assert payload.title == "🔑 Новый ключ готов!"
    assert payload.body == format_notification_message("generator_new_key", {"keys": 2})
    assert "<b>" not in payload.body
    assert payload.data["type"] == "game_notification"
    assert payload.data["section"] == "generator"
    assert payload.data["body"] == payload.body


def test_android_update_push_uses_required_app_update_contract():
    payload = build_android_push_payload("app_update", "app_update_required", {})

    assert payload.title == "⬇️ Хорошие новости!"
    assert payload.body == "⬇️ Вышло обновление, скачай новую версию, чтобы продолжить игру"
    assert payload.data["type"] == "app_update_required"
    assert payload.data["url"] == "https://t.me/extraarenamobile"
    assert payload.data["android_channel_id"] == "extraarena_updates"


def test_android_update_push_keeps_mobile_update_links_when_env_is_stale(monkeypatch):
    monkeypatch.setenv("EXTRAARENA_UPDATE_CHANNEL_URL", "https://t.me/extraarena")

    payload = build_android_push_payload("app_update", "app_update_required", {})

    assert payload.data["url"] == "https://t.me/extraarenamobile"
    assert payload.data["apk_url"] == "https://apk.laveqox.ru"


def test_android_friendly_invite_push_preserves_accept_metadata():
    payload = build_android_push_payload(
        "game_invites",
        "friendly_battle_invite",
        {"from_name": "Alice", "invite_id": 77, "invite_action": "accept"},
    )

    assert payload.data["section"] == "friends"
    assert payload.title == "⚔️ Alice вызвал тебя на бой!"
    assert payload.data["invite_id"] == "77"
    assert payload.data["invite_action"] == "accept"
    assert payload.data["event_type"] == "friendly_battle_invite"


def test_android_returnclock_push_preserves_attribution_metadata():
    payload = build_android_push_payload(
        "reminders",
        "daily_reminder",
        {
            "text": "Зайди сыграть",
            "section": "arena",
            "decision_id": "rcd-20260729-001",
            "notification_id": 9182,
            "delivery_id": "rc-delivery-9182-android",
        },
    )

    assert payload.data["entrypoint"] == "notification"
    assert payload.data["rc_decision_id"] == "rcd-20260729-001"
    assert payload.data["notification_id"] == "9182"
    assert payload.data["delivery_id"] == "rc-delivery-9182-android"


def test_android_returnclock_push_accepts_canonical_decision_id():
    payload = build_android_push_payload(
        "reminders",
        "daily_reminder",
        {
            "rc_decision_id": "rcd-canonical",
            "decision_id": "rcd-legacy",
            "notification_id": "9183",
        },
    )

    assert payload.data["rc_decision_id"] == "rcd-canonical"
    assert "decision_id" not in payload.data


def test_android_notification_attribution_reaches_launch_url_contract():
    service = Path(
        "android-app/app/src/main/java/ru/extraarena/app/ExtraArenaMessagingService.java"
    ).read_text(encoding="utf-8")
    activity = Path(
        "android-app/app/src/main/java/ru/extraarena/app/MainActivity.java"
    ).read_text(encoding="utf-8")

    assert 'data.get("rc_decision_id")' in service
    assert 'data.get("decision_id")' in service
    assert 'intent.putExtra("rc_decision_id", decisionId)' in service
    assert 'intent.putExtra("notification_id", outboxNotificationId)' in service
    assert 'intent.putExtra("delivery_id", deliveryId)' in service
    assert 'intent.putExtra("entrypoint", "notification")' in service
    assert 'getStringExtra("rc_decision_id")' in activity
    assert 'getStringExtra("notification_id")' in activity
    assert 'getStringExtra("delivery_id")' in activity
    assert 'query.put("rc_decision_id", decisionId)' in activity
    assert 'query.put("notification_id", outboxNotificationId)' in activity
    assert 'query.put("delivery_id", deliveryId)' in activity
    assert 'query.put("entrypoint", "notification")' in activity


def test_android_notification_attribution_is_consumed_once_without_dropping_invites():
    activity = Path(
        "android-app/app/src/main/java/ru/extraarena/app/MainActivity.java"
    ).read_text(encoding="utf-8")
    consume_block = activity.split(
        "private void consumeReturnClockAttribution",
        1,
    )[1].split(
        "private String buildLaunchUrl",
        1,
    )[0]
    probe_block = activity.split(
        "private void probeAndLoadArena",
        1,
    )[1].split(
        "private boolean isServerAvailable",
        1,
    )[0]

    assert "private void consumeReturnClockAttribution(Intent intent)" in activity
    assert 'intent.removeExtra("rc_decision_id")' in consume_block
    assert 'intent.removeExtra("decision_id")' in consume_block
    assert 'intent.removeExtra("notification_id")' in consume_block
    assert 'intent.removeExtra("delivery_id")' in consume_block
    assert '"notification".equals(intent.getStringExtra("entrypoint"))' in consume_block
    assert 'intent.removeExtra("entrypoint")' in consume_block
    assert "data.buildUpon().clearQuery()" in consume_block
    assert "data.getQueryParameterNames()" in consume_block
    assert "data.getQueryParameters(name)" in consume_block
    assert "sanitized.appendQueryParameter(name, value)" in consume_block
    assert '"section".equals(name)' not in consume_block
    assert '"invite_id".equals(name)' not in consume_block
    assert '"invite_action".equals(name)' not in consume_block
    assert "Intent sourceIntent" in probe_block
    assert "String launchUrl = buildLaunchUrl(" in probe_block
    assert "consumeReturnClockAttribution(sourceIntent);" in probe_block
    assert "webView.loadUrl(launchUrl);" in probe_block
    assert probe_block.index("String launchUrl = buildLaunchUrl(") < probe_block.index(
        "consumeReturnClockAttribution(sourceIntent);"
    )
    assert probe_block.index("consumeReturnClockAttribution(sourceIntent);") < probe_block.index(
        "webView.loadUrl(launchUrl);"
    )


def test_android_titles_are_event_specific():
    assert build_android_push_payload("generator", "generator_full_blocked_key", {}).title == "🚫 Генератор переполнен!"
    assert build_android_push_payload("shop", "shop_particles", {}).title == "✨ Новые частицы карт в магазине!"
    assert build_android_push_payload(
        "extra_arena_modifier",
        "extra_arena_modifier_changed",
        {"label": "Двойная энергия"},
    ).title == "🌀 Двойная энергия в ExtraArena!"
    assert build_android_push_payload(
        "friend_requests",
        "friend_request_received",
        {"from_name": "Alice"},
    ).title == "👋 Заявка в друзья"
    assert build_android_push_payload(
        "squad_new_member",
        "squad_new_member",
        {"squad_name": "Squad1"},
    ).title == "Squad1: 🎉 новый участник!"


def test_android_weekly_squad_tokens_push_uses_requested_copy():
    payload = build_android_push_payload("squad_weekly_tokens", "squad_weekly_tokens", {})

    assert payload.title == "🌟 Ты получил токены сквада!"
    assert payload.body == "🌟 Недельный рассчет твоего вклада в CBRP сквада завершен - ты получил токены. Скорее потрать их в магазине сквада!"
    assert payload.data["section"] == "squads"
    assert payload.data["event_type"] == "squad_weekly_tokens"


@pytest.mark.asyncio
async def test_outbox_delivery_prefers_android_push_without_telegram_when_device_registered():
    from main import _deliver_notification

    class FakeBot:
        def __init__(self):
            self.messages = []

        async def send_message(self, *, chat_id, text, parse_mode=None, reply_markup=None):
            self.messages.append({"chat_id": chat_id, "text": text, "parse_mode": parse_mode, "reply_markup": reply_markup})

    class FakeDb:
        def __init__(self):
            self.sent = []
            self.device_errors = []

        async def get_push_devices(self, user_id, *, platform="android"):
            assert user_id == 42
            assert platform == "android"
            return [{"token": "fcm-token"}]

        async def mark_push_device_error(self, token, error, *, permanent=False):
            self.device_errors.append((token, error, permanent))

        async def mark_notification_sent(self, notification_id):
            self.sent.append(notification_id)

        async def mark_notification_blocked(self, notification_id):
            raise AssertionError("notification should not be blocked")

        async def mark_notification_failed(self, notification_id):
            raise AssertionError("notification should not fail")

    class FakeSender:
        def __init__(self):
            self.calls = []

        async def send(self, *, token, title, body, data):
            self.calls.append({"token": token, "title": title, "body": body, "data": data})
            return type("Result", (), {"ok": True, "error": None, "permanent": False})()

    bot = FakeBot()
    db = FakeDb()
    sender = FakeSender()
    notif = {
        "id": 77,
        "user_id": 42,
        "category": "generator",
        "event_type": "generator_new_key",
        "payload": {"keys": 2},
    }

    await _deliver_notification(bot, db, "https://example.com/game", notif, push_sender=sender)

    assert sender.calls[0]["token"] == "fcm-token"
    assert sender.calls[0]["title"] == "🔑 Новый ключ готов!"
    assert sender.calls[0]["body"] == "🔑 Новый ключ уже готов! - скорее открой кейс! В генераторе уже 2 ключ(ей)."
    assert sender.calls[0]["data"]["section"] == "generator"
    assert bot.messages == []
    assert db.sent == [77]
    assert db.device_errors == []


@pytest.mark.asyncio
async def test_returnclock_android_delivery_records_provider_event_and_status():
    from main import _deliver_notification

    class FakeBot:
        async def send_message(self, **kwargs):
            raise AssertionError("successful Android delivery must not fall back")

    class FakeDb:
        def __init__(self):
            self.events = []
            self.decisions = []
            self.sent = []

        async def get_push_devices(self, user_id, *, platform="android"):
            return [{"id": 9, "token": "fcm-token"}]

        async def record_returnclock_delivery_event(
            self,
            user_id,
            decision_id,
            **kwargs,
        ):
            self.events.append((user_id, decision_id, kwargs))

        async def update_returnclock_decision(self, user_id, decision_id, **kwargs):
            self.decisions.append((user_id, decision_id, kwargs))

        async def mark_notification_sent(self, notification_id):
            self.sent.append(notification_id)

    class FakeSender:
        async def send(self, *, token, title, body, data):
            assert data["rc_decision_id"] == "decision-77"
            assert data["notification_id"] == "77"
            assert data["delivery_id"] == "delivery-77"
            return type(
                "Result",
                (),
                {
                    "ok": True,
                    "message_id": "fcm-message-77",
                    "error": None,
                    "permanent": False,
                },
            )()

    db = FakeDb()
    await _deliver_notification(
        FakeBot(),
        db,
        "https://example.com/game",
        {
            "id": 77,
            "user_id": 42,
            "category": "reminders",
            "event_type": "daily_reminder",
            "payload": {},
            "returnclock_decision_id": "decision-77",
            "returnclock_delivery_id": "delivery-77",
            "attempts": 1,
        },
        push_sender=FakeSender(),
    )

    assert db.sent == [77]
    assert len(db.events) == 1
    event_user, event_decision, event = db.events[0]
    assert (event_user, event_decision) == (42, "decision-77")
    assert event["event_type"] == "provider_accepted"
    assert event["delivery_id"] == "delivery-77"
    assert event["provider_message_id"] == "fcm-message-77"
    assert db.decisions == [
        (42, "decision-77", {"status": "sent", "outbox_id": 77})
    ]


@pytest.mark.asyncio
async def test_outbox_delivery_falls_back_to_telegram_when_no_android_device():
    from main import _deliver_notification

    class FakeBot:
        def __init__(self):
            self.messages = []

        async def send_message(self, *, chat_id, text, parse_mode=None, reply_markup=None):
            self.messages.append({"chat_id": chat_id, "text": text, "parse_mode": parse_mode, "reply_markup": reply_markup})

    class FakeDb:
        def __init__(self):
            self.sent = []

        async def get_push_devices(self, user_id, *, platform="android"):
            return []

        async def mark_notification_sent(self, notification_id):
            self.sent.append(notification_id)

        async def mark_notification_blocked(self, notification_id):
            raise AssertionError("notification should not be blocked")

        async def mark_notification_failed(self, notification_id):
            raise AssertionError("notification should not fail")

    class FakeSender:
        async def send(self, *, token, title, body, data):
            raise AssertionError("push should not be sent without devices")

    bot = FakeBot()
    db = FakeDb()
    notif = {
        "id": 78,
        "user_id": 42,
        "category": "generator",
        "event_type": "generator_new_key",
        "payload": {"keys": 1},
    }

    await _deliver_notification(bot, db, "https://example.com/game", notif, push_sender=FakeSender())

    assert bot.messages[0]["text"] == "🔑 <b>Новый ключ уже готов!</b> - скорее открой кейс! В генераторе уже 1 ключ(ей)."
    assert bot.messages[0]["parse_mode"] == "HTML"
    assert db.sent == [78]


@pytest.mark.asyncio
async def test_friendly_invite_telegram_fallback_uses_accept_deep_link():
    from main import _deliver_notification

    class FakeBot:
        def __init__(self):
            self.messages = []

        async def send_message(self, *, chat_id, text, parse_mode=None, reply_markup=None):
            self.messages.append({"chat_id": chat_id, "text": text, "parse_mode": parse_mode, "reply_markup": reply_markup})

    class FakeDb:
        def __init__(self):
            self.sent = []

        async def get_push_devices(self, user_id, *, platform="android"):
            return []

        async def mark_notification_sent(self, notification_id):
            self.sent.append(notification_id)

        async def mark_notification_blocked(self, notification_id):
            raise AssertionError("notification should not be blocked")

        async def mark_notification_failed(self, notification_id):
            raise AssertionError("notification should not fail")

    class FakeSender:
        async def send(self, *, token, title, body, data):
            raise AssertionError("push should not be sent without devices")

    bot = FakeBot()
    db = FakeDb()
    notif = {
        "id": 81,
        "user_id": 42,
        "category": "game_invites",
        "event_type": "friendly_battle_invite",
        "payload": {"from_name": "Alice", "invite_id": 77, "invite_action": "accept", "section": "friends"},
    }

    await _deliver_notification(bot, db, "https://example.com/game?foo=bar", notif, push_sender=FakeSender())

    button = bot.messages[0]["reply_markup"].inline_keyboard[0][0]
    query = parse_qs(urlparse(button.web_app.url).query)
    assert query["foo"] == ["bar"]
    assert query["section"] == ["friends"]
    assert query["invite_id"] == ["77"]
    assert query["invite_action"] == ["accept"]
    assert query["notification_id"] == ["81"]
    assert query["entrypoint"] == ["notification"]
    assert "Alice" in bot.messages[0]["text"]
    assert db.sent == [81]


@pytest.mark.asyncio
async def test_outbox_delivery_respects_telegram_only_mode():
    from main import _deliver_notification

    class FakeBot:
        def __init__(self):
            self.messages = []

        async def send_message(self, *, chat_id, text, parse_mode=None, reply_markup=None):
            self.messages.append({"chat_id": chat_id, "text": text, "parse_mode": parse_mode, "reply_markup": reply_markup})

    class FakeDb:
        def __init__(self):
            self.sent = []

        async def get_notification_delivery_mode(self, user_id):
            return "telegram_only"

        async def get_push_devices(self, user_id, *, platform="android"):
            raise AssertionError("telegram_only must not query Android devices")

        async def mark_notification_sent(self, notification_id):
            self.sent.append(notification_id)

        async def mark_notification_blocked(self, notification_id):
            raise AssertionError("notification should not be blocked")

        async def mark_notification_failed(self, notification_id):
            raise AssertionError("notification should not fail")

    class FakeSender:
        async def send(self, *, token, title, body, data):
            raise AssertionError("telegram_only must not send push")

    bot = FakeBot()
    db = FakeDb()
    notif = {
        "id": 79,
        "user_id": 42,
        "category": "generator",
        "event_type": "generator_new_key",
        "payload": {"keys": 1},
    }

    await _deliver_notification(bot, db, "https://example.com/game", notif, push_sender=FakeSender())

    assert bot.messages[0]["text"] == "🔑 <b>Новый ключ уже готов!</b> - скорее открой кейс! В генераторе уже 1 ключ(ей)."
    assert bot.messages[0]["parse_mode"] == "HTML"
    assert db.sent == [79]


@pytest.mark.asyncio
async def test_outbox_delivery_respects_app_only_without_telegram_fallback():
    from main import _deliver_notification

    class FakeBot:
        async def send_message(self, *, chat_id, text, reply_markup):
            raise AssertionError("app_only must not fall back to Telegram")

    class FakeDb:
        def __init__(self):
            self.failed = []

        async def get_notification_delivery_mode(self, user_id):
            return "app_only"

        async def get_push_devices(self, user_id, *, platform="android"):
            return []

        async def mark_notification_sent(self, notification_id):
            raise AssertionError("notification should not be marked sent")

        async def mark_notification_blocked(self, notification_id):
            raise AssertionError("notification should not be blocked")

        async def mark_notification_failed(self, notification_id):
            self.failed.append(notification_id)

    class FakeSender:
        async def send(self, *, token, title, body, data):
            raise AssertionError("push should not be sent without devices")

    db = FakeDb()
    notif = {
        "id": 80,
        "user_id": 42,
        "category": "generator",
        "event_type": "generator_new_key",
        "payload": {"keys": 1},
    }

    await _deliver_notification(FakeBot(), db, "https://example.com/game", notif, push_sender=FakeSender())

    assert db.failed == [80]


@pytest.mark.asyncio
async def test_app_reminder_push_is_postponed_to_morning_during_device_quiet_hours(monkeypatch):
    import main
    from main import _deliver_notification

    monkeypatch.setattr(
        main,
        "_notification_utc_now",
        lambda: datetime(2026, 5, 25, 23, 15, tzinfo=timezone.utc),
    )

    class FakeBot:
        async def send_message(self, *, chat_id, text, reply_markup):
            raise AssertionError("app_only quiet reminder must not fall back to Telegram")

    class FakeDb:
        def __init__(self):
            self.sent = []
            self.deferred = []

        async def get_notification_delivery_mode(self, user_id):
            return "app_only"

        async def get_push_devices(self, user_id, *, platform="android"):
            return [{
                "token": "sleeping-token",
                "timezone": "Europe/Moscow",
                "utc_offset_minutes": 180,
            }]

        async def mark_notification_sent(self, notification_id):
            raise AssertionError("quiet reminder should be postponed, not marked sent")

        async def mark_notification_blocked(self, notification_id):
            raise AssertionError("quiet reminder should be handled, not blocked")

        async def mark_notification_failed(self, notification_id):
            raise AssertionError("quiet reminder should be handled, not failed")

        async def postpone_notification(self, notification_id, not_before_at):
            self.deferred.append((notification_id, not_before_at))

    class FakeSender:
        async def send(self, *, token, title, body, data):
            raise AssertionError("reminder push must not be sent at 02:15 local time")

    db = FakeDb()
    notif = {
        "id": 81,
        "user_id": 42,
        "category": "reminders",
        "event_type": "daily_reminder",
        "payload": {"section": "arena", "text": "Зайди сыграть"},
    }

    await _deliver_notification(FakeBot(), db, "https://example.com/game", notif, push_sender=FakeSender())

    assert db.sent == []
    assert db.deferred == [(81, datetime(2026, 5, 26, 6, 0, tzinfo=timezone.utc))]


@pytest.mark.asyncio
async def test_android_broadcast_marks_permanent_token_errors():
    class FakeDb:
        def __init__(self):
            self.errors = []

        async def get_push_devices_for_broadcast(self, *, platform="android", limit=10000):
            assert platform == "android"
            assert limit == 10000
            return [{"token": "ok-token"}, {"token": "bad-token"}]

        async def mark_push_device_error(self, token, error, *, permanent=False):
            self.errors.append((token, error, permanent))

    class FakeSender:
        async def send(self, *, token, title, body, data):
            if token == "ok-token":
                return type("Result", (), {"ok": True, "error": None, "permanent": False})()
            return type("Result", (), {"ok": False, "error": "unregistered", "permanent": True})()

    db = FakeDb()
    payload = build_android_push_payload("app_update", "app_update_required", {})
    result = await send_android_broadcast(db=db, push_sender=FakeSender(), payload=payload)

    assert result.total == 2
    assert result.sent == 1
    assert result.failed == 1
    assert db.errors == [("bad-token", "unregistered", True)]


@pytest.mark.asyncio
async def test_android_broadcast_records_per_user_returnclock_envelopes_and_device_events():
    class FakeDb:
        def __init__(self):
            self.decisions = []
            self.events = []
            self.updates = []
            self.errors = []

        async def get_push_devices_for_broadcast(self, *, platform="android", limit=10000):
            return [
                {"id": 11, "user_id": 42, "token": "user-42-ok"},
                {"id": 12, "user_id": 42, "token": "user-42-failed"},
                {"id": 21, "user_id": 99, "token": "user-99-failed"},
            ]

        async def create_returnclock_decision(self, user_id, **kwargs):
            self.decisions.append((user_id, kwargs))

        async def record_returnclock_delivery_event(
            self,
            user_id,
            decision_id,
            **kwargs,
        ):
            self.events.append((user_id, decision_id, kwargs))

        async def update_returnclock_decision(self, user_id, decision_id, **kwargs):
            self.updates.append((user_id, decision_id, kwargs))

        async def mark_push_device_error(self, token, error, *, permanent=False):
            self.errors.append((token, error, permanent))

    class FakeSender:
        def __init__(self):
            self.calls = []

        async def send(self, *, token, title, body, data):
            self.calls.append({"token": token, "data": data})
            if token == "user-42-ok":
                return type(
                    "Result",
                    (),
                    {
                        "ok": True,
                        "message_id": "fcm-accepted-42",
                        "error": None,
                        "permanent": False,
                    },
                )()
            return type(
                "Result",
                (),
                {
                    "ok": False,
                    "message_id": None,
                    "error": "unregistered",
                    "permanent": True,
                },
            )()

    db = FakeDb()
    sender = FakeSender()
    payload = build_android_push_payload("app_update", "app_update_required", {})

    result = await send_android_broadcast(db=db, push_sender=sender, payload=payload)

    assert result.total == 3
    assert result.sent == 1
    assert result.failed == 2
    assert len(db.decisions) == 2
    decisions = {user_id: kwargs for user_id, kwargs in db.decisions}
    decision_42 = decisions[42]["decision_id"]
    decision_99 = decisions[99]["decision_id"]
    assert decision_42.rsplit(":", 1)[0] == decision_99.rsplit(":", 1)[0]
    for decision in decisions.values():
        assert decision["decision"] == "send"
        assert decision["schedule_type"] == "broadcast"
        assert decision["decision_source"] == "admin_broadcast"
        assert decision["policy_version"] == "manual-broadcast-v1"
        assert decision["treatment_arm"] == "observational"
        assert decision["assignment_probability"] == 1.0

    sent_data = {call["token"]: call["data"] for call in sender.calls}
    assert sent_data["user-42-ok"]["rc_decision_id"] == decision_42
    assert sent_data["user-42-failed"]["rc_decision_id"] == decision_42
    assert sent_data["user-99-failed"]["rc_decision_id"] == decision_99
    assert all(call["data"]["entrypoint"] == "notification" for call in sender.calls)
    delivery_ids = {call["data"]["delivery_id"] for call in sender.calls}
    assert len(delivery_ids) == 3
    assert "rc_decision_id" not in payload.data
    assert "delivery_id" not in payload.data

    assert len(db.events) == 3
    events = {(user_id, event["event_type"]): event for user_id, _, event in db.events}
    assert events[(42, "provider_accepted")]["provider_message_id"] == "fcm-accepted-42"
    assert events[(42, "provider_accepted")]["channel"] == "android"
    assert events[(42, "provider_failed")]["metadata"]["device_id"] == 12
    assert events[(99, "provider_failed")]["metadata"]["device_id"] == 21
    assert {
        event["delivery_id"]
        for _, _, event in db.events
    } == delivery_ids

    updates = {user_id: kwargs for user_id, _, kwargs in db.updates}
    assert updates[42] == {"status": "sent"}
    assert updates[99] == {"status": "failed"}
    assert db.errors == [
        ("user-42-failed", "unregistered", True),
        ("user-99-failed", "unregistered", True),
    ]


@pytest.mark.asyncio
async def test_android_broadcast_delivery_survives_returnclock_telemetry_failure():
    class FakeDb:
        async def get_push_devices_for_broadcast(self, *, platform="android", limit=10000):
            return [{"id": 11, "user_id": 42, "token": "working-token"}]

        async def create_returnclock_decision(self, user_id, **kwargs):
            raise RuntimeError("telemetry unavailable")

        async def record_returnclock_delivery_event(self, user_id, decision_id, **kwargs):
            raise RuntimeError("telemetry unavailable")

        async def update_returnclock_decision(self, user_id, decision_id, **kwargs):
            raise RuntimeError("telemetry unavailable")

    class FakeSender:
        async def send(self, *, token, title, body, data):
            assert data["rc_decision_id"].startswith("android-broadcast:")
            assert data["delivery_id"]
            assert data["entrypoint"] == "notification"
            return type(
                "Result",
                (),
                {
                    "ok": True,
                    "message_id": "accepted-despite-telemetry",
                    "error": None,
                    "permanent": False,
                },
            )()

    result = await send_android_broadcast(
        db=FakeDb(),
        push_sender=FakeSender(),
        payload=build_android_push_payload("app_update", "app_update_required", {}),
    )

    assert result.total == 1
    assert result.sent == 1
    assert result.failed == 0


def test_service_account_can_be_loaded_from_base64_env(monkeypatch):
    monkeypatch.delenv("FIREBASE_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.delenv("FIREBASE_SERVICE_ACCOUNT_FILE", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    data = {"type": "service_account", "project_id": "extraarena-94cd6"}
    encoded = base64.b64encode(json.dumps(data).encode("utf-8")).decode("ascii")
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT_B64", encoded)

    loaded, path = _service_account_from_env()

    assert loaded == data
    assert path is None

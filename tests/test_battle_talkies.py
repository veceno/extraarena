import pytest

from battle_engine import BattleEngine


def _engine(extra_pass="inactive"):
    engine = BattleEngine(match_id="m-talkie", player_ids=[101, 202])
    engine.current_player_id = 101
    engine.turn = 7
    engine._p1_extra_pass = extra_pass
    engine._p2_extra_pass = "inactive"
    return engine


def test_f2p_can_send_one_talkie_per_own_turn():
    engine = _engine("inactive")

    first = engine.register_talkie(101, "5", now=10.0)
    second = engine.register_talkie(101, "7", now=10.2)

    assert first["success"] is True
    assert isinstance(first["event"]["event_id"], str)
    assert first["event"]["event_id"] == "m-talkie:7:101:1"
    assert first["event"]["match_id"] == "m-talkie"
    assert first["event"]["sender_id"] == 101
    assert first["event"]["turn"] == 7
    assert first["event"]["talkie_id"] == "5"
    assert first["event"]["sound"] == "happy"
    assert first["event"]["remaining"] == 0
    assert second["success"] is False
    assert second["error"] == "talkie_limit_reached"
    assert second["remaining"] == 0


def test_extra_pass_limit_and_cooldown_reset_next_turn():
    engine = _engine("active")

    assert engine.register_talkie(101, "1", now=10.0)["success"] is True
    cooldown = engine.register_talkie(101, "2", now=10.5)
    assert cooldown["success"] is False
    assert cooldown["error"] == "talkie_cooldown"
    assert cooldown["retry_after"] == 0.5
    assert cooldown["remaining"] == 1

    assert engine.register_talkie(101, "2", now=11.0)["success"] is True
    limit = engine.register_talkie(101, "3", now=12.1)
    assert limit["success"] is False
    assert limit["error"] == "talkie_limit_reached"
    assert limit["remaining"] == 0

    engine.turn = 8
    assert engine.register_talkie(101, "3", now=13.2)["success"] is True


def test_ultra_can_send_three_per_turn():
    engine = _engine("ultra")

    assert engine.register_talkie(101, "1", now=10.0)["success"] is True
    assert engine.register_talkie(101, "2", now=11.0)["success"] is True
    assert engine.register_talkie(101, "3", now=12.0)["success"] is True
    assert engine.register_talkie(101, "4", now=13.0)["error"] == "talkie_limit_reached"


def test_talkie_requires_current_turn_and_valid_id():
    engine = _engine("ultra")

    assert engine.register_talkie(202, "1", now=10.0)["error"] == "not_your_turn"
    assert engine.register_talkie(101, "999", now=10.0)["error"] == "invalid_talkie"

    engine.set_talkie_enabled(101, False)
    assert engine.register_talkie(101, "999", now=10.0)["error"] == "invalid_talkie"


def test_talkie_toggle_blocks_send_and_receive():
    engine = _engine("ultra")

    engine.set_talkie_enabled(101, False)

    assert engine.talkie_enabled_for(101) is False
    assert engine.should_deliver_talkie_to(101) is False
    assert engine.register_talkie(101, "5", now=10.0)["error"] == "talkie_disabled"

    engine.set_talkie_enabled(101, True)
    assert engine.register_talkie(101, "5", now=10.0)["success"] is True


def test_talkie_uses_monotonic_when_now_is_omitted(monkeypatch):
    engine = _engine("active")

    monkeypatch.setattr("battle_engine.time.monotonic", lambda: 42.25)

    result = engine.register_talkie(101, "6")

    assert result["success"] is True
    assert engine._talkie_usage_by_turn[(101, 7)] == [42.25]

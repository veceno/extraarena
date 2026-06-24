"""Тесты log_schema: BATTLE_LOG_VERSION / MANIFEST_VERSION / валидация."""
from __future__ import annotations

from rlhf_env.components.log_schema import (
    BATTLE_LOG_VERSION,
    MANIFEST_VERSION,
    new_battle_log,
    summarize_state,
    validate_battle_log,
    validate_manifest,
)


def test_versions_are_set():
    assert BATTLE_LOG_VERSION == "1.0"
    assert MANIFEST_VERSION == "1.0"


def test_new_battle_log_has_required_keys():
    log = new_battle_log(
        battle_id="b1", group_id="g1", started_at="2026-06-24T12:00:00Z",
        models={"p1": {"name": "v4-max"}, "p2": {"name": "end_turn"}},
        decks={"p1": [1, 14], "p2": [3, 18]},
    )
    assert log["log_version"] == BATTLE_LOG_VERSION
    assert log["battle_id"] == "b1"
    assert log["group_id"] == "g1"
    assert log["actions"] == []
    assert log["result"]["status"] == "ONGOING"
    errors = validate_battle_log(log)
    assert errors == []


def test_validate_battle_log_missing_keys():
    bad = {"log_version": BATTLE_LOG_VERSION, "battle_id": "x"}
    errors = validate_battle_log(bad)
    assert any("missing keys" in e for e in errors)


def test_validate_battle_log_wrong_version():
    log = new_battle_log(
        battle_id="b1", group_id="g1", started_at="2026-06-24T12:00:00Z",
        models={}, decks={},
    )
    log["log_version"] = "0.9"
    errors = validate_battle_log(log)
    assert any("log_version mismatch" in e for e in errors)


def test_validate_manifest_missing_keys():
    errors = validate_manifest({"manifest_version": MANIFEST_VERSION})
    assert any("missing keys" in e for e in errors)


def test_validate_manifest_ok():
    errors = validate_manifest({
        "manifest_version": MANIFEST_VERSION,
        "group_id": "g1",
        "created_at": "2026-06-24T12:00:00Z",
        "spec": {},
        "env": {},
    })
    assert errors == []


def test_summarize_state_normalizes():
    summary = summarize_state({
        "turn_number": 5, "p1_hp": 12, "p2_hp": 0,
        "p1_mana": 3, "p2_mana": 1,
        "p1_max_mana": 5, "p2_max_mana": 4,
        "p1_board_count": 3, "p2_board_count": 0,
        "extra": "ignored",
    })
    assert summary == {
        "turn_number": 5, "p1_hp": 12, "p2_hp": 0,
        "p1_mana": 3, "p2_mana": 1,
        "p1_max_mana": 5, "p2_max_mana": 4,
        "p1_board_count": 3, "p2_board_count": 0,
    }
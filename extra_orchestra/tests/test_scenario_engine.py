"""Тесты сценарного движка: soldatik-demo → aoe_silence снимает 3 механики."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from extra_orchestra.components.cards_catalog import CardsCatalog
from extra_orchestra.components.scenario_engine import run_scenario, validate_scenario, ScenarioError

HERE = Path(__file__).resolve().parent
SCENARIOS = HERE.parent / "scenarios"
SOLDATIK = SCENARIOS / "soldatik-demo.json"


@pytest.fixture
def soldatik_scenario():
    return json.loads(SOLDATIK.read_text(encoding="utf-8"))


@pytest.fixture
def catalog():
    return CardsCatalog()


def test_soldatik_strips_three_mechanics(soldatik_scenario, catalog):
    res = run_scenario(soldatik_scenario, catalog)
    assert res["error"] is None, res["error"]
    assert res["frame_count"] >= 3

    init = res["frames"][0]["snapshot"]
    # init — ход 15, персона оппонента видна
    assert init["turn"] == 15
    assert init["opponent"]["name"] == "Режиссёр"
    # на поле оппонента 3 юнита с механиками
    opp_board = init["opponent"]["board"]
    assert len(opp_board) == 3
    init_mechs = [c["mechanics"] for c in opp_board]
    assert init_mechs == [["shield", "shield_refresh"], ["taunt"], ["deathrattle_aoe_damage_2"]]

    # найдём кадр после play_card Солдатика
    play = next(f for f in res["frames"] if f["action_kind"] == "play_card")
    assert play is not None
    # aoe_silence снял все 3 механики
    after_mechs = [c["mechanics"] for c in play["snapshot"]["opponent"]["board"]]
    assert after_mechs == [[], [], []], after_mechs
    # Солдатик на поле игрока, с механикой aoe_silence
    pboard = play["snapshot"]["player"]["board"]
    assert any(c["name"] == "Солдатик" and "aoe_silence" in c["mechanics"] for c in pboard)


def test_soldatik_validate_ok(soldatik_scenario, catalog):
    v = validate_scenario(soldatik_scenario, catalog)
    assert v["ok"], v["error"]
    assert v["frame_count"] >= 3
    assert v["total_ms"] > 0


def test_determinism_same_seed_same_frames(soldatik_scenario, catalog):
    r1 = run_scenario(soldatik_scenario, catalog)
    r2 = run_scenario(soldatik_scenario, catalog)
    # покадровое сравнение снимков (без учёта display_ms-таймингов — они детерминированы тоже)
    assert r1["frame_count"] == r2["frame_count"]
    for a, b in zip(r1["frames"], r2["frames"]):
        assert a["snapshot"] == b["snapshot"]
        assert a["sound_events"] == b["sound_events"]
        assert a["display_ms"] == b["display_ms"]


def test_bad_side_raises_scenario_error(catalog):
    sc = {
        "name": "x", "seed": 1, "viewer_side": "p1", "match_id": "x",
        "init_scene": {
            "turn_number": 1, "starting_side": "p1",
            "p1": {"user_id": 11, "nickname": "A", "hero": {"card_id": 1, "level": 1}, "mana": 5, "max_mana": 5, "hand": [], "board": [], "deck": []},
            "p2": {"user_id": 22, "is_bot": True, "nickname": "B", "hero": {"card_id": 3, "level": 1}, "mana": 5, "max_mana": 5, "hand": [], "board": [], "deck": []},
        },
        "turns": [{"id": "t1", "side": "p2", "end_with_end_turn": False, "nodes": []}],
    }
    res = run_scenario(sc, catalog)
    # starting_side p1, но первый ход заявлен за p2 → ScenarioError
    assert res["error"] is not None
    assert "not the current turn owner" in res["error"] or "side" in res["error"]
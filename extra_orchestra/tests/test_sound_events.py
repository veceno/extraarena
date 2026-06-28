"""Порт sound_events: play Солдатика → deploy-событие (aoe_silence НЕ входит в
``_is_play_sound_mechanic`` → отдельного mechanic-события нет, как и в prod
``battle_engine.py``). event_id детерминированный.
"""
from __future__ import annotations

import json
from pathlib import Path

from extra_orchestra.components.cards_catalog import CardsCatalog
from extra_orchestra.components.scenario_engine import run_scenario

HERE = Path(__file__).resolve().parent
SOLDATIK = HERE.parent / "scenarios" / "soldatik-demo.json"


def test_soldatik_deploy_sound_event():
    sc = json.loads(SOLDATIK.read_text(encoding="utf-8"))
    res = run_scenario(sc, CardsCatalog())
    assert res["error"] is None
    play = next(f for f in res["frames"] if f["action_kind"] == "play_card")
    evs = play["sound_events"]
    # ровно одно deploy-событие
    assert len(evs) == 1, evs
    e = evs[0]
    assert e["event"] == "deploy"
    assert e["card_id"] == 47
    assert e["card_name"] == "Солдатик"
    assert e["source"] == "action"
    # event_id детерминированный по шаблону orchestra:<turn>:play_card:<inst>:deploy
    assert e["event_id"].startswith("orchestra:15:play_card:")
    assert e["event_id"].endswith(":deploy")
    # aoe_silence не входит в _is_play_sound_mechanic → mechanic-события нет
    assert not any(e.get("event") == "mechanic" for e in evs)


def test_is_play_sound_mechanic_matches_prod():
    from extra_orchestra.components.arena_engine import OrchestraBattleEngine as E
    # verbatim-копия battle_engine.py:511 — те же правила
    assert E._is_play_sound_mechanic("aoe_silence") is False
    assert E._is_play_sound_mechanic("deathrattle_aoe_damage_2") is False  # deathrattle-префикс
    assert E._is_play_sound_mechanic("battlecry_damage_3") is True
    assert E._is_play_sound_mechanic("cast_random_spell") is True
    assert E._is_play_sound_mechanic("aoe_freeze") is True
    assert E._is_play_sound_mechanic("taunt") is False
    assert E._is_play_sound_mechanic("") is False


def test_attack_sound_event():
    # построим мини-сценарий с атакой: p1 board уже имеет готового юнита, атакует героя оппонента
    sc = {
        "name": "atk", "seed": 1, "viewer_side": "p1", "match_id": "atk",
        "classic_params": {"sudden_death_enabled": False},
        "init_scene": {
            "turn_number": 5, "starting_side": "p1",
            "p1": {"user_id": 11, "nickname": "A", "hero": {"card_id": 1, "level": 1}, "mana": 5, "max_mana": 5,
                   "hand": [], "board": [{"card_id": 25, "level": 1, "is_ready": True}], "deck": []},
            "p2": {"user_id": 22, "is_bot": True, "nickname": "B", "hero": {"card_id": 3, "level": 1}, "mana": 5, "max_mana": 5,
                   "hand": [], "board": [], "deck": []},
        },
        "turns": [{"id": "t1", "side": "p1", "end_with_end_turn": False,
                   "nodes": [{"id": "a", "type": "attack", "attacker_index": 0, "target_is_hero": True, "delay_ms": 300}]}],
    }
    res = run_scenario(sc, CardsCatalog())
    assert res["error"] is None, res["error"]
    atk = next(f for f in res["frames"] if f["action_kind"] == "attack")
    evs = atk["sound_events"]
    assert len(evs) == 1
    assert evs[0]["event"] == "attack"
    assert evs[0]["event_id"].startswith("orchestra:5:attack:")
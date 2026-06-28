"""Wire-drift guard: сериализатор OrchestraBattleEngine.get_full_state
проверяется против ожидаемого prod-контракта (ключи + значения) на фиксированном
init-состоянии soldatik-demo.

Прямое byte-for-byte сравнение с ``BattleEngine`` требует БД/match-менеджера
(тяжёлый setup); вместо этого фиксируем форму контракта, которую читает
``arena.js``: поля viewer-perspective, persona, hero, board, mechanics,
image-url, mana_draw_count_this_turn (которого нет в rlhf-шиме — отличаемся
от него намеренно в сторону prod'а).
"""
from __future__ import annotations

import json
from pathlib import Path

from extra_orchestra.components.cards_catalog import CardsCatalog
from extra_orchestra.components.scenario_engine import build_initial_state

HERE = Path(__file__).resolve().parent
SOLDATIK = HERE.parent / "scenarios" / "soldatik-demo.json"


def _build():
    sc = json.loads(SOLDATIK.read_text(encoding="utf-8"))
    env, engine, viewer_uid, side_uids = build_initial_state(sc, CardsCatalog())
    return engine, viewer_uid


EXPECTED_TOP_KEYS = {
    "match_id", "turn", "current_player_id", "is_my_turn", "player_ids", "viewer_id",
    "match_status", "battle_started", "is_ended", "game_over", "winner_id",
    "turn_time_remaining", "turn_duration", "game_mode", "ruleset", "mode_config",
    "sudden_death", "player1_hp", "player2_hp", "player", "opponent",
    "legal_actions", "action_history",
}
EXPECTED_PLAYER_KEYS = {
    "user_id", "name", "avatar_url", "title", "rarity", "extra_pass",
    "nickname_glow_disabled", "hide_player_id_public", "background_url", "is_bot",
    "replacement_status", "mana", "max_mana", "mana_draw_count_this_turn", "trophies",
    "hero", "hand", "hand_count", "deck_count", "board",
}
EXPECTED_CARD_KEYS = {
    "instance_id", "card_id", "name", "description", "card_type", "rarity", "level",
    "mana", "mana_cost", "attack", "atk", "hp", "hp_current", "max_hp", "mechanics",
    "is_ready", "can_attack", "is_asleep", "is_frozen", "mechanics_desc", "owner_id", "image",
}


def test_get_full_state_contract():
    engine, viewer_uid = _build()
    state = engine.get_full_state(viewer_uid)
    assert set(state.keys()) >= EXPECTED_TOP_KEYS
    assert state["turn"] == 15
    assert state["viewer_id"] == viewer_uid
    assert state["player_ids"] == [1001, 2002]
    # viewer=p1 → player.name из persona (nickname)
    assert state["player"]["name"] == "Демо"
    assert state["opponent"]["name"] == "Режиссёр"
    # opponent hand скрыт (в soldatik p2.hand пуст → []; при непустой руке — [{hidden:True}…])
    assert state["opponent"]["hand"] == []
    # player hand видим (Солдатик)
    assert state["player"]["hand"][0]["name"] == "Солдатик"
    # image-url формата /DesignAssets/Cards/<id>.<ext>
    assert state["player"]["hand"][0]["image"].startswith("/DesignAssets/Cards/")
    # mana_draw_count_this_turn присутствует (отличие от rlhf-шима — как в prod)
    assert "mana_draw_count_this_turn" in state["player"]
    # board-карта несёт owner_id
    b = state["opponent"]["board"][0]
    assert b["owner_id"] == 2002
    assert set(b.keys()) >= EXPECTED_CARD_KEYS


def test_player_state_keys_exact():
    engine, viewer_uid = _build()
    state = engine.get_full_state(viewer_uid)
    assert set(state["player"].keys()) == EXPECTED_PLAYER_KEYS
    assert set(state["opponent"].keys()) == EXPECTED_PLAYER_KEYS


def test_viewer_p2_swaps_player_opponent():
    sc = json.loads(SOLDATIK.read_text(encoding="utf-8"))
    sc["viewer_side"] = "p2"
    env, engine, viewer_uid, _ = build_initial_state(sc, CardsCatalog())
    state = engine.get_full_state(viewer_uid)
    assert state["viewer_id"] == 2002
    assert state["player"]["name"] == "Режиссёр"
    assert state["opponent"]["name"] == "Демо"
    # у p2 рука пустая → player.hand == []
    assert state["player"]["hand"] == []
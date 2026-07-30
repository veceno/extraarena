"""Полнота V5-входов в v5_trace (WS2 + WS4 плана joyful-churning-bird).

Проверяет, что v5-снапшот захватывает всё, что encode_observation_v5 (Block 0)
будет читать из реконструированного GameState:
  - state.action_history (List) — UI history;
  - state.history (List[Dict]) — native actions;
  - state.v5_history_events (List[Dict]) — model history-window 20×162;
  - mechanics НОВЫХ карт 47-52 (5 новых mechanic-семейств) в _snapshot_card.

Все три history-поля кладутся в pre_state каждой action-строки; bridge
реконструирует полный GameState из pre_state.
"""
from __future__ import annotations

import asyncio

from rlhf_env.tests._v5_helpers import make_manager, read_jsonl, v5_dir_for

# Кастомная колода p1 с НОВЫМИ картами 47/50/52 (3 из 5 новых mechanic-семейств)
# + hero 49 (crime_and_punishment_2). Current ArenaEnv contract is exactly
# 1 hero + 8 unique cards = 9.
P1_CUSTOM_DECK = {
    "hero": 49,
    "cards": [47, 50, 52, 8, 10, 13, 18, 19],
}
NEW_CARD_MECHANICS = {
    47: "aoe_silence",
    48: "team_wide_shield",
    49: "crime_and_punishment",
    50: "rebirth",
    52: "target_ally_max_hp_plus_universal",
}


def _has_mechanic_family(mechs, family):
    """Механика может нести magnitude-суффикс по уровню карты (rebirth_1 → rebirth_2,
    crime_and_punishment_2 и т.д.) — сверяем по family-префиксу, не по точной строке."""
    return any(m == family or m.startswith(family + "_") for m in (mechs or []))


def _play_custom_deck(tmp_path, *, seed=7):
    from rlhf_env.components.arena_match_manager import ArenaMatchManager
    from rlhf_env.components.match_runner import MatchRunner

    mgr = make_manager(tmp_path)
    spec = {
        "p2_model": "random",
        "battles_planned": 1,
        "seed": seed,
        "p1_actor_type": "llm",
        "deck_strategy_p1": "custom",
        "custom_deck_p1": P1_CUSTOM_DECK,
    }
    match = mgr.create_series(spec)
    engine = match.engine
    runner = MatchRunner(match)

    async def go():
        for _ in range(8):
            if engine.is_ended:
                break
            if engine.get_current_player_id() == engine.human_user_id:
                await runner.execute_human_action(
                    {"type": "end_turn", "client_action_id": f"c_et_{_}"}
                )
            else:
                await runner.run_bot_turn()
        if not engine.is_ended:
            await runner.surrender()

    asyncio.run(go())
    return match, engine


def _all_card_snapshots(rows):
    """Собирает все card-снапшоты из p1/p2 deck/hand/board/graveyard всех строк."""
    out = []
    for r in rows:
        for side in ("p1", "p2"):
            ps = r.get("pre_state", {}).get(side, {}) or {}
            for field in ("deck", "hand", "board", "graveyard"):
                for c in ps.get(field, []) or []:
                    if isinstance(c, dict):
                        out.append(c)
    return out


def test_all_history_contracts_in_pre_state(tmp_path):
    """pre_state keeps native/UI logs and the dedicated V5 model tape."""
    from rlhf_env.tests._v5_helpers import create_match

    mgr = make_manager(tmp_path)
    match, engine, runner = create_match(mgr, p1_actor_type="llm", p2_model="random", seed=3)

    async def go():
        for _ in range(6):
            if engine.is_ended:
                break
            if engine.get_current_player_id() == engine.human_user_id:
                await runner.execute_human_action(
                    {"type": "end_turn", "client_action_id": f"c_et_{_}"}
                )
            else:
                await runner.run_bot_turn()
        if not engine.is_ended:
            await runner.surrender()

    asyncio.run(go())
    rows = read_jsonl(v5_dir_for(match, tmp_path) / "actions.jsonl")
    assert rows, "actions.jsonl пуст"
    for r in rows:
        ps = r["pre_state"]
        assert "action_history" in ps, "action_history отсутствует в pre_state"
        assert isinstance(ps["action_history"], list), "action_history не list"
        assert "history" in ps, "history отсутствует в pre_state"
        assert isinstance(ps["history"], list), "history не list"
        assert "v5_history_events" in ps, "v5_history_events отсутствует в pre_state"
        assert isinstance(ps["v5_history_events"], list), "v5_history_events не list"
        assert len(ps["v5_history_events"]) <= 20


def test_action_history_grows_monotonically(tmp_path):
    """action_history накапливается: поздние строки >= ранних (bridge читает окно 20)."""
    from rlhf_env.tests._v5_helpers import create_match

    mgr = make_manager(tmp_path)
    match, engine, runner = create_match(mgr, p1_actor_type="llm", p2_model="random", seed=5)

    async def go():
        for _ in range(10):
            if engine.is_ended:
                break
            if engine.get_current_player_id() == engine.human_user_id:
                await runner.execute_human_action(
                    {"type": "end_turn", "client_action_id": f"c_et_{_}"}
                )
            else:
                await runner.run_bot_turn()
        if not engine.is_ended:
            await runner.surrender()

    asyncio.run(go())
    rows = read_jsonl(v5_dir_for(match, tmp_path) / "actions.jsonl")
    lengths = [len(r["pre_state"]["action_history"]) for r in rows]
    assert lengths, "нет строк"
    # хотя бы одна поздняя строка длиннее первой (не все равны)
    assert max(lengths) > lengths[0], (
        f"action_history не растёт: {lengths} (bridge не получит history-window)"
    )


def test_new_card_mechanics_captured_in_snapshot(tmp_path):
    """_snapshot_card пишет mechanics новых карт 47/50/52 (3 новых семейства)."""
    match, _ = _play_custom_deck(tmp_path)
    rows = read_jsonl(v5_dir_for(match, tmp_path) / "actions.jsonl")
    assert rows, "actions.jsonl пуст для кастомной колоды"
    cards = _all_card_snapshots(rows)
    by_id = {c["id"]: c for c in cards if "id" in c}
    # колода p1 содержит 47/50/52 → они обязаны быть в deck-снапшоте
    for cid, mech in NEW_CARD_MECHANICS.items():
        if cid == 49:
            continue  # hero 49 — отдельный field, проверим ниже
        assert cid in by_id, f"карта {cid} не попала ни в один снапшот (колода должна её содержать)"
        mechs = by_id[cid].get("mechanics") or []
        assert _has_mechanic_family(mechs, mech), (
            f"карта {cid}: семейство {mech!r} не захвачено, есть {mechs}"
        )


def test_new_hero_mechanics_captured(tmp_path):
    """Hero 49 (crime_and_punishment_2) — механика в hero-снапшоте p1."""
    match, _ = _play_custom_deck(tmp_path)
    rows = read_jsonl(v5_dir_for(match, tmp_path) / "actions.jsonl")
    assert rows
    # hero — в pre_state.p1.hero (и p2.hero)
    found = False
    for r in rows:
        hero = r["pre_state"]["p1"].get("hero")
        if isinstance(hero, dict) and hero.get("id") == 49:
            assert _has_mechanic_family(hero.get("mechanics"), "crime_and_punishment"), (
                f"hero 49 mechanics: {hero.get('mechanics')}"
            )
            found = True
            break
    assert found, "hero 49 не найден ни в одном p1-снапшоте"


def test_v5_trace_present_flag_in_meta(tmp_path):
    """meta.json выставляет v5_trace_present=True (G4-сигнал для collection-фильтра)."""
    from rlhf_env.tests._v5_helpers import create_match, read_json

    mgr = make_manager(tmp_path)
    match, engine, runner = create_match(mgr, p1_actor_type="llm", p2_model="random", seed=1)

    async def go():
        if not engine.is_ended:
            await runner.surrender()

    asyncio.run(go())
    meta = read_json(v5_dir_for(match, tmp_path) / "meta.json")
    assert meta.get("v5_trace_present") is True

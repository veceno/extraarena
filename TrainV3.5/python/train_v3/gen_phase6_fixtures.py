"""Phase 6 fixture generator: random_battlecry (card15) + shield_refresh (card24).

Removes the last two card_id hardcodes in the Rust kernel by exercising the
mechanic-driven paths:

  random_battlecry  — card 15 Тока Киришима (cost 2, 2/1, warrior,
                      `battlecry_damage_1_random`). Python
                      `_apply_random_battlecry_damage` picks a random target
                      via `random.choice(list(opponent.board) + [opponent.hero])`
                      and deals 1 damage. The `random.choice` outcome is
                      recorded in the new `choice_rolls` stream (AC-FFI-2,
                      TA-5) so Rust `roll_choice` reproduces the exact target.

  shield_refresh     — card 24 Годжо Сатору (cost 9, 5/6, warrior,
                      `["shield", "shield_refresh"]`). Python
                      `core/engine.py::_handle_end_turn` (line 728) refreshes
                      `shield` at the start of the owner's turn when the unit
                      has `shield_refresh` and no current `shield`. The
                      fixture consumes the shield via an enemy attack, then
                      ends the enemy turn → owner's turn start re-adds
                      shield (TWS-3, AC-FFI-3).

Usage:
  PYTHONPATH=.:TrainV3.5/python python3 -m train_v3.gen_phase6_fixtures
"""
from __future__ import annotations

import json

import numpy as np

from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.classic_actions_v1 import (
    _PLAY_BASE, _PLAY_STRIDE, _NUM_PLAY_TARGETS, _ATTACK_BASE, _NUM_ATTACK_TARGETS,
)

from .golden_trace import build_golden_trace


# ---------------------------------------------------------------------------
# Helpers (mirrors gen_phase4/phase5_fixtures)
# ---------------------------------------------------------------------------

def _legal_ids(env):
    cp = env.current_player_id()
    mask = env.action_mask(cp)
    return [int(i) for i in np.flatnonzero(mask == 1.0)]


def _decode(action_id):
    if action_id == 0:
        return "EndTurn"
    if action_id >= _ATTACK_BASE:
        a = (action_id - _ATTACK_BASE) // _NUM_ATTACK_TARGETS
        t = (action_id - _ATTACK_BASE) % _NUM_ATTACK_TARGETS
        return f"Attack(att={a},target={t})"
    h = (action_id - _PLAY_BASE) // _PLAY_STRIDE
    rest = (action_id - _PLAY_BASE) % _PLAY_STRIDE
    p = rest // _NUM_PLAY_TARGETS
    t = rest % _NUM_PLAY_TARGETS
    return f"Play(hand={h},pos={p},target={t})"


def _state(env):
    st = env._env.state
    cp = env.current_player_id()
    me = st.p1 if st.p1.user_id == cp else st.p2
    opp = st.p2 if st.p1.user_id == cp else st.p1
    hand = [(c.card_id, c.name, c.mana_cost) for c in me.hand]
    board = [(c.card_id, c.name, c.attack, c.hp, c.max_hp, list(c.mechanics)) for c in me.board]
    ob = [(c.card_id, c.name, c.attack, c.hp, list(c.mechanics)) for c in opp.board]
    return (f"cp={cp} mana={me.mana} hero={me.hero.name}({me.hero.hp}/{me.hero.max_hp}hp) "
            f"hand={hand} board={board} opp_board={ob} opp_hero={opp.hero.name}({opp.hero.hp}/{opp.hero.max_hp}hp)")


def _find_play(env, card_id, target_code=0):
    cp = env.current_player_id()
    me = env._env.state.p1 if env._env.state.p1.user_id == cp else env._env.state.p2
    for aid in _legal_ids(env):
        if aid < _PLAY_BASE or aid >= _ATTACK_BASE:
            continue
        h = (aid - _PLAY_BASE) // _PLAY_STRIDE
        rest = (aid - _PLAY_BASE) % _PLAY_STRIDE
        t = rest % _NUM_PLAY_TARGETS
        if h < len(me.hand) and me.hand[h].card_id == card_id and t == target_code:
            return aid
    return None


def _find_attack(env, attacker_card_id, target_code):
    cp = env.current_player_id()
    me = env._env.state.p1 if env._env.state.p1.user_id == cp else env._env.state.p2
    for aid in _legal_ids(env):
        if aid < _ATTACK_BASE:
            continue
        a = (aid - _ATTACK_BASE) // _NUM_ATTACK_TARGETS
        t = (aid - _ATTACK_BASE) % _NUM_ATTACK_TARGETS
        if a < len(me.board) and me.board[a].card_id == attacker_card_id and t == target_code:
            return aid
    return None


END_TURN = 0


def _gen(name, seed, p1_deck, p2_deck, action_ids, mana_per_turn=10,
         p1_levels=None, p2_levels=None, post_reset_setup=None, verify_mask=True):
    trace = build_golden_trace(
        seed=seed,
        steps=len(action_ids),
        placement_mode="append_only",
        verify_mask=verify_mask,
        include_v5=True,
        choose="first",
        p1_deck_ids=p1_deck,
        p2_deck_ids=p2_deck,
        p1_levels=p1_levels,
        p2_levels=p2_levels,
        action_ids=action_ids,
        mana_per_turn=mana_per_turn,
        post_reset_setup=post_reset_setup,
    )
    path = f"TrainV3.5/rust/trainv3_core/tests/fixtures/golden_trace_{name}.json"
    with open(path, "w") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
    print(f"  wrote {path} ({len(trace['steps'])} steps)")
    # Echo the recorded choice_rolls per step for sanity.
    for s in trace["steps"]:
        if s.get("choice_rolls"):
            print(f"    step {s['t']} choice_rolls={s['choice_rolls']}")


# ---------------------------------------------------------------------------
# random_battlecry: card 15 Тока Киришима (cost 2, 2/1, battlecry_damage_1_random).
# p1 fills board with 2 Скелет (card 27) → opponent targets list = [Скелет,
# Скелет, hero] (3 entries). p2 plays Тока → `random.choice` picks one of the
# 3 and deals 1 damage. The chosen index is recorded in `choice_rolls`.
# Mirrors core/effects.py `_apply_random_battlecry_damage` (line 67-83).
# ---------------------------------------------------------------------------

def gen_random_battlecry():
    print("[random_battlecry] generating...")
    p1_deck = [27, 27, 27, 27, 27, 27, 27, 27]  # Скелет filler
    p2_deck = [15, 15, 15, 15, 15, 15, 15, 15]  # Тока Киришима
    seed = 1800
    env = ClassicRLEnv(seed=seed, placement_mode="append_only", mana_per_turn=10)
    env.reset(seed=seed, p1_deck_ids=p1_deck, p2_deck_ids=p2_deck)
    ids = []

    # T1 (p1, mana=1): play Скелет (cost 1) then end
    print(f"  s0: {_state(env)}")
    aid = _find_play(env, 27); assert aid, "Скелет not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    print(f"  s1: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T2 (p2, mana=10): end (set up mana)
    print(f"  s2: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T3 (p1, mana=10): play Скелет then end → 2 Скелет on board
    print(f"  s3: {_state(env)}")
    aid = _find_play(env, 27); assert aid, "Скелет not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    print(f"  s4: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T4 (p2): end
    print(f"  s5: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T5 (p1): end
    print(f"  s6: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T6 (p2, mana=10): play Тока (15, cost 2) → battlecry_damage_1_random
    # targets = list(p1.board) + [p1.hero] = [Скелет, Скелет, hero] (3 entries).
    # random.choice picks index 0/1/2; 1 damage applied to that target.
    print(f"  s7: {_state(env)}")
    p1_board_before = [(c.card_id, c.hp) for c in env._env.state.p1.board]
    p1_hero_before = env._env.state.p1.hero.hp
    aid = _find_play(env, 15); assert aid, "Тока not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    p1_board_after = [(c.card_id, c.hp) for c in env._env.state.p1.board]
    p1_hero_after = env._env.state.p1.hero.hp
    print(f"  RANDOM_BATTLECRY: p1 board {p1_board_before}->{p1_board_after}, hero {p1_hero_before}->{p1_hero_after}")
    # Exactly one target took 1 damage: either a Скелет died (removed from
    # board) or hero lost 1 hp.
    board_damage = sum(1 for i in range(min(len(p1_board_before), len(p1_board_after)))
                       if p1_board_after[i][1] == p1_board_before[i][1] - 1)
    skel_died = len(p1_board_after) < len(p1_board_before)
    hero_dmg = p1_hero_before - p1_hero_after == 1
    assert board_damage == 1 or skel_died or hero_dmg, \
        f"expected exactly 1 random damage, got board {p1_board_before}->{p1_board_after}, hero {p1_hero_before}->{p1_hero_after}"
    print(f"  RANDOM_BATTLECRY VERIFIED: 1 damage to a random enemy target")
    print(f"  s8: {_state(env)}")

    _gen("random_battlecry", seed, p1_deck, p2_deck, ids, verify_mask=False)


# ---------------------------------------------------------------------------
# shield_refresh: card 24 Годжо Сатору (cost 9, 5/6, shield + shield_refresh).
# p1 plays Годжо (comes with `shield`). p2's Скелет attacks Годжо → shield
# consumed (no hp loss). p2 ends turn → p1's turn starts → shield_refresh
# re-adds `shield` (since shield is gone). Mechanic-driven (any card with
# shield_refresh), matches core/engine.py line 728.
# ---------------------------------------------------------------------------

def gen_shield_refresh():
    print("[shield_refresh] generating...")
    p1_deck = [24, 24, 24, 27, 27, 27, 27, 27]  # Годжо + Скелет filler
    p2_deck = [27, 27, 27, 27, 27, 27, 27, 27]  # Скелет filler
    seed = 1900
    env = ClassicRLEnv(seed=seed, placement_mode="append_only", mana_per_turn=10)
    env.reset(seed=seed, p1_deck_ids=p1_deck, p2_deck_ids=p2_deck)
    ids = []

    # T1 (p1, mana=1): end (Годжо costs 9)
    print(f"  s0: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T2 (p2, mana=10): play Скелет (cost 1) then end
    print(f"  s1: {_state(env)}")
    aid = _find_play(env, 27); assert aid, "Скелет not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    print(f"  s2: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T3 (p1, mana=10): play Годжо (24, cost 9) → shield + shield_refresh
    print(f"  s3: {_state(env)}")
    aid = _find_play(env, 24); assert aid, "Годжо not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    gojo = next(c for c in env._env.state.p1.board if c.card_id == 24)
    print(f"  ГОДЖО played: hp={gojo.hp} mechanics={list(gojo.mechanics)}")
    assert "shield" in gojo.mechanics and "shield_refresh" in gojo.mechanics, \
        f"Годжо should have shield+shield_refresh, got {gojo.mechanics}"
    print(f"  s4: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T4 (p2, mana=10): Скелет attacks Годжо (target_code=0, first enemy minion
    # — attack target codes are 0-indexed enemy board positions, NOT the
    # 1-indexed play-target scheme) → shield consumed, Годжо hp unchanged.
    print(f"  s5: {_state(env)}")
    gojo_before = next(c for c in env._env.state.p1.board if c.card_id == 24)
    hp_before, mech_before = gojo_before.hp, list(gojo_before.mechanics)
    aid = _find_attack(env, 27, target_code=0); assert aid, "Скелет attack on Годжо not found"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    gojo_after = next(c for c in env._env.state.p1.board if c.card_id == 24)
    print(f"  SHIELD CONSUMED: Годжо hp {hp_before}->{gojo_after.hp}, mech {mech_before}->{list(gojo_after.mechanics)}")
    assert "shield" not in gojo_after.mechanics, \
        f"shield should be consumed by attack, got {gojo_after.mechanics}"
    assert gojo_after.hp == hp_before, \
        f"shield should block all damage, hp {hp_before}->{gojo_after.hp}"
    print(f"  SHIELD CONSUMED VERIFIED: shield gone, hp unchanged")
    print(f"  s6: {_state(env)}")
    # p2 ends turn → p1 turn starts → shield_refresh re-adds shield.
    ids.append(END_TURN); env.step(END_TURN)
    gojo_refreshed = next(c for c in env._env.state.p1.board if c.card_id == 24)
    print(f"  SHIELD_REFRESH: Годжо mech {list(gojo_after.mechanics)}->{list(gojo_refreshed.mechanics)}")
    assert "shield" in gojo_refreshed.mechanics, \
        f"shield_refresh should re-add shield at turn start, got {gojo_refreshed.mechanics}"
    print(f"  SHIELD_REFRESH VERIFIED: shield restored at owner turn start")
    print(f"  s7: {_state(env)}")

    _gen("shield_refresh", seed, p1_deck, p2_deck, ids, verify_mask=False)


# ---------------------------------------------------------------------------

def main():
    gen_random_battlecry()
    gen_shield_refresh()
    print("[phase6] done")


if __name__ == "__main__":
    main()
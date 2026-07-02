"""Phase 5 fixture generator: aoe_silence, team_wide_shield, target_ally_max_hp_plus.

Cards:
  47 Солдатик    (cost 7, 4/5, aoe_silence)              — AOE-SILENCE-1
  48 Соул Гудман (cost 7, 2/4, team_wide_shield)         — TWS-1/2
  52 Криста Ленц (cost 2, 1/2, target_ally_max_hp_plus_universal_1) — TAMHP-1/2/3

card52 is played by BOTH players (user decision: "playable everywhere") and
exercises BOTH target families — friendly minion (code 9) and own hero
(code 16). Drives the Python env directly to discover action_ids, then calls
build_golden_trace to produce recorded-outcome RNG fixtures for Rust
state-transition parity.

Usage:
  PYTHONPATH=.:TrainV3.5/python python3 -m train_v3.gen_phase5_fixtures
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
# Helpers (mirrors gen_phase4_fixtures)
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
    board = [(c.card_id, c.name, c.attack, c.hp, c.max_hp, c.mechanics) for c in me.board]
    ob = [(c.card_id, c.name, c.attack, c.hp, c.mechanics) for c in opp.board]
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


# ---------------------------------------------------------------------------
# aoe_silence: card 47 Солдатик (cost 7, 4/5). p1 fills board with 3 Сукуна
# (card 23, cleave_1_2 — a passive warrior attack mechanic, no play target
# required). p2 plays Солдатик → aoe_silence strips `mechanics` from up to 3
# enemy minions that have mechanics (all 3 Сукуна → cleave_1_2 removed).
# Mirrors core/effects.py `effect_aoe_silence` (limit=3, candidates =
# `[u for u in opponent.board if u.mechanics]`, `unit.mechanics = []`).
# ---------------------------------------------------------------------------

def gen_aoe_silence():
    print("[aoe_silence] generating...")
    p1_deck = [23] * 8  # Сукуна (cleave_1_2)
    p2_deck = [47] * 8  # Солдатик (aoe_silence)
    seed = 1500
    env = ClassicRLEnv(seed=seed, placement_mode="append_only", mana_per_turn=10)
    env.reset(seed=seed, p1_deck_ids=p1_deck, p2_deck_ids=p2_deck)
    ids = []

    # T1 (p1, mana=1): end (Сукуна costs 7)
    print(f"  s0: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T2 (p2, mana=10): end
    print(f"  s1: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T3 (p1, mana=10): play Сукуна (cost 7) then end
    print(f"  s2: {_state(env)}")
    aid = _find_play(env, 23); assert aid, "Сукуна not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    print(f"  s3: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T4 (p2): end
    print(f"  s4: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T5 (p1, mana=10): play Сукуна then end
    print(f"  s5: {_state(env)}")
    aid = _find_play(env, 23); assert aid, "Сукуна not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    print(f"  s6: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T6 (p2): end
    print(f"  s7: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T7 (p1, mana=10): play Сукуна — now 3 on board, each with cleave_1_2
    print(f"  s8: {_state(env)}")
    aid = _find_play(env, 23); assert aid, "Сукуна not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    print(f"  s9: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T8 (p2, mana=10): play Солдатик (47, cost 7) → aoe_silence strips
    # mechanics from up to 3 enemy Сукуна (each has cleave_1_2).
    print(f"  s10: {_state(env)}")
    p1_board_before = [(c.card_id, list(c.mechanics)) for c in env._env.state.p1.board]
    aid = _find_play(env, 47); assert aid, "Солдатик not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    p1_board_after = [(c.card_id, list(c.mechanics)) for c in env._env.state.p1.board]
    print(f"  AOE_SILENCE: p1 board mech {p1_board_before} -> {p1_board_after}")
    assert all(m == [] for _, m in p1_board_after), f"all 3 Сукуна should be silenced, got {p1_board_after}"
    print(f"  AOE_SILENCE VERIFIED: 3 enemy minions stripped of mechanics")
    print(f"  s11: {_state(env)}")

    _gen("aoe_silence", seed, p1_deck, p2_deck, ids)


# ---------------------------------------------------------------------------
# team_wide_shield: card 48 Соул Гудман (cost 7, 2/4). p1 plays 3 Скелет
# (card 27, cost 1, no mechanics), then plays Соул Гудман → team_wide_shield
# grants `shield` to up to 3 friendly minions EXCLUDING the just-played card
# (TWS-2 self-exclusion — mirrors core/effects.py `effect_team_wide_shield`:
# `targets = [u for u in owner.board if u.instance_id != card.instance_id]`).
# After: 3 Скелет gain `shield`; Соул Гудман does NOT have `shield`.
# ---------------------------------------------------------------------------

def gen_team_wide_shield():
    print("[team_wide_shield] generating...")
    p1_deck = [27, 27, 27, 48, 48, 48, 48, 48]  # 3 Скелет + 5 Соул Гудман
    p2_deck = [27] * 8  # filler
    seed = 1600
    env = ClassicRLEnv(seed=seed, placement_mode="append_only", mana_per_turn=10)
    env.reset(seed=seed, p1_deck_ids=p1_deck, p2_deck_ids=p2_deck)
    ids = []

    # T1 (p1, mana=1): play Скелет (cost 1) then end
    print(f"  s0: {_state(env)}")
    aid = _find_play(env, 27); assert aid, "Скелет not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    print(f"  s1: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T2 (p2): end
    print(f"  s2: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T3 (p1, mana=10): play Скелет then end
    print(f"  s3: {_state(env)}")
    aid = _find_play(env, 27); assert aid, "Скелет not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    print(f"  s4: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T4 (p2): end
    print(f"  s5: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T5 (p1, mana=10): play Скелет — 3 on board
    print(f"  s6: {_state(env)}")
    aid = _find_play(env, 27); assert aid, "Скелет not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    print(f"  s7: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T6 (p2): end
    print(f"  s8: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)
    # T7 (p1, mana=10): play Соул Гудман (48, cost 7) → team_wide_shield
    print(f"  s9: {_state(env)}")
    board_before = [(c.card_id, c.name, list(c.mechanics)) for c in env._env.state.p1.board]
    aid = _find_play(env, 48); assert aid, "Соул Гудман not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    board_after = [(c.card_id, c.name, list(c.mechanics)) for c in env._env.state.p1.board]
    print(f"  TEAM_WIDE_SHIELD: board {board_before} -> {board_after}")
    skeletons = [c for c in env._env.state.p1.board if c.card_id == 27]
    soul = [c for c in env._env.state.p1.board if c.card_id == 48]
    assert len(skeletons) == 3 and all("shield" in c.mechanics for c in skeletons), \
        f"3 Скелет should have shield, got {[(c.card_id, c.mechanics) for c in skeletons]}"
    assert len(soul) == 1 and "shield" not in soul[0].mechanics, \
        f"Соул Гудман should self-exclude (no shield), got {soul[0].mechanics}"
    print(f"  TEAM_WIDE_SHIELD VERIFIED: 3 Скелет shielded, Соул Гудман self-excluded")
    print(f"  s7: {_state(env)}")

    _gen("team_wide_shield", seed, p1_deck, p2_deck, ids)


# ---------------------------------------------------------------------------
# target_ally_max_hp_plus_universal_1: card 52 Криста Ленц (cost 2, 1/2).
# Played by BOTH players (user decision: "playable everywhere"). Exercises
# BOTH target families:
#   - friendly minion target (code 9): p1 and p2 each play Криста targeting
#     their Скелет → Скелет.max_hp += 1 (hp UNCHANGED — direct increase,
#     no heal_card clamp).
#   - own hero target (code 16): p1 plays Криста targeting own hero →
#     hero.max_hp += 1 (hp unchanged).
# Mirrors core/effects.py `target_ally_max_hp_plus_universal_N` handler.
# ---------------------------------------------------------------------------

def gen_tamhp():
    print("[tamhp] generating...")
    p1_deck = [27, 27, 52, 52, 52, 27, 27, 27]  # Скелет + Криста
    p2_deck = [27, 27, 52, 52, 52, 27, 27, 27]
    seed = 1700
    env = ClassicRLEnv(seed=seed, placement_mode="append_only", mana_per_turn=10)
    env.reset(seed=seed, p1_deck_ids=p1_deck, p2_deck_ids=p2_deck)
    ids = []

    # T1 (p1, mana=1): play Скелет (cost 1)
    print(f"  s0: {_state(env)}")
    aid = _find_play(env, 27); assert aid, "Скелет not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    print(f"  s1: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # T2 (p2, mana=10): play Скелет (cost 1)
    print(f"  s2: {_state(env)}")
    aid = _find_play(env, 27); assert aid, "Скелет not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    print(f"  s3: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # T3 (p1, mana=10): play Криста (52, cost 2) targeting friendly Скелет (code 9)
    print(f"  s4: {_state(env)}")
    p1_skel_before = next(c for c in env._env.state.p1.board if c.card_id == 27)
    hp_b, maxhp_b = p1_skel_before.hp, p1_skel_before.max_hp
    aid = _find_play(env, 52, target_code=9); assert aid, "Криста targeting Скелет (code 9) not found"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    p1_skel_after = next(c for c in env._env.state.p1.board if c.card_id == 27)
    print(f"  TAMHP p1->Скелет: hp {hp_b}->{p1_skel_after.hp}, max_hp {maxhp_b}->{p1_skel_after.max_hp}")
    assert p1_skel_after.max_hp == maxhp_b + 1, f"max_hp should bump +1, got {maxhp_b}->{p1_skel_after.max_hp}"
    assert p1_skel_after.hp == hp_b, f"hp should NOT change, got {hp_b}->{p1_skel_after.hp}"
    print(f"  TAMHP p1 friendly-minion VERIFIED: max_hp+1, hp unchanged")
    print(f"  s5: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # T4 (p2, mana=10): play Криста (52) targeting friendly Скелет (code 9)
    print(f"  s6: {_state(env)}")
    p2_skel_before = next(c for c in env._env.state.p2.board if c.card_id == 27)
    hp_b2, maxhp_b2 = p2_skel_before.hp, p2_skel_before.max_hp
    aid = _find_play(env, 52, target_code=9); assert aid, "Криста targeting Скелет (code 9) not found"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    p2_skel_after = next(c for c in env._env.state.p2.board if c.card_id == 27)
    print(f"  TAMHP p2->Скелет: hp {hp_b2}->{p2_skel_after.hp}, max_hp {maxhp_b2}->{p2_skel_after.max_hp}")
    assert p2_skel_after.max_hp == maxhp_b2 + 1, f"max_hp should bump +1, got {maxhp_b2}->{p2_skel_after.max_hp}"
    assert p2_skel_after.hp == hp_b2, f"hp should NOT change, got {hp_b2}->{p2_skel_after.hp}"
    print(f"  TAMHP p2 friendly-minion VERIFIED: max_hp+1, hp unchanged (BOTH SIDES)")
    print(f"  s7: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # T5 (p1, mana=10): play Криста (52) targeting own hero (code 16)
    print(f"  s8: {_state(env)}")
    hero_hp_b, hero_maxhp_b = env._env.state.p1.hero.hp, env._env.state.p1.hero.max_hp
    aid = _find_play(env, 52, target_code=16); assert aid, "Криста targeting own hero (code 16) not found"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    hero_hp_a, hero_maxhp_a = env._env.state.p1.hero.hp, env._env.state.p1.hero.max_hp
    print(f"  TAMHP p1->hero: hp {hero_hp_b}->{hero_hp_a}, max_hp {hero_maxhp_b}->{hero_maxhp_a}")
    assert hero_maxhp_a == hero_maxhp_b + 1, f"hero max_hp should bump +1, got {hero_maxhp_b}->{hero_maxhp_a}"
    assert hero_hp_a == hero_hp_b, f"hero hp should NOT change, got {hero_hp_b}->{hero_hp_a}"
    print(f"  TAMHP p1 hero-target VERIFIED: max_hp+1, hp unchanged")
    print(f"  s9: {_state(env)}")

    _gen("tamhp", seed, p1_deck, p2_deck, ids)


# ---------------------------------------------------------------------------

def main():
    gen_aoe_silence()
    gen_team_wide_shield()
    gen_tamhp()
    print("[phase5] done")


if __name__ == "__main__":
    main()
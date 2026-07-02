"""Phase 4 fixture generator: cleave, instant_kill, freeze, armor_X_Y.

Discovers the right action_ids by driving the Python env directly, then
calls build_golden_trace with those action_ids to produce recorded-outcome
RNG fixtures for Rust state-transition parity. The armor fixture populates
the new `randint_rolls` per-step stream (Phase 4 recorded-outcome RNG
extension for `random.randint`).

Usage:
  PYTHONPATH=.:TrainV3.5/python python3 -m train_v3.gen_phase4_fixtures
"""
from __future__ import annotations

import json
import sys

import numpy as np

from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.classic_actions_v1 import (
    _PLAY_BASE, _PLAY_STRIDE, _NUM_PLAY_TARGETS, _ATTACK_BASE, _NUM_ATTACK_TARGETS,
)

from .golden_trace import build_golden_trace


# ---------------------------------------------------------------------------
# Helpers (mirrors gen_phase3_fixtures)
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
    hand = [(c.card_id, c.name, c.mana_cost, c.attack, c.hp) for c in me.hand]
    board = [(c.card_id, c.name, c.attack, c.hp, c.is_ready, c.is_frozen, c.mechanics) for c in me.board]
    ob = [(c.card_id, c.name, c.attack, c.hp, c.is_ready, c.is_frozen) for c in opp.board]
    return (f"cp={cp} mana={me.mana} hero={me.hero.name}({me.hero.hp}hp,{me.hero.mechanics}) "
            f"hand={hand} board={board} opp_board={ob} opp_hero={opp.hero.name}({opp.hero.hp}hp)")


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
    # Report randint_rolls for the armor fixture.
    for st in trace["steps"]:
        if st.get("randint_rolls"):
            print(f"  step {st['t']} randint_rolls={st['randint_rolls']}")


# ---------------------------------------------------------------------------
# Cleave: card 23 Сукуна (cleave_1_2, 7 mana 7/7) attacks the middle of 3
# Скелет lvl5 (3/2). Splash 1 to neighbours (index 0 and 2).
# ---------------------------------------------------------------------------

def gen_cleave():
    print("[cleave] generating...")
    p1_deck = [27] * 8  # Скелет
    p2_deck = [23] * 8  # Сукуна (cleave_1_2)
    p1_levels = {27: 5}  # Скелет lvl5 → 3/2 (survives 1 cleave splash: 2-1=1)
    seed = 1100
    env = ClassicRLEnv(seed=seed, placement_mode="append_only", mana_per_turn=10)
    env.reset(seed=seed, p1_deck_ids=p1_deck, p2_deck_ids=p2_deck, p1_levels=p1_levels)
    ids = []

    # Turn 1 (p1, mana=1): play Скелет (1 mana) at pos 0.
    print(f"  s0: {_state(env)}")
    aid = _find_play(env, 27); assert aid, "Скелет not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)

    # p1 end turn
    print(f"  s1: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 2 (p2, mana=10): end (let p1 build board)
    print(f"  s2: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 3 (p1, mana=10): play Скелет at pos 1.
    print(f"  s3: {_state(env)}")
    aid = _find_play(env, 27); assert aid, "Скелет not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    print(f"  s4: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 4 (p2): end
    print(f"  s5: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 5 (p1, mana=10): play Скелет at pos 2.
    print(f"  s6: {_state(env)}")
    aid = _find_play(env, 27); assert aid, "Скелет not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    print(f"  s7: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 6 (p2, mana=10): play Сукуна (7 mana)
    print(f"  s8: {_state(env)}")
    aid = _find_play(env, 23); assert aid, "Сукуна not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    print(f"  s9: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 7 (p1, mana=10): end (let Сукуна become ready)
    print(f"  s10: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 8 (p2, mana=10): Сукуна attacks middle Скелет (target_code=1)
    print(f"  s11: {_state(env)}")
    aid = _find_attack(env, 23, target_code=1); assert aid, "attack not found"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)

    # Verify cleave: middle dead, neighbours at 1 hp.
    p1 = env._env.state.p1
    survivors = [(c.card_id, c.hp) for c in p1.board]
    print(f"  s12: {_state(env)}")
    print(f"  CLEAVE survivors={survivors}")
    assert len(p1.board) == 2, f"cleave should leave 2 neighbours, board={survivors}"
    assert all(c.hp == 1 for c in p1.board), f"neighbours should be 1hp, got {survivors}"
    print(f"  CLEAVE VERIFIED: 2 neighbours at 1hp")

    _gen("cleave", seed, p1_deck, p2_deck, ids, p1_levels=p1_levels)


# ---------------------------------------------------------------------------
# instant_kill: card 25 Сайтама (instant_kill, 10 mana 10/10) attacks
# Атакующий Титан lvl10 (card 42, 15/15). Normal damage leaves 5hp;
# instant_kill sets hp=0 → Титан dies (and Сайтама dies to 15 counter).
# ---------------------------------------------------------------------------

def gen_instant_kill():
    print("[instant_kill] generating...")
    p1_deck = [42] * 8  # Атакующий Титан
    p2_deck = [25] * 8  # Сайтама (instant_kill)
    p1_levels = {42: 10}  # 15/15 — HP > 10 so normal damage doesn't kill
    seed = 1200
    env = ClassicRLEnv(seed=seed, placement_mode="append_only", mana_per_turn=10)
    env.reset(seed=seed, p1_deck_ids=p1_deck, p2_deck_ids=p2_deck, p1_levels=p1_levels)
    ids = []

    # Turn 1 (p1, mana=1): end (Титан costs 6)
    print(f"  s0: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 2 (p2, mana=10): end (mana=1 actually turn 2... wait mana_per_turn=10)
    # Actually turn 1 mana=1, turn 2 mana=11? No — mana_per_turn=10 means +10/turn.
    # Turn 1 p1 mana=1. Turn 2 p2 mana=1+10=11? Let me just check state.
    print(f"  s1: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 3 (p1, mana=10): play Титан (6 mana)
    print(f"  s2: {_state(env)}")
    aid = _find_play(env, 42); assert aid, "Титан not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    print(f"  s3: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 4 (p2, mana=10): play Сайтама (10 mana)
    print(f"  s4: {_state(env)}")
    aid = _find_play(env, 25); assert aid, "Сайтама not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    print(f"  s5: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 5 (p1, mana=10): end (let Сайтама become ready)
    print(f"  s6: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 6 (p2, mana=10): Сайтама attacks Титан (target_code=0)
    print(f"  s7: {_state(env)}")
    aid = _find_attack(env, 25, target_code=0); assert aid, "attack not found"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)

    # Verify instant_kill: Титан dead (would be 5hp without instant_kill).
    p1 = env._env.state.p1
    p2 = env._env.state.p2
    titans = [c for c in p1.board if c.card_id == 42]
    assert len(titans) == 0, f"Титан should be dead via instant_kill, board={[(c.card_id,c.hp) for c in p1.board]}"
    saitama = [c for c in p2.board if c.card_id == 25]
    print(f"  INSTANT_KILL VERIFIED: Титан dead (instant_kill hp=0). Сайтама on board={len(saitama)}")
    _gen("instant_kill", seed, p1_deck, p2_deck, ids, p1_levels=p1_levels)


# ---------------------------------------------------------------------------
# Freeze: card 19 Саб-Зиро (battlecry_freeze, 4 mana 3/4) played targeting
# an enemy Скелет → is_frozen=True. End turn → thaw (is_frozen=False,
# is_ready=False — skips next activation).
# ---------------------------------------------------------------------------

def gen_freeze():
    print("[freeze] generating...")
    p1_deck = [27] * 8  # Скелет
    p2_deck = [19] * 8  # Саб-Зиро (battlecry_freeze)
    seed = 1300
    env = ClassicRLEnv(seed=seed, placement_mode="append_only", mana_per_turn=10)
    env.reset(seed=seed, p1_deck_ids=p1_deck, p2_deck_ids=p2_deck)
    ids = []

    # Turn 1 (p1, mana=1): play Скелет (1 mana)
    print(f"  s0: {_state(env)}")
    aid = _find_play(env, 27); assert aid, "Скелет not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    print(f"  s1: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 2 (p2, mana=10): play Саб-Зиро (4 mana) targeting Скелет (target_code=1)
    print(f"  s2: {_state(env)}")
    aid = _find_play(env, 19, target_code=1); assert aid, "Саб-Зиро not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)

    # Verify freeze applied
    p1 = env._env.state.p1
    skel = [c for c in p1.board if c.card_id == 27]
    assert len(skel) == 1 and skel[0].is_frozen, f"Скелет should be frozen, board={[(c.card_id,c.is_frozen) for c in p1.board]}"
    print(f"  FREEZE APPLIED: Скелет is_frozen={skel[0].is_frozen}")
    print(f"  s3: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 3 (p1, mana=10): end-turn thaw already happened — Скелет is_frozen=False, is_ready=False (skips)
    # p1 ends (Скелет can't attack).
    p1 = env._env.state.p1
    skel = [c for c in p1.board if c.card_id == 27]
    assert len(skel) == 1 and not skel[0].is_frozen and not skel[0].is_ready, (
        f"Скелет should be thawed+not ready, got is_frozen={skel[0].is_frozen} is_ready={skel[0].is_ready}")
    print(f"  FREEZE THAW+SKIP VERIFIED: is_frozen={skel[0].is_frozen} is_ready={skel[0].is_ready}")
    print(f"  s4: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    _gen("freeze", seed, p1_deck, p2_deck, ids)


# ---------------------------------------------------------------------------
# armor_X_Y: Даркнесс hero (card 5, armor_1_3, 45hp). Сукуна (7 atk) attacks
# the hero → apply_damage_modifiers rolls random.randint(1,3). The fixture
# records the roll in randint_rolls; the hero hp in the post-state reflects
# (45 - (7 - roll)).
# ---------------------------------------------------------------------------

def gen_armor():
    print("[armor] generating...")
    # p1 hero = Даркнесс (card 5, hero). The first hero-typed card in the
    # deck is extracted as the hero (core/classic_setup._extract_hero);
    # remaining 7 cards form the draw deck (Скелет, no taunt → hero attackable).
    # NOTE: core/converter._normalize_mechanic collapses `armor_1_3` →
    # `armor_1` (min) at deck-construction, so the engine never sees the
    # range form via prod decks. To exercise the engine's `random.randint`
    # armor_X_Y path (TA-4), we re-inject `armor_1_3` into the hero's
    # mechanics post-reset via the `post_reset_setup` hook. This is a
    # TEST-only injection — prod always carries the collapsed `armor_X`.
    p1_deck = [5, 27, 27, 27, 27, 27, 27, 27]
    p2_deck = [23] * 8  # Сукуна 7 atk
    seed = 1400
    env = ClassicRLEnv(seed=seed, placement_mode="append_only", mana_per_turn=10)
    env.reset(seed=seed, p1_deck_ids=p1_deck, p2_deck_ids=p2_deck)
    st = env._env.state
    p1 = st.p1 if st.p1.user_id == 1 else st.p2
    print(f"  p1 hero (pre-inject): {p1.hero.name} {p1.hero.mechanics} hp={p1.hero.hp}")
    assert p1.hero.card_id == 5, f"p1 hero should be Даркнесс, got {p1.hero.card_id}"

    def inject_armor_1_3(env):
        # Re-inject the range form so apply_damage_modifiers hits the
        # randint(1,3) path. The hero's base_mechanics are left as-is
        # (collapsed armor_1) — only the live mechanics are patched.
        s = env._env.state
        hero = s.p1.hero if s.p1.user_id == 1 else s.p2.hero
        hero.mechanics = ["armor_1_3"]

    ids = []

    # Turn 1 (p1, mana=1): end (no need to play)
    print(f"  s0: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 2 (p2, mana=10): play Сукуна (7 mana)
    print(f"  s1: {_state(env)}")
    aid = _find_play(env, 23); assert aid, "Сукуна not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    print(f"  s2: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 3 (p1, mana=10): end (let Сукуна become ready)
    print(f"  s3: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 4 (p2, mana=10): Сукуна attacks p1 hero (target_code=7)
    print(f"  s4: {_state(env)}")
    hero_hp_before = env._env.state.p1.hero.hp
    aid = _find_attack(env, 23, target_code=7); assert aid, "hero attack not found"
    print(f"  -> {_decode(aid)} (p1 hero hp before = {hero_hp_before})")
    ids.append(aid); env.step(aid)

    # Verify armor roll: hero hp = 45 - (7 - roll) = 38 + roll, roll in [1,3].
    hero_hp_after = env._env.state.p1.hero.hp
    roll = hero_hp_after - (hero_hp_before - 7)
    print(f"  ARMOR: hero hp {hero_hp_before} -> {hero_hp_after}, roll={roll}")
    assert 1 <= roll <= 3, f"armor roll out of range [1,3]: {roll}"
    print(f"  ARMOR VERIFIED: roll={roll} (hero took {7 - roll} damage)")

    # verify_mask=False: `_verify_mask` re-executes every legal action on a
    # cloned state to validate the mask. For the attack action that would
    # trigger `apply_damage_modifiers` → `random.randint(1,3)` on the CLONE,
    # polluting the recorded `randint_rolls` with phantom rolls that Rust
    # (which only rolls for the real attack) cannot replay. The state-
    # transition matcher ignores the mask, so disabling verification here
    # keeps the randint stream aligned (1 roll = the real attack) without
    # affecting parity coverage.
    _gen("armor", seed, p1_deck, p2_deck, ids, post_reset_setup=inject_armor_1_3,
         verify_mask=False)


# ---------------------------------------------------------------------------

def main():
    gen_cleave()
    gen_instant_kill()
    gen_freeze()
    gen_armor()
    print("[phase4] done")


if __name__ == "__main__":
    main()
"""Phase 3 fixture generator: rebirth, crime_and_punishment, consume_ally.

Discovers the right action_ids by driving the Python env directly, then
calls build_golden_trace with those action_ids to produce recorded-outcome
RNG fixtures for Rust state-transition parity.

Usage:
  PYTHONPATH=.:TrainV3.5/python python3 -m train_v3.gen_phase3_fixtures
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
# Helpers
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
    board = [(c.card_id, c.name, c.attack, c.hp, c.is_ready, c.mechanics) for c in me.board]
    ob = [(c.card_id, c.name, c.attack, c.hp, c.is_ready) for c in opp.board]
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


def _gen(name, seed, p1_deck, p2_deck, action_ids, mana_per_turn=10):
    trace = build_golden_trace(
        seed=seed,
        steps=len(action_ids),
        placement_mode="append_only",
        verify_mask=True,
        include_v5=True,
        choose="first",
        p1_deck_ids=p1_deck,
        p2_deck_ids=p2_deck,
        action_ids=action_ids,
        mana_per_turn=mana_per_turn,
    )
    path = f"TrainV3.5/rust/trainv3_core/tests/fixtures/golden_trace_{name}.json"
    with open(path, "w") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
    print(f"  wrote {path} ({len(trace['steps'])} steps)")


# ---------------------------------------------------------------------------
# Rebirth: card 50 Бан (rebirth_1, 8 mana 3/7) killed by Сайтама instant_kill
# ---------------------------------------------------------------------------

def gen_rebirth():
    print("[rebirth] generating...")
    p1_deck = [50] * 8  # all Бан (rebirth_1, 8 mana 3/7)
    p2_deck = [25] * 8  # all Сайтама (instant_kill, 10 mana 10/10)
    seed = 100
    env = ClassicRLEnv(seed=seed, placement_mode="append_only", mana_per_turn=10)
    env.reset(seed=seed, p1_deck_ids=p1_deck, p2_deck_ids=p2_deck)
    ids = []

    # Turn 1 (p1, mana=1): can't play Бан (8 mana) → end turn
    print(f"  s0: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 2 (p2, mana=10): play Сайтама (10 mana)
    print(f"  s1: {_state(env)}")
    aid = _find_play(env, 25); assert aid, "Сайтама not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)

    # p2 end turn
    print(f"  s2: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 3 (p1, mana=10): play Бан (8 mana)
    print(f"  s3: {_state(env)}")
    aid = _find_play(env, 50); assert aid, "Бан not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)

    # p1 end turn
    print(f"  s4: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 4 (p2, mana=10): Сайтама attacks Бан → instant_kill → rebirth
    print(f"  s5: {_state(env)}")
    aid = _find_attack(env, 25, target_code=0); assert aid, "attack not found"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)

    # Verify rebirth
    p1 = env._env.state.p1
    ban = [c for c in p1.board if c.card_id == 50]
    assert len(ban) == 1, f"Бан should be on board after rebirth, board={[(c.card_id,c.hp) for c in p1.board]}"
    assert ban[0].hp == 1, f"Бан should have 1 HP, got {ban[0].hp}"
    assert not any(m.startswith("rebirth_") for m in ban[0].mechanics), "rebirth consumed"
    print(f"  REBIRTH VERIFIED: Бан hp={ban[0].hp} mechanics={ban[0].mechanics}")

    _gen("rebirth", seed, p1_deck, p2_deck, ids)


# ---------------------------------------------------------------------------
# CAP: card 49 Достоевский hero (crime_and_punishment_2), friendly minion dies
# ---------------------------------------------------------------------------

def gen_cap():
    print("[cap] generating...")
    p1_deck = [49, 27, 27, 27, 27, 27, 27, 27]  # Достоевский hero + Скелет
    p2_deck = [25, 25, 25, 25, 25, 25, 25, 25]  # all Сайтама
    seed = 200
    env = ClassicRLEnv(seed=seed, placement_mode="append_only", mana_per_turn=10)
    env.reset(seed=seed, p1_deck_ids=p1_deck, p2_deck_ids=p2_deck)
    st = env._env.state
    print(f"  p1 hero: {st.p1.hero.name} {st.p1.hero.mechanics} hp={st.p1.hero.hp}")
    assert st.p1.hero.card_id == 49, f"p1 hero should be Достоевский, got {st.p1.hero.card_id}"
    ids = []

    # Turn 1 (p1, mana=1): play Скелет (1 mana 2/1)
    print(f"  s0: {_state(env)}")
    aid = _find_play(env, 27); assert aid, "Скелет not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)

    # p1 end turn
    print(f"  s1: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 2 (p2, mana=10): play Сайтама (10 mana 10/10)
    print(f"  s2: {_state(env)}")
    aid = _find_play(env, 25); assert aid, "Сайтама not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)

    # p2 end turn
    print(f"  s3: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 3 (p1, mana=10): end turn (let p2's Сайтама become ready)
    print(f"  s4: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 4 (p2, mana=10): Сайтама attacks Скелет → instant_kill → Скелет dies
    # CAP: Достоевский deals 2 to p2 hero (direct subtraction)
    print(f"  s5: {_state(env)}")
    p2_hero_before = env._env.state.p2.hero.hp
    aid = _find_attack(env, 25, target_code=0); assert aid, "attack not found"
    print(f"  -> {_decode(aid)} (p2 hero hp before = {p2_hero_before})")
    ids.append(aid); env.step(aid)

    # Verify CAP
    p1 = env._env.state.p1
    p2 = env._env.state.p2
    skeleton = [c for c in p1.board if c.card_id == 27]
    assert len(skeleton) == 0, "Скелет should be dead"
    assert p2.hero.hp == p2_hero_before - 2, f"CAP: p2 hero should be {p2_hero_before-2}, got {p2.hero.hp}"
    print(f"  CAP VERIFIED: p2 hero hp={p2.hero.hp} (was {p2_hero_before}, -2 CAP direct subtraction)")

    _gen("cap", seed, p1_deck, p2_deck, ids)


# ---------------------------------------------------------------------------
# consume_ally: card 20 Канеки (consume_ally, 3 mana 2/2) consumes friendly
# ---------------------------------------------------------------------------

def gen_consume_ally():
    print("[consume_ally] generating...")
    p1_deck = [20] * 8  # all Канеки (consume_ally, 3 mana 2/2)
    p2_deck = [27] * 8  # all Скелет (1 mana 2/1)
    seed = 300
    env = ClassicRLEnv(seed=seed, placement_mode="append_only", mana_per_turn=10)
    env.reset(seed=seed, p1_deck_ids=p1_deck, p2_deck_ids=p2_deck)
    ids = []

    # Turn 1 (p1, mana=1): can't play Канеки (3 mana) → end turn
    print(f"  s0: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 2 (p2, mana=10): play Скелет, end turn
    print(f"  s1: {_state(env)}")
    aid = _find_play(env, 27); assert aid, "Скелет not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    print(f"  s2: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 3 (p1, mana=10): play Канеки #1 (3 mana, 2/2, no consume target needed yet)
    # Wait — Канеки has consume_ally which requires a target. But board is empty.
    # consume_ally requires target → target_required error if board is empty.
    # So I need a friendly minion on board first. Let me play Канеки without
    # consume... but Канеки always has consume_ally. I need a different approach:
    # use a deck with some non-consume minions to put on board first.

    # Actually, with all-Канеки deck, I can't play the first Канеки (no target).
    # Let me use a mixed deck: some Скелет + some Канеки.
    print("  [redirect] need non-consume minion on board first — using mixed deck")
    return gen_consume_ally_mixed()


def gen_consume_ally_mixed():
    print("[consume_ally] generating (mixed deck)...")
    # Скелет (27, 1 mana 2/1) + Канеки (20, 3 mana 2/2 consume_ally)
    # _deal_starting_hand sorts warriors by mana_cost, takes 3 cheapest.
    # With deck [27, 27, 20, 20, 20, 20, 20, 20]:
    #   warriors sorted: [Скелет(1), Скелет(1), Канеки(3)×6]
    #   hand = [Скелет, Скелет, Канеки]  (3 cheapest)
    #   deck = [Канеки×5]  (remaining warriors, shuffled)
    p1_deck = [27, 27, 20, 20, 20, 20, 20, 20]
    p2_deck = [27, 27, 27, 27, 27, 27, 27, 27]
    seed = 301
    env = ClassicRLEnv(seed=seed, placement_mode="append_only", mana_per_turn=10)
    env.reset(seed=seed, p1_deck_ids=p1_deck, p2_deck_ids=p2_deck)
    ids = []

    # Turn 1 (p1, mana=1): play Скелет (1 mana 2/1)
    print(f"  s0: {_state(env)}")
    aid = _find_play(env, 27); assert aid, "Скелет not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)

    # p1 end turn
    print(f"  s1: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 2 (p2, mana=10): end turn (we don't care about p2)
    print(f"  s2: {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 3 (p1, mana=10): play Канеки (3 mana, 2/2, consume_ally)
    # targeting friendly Скелет (board[0]) → target_code=9
    print(f"  s3: {_state(env)}")
    aid = _find_play(env, 20, target_code=9); assert aid, "Канеки consume action not found"
    print(f"  -> {_decode(aid)}")
    skeleton_before = [c for c in env._env.state.p1.board if c.card_id == 27]
    kaneki_hand = [c for c in env._env.state.p1.hand if c.card_id == 20]
    print(f"  Скелет on board: {[(c.card_id,c.attack,c.hp) for c in skeleton_before]}")
    print(f"  Канеки in hand: {[(c.card_id,c.attack,c.hp) for c in kaneki_hand]}")
    ids.append(aid); env.step(aid)

    # Verify consume_ally
    p1 = env._env.state.p1
    skeletons = [c for c in p1.board if c.card_id == 27]
    kanekis = [c for c in p1.board if c.card_id == 20]
    assert len(skeletons) == 0, f"Скелет should be consumed (removed from board), board={[(c.card_id,c.attack,c.hp) for c in p1.board]}"
    assert len(kanekis) == 1, f"Канеки should be on board, board={[(c.card_id,c.attack,c.hp) for c in p1.board]}"
    # Канеки base 2/2 + Скелет 2/1 = 4/3
    assert kanekis[0].attack == 4, f"Канеки attack should be 2+2=4, got {kanekis[0].attack}"
    assert kanekis[0].hp == 3, f"Канеки hp should be 2+1=3, got {kanekis[0].hp}"
    assert kanekis[0].max_hp == 3, f"Канеки max_hp should be 2+1=3, got {kanekis[0].max_hp}"
    assert len(p1.graveyard) == 1 and p1.graveyard[0].card_id == 27, "Скелет in graveyard"
    print(f"  CONSUME_ALLY VERIFIED: Канеки {kanekis[0].attack}/{kanekis[0].hp} on board, Скелет in graveyard")

    _gen("consume_ally", seed, p1_deck, p2_deck, ids)


if __name__ == "__main__":
    gen_rebirth()
    gen_cap()
    gen_consume_ally()
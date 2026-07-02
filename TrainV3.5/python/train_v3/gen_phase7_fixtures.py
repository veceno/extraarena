"""Phase 7 fixture generator: consume_ally_full + mana_drain.

Discovers the right action_ids by driving the Python env directly, then
calls build_golden_trace with those action_ids to produce recorded-outcome
RNG fixtures for Rust state-transition parity.

consume_ally_full (CLN-3 full-board): board.len()==5 + a consume_ally warrior
(Канеки, card 20) in hand. The frozen codec mask
(classic_actions_v1._mask_play_actions:228) does NOT exempt consume_ally, so
the consume play is masked OUT at a full board — the model never sees it.
The engine apply path (core/engine.py:1228) and the Rust apply_play_card
board-full guard STILL exempt consume_ally, so the play succeeds when FORCED.
The generator constructs the consume play action_id directly (the mask will
not expose it) and force-applies it via the engine apply path
(build_golden_trace force_steps → _forced_engine_step, bypassing the TrainV2
mask check). The fixture's recorded mask therefore has the consume play bit
DISABLED (0.0) — matching the reverted Rust action_mask byte-for-byte. The
Rust state-transition parity test forces the action via apply_action_unchecked
(mask-bypassing) and compares post-state JSON only.

FIX 3 (mana_drain two-stage): card 12 Кража Маны (mana_drain_2) — exercises
the immediate drain (core/effects.py:540-580) AND the pending end-turn drain
(core/engine.py:700-703). Needs NO monkeypatch — it does not use the consume
mask.

Usage:
  PYTHONPATH=.:TrainV3.5/python python3 -m train_v3.gen_phase7_fixtures
"""
from __future__ import annotations

import json
import sys

import numpy as np

from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.classic_actions_v1 import (
    _PLAY_BASE, _PLAY_STRIDE, _NUM_PLAY_TARGETS, _ATTACK_BASE, _NUM_ATTACK_TARGETS,
    _NUM_PLAY_POS, decode_action,
)
from core.state import CardType
from core.effects import requires_target, is_random_battlecry_damage_card

from .golden_trace import build_golden_trace


# ---------------------------------------------------------------------------
# Helpers (mirrors gen_phase3_fixtures.py)
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
    return (f"cp={cp} mana={me.mana}/{me.max_mana} opp_mana={opp.mana}/{opp.max_mana} "
            f"hand={hand} board({len(board)})={board} opp_board({len(ob)})={ob}")


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


def _force_engine_step(env, action_id):
    """Apply a (possibly mask-illegal) action_id via the engine apply path
    (env._env.step), bypassing the TrainV2 action_mask. Used in the driver
    loop to advance state + verify post-conditions for the consume_ally play
    at a full board (the frozen mask masks it out, but the engine exempts
    consume_ally). build_golden_trace replays the same action via force_steps.
    """
    cp = env.current_player_id()
    st = env._env.state
    action = decode_action(st, cp, action_id)
    if action is None:
        return False, "decode_failed"
    success, error = env._env.step(cp, action)
    if success and env._cache is not None:
        env._cache.set_state(env._env.state, env.current_player_id())
    return success, error


END_TURN = 0


def _gen(name, seed, p1_deck, p2_deck, action_ids, mana_per_turn=10, force_steps=None):
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
        force_steps=force_steps,
    )
    path = f"TrainV3.5/rust/trainv3_core/tests/fixtures/golden_trace_{name}.json"
    with open(path, "w") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
    print(f"  wrote {path} ({len(trace['steps'])} steps)")


# ---------------------------------------------------------------------------
# consume_ally_full: board.len()==5 + consume_ally in hand.
# The frozen mask masks the consume play OUT at a full board, so the action_id
# is constructed directly and force-applied via the engine apply path.
# ---------------------------------------------------------------------------

def gen_consume_ally_full():
    print("[consume_ally_full] generating...")
    # 5 Скелет (27, 1 mana) + 3 Канеки (20, 3 mana consume_ally).
    # Starting hand = 3 cheapest warriors = [27,27,27]. deck = [27,27, 20,20,20].
    p1_deck = [27, 27, 27, 27, 27, 20, 20, 20]
    p2_deck = [27, 27, 27, 27, 27, 27, 27, 27]
    seed = 310
    env = ClassicRLEnv(seed=seed, placement_mode="append_only", mana_per_turn=10)
    env.reset(seed=seed, p1_deck_ids=p1_deck, p2_deck_ids=p2_deck)
    ids = []
    force_steps = set()

    # Drive: fill p1 board to 5 with Скелет across turns, then play Канеки
    # (consume_ally) targeting board[0]. p2 just ends turns.
    for _ in range(40):
        cp = env.current_player_id()
        st = env._env.state
        me = st.p1 if st.p1.user_id == cp else st.p2
        print(f"  {_state(env)}")
        if cp == 1:
            # If a Канеки (consume_ally) is in hand and board is full -> play it.
            if len(me.board) >= 5 and any(c.card_id == 20 for c in me.hand):
                hand_idx = next(i for i, c in enumerate(me.hand) if c.card_id == 20)
                # Append position (board.len() capped to the play-position
                # range). The consume happens BEFORE insert, so the new card
                # lands at board.len() after the consumed ally is removed.
                pos = min(len(me.board), _NUM_PLAY_POS - 1)
                target_code = 9  # consume friendly board[0]
                aid = _PLAY_BASE + hand_idx * _PLAY_STRIDE + pos * _NUM_PLAY_TARGETS + target_code
                print(f"  -> {_decode(aid)} (consume_ally on full board, FORCED via engine)")
                ok, err = _force_engine_step(env, aid)
                assert ok, f"consume play failed via engine: {err}"
                ids.append(aid)
                force_steps.add(len(ids) - 1)
                # verify consume happened
                p1 = env._env.state.p1
                kanekis = [c for c in p1.board if c.card_id == 20]
                assert len(kanekis) == 1, f"Канеки not on board: {[(c.card_id,c.attack,c.hp) for c in p1.board]}"
                assert len(p1.board) == 5, f"board should stay 5 after consume, got {len(p1.board)}"
                print(f"  CONSUME_ALLY_FULL VERIFIED: board stays 5, Канеки {kanekis[0].attack}/{kanekis[0].hp}")
                break
            # else play a Скелет to fill board
            aid = _find_play(env, 27)
            if aid is not None and len(me.board) < 5:
                print(f"  -> {_decode(aid)} (fill Скелет)")
                ids.append(aid); env.step(aid); continue
            # else end turn
            print("  -> EndTurn (p1)")
            ids.append(END_TURN); env.step(END_TURN); continue
        else:
            # p2: end turn
            print("  -> EndTurn (p2)")
            ids.append(END_TURN); env.step(END_TURN); continue

    _gen("consume_ally_full", seed, p1_deck, p2_deck, ids, force_steps=force_steps)


# ---------------------------------------------------------------------------
# mana_drain: card 12 Кража Маны (mana_drain_2) — immediate + pending drain
# ---------------------------------------------------------------------------

def gen_mana_drain():
    print("[mana_drain] generating...")
    # p1: 5 Кража Маны (12, potion 3 mana, mana_drain_2) + 3 Скелет (27, 1 mana).
    #   Starting hand = 3 cheapest warriors = [27,27,27]. deck = [12,12,12,12,12]
    #   (all Кража) -> turn-1-end draw guarantees a Кража in hand by turn 3.
    # p2: Годжо (24, 9 mana) -> p2 spends 9 mana on turn 2, leaving mana=1.
    #   Кража (drain 2) vs p2 mana=1 -> current_drained=1 (immediate) +
    #   future_drained=1 (pending). End-turn then applies the pending 1.
    p1_deck = [12, 12, 12, 12, 12, 27, 27, 27]
    p2_deck = [24, 24, 24, 24, 24, 24, 24, 24]
    seed = 320
    env = ClassicRLEnv(seed=seed, placement_mode="append_only", mana_per_turn=10)
    env.reset(seed=seed, p1_deck_ids=p1_deck, p2_deck_ids=p2_deck)
    ids = []

    # Turn 1 (p1 mana=1): play Скелет (filler). end turn.
    print(f"  {_state(env)}")
    aid = _find_play(env, 27); assert aid, "Скелет not in hand"
    print(f"  -> {_decode(aid)}"); ids.append(aid); env.step(aid)
    print(f"  {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 2 (p2 mana=10): p2 plays Годжо (9 mana -> mana=1). end turn.
    print(f"  {_state(env)}")
    aid = _find_play(env, 24); assert aid, "Годжо not in hand for p2"
    print(f"  -> {_decode(aid)} (p2 spends 9 mana)"); ids.append(aid); env.step(aid)
    p2 = env._env.state.p2
    assert p2.mana == 1, f"p2 mana should be 1 after Годжо, got {p2.mana}"
    print(f"  p2 mana after Годжо = {p2.mana}/{p2.max_mana}")
    print(f"  {_state(env)}")
    ids.append(END_TURN); env.step(END_TURN)

    # Turn 3 (p1 mana=10): play Кража Маны (mana_drain_2) vs p2 mana=1.
    print(f"  {_state(env)}")
    aid = _find_play(env, 12); assert aid, "Кража Маны not in hand"
    print(f"  -> {_decode(aid)} (mana_drain_2)")
    p2_mana_before = env._env.state.p2.mana
    p1_mana_before = env._env.state.p1.mana
    ids.append(aid); env.step(aid)
    p2 = env._env.state.p2
    p1 = env._env.state.p1
    assert p2.mana == 0, f"p2 mana should be 0 after immediate drain of 1, got {p2.mana}"
    print(f"  IMMEDIATE DRAIN: p2 mana {p2_mana_before}->0, p1 mana {p1_mana_before}->{p1.mana}")
    # pending scheduled for p2 next turn
    pending = env._env.state.pending_mana_drain_by_player.get(p2.user_id, 0)
    assert pending == 1, f"pending mana drain should be 1, got {pending}"
    print(f"  PENDING SCHEDULED: pending_mana_drain_by_player[{p2.user_id}]={pending}")
    print(f"  {_state(env)}")

    # p1 end turn -> p2 turn 4: mana restored to 10 then pending 1 applied -> 9.
    ids.append(END_TURN); env.step(END_TURN)
    p2 = env._env.state.p2
    assert p2.mana == 9, f"p2 mana should be 9 after restore(10)-pending(1), got {p2.mana}"
    assert p2.user_id not in env._env.state.pending_mana_drain_by_player, "pending should be popped"
    print(f"  PENDING APPLIED at p2 turn start: p2 mana={p2.mana}/{p2.max_mana} (10-1=9)")
    print(f"  {_state(env)}")

    _gen("mana_drain", seed, p1_deck, p2_deck, ids)


if __name__ == "__main__":
    gen_consume_ally_full()
    gen_mana_drain()
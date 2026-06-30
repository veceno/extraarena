"""Phase 9 fixture generator: cast_random_spell (card 26 Мидория).

Exercises all 4 cast_random_spell variants (Texas Smash / Recovery / Blackwhip
/ Full Cowl) via recorded-outcome RNG so Rust state-transition parity can pin
the spell choice + choice_rolls (spell 1 target) + sample_rolls (spell 3 freeze
targets).

card 26: warrior, 5 mana, 5/5, mechanics=[cast_random_spell], level=1 in the
classic training env → scaling is dmg=4, heal=5, freeze_count=1, buff=2.

The frozen mask exposes cast_random_spell at the no-target slot (base+0), so
every play is mask-legal — NO force_steps needed. verify_mask=False isolates
the trace from the _verify_mask clone-re-exec (established RNG-stream pattern).

Usage:
  PYTHONPATH=.:TrainV3.5/python python3 -m train_v3.gen_phase9_fixtures
"""
from __future__ import annotations

import json
import sys

import numpy as np

from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.classic_actions_v1 import (
    _PLAY_BASE, _PLAY_STRIDE, _NUM_PLAY_TARGETS, _ATTACK_BASE, _NUM_ATTACK_TARGETS,
)
from core.state import CardType, GameStatus

from .golden_trace import build_golden_trace


SPELL_MARKERS = {
    1: "Техасский удар",
    2: "Восстановление",
    3: "Чёрный кнут",
    4: "Полный покров",
}


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


def _find_attack(env, attacker_idx, target_idx):
    """Encode the attack action_id for me.board[attacker_idx] attacking
    enemy.board[target_idx] (target_idx 0-based). Mirrors the 601 codec:
    ATTACK_BASE + attacker_idx*NUM_ATTACK_TARGETS + (target_idx+1)."""
    from ai.train_v2.classic_actions_v1 import _ATTACK_BASE as AB, _NUM_ATTACK_TARGETS as NAT
    return AB + attacker_idx * NAT + (target_idx + 1)


def _find_legal_attack(env, attacker_idx, enemy_minion_idx):
    """Find a legal attack action_id matching me.board[attacker_idx] →
    enemy.board[enemy_minion_idx], or None."""
    cp = env.current_player_id()
    me = env._env.state.p1 if env._env.state.p1.user_id == cp else env._env.state.p2
    if attacker_idx >= len(me.board) or not me.board[attacker_idx].is_ready:
        return None
    target_code = enemy_minion_idx + 1
    for aid in _legal_ids(env):
        if aid < _ATTACK_BASE:
            continue
        a = (aid - _ATTACK_BASE) // _NUM_ATTACK_TARGETS
        t = (aid - _ATTACK_BASE) % _NUM_ATTACK_TARGETS
        if a == attacker_idx and t == target_code:
            return aid
    return None


def _last_spell(state):
    """Return the spell number (1-4) just cast, or None, from action_history.

    action_history is a deque of (log_type, text) tuples. The cast_random_spell
    effect appends its spell line BEFORE the engine appends the
    'Мидория выставлен' play event, so the spell line sits at index -2 (or
    -3 when a turn-header system line interleaves). Scan the last few entries
    for a spell marker. NOTE: spell 3 (Blackwhip) only appends when
    unfrozen_enemies is non-empty, so p2 must keep a populated board for spell
    3 to be detectable.
    """
    ah = state.action_history
    if not ah:
        return None
    n = len(ah)
    for i in range(max(0, n - 4), n):
        entry = ah[i]
        text = entry[1] if isinstance(entry, (tuple, list)) else str(entry)
        for num, marker in SPELL_MARKERS.items():
            if marker in text:
                return num
    return None


END_TURN = 0


def _gen(name, seed, p1_deck, p2_deck, action_ids, mana_per_turn=10):
    trace = build_golden_trace(
        seed=seed,
        steps=len(action_ids),
        placement_mode="append_only",
        verify_mask=False,
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


def gen_cast_random_spell():
    print("[cast_random_spell] generating...")
    # p1: 8x Мидория (26, 5 mana cast_random_spell, 5/5). Single card type →
    # weighted draw always picks card 26 (deterministic draw).
    # p2: 8x Годжо (24, 9 mana, 5/6, shield+shield_refresh) — p2 keeps a 5/6
    # minion on board. p1 suicides its ready Мидория into Годжо to free board
    # slots (Мидория 5/5 vs Годжо 5/6: Мидория dies, Годжо survives at 1 hp).
    # This lets p1 keep playing Мидория across many turns → many spell casts.
    # p2's Годжо also serves as the enemy minion target for spells 1 (Texas
    # Smash) and 3 (Blackwhip).
    p1_deck = [26, 26, 26, 26, 26, 26, 26, 26]
    p2_deck = [24, 24, 24, 24, 24, 24, 24, 24]

    chosen_seed = None
    chosen_ids = None
    chosen_seen = None
    # Try seeds until all 4 spells are exercised within a bounded turn budget.
    for seed in range(400, 700):
        env = ClassicRLEnv(seed=seed, placement_mode="append_only", mana_per_turn=10)
        env.reset(seed=seed, p1_deck_ids=p1_deck, p2_deck_ids=p2_deck)
        ids = []
        seen: set[int] = set()
        ok = False
        for _ in range(80):
            cp = env.current_player_id()
            st = env._env.state
            if cp == 1:
                me = st.p1
                # First, if the board is full and a ready Мидория exists, suicide
                # it into p2's biggest minion to free a slot for more casts.
                if len(me.board) >= 5:
                    # find a ready Мидория
                    att_idx = next(
                        (i for i, c in enumerate(me.board)
                         if c.card_id == 26 and c.is_ready),
                        None,
                    )
                    # find p2 minion with highest attack to suicide into
                    enemy = st.p2
                    tgt_idx = None
                    if enemy.board:
                        tgt_idx = max(
                            range(len(enemy.board)),
                            key=lambda j: enemy.board[j].attack,
                        )
                    if att_idx is not None and tgt_idx is not None:
                        aid = _find_legal_attack(env, att_idx, tgt_idx)
                        if aid is not None:
                            ids.append(aid); env.step(aid)
                            st = env._env.state
                            me = st.p1
                # Play Мидория if mana + hand + board space.
                if me.mana >= 5 and len(me.board) < 5:
                    aid = _find_play(env, 26)
                    if aid is not None:
                        ids.append(aid); env.step(aid)
                        st = env._env.state
                        spell = _last_spell(st)
                        if spell is not None:
                            seen.add(spell)
                        me = st.p1
                ids.append(END_TURN); env.step(END_TURN)
            else:
                me2 = st.p2
                # p2: play Годжо if mana + board space, else attack p1's biggest
                # minion with a ready Годжо (clears p1 board → more p1 casts),
                # else end turn.
                played = False
                if me2.mana >= 9 and len(me2.board) < 5:
                    aid = _find_play(env, 24)
                    if aid is not None:
                        ids.append(aid); env.step(aid); st = env._env.state
                        played = True
                if not played:
                    me2 = st.p2
                    # find a ready Годжо to attack p1's biggest minion
                    att_idx = next(
                        (i for i, c in enumerate(me2.board)
                         if c.card_id == 24 and c.is_ready),
                        None,
                    )
                    enemy = st.p1
                    tgt_idx = None
                    if enemy.board:
                        tgt_idx = max(
                            range(len(enemy.board)),
                            key=lambda j: enemy.board[j].attack,
                        )
                    if att_idx is not None and tgt_idx is not None:
                        aid = _find_legal_attack(env, att_idx, tgt_idx)
                        if aid is not None:
                            ids.append(aid); env.step(aid); st = env._env.state
                            played = True
                if not played:
                    ids.append(END_TURN); env.step(END_TURN)
            if len(seen) == 4:
                ok = True
                break
            if st.status != GameStatus.ONGOING:
                break
        if ok:
            chosen_seed = seed
            chosen_ids = ids
            chosen_seen = seen
            print(f"  seed={seed} exercised all 4 spells: {sorted(seen)} in {len(ids)} steps")
            break
    if chosen_seed is None:
        raise RuntimeError("could not find a seed exercising all 4 cast_random_spell variants")

    _gen("cast_random_spell", chosen_seed, p1_deck, p2_deck, chosen_ids)

    # Sanity: report the recorded RNG streams per cast_random_spell step.
    trace = json.load(
        open("TrainV3.5/rust/trainv3_core/tests/fixtures/golden_trace_cast_random_spell.json")
    )
    print("  per-step RNG (action_id, randint_rolls, choice_rolls, sample_rolls):")
    for step in trace["steps"]:
        if step["action_id"] == 0:
            continue
        r = step["randint_rolls"]
        c = step["choice_rolls"]
        s = step["sample_rolls"]
        if r or c or s:
            print(f"    t={step['t']} aid={step['action_id']} randint={r} choice={c} sample={s}")
    print(f"  spells exercised: {sorted(chosen_seen)}")


if __name__ == "__main__":
    gen_cast_random_spell()
"""Block 0 component 5 — offline-bridge tests (TRACKED ``test_train_v2_*``).

Six gates for ``ai/train_v2/offline_dataset_loader.py`` (the recorded v5/
trace -> AWAC/CRR offline-PPO replay bridge):

  (a) ``test_deserializer_round_trip_obs_integrity`` — CANONICAL gate
      (spec §6.185): for a recorded battle, ``obs(reconstruct(post_N)) ==
      obs(reconstruct(pre_{N+1}))`` within float tol. Proves deserializer
      DETERMINISM + snapshot continuity.
  (b) ``test_deserializer_correctness_vs_known_state`` — SOURCE-VS-SOURCE
      gate (NOT self-referential): a known LIVE GameState (driven via
      ClassicRLEnv) is snapshotted with the REAL ``v5_trace._snapshot_state``
      serializer (driven via a stub recorder), reconstructed by the bridge,
      and ``obs(reconstructed) == obs(live)`` within tol. The live engine is
      the ORACLE; the deserializer is the UUT.
  (c) ``test_reward_byte_matches_classic_rl_env`` —
      ``compute_offline_reward`` EXACTLY equals
      ``classic_rl_env._compute_reward`` for invalid/win/loss/draw/shaped
      cases.
  (d) ``test_orphans_skipped`` — a battle with ``meta.status='ongoing'`` is
      skipped; a terminal battle is included.
  (e) ``test_surrender_row_terminal`` — a surrender action row produces a
      terminal transition (``terminal=True``) with the terminal reward
      (±1.0/0.0 from status) and is the last row of its battle.
  (f) ``test_mana_draw_legal_flag_populated`` — for a mana_draw-legal row +
      a mana_draw-illegal row, ``OfflineTransition.mana_draw_legal`` matches
      ``mana_draw_head_v5.mana_draw_legal_mask`` on the reconstructed state.

Oracle strategy (b): the REAL ``V5TraceRecorder._snapshot_state`` /
``_snapshot_player`` / ``RlhfBattleEngine._snapshot_card`` serializer is
driven against a live ``ClassicRLEnv`` state via ``StubRecorder`` (the
recorder is coupled to ``RlhfBattleEngine`` via ``engine._arena.state`` +
``engine._snapshot_card``, NOT to ``ClassicRLEnv``; the stub wires
``engine._snapshot_card`` to the real staticmethod and
``engine._arena.state`` to the ClassicRLEnv live state, so the unbound
``V5TraceRecorder._snapshot_state`` runs against the live state). This
produces a REAL-schema snapshot (the exact field set the bridge deserializer
must consume) WITHOUT the full rlhf match_runner/DB/socket infra — the
strongest oracle short of a prod-recorded battle.

Run: ``PYTHONPATH=.:TrainV3.5/python python3 -m pytest tests/test_train_v2_offline_bridge.py``
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from ai.train_v2.classic_actions_v1 import encode_action_features
from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.offline_dataset_loader import (
    OfflineTransition,
    compute_offline_reward,
    iter_offline_transitions,
    load_offline_dataset,
    reconstruct_gamestate,
    reward_view_from_snapshot,
)
from core.state import GameStatus
from rlhf_env.components.arena_engine import RlhfBattleEngine
from rlhf_env.components.v5_trace import V5TraceRecorder
from train_v3.contracts import AssistModeV5, InfoModeV5
from train_v3.mana_draw_head_v5 import mana_draw_legal_mask
from train_v3.obs_v5 import encode_observation_v5


# ---------------------------------------------------------------------------
# Oracle: drive the REAL v5_trace serializer against a live ClassicRLEnv state
# ---------------------------------------------------------------------------


class StubRecorder:
    """Drive the REAL ``V5TraceRecorder`` serializer methods against a live
    state (ClassicRLEnv or any ``GameState``).

    ``V5TraceRecorder`` is coupled to ``RlhfBattleEngine`` via
    ``engine._arena.state`` (v5_trace.py:299) + ``engine._snapshot_card``
    (the staticmethod at arena_engine.py:914). The stub wires
    ``engine._arena.state`` to the live state and ``engine._snapshot_card``
    to the real staticmethod, so the unbound ``V5TraceRecorder._snapshot_state``
    / ``_snapshot_player`` / ``_snapshot_card`` run against the live state —
    producing a snapshot with the EXACT v5_trace field set, WITHOUT the full
    rlhf match_runner/DB/socket infra.
    """

    def __init__(self, live_state):
        self.engine = SimpleNamespace(
            _arena=SimpleNamespace(state=live_state),
            _snapshot_card=RlhfBattleEngine._snapshot_card,
        )

    def _snapshot_card(self, card):
        return V5TraceRecorder._snapshot_card(self, card)

    def _snapshot_player(self, p):
        return V5TraceRecorder._snapshot_player(self, p)

    def _snapshot_v5_history_event(self, event):
        return V5TraceRecorder._snapshot_v5_history_event(self, event)

    def snapshot_state(self):
        return V5TraceRecorder._snapshot_state(self)


def _drive_env(env: ClassicRLEnv, n_steps: int):
    """Drive ``n_steps`` valid steps. Returns the list of (action_id, ) taken."""
    taken = []
    for _ in range(n_steps):
        mask = env.action_mask()
        ids = [i for i in range(601) if mask[i] == 1.0]
        if not ids:
            break
        env.step(ids[0])
        taken.append(ids[0])
    return taken


def _action_type_for(action_id: int) -> str:
    if action_id == 0:
        return "end_turn"
    if 1 <= action_id <= 544:
        return "play_card"
    if 545 <= action_id <= 600:
        return "attack"
    return "unknown"


def _make_omniscient_info_mode() -> InfoModeV5:
    """The trace is omniscient; use an omniscient info mode for the oracle obs."""
    return InfoModeV5(
        own_hand_identity_known=True,
        own_deck_known=True,
        enemy_hand_known=True,
        enemy_deck_known=True,
        enemy_deck_order_known=True,
    )


# ---------------------------------------------------------------------------
# (b) SOURCE-VS-SOURCE correctness gate (live = oracle)
# ---------------------------------------------------------------------------


def test_deserializer_correctness_vs_known_state():
    """obs(reconstructed) == obs(live) for a known live ClassicRLEnv state.

    NOT self-referential: the live state is driven through the real engine
    (ClassicRLEnv) and snapshotted with the REAL v5_trace serializer
    (StubRecorder); the bridge deserializer is the UUT. A few different turn
    states are checked (different boards/hands/history).
    """
    env = ClassicRLEnv(seed=11)
    env.reset()
    info = _make_omniscient_info_mode()
    assist = AssistModeV5()

    # Check several states across the early game so board/hand/history vary.
    for expected_steps in (0, 2, 4, 6, 9):
        # Re-drive from a fresh env to land on distinct states.
        env = ClassicRLEnv(seed=11)
        env.reset()
        _drive_env(env, expected_steps)
        live = env._env.state
        actor = live.current_turn_owner_id
        if live.status != GameStatus.ONGOING:
            continue  # game ended early; skip

        snap = StubRecorder(live).snapshot_state()
        recon = reconstruct_gamestate(snap)

        obs_live = encode_observation_v5(
            live, actor, info_mode=info, assist_mode=assist,
            history_events=list(live.v5_history_events),
        )
        obs_recon = encode_observation_v5(
            recon, actor, info_mode=info, assist_mode=assist,
            history_events=snap["v5_history_events"],
        )
        assert obs_recon.shape == (7128,)
        assert np.allclose(obs_live, obs_recon, atol=1e-6), (
            f"obs mismatch at steps={expected_steps}: "
            f"max abs diff={float(np.max(np.abs(obs_live - obs_recon)))}"
        )
        # Also assert the reconstructed action_history round-trips as tuples.
        assert list(map(tuple, snap["action_history"])) == list(recon.action_history)
        # mana_draw_legal parity (reconstructed vs live).
        assert mana_draw_legal_mask(recon, actor) == mana_draw_legal_mask(live, actor)


# ---------------------------------------------------------------------------
# (a) CANONICAL round-trip obs-integrity gate (spec §6.185)
# ---------------------------------------------------------------------------


def _write_real_trace(group_dir: Path, battle_id: str, *, n_steps: int,
                      seed: int, meta_status: str = "p2_win",
                      append_surrender: bool = False,
                      surrender_actor_player: int = 2) -> int:
    """Drive a real ClassicRLEnv battle, write manifest+meta+actions.jsonl,
    return the number of action rows written.

    Each action row mirrors the v5_trace schema: pre_state/post_state via the
    REAL serializer, action_type/actor_user_id/actor_player/seq/accepted/
    legal_action_index. ``post_state`` of row N == ``pre_state`` of row N+1
    by construction (the continuity invariant) because each is snapshotted
    from the same live state at the same point.
    """
    env = ClassicRLEnv(seed=seed)
    env.reset()
    live = env._env.state
    stub = StubRecorder(live)

    rows = []
    seq = 0
    for _ in range(n_steps):
        if live.status != GameStatus.ONGOING:
            break
        actor = live.current_turn_owner_id
        pre_snap = stub.snapshot_state()
        mask = env.action_mask()
        ids = [i for i in range(601) if mask[i] == 1.0]
        if not ids:
            break
        action_id = ids[0]
        # legal_action_index into the engine's raw legal list (best-effort;
        # not a binding gate — find the decoded action in get_legal_actions).
        legal_raw = env._env.get_legal_actions(actor)
        from ai.train_v2.classic_actions_v1 import decode_action
        chosen = decode_action(live, actor, action_id)
        legal_index = None
        if chosen is not None:
            for i, a in enumerate(legal_raw):
                if a.__class__ == chosen.__class__ and a.to_dict() == chosen.to_dict():
                    legal_index = i
                    break
        actor_player = 1 if actor == live.p1.user_id else 2
        env.step(action_id)
        post_snap = stub.snapshot_state()
        seq += 1
        rows.append({
            "seq": seq,
            "battle_id": battle_id,
            "turn_number": int(pre_snap["turn_number"]),
            "actor_user_id": int(actor),
            "actor_player": actor_player,
            "decision_source": "test",
            "legal_action_index": legal_index,
            "action_type": _action_type_for(action_id),
            "action_json": {"type": _action_type_for(action_id)},
            "action_native": chosen.to_dict() if chosen else None,
            "source_card": None,
            "target_card": None,
            "legal_actions": [a.to_dict() for a in legal_raw],
            "legal_action_count": len(legal_raw),
            "pre_state": pre_snap,
            "post_state": post_snap,
            "deltas": None,
            "accepted": True,
            "error": None,
            "timestamp_ms": 0,
            "visibility": "omniscient_offline_only",
        })

    if append_surrender:
        # Synthesize a terminal surrender row (mirrors v5_trace.record_terminal
        # v5_trace.py:552-599): pre_state snapshotted, post_state == pre_state
        # (mark_surrender does not mutate state.status; the bridge resolves
        # status from meta.status). legal_action_index=None, action_native=None.
        actor = live.current_turn_owner_id
        actor_player = surrender_actor_player
        actor_uid = live.p1.user_id if actor_player == 1 else live.p2.user_id
        pre_snap = stub.snapshot_state()
        seq += 1
        rows.append({
            "seq": seq,
            "battle_id": battle_id,
            "turn_number": int(pre_snap["turn_number"]),
            "actor_user_id": int(actor_uid),
            "actor_player": actor_player,
            "decision_source": "human",
            "legal_action_index": None,
            "action_type": "surrender",
            "action_json": {"type": "surrender", "reason": "test"},
            "action_native": None,
            "source_card": None,
            "target_card": None,
            "legal_actions": [],
            "legal_action_count": 0,
            "pre_state": pre_snap,
            "post_state": pre_snap,  # mark_surrender leaves status ongoing
            "deltas": None,
            "accepted": True,
            "error": None,
            "timestamp_ms": 0,
            "visibility": "omniscient_offline_only",
        })

    # Write manifest + meta + actions.jsonl.
    v5_dir = group_dir / "battles" / battle_id / "v5"
    v5_dir.mkdir(parents=True, exist_ok=True)
    with (v5_dir / "actions.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "schema_version": "rlhf_v5_storage_v1",
        "visibility": "omniscient_offline_only",
        "group_id": "test_group",
        "battle_id": battle_id,
        "status": meta_status,
        "winner_user_id": None,
        "turns": int(live.turn_number),
        "p1_user_id": int(live.p1.user_id),
        "p2_user_id": int(live.p2.user_id),
        "p1_actor_type": "human",
        "p2_actor_type": "bot",
        "v5_trace_present": True,
    }
    (v5_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "manifest_version": "rlhf_manifest_v1",
        "group_id": "test_group",
        "battles_results": [
            {
                "battle_id": battle_id,
                "battle_log_path": f"battles/{battle_id}/b_{battle_id}.json",
                "winner_user_id": None,
                "loser_user_id": None,
                "status": meta_status.upper(),
                "turns": int(live.turn_number),
                "duration_seconds": 0.0,
                "v5_dir": f"battles/{battle_id}/v5",
                "v5_meta_path": f"battles/{battle_id}/v5/meta.json",
                "v5_trace_ok": True,
            },
        ],
        "battle_ids": [battle_id],
    }
    (group_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return len(rows)


def test_deserializer_round_trip_obs_integrity(tmp_path):
    """CANONICAL gate (spec §6.185): for a recorded battle,
    obs(reconstruct(post_N)) == obs(reconstruct(pre_{N+1})).

    The continuity invariant (v5_trace_validate.py:244-386) guarantees
    ``post_state[N] == pre_state[N+1]`` byte-identically. A DETERMINISTIC
    deserializer therefore yields identical obs from the SAME perspective.
    The gate is evaluated at a FIXED perspective (p1_user_id): the loader's
    per-row transitions use each row's ACTOR perspective, which FLIPS when an
    ``end_turn`` passes the turn — so the loader's ``next_obs[N]`` vs
    ``obs[N+1]`` legitimately differ across a turn flip (a perspective change,
    NOT a deserializer bug). The canonical deserializer+continuity gate holds
    the perspective fixed and compares the reconstructed states directly.

    This test asserts BOTH:
      (1) the byte-identical continuity invariant ``post_N == pre_{N+1}``
          (the recorded snapshots themselves, before any reconstruction), and
      (2) ``obs(reconstruct(post_N), P) == obs(reconstruct(pre_{N+1}), P)``
          at a fixed perspective P = p1_user_id, within float tol.
    It additionally checks the loader's per-actor transitions agree across
    NON-turn-flip boundaries (``actor_N == actor_{N+1}``).
    """
    n_rows = _write_real_trace(tmp_path, "b_roundtrip", n_steps=10, seed=21,
                               meta_status="p2_win")
    info = _make_omniscient_info_mode()
    assist = AssistModeV5()

    # Read the recorded actions.jsonl + meta directly.
    v5_dir = tmp_path / "battles" / "b_roundtrip" / "v5"
    meta = json.loads((v5_dir / "meta.json").read_text(encoding="utf-8"))
    p1_uid = int(meta["p1_user_id"])
    rows = [json.loads(line) for line in (v5_dir / "actions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 2

    # (1) Byte-identical continuity invariant on the recorded snapshots.
    for i in range(len(rows) - 1):
        post_n = rows[i]["post_state"]
        pre_next = rows[i + 1]["pre_state"]
        assert post_n == pre_next, (
            f"continuity invariant broken at i={i}: post_state != next pre_state"
        )

    # (2) Fixed-perspective obs round-trip through the deserializer.
    for i in range(len(rows) - 1):
        post_n = rows[i]["post_state"]
        pre_next = rows[i + 1]["pre_state"]
        recon_post = reconstruct_gamestate(post_n)
        recon_pre_next = reconstruct_gamestate(pre_next)
        obs_post = encode_observation_v5(
            recon_post, p1_uid, info_mode=info, assist_mode=assist,
            history_events=post_n["v5_history_events"],
        )
        obs_pre_next = encode_observation_v5(
            recon_pre_next, p1_uid, info_mode=info, assist_mode=assist,
            history_events=pre_next["v5_history_events"],
        )
        assert np.allclose(obs_post, obs_pre_next, atol=1e-6), (
            f"round-trip obs broken at i={i} (fixed perspective p1): "
            f"max abs diff={float(np.max(np.abs(obs_post - obs_pre_next)))}"
        )

    # Loader-level: transitions are well-formed, and across NON-turn-flip
    # boundaries (actor_N == actor_{N+1}) next_obs[N] == obs[N+1].
    transitions = load_offline_dataset(tmp_path, info_mode=info)
    assert len(transitions) >= 2
    for t in transitions:
        assert t.obs.shape == (7128,)
        assert t.next_obs.shape == (7128,)
        assert np.all(np.isfinite(t.obs)) and np.all(np.isfinite(t.next_obs))
        assert t.action_features.shape == (601, 171)
        assert isinstance(t.reward, float)
        assert isinstance(t.mana_draw_legal, bool)
    for i in range(len(transitions) - 1):
        if transitions[i].meta["actor_user_id"] == transitions[i + 1].meta["actor_user_id"]:
            a = transitions[i].next_obs
            b = transitions[i + 1].obs
            assert np.allclose(a, b, atol=1e-6), (
                f"loader continuity broken at i={i} (same actor): "
                f"max abs diff={float(np.max(np.abs(a - b)))}"
            )


# ---------------------------------------------------------------------------
# (c) reward byte-matches classic_rl_env._compute_reward
# ---------------------------------------------------------------------------


def _reward_dict(my_hp, enemy_hp, my_board, enemy_board, my_mana, enemy_mana,
                 p1_uid=1, p2_uid=2):
    return {
        "my_hero_hp": my_hp,
        "enemy_hero_hp": enemy_hp,
        "my_board_hp": list(my_board),
        "enemy_board_hp": list(enemy_board),
        "my_mana": my_mana,
        "enemy_mana": enemy_mana,
        "opponent_id": p2_uid,
        "p1_user_id": p1_uid,
        "p2_user_id": p2_uid,
    }


def test_reward_byte_matches_classic_rl_env():
    """compute_offline_reward EXACTLY equals classic_rl_env._compute_reward
    for invalid / win / loss / draw / shaped cases."""
    env = ClassicRLEnv(seed=3)
    env.reset()
    # ClassicRLEnv reset gives p1_user_id=1, p2_user_id=2.
    assert env._env.state.p1.user_id == 1
    assert env._env.state.p2.user_id == 2

    cases = []

    # invalid action (not accepted) -> -0.05
    pre = _reward_dict(30, 30, [5], [4], 5, 3)
    post = _reward_dict(30, 30, [5], [4], 5, 3)
    cases.append(("invalid", 1, pre, post, False, "ongoing", -0.05))

    # win (P1_WIN, actor=p1=1) -> +1.0
    cases.append(("win", 1, pre, post, True, "p1_win", 1.0))
    # loss (P1_WIN, actor=p2=2) -> -1.0
    cases.append(("loss", 2, pre, post, True, "p1_win", -1.0))
    # draw -> 0.0
    cases.append(("draw", 1, pre, post, True, "draw", 0.0))
    # stalemate -> 0.0 (no-winner terminal, documented). classic_rl_env has no
    # STALEMATE enum member (core/state.py:64-69), so it never observes
    # stalemate; if handed ONGOING it would SHAPE. The bridge explicitly maps
    # stalemate -> 0.0 (draw-equivalent terminal). Use delta-bearing pre/post
    # so classic's shaped path is NON-zero, demonstrating the divergence.
    pre_st = _reward_dict(30, 30, [5], [4, 4], 5, 3)
    post_st = _reward_dict(30, 20, [5], [4], 1, 3)  # enemy_hp +10, enemy_killed 1, mana 4
    shaped_for_stalemate = 0.02 * 10 + 0.03 * 1 + min(0.02, 0.005 * 4)  # 0.25
    cases.append(("stalemate", 1, pre_st, post_st, True, "stalemate", 0.0))

    # shaped: enemy_hp +8, own_hp +2, enemy_killed 2, own_killed 1, mana 4
    # reward = 0.02*8 - 0.01*2 + 0.03*2 - 0.02*1 + min(0.02, 0.005*4)
    #        = 0.16 - 0.02 + 0.06 - 0.02 + 0.02 = 0.20
    pre_s = _reward_dict(30, 30, [5, 3], [4, 4, 2], 5, 3)
    post_s = _reward_dict(28, 22, [5], [4], 1, 3)
    expected_shaped = 0.02 * 8 - 0.01 * 2 + 0.03 * 2 - 0.02 * 1 + min(0.02, 0.005 * 4)
    cases.append(("shaped", 1, pre_s, post_s, True, "ongoing", expected_shaped))

    for name, actor_id, pre_d, post_d, accepted, status, _exp in cases:
        # Set the live env state status so classic_rl_env._compute_reward
        # takes the same branch. p1/p2 user ids are 1/2 (from reset).
        st = env._env.state
        status_map = {
            "ongoing": GameStatus.ONGOING,
            "p1_win": GameStatus.P1_WIN,
            "p2_win": GameStatus.P2_WIN,
            "draw": GameStatus.DRAW,
            "stalemate": GameStatus.ONGOING,  # no enum member; classic never sees it
        }
        st.status = status_map[status]

        # classic_rl_env._compute_reward(self, actor_id, pre, post, success)
        classic_reward = env._compute_reward(actor_id, pre_d, post_d, bool(accepted))
        # compute_offline_reward(actor_id, pre, post, accepted, status)
        offline_reward = compute_offline_reward(
            actor_id, pre_d, post_d, accepted, status,
        )
        # For stalemate, classic_rl_env has no STALEMATE enum member; the live
        # env state is ONGOING so classic SHAPES (non-zero here). The offline
        # bridge explicitly maps stalemate -> 0.0 (no-winner terminal). Assert
        # the documented divergence: bridge == 0.0 while classic == shaped.
        if status == "stalemate":
            assert offline_reward == 0.0, f"{name}: offline={offline_reward}"
            assert classic_reward == pytest.approx(shaped_for_stalemate), (
                f"{name}: classic shaped={classic_reward} != {shaped_for_stalemate}"
            )
        else:
            assert offline_reward == classic_reward, (
                f"{name}: offline={offline_reward} != classic={classic_reward} "
                f"(status={status})"
            )
        # Also check the explicit expected value where given.
        assert offline_reward == pytest.approx(_exp), (
            f"{name}: offline={offline_reward} != expected={_exp}"
        )

    # Extra: reward_view_from_snapshot mirrors classic_rl_env._snapshot field
    # shape (my_hero_hp/enemy_hero_hp/my_board_hp/enemy_board_hp/my_mana/
    # enemy_mana/opponent_id) — verify against a live state.
    env2 = ClassicRLEnv(seed=5)
    env2.reset()
    _drive_env(env2, 3)
    live = env2._env.state
    actor = live.current_turn_owner_id
    actor_player = 1 if actor == live.p1.user_id else 2
    snap = StubRecorder(live).snapshot_state()
    view = reward_view_from_snapshot(snap, actor_player)
    classic_snap = env2._snapshot(actor)
    for k in ("my_hero_hp", "enemy_hero_hp", "my_board_hp", "enemy_board_hp",
              "my_mana", "enemy_mana", "opponent_id"):
        assert view[k] == classic_snap[k], f"reward view field {k}: {view[k]} != {classic_snap[k]}"
    assert view["p1_user_id"] == live.p1.user_id
    assert view["p2_user_id"] == live.p2.user_id


# ---------------------------------------------------------------------------
# (d) orphans skipped
# ---------------------------------------------------------------------------


def _write_battle_index(group_dir: Path, battle_id: str, *, meta_status: str,
                         v5_trace_ok: bool, rows=None):
    """Write a battle with the given meta.status + optional action rows."""
    v5_dir = group_dir / "battles" / battle_id / "v5"
    v5_dir.mkdir(parents=True, exist_ok=True)
    if rows is not None:
        with (v5_dir / "actions.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "schema_version": "rlhf_v5_storage_v1", "group_id": "g", "battle_id": battle_id,
        "status": meta_status, "winner_user_id": None, "turns": 1,
        "p1_user_id": 1, "p2_user_id": 2, "p1_actor_type": "human", "p2_actor_type": "bot",
        "v5_trace_present": True,
    }
    (v5_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return {
        "battle_id": battle_id, "battle_log_path": f"b_{battle_id}.json",
        "winner_user_id": None, "loser_user_id": None,
        "status": meta_status.upper(), "turns": 1, "duration_seconds": 0.0,
        "v5_dir": f"battles/{battle_id}/v5",
        "v5_meta_path": f"battles/{battle_id}/v5/meta.json",
        "v5_trace_ok": v5_trace_ok,
    }


def _minimal_state_snapshot(*, turn=1, owner=1, p1_hand=2, p1_mana=5,
                             status="ongoing"):
    """Hand-build a minimal v5_trace-schema state snapshot for orphan/
    surrender/mana_draw tests (real schema via StubRecorder on a hand-built
    live state). The hand is set to EXACTLY ``p1_hand`` cards by pulling from
    the deck (or duplicating a hand card with a fresh instance_id if the deck
    is exhausted) so the mana_draw hand_full guard (HAND_CAP=4) is exercisable.
    """
    env = ClassicRLEnv(seed=1)
    env.reset()
    _drive_env(env, 1)
    live = env._env.state
    # Mutate to the desired mana_draw legality config.
    live.p1.mana = p1_mana
    live.p1.max_mana = max(p1_mana, live.p1.max_mana)
    # Set p1 hand to EXACTLY p1_hand cards (shrink to deck, grow from deck /
    # duplicate with fresh instance_id). Structure stays valid for obs.
    hand = list(live.p1.hand)
    deck = list(live.p1.deck)
    if len(hand) > p1_hand:
        deck = hand[p1_hand:] + deck
        hand = hand[:p1_hand]
    while len(hand) < p1_hand:
        if deck:
            hand.append(deck.pop(0))
        elif hand:
            c = copy.deepcopy(hand[0])
            c.instance_id = uuid4()
            hand.append(c)
        else:
            break
    live.p1.hand = hand
    live.p1.deck = deck
    # owner: ensure current_turn_owner_id matches.
    live.current_turn_owner_id = live.p1.user_id if owner == 1 else live.p2.user_id
    live.turn_number = turn
    live.status = GameStatus.ONGOING if status == "ongoing" else GameStatus.P1_WIN
    return StubRecorder(live).snapshot_state(), live


def test_orphans_skipped(tmp_path):
    """A battle with meta.status='ongoing' is skipped; a terminal battle is
    included."""
    # Orphan battle (ongoing) with a real row.
    snap_orphan, _ = _minimal_state_snapshot(p1_hand=2, p1_mana=5, status="ongoing")
    orphan_rows = [{
        "seq": 1, "battle_id": "b_orphan", "turn_number": 1,
        "actor_user_id": 1, "actor_player": 1, "decision_source": "test",
        "legal_action_index": 0, "action_type": "end_turn",
        "action_json": {"type": "end_turn"}, "action_native": None,
        "source_card": None, "target_card": None, "legal_actions": [],
        "legal_action_count": 0, "pre_state": snap_orphan, "post_state": snap_orphan,
        "deltas": None, "accepted": True, "error": None, "timestamp_ms": 0,
        "visibility": "omniscient_offline_only",
    }]
    # Terminal battle (p2_win) with a real row.
    snap_term, _ = _minimal_state_snapshot(p1_hand=2, p1_mana=5, status="ongoing")
    term_rows = [dict(orphan_rows[0], battle_id="b_term", seq=1)]

    rec_orphan = _write_battle_index(tmp_path, "b_orphan", meta_status="ongoing",
                                     v5_trace_ok=True, rows=orphan_rows)
    rec_term = _write_battle_index(tmp_path, "b_term", meta_status="p2_win",
                                   v5_trace_ok=True, rows=term_rows)
    manifest = {
        "manifest_version": "rlhf_manifest_v1", "group_id": "g",
        "battles_results": [rec_orphan, rec_term],
        "battle_ids": ["b_orphan", "b_term"],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    transitions = load_offline_dataset(tmp_path)
    battle_ids = {t.meta["battle_id"] for t in transitions}
    assert "b_orphan" not in battle_ids, "orphan (ongoing) battle must be skipped"
    assert "b_term" in battle_ids, "terminal battle must be included"
    # All transitions come from the terminal battle only.
    assert all(t.meta["battle_id"] == "b_term" for t in transitions)


# ---------------------------------------------------------------------------
# (e) surrender row terminal
# ---------------------------------------------------------------------------


def test_surrender_row_terminal(tmp_path):
    """A surrender action row produces a terminal transition (terminal=True)
    with the terminal reward (±1.0/0.0 from status) and is the last row."""
    # Drive a real trace WITH an appended surrender row. The surrendering
    # player is p2 (actor_player=2); when p2 surrenders the NON-surrendering
    # player p1 wins, so meta.status='p1_win' (mirrors match_runner semantics).
    n_rows = _write_real_trace(
        tmp_path, "b_surrender", n_steps=6, seed=33,
        meta_status="p1_win", append_surrender=True, surrender_actor_player=2,
    )
    transitions = load_offline_dataset(tmp_path, info_mode=_make_omniscient_info_mode())
    assert len(transitions) >= 1
    last = transitions[-1]
    # The last row is the surrender row.
    assert last.meta["action_type"] == "surrender", (
        f"last row action_type={last.meta['action_type']!r}, expected 'surrender'"
    )
    assert last.terminal is True, "surrender row must be terminal"
    assert last.meta["status"] == "p1_win"
    # Terminal reward: the surrendering actor is p2 (actor_user_id == 2 ==
    # p2_user_id); status p1_win -> actor_id(2) != p1_user_id(1) -> -1.0
    # (the surrendering player LOST). p1_user_id == 1 (ClassicRLEnv reset).
    assert last.reward == -1.0, (
        f"surrender reward={last.reward}, expected -1.0 "
        f"(p2 surrendered -> p1 wins -> p2 lost)"
    )
    # action_tcode_or_index is None for terminal rows.
    assert last.action_tcode_or_index is None
    # Surrender row is the last row of its battle (no transition after).
    assert all(t.meta["battle_id"] == "b_surrender" for t in transitions)
    # The non-surrender rows are NOT terminal (ongoing).
    non_terminal = [t for t in transitions if t.meta["action_type"] != "surrender"]
    assert all(not t.terminal for t in non_terminal), (
        "non-surrender rows in an ongoing game must not be terminal"
    )


# ---------------------------------------------------------------------------
# (f) mana_draw_legal flag populated
# ---------------------------------------------------------------------------


def test_mana_draw_legal_flag_populated(tmp_path):
    """For a mana_draw-legal row + a mana_draw-illegal row,
    OfflineTransition.mana_draw_legal matches
    mana_draw_head_v5.mana_draw_legal_mask on the reconstructed state."""
    # mana_draw-legal: hand < 4, mana >= 2*(0+1)=2 -> hand=2, mana=5 -> legal.
    snap_legal, live_legal = _minimal_state_snapshot(p1_hand=2, p1_mana=5, owner=1)
    # mana_draw-illegal: hand >= 4 -> hand=4 -> illegal (hand_full guard).
    snap_illegal, live_illegal = _minimal_state_snapshot(p1_hand=4, p1_mana=10, owner=1)

    # Independent oracle: mana_draw_legal_mask on the LIVE state.
    legal_expected = mana_draw_legal_mask(live_legal, live_legal.current_turn_owner_id)
    illegal_expected = mana_draw_legal_mask(live_illegal, live_illegal.current_turn_owner_id)
    assert legal_expected is True, "fixture: hand=2,mana=5 must be mana_draw-legal"
    assert illegal_expected is False, "fixture: hand=4 must be mana_draw-illegal"

    def _row(bid, seq, snap, action_type):
        return {
            "seq": seq, "battle_id": bid, "turn_number": int(snap["turn_number"]),
            "actor_user_id": int(snap["p1"]["user_id"]), "actor_player": 1,
            "decision_source": "test", "legal_action_index": None,
            "action_type": action_type,
            "action_json": {"type": action_type}, "action_native": None,
            "source_card": None, "target_card": None, "legal_actions": [],
            "legal_action_count": 0, "pre_state": snap, "post_state": snap,
            "deltas": None, "accepted": True, "error": None, "timestamp_ms": 0,
            "visibility": "omniscient_offline_only",
        }

    rows = [
        _row("b_md", 1, snap_legal, "mana_draw"),
        _row("b_md", 2, snap_illegal, "end_turn"),
    ]
    rec = _write_battle_index(tmp_path, "b_md", meta_status="p2_win",
                              v5_trace_ok=True, rows=rows)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "manifest_version": "rlhf_manifest_v1", "group_id": "g",
        "battles_results": [rec], "battle_ids": ["b_md"],
    }), encoding="utf-8")

    transitions = load_offline_dataset(tmp_path)
    assert len(transitions) == 2
    t_legal, t_illegal = transitions
    assert t_legal.meta["action_type"] == "mana_draw"
    assert t_illegal.meta["action_type"] == "end_turn"

    # Parity: loader's mana_draw_legal == independent mask on reconstructed state.
    recon_legal = reconstruct_gamestate(snap_legal)
    recon_illegal = reconstruct_gamestate(snap_illegal)
    assert t_legal.mana_draw_legal == mana_draw_legal_mask(
        recon_legal, snap_legal["p1"]["user_id"]
    )
    assert t_illegal.mana_draw_legal == mana_draw_legal_mask(
        recon_illegal, snap_illegal["p1"]["user_id"]
    )
    # And the values match the live-oracle expectations.
    assert t_legal.mana_draw_legal is True
    assert t_illegal.mana_draw_legal is False

"""Block C component C2 -- offline replay bridge tests (TRACKED
``test_offline_replay_bridge``).

Ten SYNTHETIC gates for ``TrainV3.5/python/train_v3/offline_replay_bridge.py``
(the recorded v5/trace -> RustTransitionBatch-shaped DENSE offline-replay
bridge for the C3 AWAC/CRR offline-PPO replay):

  (1) ``test_human_filter_excludes_bot_rows`` -- D-C7: rows with
      decision_source!='human' are excluded (a bot row is dropped).
  (2) ``test_omniscient_info_mode_passed`` -- D11: the bridge calls
      iter_offline_transitions with InfoModeV5(enemy_hand_known=True,
      enemy_deck_known=True).
  (3) ``test_tcode_resolves_via_engine_action_native`` -- 601-tcode resolves via
      the ENGINE-sourced action_native (a normal play_card/end_turn/attack row
      -> target_tcode in 0..600 matching the engine legal set); mana_draw row
      -> target_tcode=None (-1), is_mana_draw=True.
  (4) ``test_old_log_prob_and_value_from_current_policy`` -- D-C10: with a fake
      policy returning known logits, old_log_prob == log(softmax(masked_logits)
      [target_tcode]+1e-10) and value == fake_values[i] for EVERY row (not just
      the tail); mana_draw rows old_log_prob=0 but value still populated.
  (5) ``test_per_game_terminated_and_bootstrap`` -- per-game GAE episode
      boundaries: a 2-game batch -> terminated is True at each game's last real
      step; padded steps terminated; bootstrap_values shape == (num_games,) ==
      V(next_obs) of each game's final transition.
  (6) ``test_batch_flows_through_prepare_rust_ppo_batch`` -- the batch is
      RustTransitionBatch-shaped with dense action_features+action_mask and
      flows through prepare_rust_ppo_batch WITHOUT a shape error (advantages +
      returns computed).
  (7) ``test_terminal_and_orphan_rows_skipped`` -- terminal/orphan rows skipped
      (no crash on an empty/ongoing group).
  (8) ``test_action_native_sourced_from_loader_field`` -- source-vs-source:
      action_native sourced from the loader engine-sourced field, NOT
      decode_action (assert resolve_v5_tcode is called with t.action_native, not
      a decode_action output).
  (9) ``test_empty_collection_no_crash`` -- empty-collection no-crash (0 human
      rows -> empty batch, mana_draw_row_count=0).
  (10) ``test_mana_draw_rows_carry_head_fields`` -- mana_draw rows carry
       is_mana_draw=True + mana_draw_legal (for the C3 BCE term).

All SYNTHETIC: a FAKE policy_fn returning canned (logits, values,
mana_draw_logit) numpy arrays + REAL-engine-driven v5_trace fixtures (drive a
real ClassicRLEnv via the StubRecorder pattern from test_bc_dataset.py; NO real
MLX/Rust/ONNX). Asserts SPECIFIC values (target_tcode, old_log_prob numeric,
terminated flags, bootstrap_values shape) -- not trivially-true shape checks.

Run: ``PYTHONPATH=.:TrainV3.5/python PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
TrainV3.5/python/train_v3/tests/test_offline_replay_bridge.py``
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from ai.train_v2.classic_actions_v1 import (
    MAX_CANDIDATE_ACTIONS,
    decode_action,
)
from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.offline_dataset_loader import (
    iter_offline_transitions,
    reconstruct_gamestate,
)
from core.actions import ManaDrawAction
from core.state import GameStatus
from rlhf_env.components.arena_engine import RlhfBattleEngine
from rlhf_env.components.v5_trace import V5TraceRecorder
from train_v3 import offline_replay_bridge as orb
from train_v3.contracts import AssistModeV5, InfoModeV5
from train_v3.offline_replay_bridge import (
    build_offline_replay_batch,
    make_policy_fn_from_checkpoint,
)
from train_v3.rust_ppo import prepare_rust_ppo_batch


# ---------------------------------------------------------------------------
# Oracle helpers -- copied from test_bc_dataset.py (StubRecorder pattern +
# _write_real_trace_bc engine-sourced action_native forked helper).
# ---------------------------------------------------------------------------


class StubRecorder:
    """Drive the REAL ``V5TraceRecorder`` serializer against a live state."""

    def __init__(self, live_state):
        self.engine = SimpleNamespace(
            _arena=SimpleNamespace(state=live_state),
            _snapshot_card=RlhfBattleEngine._snapshot_card,
        )

    def _snapshot_card(self, card):
        return V5TraceRecorder._snapshot_card(self, card)

    def _snapshot_player(self, p):
        return V5TraceRecorder._snapshot_player(self, p)

    def snapshot_state(self):
        return V5TraceRecorder._snapshot_state(self)


def _make_omniscient_info_mode() -> InfoModeV5:
    return InfoModeV5(
        own_hand_identity_known=True,
        own_deck_known=True,
        enemy_hand_known=True,
        enemy_deck_known=True,
        enemy_deck_order_known=True,
    )


def _pick_action_nonend(legal_raw):
    for i, a in enumerate(legal_raw):
        if a.to_dict()["type"] != "end_turn":
            return a, i
    return legal_raw[0], 0


def _write_real_trace_bc(
    group_dir: Path,
    battle_id: str,
    *,
    n_steps: int,
    seed: int,
    decision_source: str = "human",
    meta_status: str = "p2_win",
) -> list[dict]:
    """Drive a REAL ClassicRLEnv battle, write manifest+meta+actions.jsonl,
    return the recorded rows. action_native = legal_raw[idx].to_dict() (ENGINE
    oracle, CRITICAL FIX A -- NOT decode_action)."""
    env = ClassicRLEnv(seed=seed)
    env.reset()
    live = env._env.state
    stub = StubRecorder(live)
    rows: list[dict] = []
    seq = 0
    for _ in range(n_steps):
        if live.status != GameStatus.ONGOING:
            break
        actor = live.current_turn_owner_id
        pre_snap = stub.snapshot_state()
        legal_raw = env._env.get_legal_actions(actor)
        if not legal_raw:
            break
        action, idx = _pick_action_nonend(legal_raw)
        action_native = action.to_dict()
        action_type = action_native["type"]
        actor_player = 1 if actor == live.p1.user_id else 2
        env.step_core_action(action)
        post_snap = stub.snapshot_state()
        seq += 1
        rows.append({
            "seq": seq, "battle_id": battle_id,
            "turn_number": int(pre_snap["turn_number"]),
            "actor_user_id": int(actor), "actor_player": actor_player,
            "decision_source": decision_source,
            "legal_action_index": idx, "action_type": action_type,
            "action_json": action_native, "action_native": action_native,
            "source_card": None, "target_card": None,
            "legal_actions": [a.to_dict() for a in legal_raw],
            "legal_action_count": len(legal_raw),
            "pre_state": pre_snap, "post_state": post_snap,
            "deltas": None, "accepted": True, "error": None,
            "timestamp_ms": 0, "visibility": "omniscient_offline_only",
        })
    _write_battle_files(group_dir, battle_id, rows, live, meta_status)
    return rows


def _write_battle_files(
    group_dir: Path, battle_id: str, rows: list[dict], live, meta_status: str,
) -> None:
    v5_dir = group_dir / "battles" / battle_id / "v5"
    v5_dir.mkdir(parents=True, exist_ok=True)
    with (v5_dir / "actions.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "schema_version": "rlhf_v5_storage_v1", "visibility": "omniscient_offline_only",
        "group_id": "test_group", "battle_id": battle_id, "status": meta_status,
        "winner_user_id": None, "turns": int(live.turn_number),
        "p1_user_id": int(live.p1.user_id), "p2_user_id": int(live.p2.user_id),
        "p1_actor_type": "human", "p2_actor_type": "bot", "v5_trace_present": True,
    }
    (v5_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "manifest_version": "rlhf_manifest_v1", "group_id": "test_group",
        "battles_results": [{
            "battle_id": battle_id, "battle_log_path": f"battles/{battle_id}/b_{battle_id}.json",
            "winner_user_id": None, "loser_user_id": None, "status": meta_status.upper(),
            "turns": int(live.turn_number), "duration_seconds": 0.0,
            "v5_dir": f"battles/{battle_id}/v5",
            "v5_meta_path": f"battles/{battle_id}/v5/meta.json", "v5_trace_ok": True,
        }],
        "battle_ids": [battle_id],
    }
    (group_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")


def _minimal_state_snapshot(*, turn=1, owner=1, p1_hand=2, p1_mana=5,
                             status="ongoing"):
    env = ClassicRLEnv(seed=1)
    env.reset()
    _drive_env_minimal(env, 1)
    live = env._env.state
    live.p1.mana = p1_mana
    live.p1.max_mana = max(p1_mana, live.p1.max_mana)
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
    live.current_turn_owner_id = live.p1.user_id if owner == 1 else live.p2.user_id
    live.turn_number = turn
    live.status = GameStatus.ONGOING if status == "ongoing" else GameStatus.P1_WIN
    return StubRecorder(live).snapshot_state(), live


def _drive_env_minimal(env: ClassicRLEnv, n_steps: int) -> None:
    for _ in range(n_steps):
        mask = env.action_mask()
        ids = [i for i in range(601) if mask[i] == 1.0]
        if not ids:
            break
        env.step(ids[0])


def _write_battle_index(group_dir: Path, battle_id: str, *, meta_status: str,
                        v5_trace_ok: bool, rows=None):
    v5_dir = group_dir / "battles" / battle_id / "v5"
    v5_dir.mkdir(parents=True, exist_ok=True)
    if rows is not None:
        with (v5_dir / "actions.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "schema_version": "rlhf_v5_storage_v1", "group_id": "g", "battle_id": battle_id,
        "status": meta_status, "winner_user_id": None, "turns": 1,
        "p1_user_id": 1, "p2_user_id": 2, "p1_actor_type": "human",
        "p2_actor_type": "bot", "v5_trace_present": True,
    }
    (v5_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return {
        "battle_id": battle_id, "battle_log_path": f"b_{battle_id}.json",
        "winner_user_id": None, "loser_user_id": None, "status": meta_status.upper(),
        "turns": 1, "duration_seconds": 0.0, "v5_dir": f"battles/{battle_id}/v5",
        "v5_meta_path": f"battles/{battle_id}/v5/meta.json", "v5_trace_ok": v5_trace_ok,
    }


def _write_manifest(group_dir: Path, recs):
    (group_dir / "manifest.json").write_text(json.dumps({
        "manifest_version": "rlhf_manifest_v1", "group_id": "g",
        "battles_results": recs, "battle_ids": [r["battle_id"] for r in recs],
    }, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# FAKE policy (D-C10): returns canned (logits, values, mana_draw_logit).
#   logits[i, :] = i for every candidate  -> masked softmax is UNIFORM over the
#     legal candidates (all legal logits equal) -> probs[target] = 1/count_legal
#     -> old_log_prob = log(1/count_legal + 1e-10) for normal rows.
#   values[i] = obs_batch[i, 0]  -> value == observations[step, env, 0] for every
#     row, and bootstrap_values[env] == next_observations[last_step, env, 0].
# ---------------------------------------------------------------------------


def _fake_policy(obs_batch, action_features_batch):
    n = int(obs_batch.shape[0])
    logits = np.broadcast_to(
        np.arange(n, dtype=np.float32)[:, None], (n, MAX_CANDIDATE_ACTIONS)
    ).copy()
    values = np.asarray(obs_batch[:, 0], dtype=np.float32).copy()
    mana_draw_logit = np.zeros((n,), dtype=np.float32)
    return logits, values, mana_draw_logit


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask == 1.0, logits, np.float32(-1.0e9))
    m = masked.max(axis=-1, keepdims=True)
    ex = np.exp(masked - m)
    return ex / ex.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# (1) D-C7 human filter -- bot/rl rows excluded
# ---------------------------------------------------------------------------


def test_human_filter_excludes_bot_rows(tmp_path):
    """A synthetic battle with mixed decision_source rows (human + bot + rl):
    only decision_source=='human' rows are replayed; bot/rl rows EXCLUDED."""
    snap, _ = _minimal_state_snapshot(p1_hand=2, p1_mana=5, owner=1)
    p1_uid = int(snap["p1"]["user_id"])
    p2_uid = int(snap["p2"]["user_id"])

    def _row(seq, ds, atype, anative, uid, player):
        return {
            "seq": seq, "battle_id": "b_ds", "turn_number": int(snap["turn_number"]),
            "actor_user_id": uid, "actor_player": player, "decision_source": ds,
            "legal_action_index": 0 if atype == "end_turn" else None,
            "action_type": atype, "action_json": anative, "action_native": anative,
            "source_card": None, "target_card": None, "legal_actions": [],
            "legal_action_count": 0, "pre_state": snap, "post_state": snap,
            "deltas": None, "accepted": True, "error": None, "timestamp_ms": 0,
            "visibility": "omniscient_offline_only",
        }

    rows = [
        _row(1, "human", "end_turn", {"type": "end_turn"}, p1_uid, 1),
        _row(2, "bot", "end_turn", {"type": "end_turn"}, p2_uid, 2),
        _row(3, "rl", "attack",
             {"type": "attack", "attacker_id": "x", "target_id": "y",
              "target_is_hero": False}, p2_uid, 2),
    ]
    rec = _write_battle_index(tmp_path, "b_ds", meta_status="p2_win",
                              v5_trace_ok=True, rows=rows)
    _write_manifest(tmp_path, [rec])

    out = build_offline_replay_batch(_fake_policy, group_dirs=tmp_path)
    # Only the single human row is replayed -> 1 game, 1 step.
    assert out.num_games == 1, f"expected 1 game (human rows only), got {out.num_games}"
    assert out.num_rows == 1, f"expected 1 human row, got {out.num_rows}"
    # The human end_turn row resolves to tcode 0.
    assert out.target_tcodes[0, 0] == 0, (
        f"human end_turn must resolve to tcode 0, got {out.target_tcodes[0, 0]}"
    )
    assert out.is_mana_draw[0, 0] is False or bool(out.is_mana_draw[0, 0]) is False
    # bot/rl rows (seq 2,3) EXCLUDED.
    assert out.num_rows == 1
    # Cross-check: the loader emits all 3 rows; the bridge filters to 1.
    loader_rows = list(iter_offline_transitions(tmp_path))
    assert len(loader_rows) == 3, "loader emits all rows (no filter)"


# ---------------------------------------------------------------------------
# (2) D11 omniscient InfoModeV5 passed to iter_offline_transitions
# ---------------------------------------------------------------------------


def test_omniscient_info_mode_passed(tmp_path):
    """The bridge calls iter_offline_transitions with an omniscient InfoModeV5
    (enemy_hand_known=True, enemy_deck_known=True). Spy on the loader call."""
    _write_real_trace_bc(tmp_path, "b_omni", n_steps=20, seed=21, meta_status="p2_win")

    captured: dict[str, InfoModeV5] = {}
    real_iter = orb.iter_offline_transitions

    def _spy(group_dir, *, info_mode=None, assist_mode=None, max_battles=None):
        captured["info_mode"] = info_mode
        # Forward to the real loader so the bridge gets real transitions.
        yield from real_iter(
            group_dir, info_mode=info_mode, assist_mode=assist_mode,
            max_battles=max_battles,
        )

    original = orb.iter_offline_transitions
    orb.iter_offline_transitions = _spy
    try:
        build_offline_replay_batch(_fake_policy, group_dirs=tmp_path)
    finally:
        orb.iter_offline_transitions = original

    assert "info_mode" in captured, "bridge did not call iter_offline_transitions"
    im = captured["info_mode"]
    assert im is not None, "bridge passed info_mode=None to the loader"
    assert im.enemy_hand_known is True, (
        "D11: bridge must pass enemy_hand_known=True (omniscient)"
    )
    assert im.enemy_deck_known is True, (
        "D11: bridge must pass enemy_deck_known=True (omniscient)"
    )


# ---------------------------------------------------------------------------
# (3) 601-tcode resolves via ENGINE-sourced action_native
# ---------------------------------------------------------------------------


def test_tcode_resolves_via_engine_action_native(tmp_path):
    """A normal play_card/end_turn/attack row -> target_tcode is a valid 0..600
    int matching the engine legal set (decode_action(tcode).to_dict() ==
    ENGINE action_native); a mana_draw row -> target_tcode=None (-1),
    is_mana_draw=True."""
    rows = _write_real_trace_bc(tmp_path, "b_tcode", n_steps=40, seed=21,
                                meta_status="p2_win")
    assert len(rows) >= 10, "fixture: battle should produce >=10 rows"

    out = build_offline_replay_batch(_fake_policy, group_dirs=tmp_path)
    assert out.num_games == 1, "single battle -> single game (env)"
    assert out.num_rows >= 1
    steps = out.batch.actions.shape[0]

    # Map seq -> (step, env) via the loader (human rows, in order).
    human_rows = [r for r in rows if r["decision_source"] == "human"]
    assert len(human_rows) == out.num_rows

    found_mana_draw = False
    found_normal = False
    for step, row in enumerate(human_rows):
        tc = int(out.target_tcodes[step, 0])
        is_md = bool(out.is_mana_draw[step, 0])
        atype = row["action_type"]
        if atype == "mana_draw":
            assert is_md is True, f"seq={row['seq']}: mana_draw -> is_mana_draw=True"
            assert tc == -1, f"seq={row['seq']}: mana_draw -> target_tcode=None (-1)"
            found_mana_draw = True
            continue
        # normal row: target_tcode in 0..600, in the legal mask, and
        # decode_action(tcode).to_dict() == ENGINE action_native.
        assert 0 <= tc <= 600, f"seq={row['seq']}: tcode out of range: {tc}"
        mask = out.batch.action_mask[step, 0]
        assert mask[tc] == 1.0, f"seq={row['seq']}: tcode {tc} not in legal mask"
        pre = reconstruct_gamestate(row["pre_state"])
        actor = row["actor_user_id"]
        decoded = decode_action(pre, actor, tc)
        assert decoded is not None
        assert decoded.to_dict() == row["action_native"], (
            f"seq={row['seq']} SOURCE-VS-SOURCE: decode_action(tcode={tc}).to_dict() "
            f"= {decoded.to_dict()} != ENGINE action_native {row['action_native']}"
        )
        found_normal = True
    assert found_normal, "fixture must produce at least one normal row"
    assert found_mana_draw, "fixture must produce at least one mana_draw row"


# ---------------------------------------------------------------------------
# (4) D-C10 old_log_prob + value from the current policy at bridge time
# ---------------------------------------------------------------------------


def test_old_log_prob_and_value_from_current_policy(tmp_path):
    """With a fake policy returning known logits (uniform over legal candidates)
    + values = obs[:, 0]: old_log_prob == log(1/count_legal + 1e-10) for normal
    rows, value == observations[step, env, 0] for EVERY row; mana_draw rows
    old_log_prob=0 but value still populated."""
    _write_real_trace_bc(tmp_path, "b_olp", n_steps=40, seed=21, meta_status="p2_win")

    out = build_offline_replay_batch(_fake_policy, group_dirs=tmp_path)
    steps, env_count = out.batch.actions.shape
    assert env_count == 1

    for step in range(steps):
        if step >= out.num_rows:
            # padded step: value=0, old_log_prob=0
            assert out.batch.values[step, 0] == 0.0
            assert out.batch.log_probs[step, 0] == 0.0
            continue
        mask = out.batch.action_mask[step, 0]
        count_legal = int(mask.sum())
        # value == observations[step, env, 0] for EVERY real row (D-C10).
        assert out.batch.values[step, 0] == pytest.approx(
            float(out.batch.observations[step, 0, 0]), abs=1e-6
        ), f"step={step}: value must equal obs[0] for every row (D-C10)"
        tc = int(out.target_tcodes[step, 0])
        is_md = bool(out.is_mana_draw[step, 0])
        if is_md or tc == -1:
            # mana_draw / terminal: old_log_prob = 0.0 (no 601 action).
            assert out.batch.log_probs[step, 0] == 0.0, (
                f"step={step}: mana_draw/terminal old_log_prob must be 0.0"
            )
        else:
            expected = float(np.log(1.0 / count_legal + 1.0e-10))
            assert out.batch.log_probs[step, 0] == pytest.approx(
                expected, abs=1e-5
            ), (
                f"step={step}: old_log_prob {out.batch.log_probs[step, 0]} != "
                f"log(1/count_legal={count_legal} + 1e-10) = {expected}"
            )


# ---------------------------------------------------------------------------
# (5) per-game GAE episode boundaries + bootstrap_values
# ---------------------------------------------------------------------------


def test_per_game_terminated_and_bootstrap(tmp_path):
    """A 2-game batch -> terminated is True at each game's last real step;
    padded steps terminated; bootstrap_values shape == (num_games,) ==
    V(next_obs) of each game's final transition."""
    rows1 = _write_real_trace_bc(tmp_path, "b_g1", n_steps=15, seed=11,
                                 meta_status="p2_win")
    # Second battle in the SAME group dir -- append to manifest.
    rows2 = _write_real_trace_bc(tmp_path, "b_g2", n_steps=25, seed=29,
                                 meta_status="p1_win")
    # _write_real_trace_bc overwrites manifest for a single battle; rebuild the
    # manifest with BOTH battles.
    rec1 = _write_battle_index(tmp_path, "b_g1", meta_status="p2_win",
                               v5_trace_ok=True)
    rec2 = _write_battle_index(tmp_path, "b_g2", meta_status="p1_win",
                               v5_trace_ok=True)
    _write_manifest(tmp_path, [rec1, rec2])

    out = build_offline_replay_batch(_fake_policy, group_dirs=tmp_path)
    assert out.num_games == 2, f"expected 2 games, got {out.num_games}"
    steps, env_count = out.batch.actions.shape
    assert env_count == 2

    human1 = [r for r in rows1 if r["decision_source"] == "human"]
    human2 = [r for r in rows2 if r["decision_source"] == "human"]
    len1, len2 = len(human1), len(human2)
    assert len1 == out.num_rows - len2  # sanity

    for env, ln in enumerate((len1, len2)):
        # intermediate real steps: terminated = t.terminal (False for a mid-game
        # human action in an ONGOING game). This is a DYNAMIC regression guard -- a
        # bridge that regressed to terminated=True on ALL real steps would FAIL here
        # (GAE would then zero every step's next-value term and leak no credit).
        for step in range(ln - 1):
            assert bool(out.batch.terminated[step, env]) is False, (
                f"env={env} step={step}: intermediate real step must NOT be "
                f"terminated (GAE continues to the next step); got True"
            )
        # last real step: terminated True (episode boundary).
        assert bool(out.batch.terminated[ln - 1, env]) is True, (
            f"env={env}: last real step must be terminated (episode boundary)"
        )
        # padded steps: terminated True.
        for step in range(ln, steps):
            assert bool(out.batch.terminated[step, env]) is True, (
                f"env={env} step={step}: padded step must be terminated"
            )
            assert out.batch.values[step, env] == 0.0
            assert out.batch.rewards[step, env] == 0.0

    # bootstrap_values shape == (num_games,) == V(next_obs) of each game's final
    # real transition. With _fake_policy, V(s) = s[0].
    assert out.bootstrap_values.shape == (2,)
    for env, ln in enumerate((len1, len2)):
        expected = float(out.batch.next_observations[ln - 1, env, 0])
        assert out.bootstrap_values[env] == pytest.approx(expected, abs=1e-6), (
            f"env={env}: bootstrap_values must equal V(next_obs) of the game's "
            f"final transition ({expected}), got {out.bootstrap_values[env]}"
        )


# ---------------------------------------------------------------------------
# (6) RustTransitionBatch-shaped + flows through prepare_rust_ppo_batch
# ---------------------------------------------------------------------------


def test_batch_flows_through_prepare_rust_ppo_batch(tmp_path):
    """The batch is RustTransitionBatch-shaped with dense action_features +
    action_mask and flows through prepare_rust_ppo_batch WITHOUT a shape error
    (advantages + returns computed)."""
    _write_real_trace_bc(tmp_path, "b_prep", n_steps=30, seed=21, meta_status="p2_win")

    out = build_offline_replay_batch(_fake_policy, group_dirs=tmp_path)
    b = out.batch
    steps, env_count = b.actions.shape
    assert env_count == 1
    # Dense shape contracts.
    assert b.observations.shape == (steps, env_count, 7128)
    assert b.action_mask.shape == (steps, env_count, 601)
    assert b.action_features.shape == (steps, env_count, 601, 171)
    assert b.actions.shape == (steps, env_count)
    assert b.rewards.shape == (steps, env_count)
    assert b.terminated.shape == (steps, env_count)
    assert b.values.shape == (steps, env_count)
    assert b.log_probs.shape == (steps, env_count)
    assert b.legal_action_counts.shape == (steps, env_count)
    assert b.legal_action_offsets.shape == (steps, env_count)
    # legal_action_ids is a flat tape; counts sum equals its length.
    assert int(b.legal_action_counts.sum()) == int(b.legal_action_ids.shape[0])
    # values + log_probs required for PPO prep.
    assert b.values is not None and b.log_probs is not None

    ppo = prepare_rust_ppo_batch(
        b,
        gamma=0.99,
        gae_lambda=0.95,
        bootstrap_values=out.bootstrap_values,
        advantage_backend="python",
        selected_local_backend="python",
        prepare_backend="separate",
    )
    # advantages + returns computed (no shape error).
    assert ppo.advantages.shape == (steps, env_count)
    assert ppo.returns.shape == (steps, env_count)
    assert np.all(np.isfinite(ppo.advantages))
    assert np.all(np.isfinite(ppo.returns))
    # selected_local_indices computed for every row (python backend).
    assert ppo.selected_local_indices.shape == (steps, env_count)
    assert np.all(ppo.selected_local_indices >= 0)


# ---------------------------------------------------------------------------
# (7) terminal / orphan rows skipped (no crash)
# ---------------------------------------------------------------------------


def test_terminal_and_orphan_rows_skipped(tmp_path):
    """Orphan (ongoing) battle excluded by the loader; surrender-terminal rows
    handled (target_tcode=-1, is_mana_draw=False). No crash on the group."""
    snap_orphan, _ = _minimal_state_snapshot(p1_hand=2, p1_mana=5, status="ongoing")
    orphan_rows = [{
        "seq": 1, "battle_id": "b_orphan", "turn_number": 1,
        "actor_user_id": int(snap_orphan["p1"]["user_id"]), "actor_player": 1,
        "decision_source": "human", "legal_action_index": 0,
        "action_type": "end_turn", "action_json": {"type": "end_turn"},
        "action_native": {"type": "end_turn"}, "source_card": None,
        "target_card": None, "legal_actions": [], "legal_action_count": 0,
        "pre_state": snap_orphan, "post_state": snap_orphan, "deltas": None,
        "accepted": True, "error": None, "timestamp_ms": 0,
        "visibility": "omniscient_offline_only",
    }]
    rec_orphan = _write_battle_index(tmp_path, "b_orphan", meta_status="ongoing",
                                     v5_trace_ok=True, rows=orphan_rows)

    snap_t, _ = _minimal_state_snapshot(p1_hand=2, p1_mana=5, status="ongoing")
    p2_uid = int(snap_t["p2"]["user_id"])
    term_rows = [{
        "seq": 1, "battle_id": "b_surr", "turn_number": 1,
        "actor_user_id": p2_uid, "actor_player": 2, "decision_source": "human",
        "legal_action_index": None, "action_type": "surrender",
        "action_json": {"type": "surrender"}, "action_native": None,
        "source_card": None, "target_card": None, "legal_actions": [],
        "legal_action_count": 0, "pre_state": snap_t, "post_state": snap_t,
        "deltas": None, "accepted": True, "error": None, "timestamp_ms": 0,
        "visibility": "omniscient_offline_only",
    }]
    rec_surr = _write_battle_index(tmp_path, "b_surr", meta_status="p1_win",
                                   v5_trace_ok=True, rows=term_rows)
    _write_manifest(tmp_path, [rec_orphan, rec_surr])

    out = build_offline_replay_batch(_fake_policy, group_dirs=tmp_path)
    # Orphan excluded by the loader -> only the surrender battle (1 game).
    assert out.num_games == 1, "orphan battle must be skipped"
    assert out.num_rows == 1
    # Surrender-terminal row: target_tcode=None (-1), is_mana_draw=False.
    assert out.target_tcodes[0, 0] == -1, "surrender row: target_tcode=None (-1)"
    assert bool(out.is_mana_draw[0, 0]) is False, "surrender is not mana_draw"
    # No crash: flows through prepare.
    ppo = prepare_rust_ppo_batch(
        out.batch, gamma=0.99, gae_lambda=0.95,
        bootstrap_values=out.bootstrap_values, advantage_backend="python",
        selected_local_backend="python", prepare_backend="separate",
    )
    assert ppo.advantages.shape == out.batch.actions.shape


# ---------------------------------------------------------------------------
# (8) source-vs-source: action_native from the LOADER engine field, not
# decode_action.
# ---------------------------------------------------------------------------


def test_action_native_sourced_from_loader_field(tmp_path):
    """resolve_v5_tcode is called with t.action_native (the loader's ENGINE-
# sourced field, v5_trace.py:481 legal[legal_index].to_dict()), NOT a
# decode_action output. Spy on the bridge's resolve_v5_tcode and assert the
# action_native arg matches the recorded ENGINE action_native."""
    rows = _write_real_trace_bc(tmp_path, "b_src", n_steps=20, seed=21,
                                meta_status="p2_win")
    expected_natives = {
        r["seq"]: r["action_native"] for r in rows
        if r["decision_source"] == "human" and r["action_type"] != "mana_draw"
    }
    assert expected_natives, "fixture must produce >=1 normal human row"

    captured: list[tuple[int | None, dict | None]] = []
    real_resolve = orb.resolve_v5_tcode

    def _spy(pre_state, actor, action_native, *, mask=None, strict=False):
        captured.append((actor, action_native))
        return real_resolve(pre_state, actor, action_native, mask=mask, strict=strict)

    original = orb.resolve_v5_tcode
    orb.resolve_v5_tcode = _spy
    try:
        build_offline_replay_batch(_fake_policy, group_dirs=tmp_path)
    finally:
        orb.resolve_v5_tcode = original

    # The bridge must have called resolve_v5_tcode for normal human rows, passing
    # the ENGINE-sourced action_native (t.action_native == recorded
    # action_native == legal_raw[idx].to_dict()).
    captured_natives = [an for (_a, an) in captured if an is not None]
    assert len(captured_natives) >= 1, (
        "bridge must call resolve_v5_tcode for normal human rows"
    )
    expected_set = [dict(en) for en in expected_natives.values()]
    for an in captured_natives:
        assert an in expected_set, (
            f"resolve_v5_tcode called with action_native={an} which is NOT one "
            f"of the recorded ENGINE action_natives {expected_set} -- the bridge "
            f"is sourcing action_native from decode_action (self-referential trap)"
        )
    # Every expected normal human row's action_native was passed at least once.
    for en in expected_set:
        assert en in captured_natives, (
            f"ENGINE action_native {en} was never passed to resolve_v5_tcode"
        )


# ---------------------------------------------------------------------------
# (9) empty-collection no-crash (0 human rows)
# ---------------------------------------------------------------------------


def test_empty_collection_no_crash(tmp_path):
    """0 human rows (all bot) -> empty batch, mana_draw_row_count=0; no crash."""
    snap, _ = _minimal_state_snapshot(p1_hand=2, p1_mana=5, owner=1)
    p2_uid = int(snap["p2"]["user_id"])
    rows = [{
        "seq": 1, "battle_id": "b_bot", "turn_number": int(snap["turn_number"]),
        "actor_user_id": p2_uid, "actor_player": 2, "decision_source": "bot",
        "legal_action_index": 0, "action_type": "end_turn",
        "action_json": {"type": "end_turn"}, "action_native": {"type": "end_turn"},
        "source_card": None, "target_card": None, "legal_actions": [],
        "legal_action_count": 0, "pre_state": snap, "post_state": snap,
        "deltas": None, "accepted": True, "error": None, "timestamp_ms": 0,
        "visibility": "omniscient_offline_only",
    }]
    rec = _write_battle_index(tmp_path, "b_bot", meta_status="p2_win",
                              v5_trace_ok=True, rows=rows)
    _write_manifest(tmp_path, [rec])

    out = build_offline_replay_batch(_fake_policy, group_dirs=tmp_path)
    assert out.num_games == 0, "0 human rows -> 0 games"
    assert out.num_rows == 0
    assert out.mana_draw_row_count == 0
    assert out.bootstrap_values.shape == (0,)
    # Empty batch flows through prepare WITHOUT a shape error. The empty GAE
    # normalization (mean/std of an empty slice) emits harmless numpy
    # RuntimeWarnings from the READ-ONLY rust_ppo.py path; silence them here.
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)
        ppo = prepare_rust_ppo_batch(
            out.batch, gamma=0.99, gae_lambda=0.95,
            bootstrap_values=out.bootstrap_values, advantage_backend="python",
            selected_local_backend="python", prepare_backend="separate",
        )
    assert ppo.advantages.shape == (0, 0)


# ---------------------------------------------------------------------------
# (10) mana_draw rows carry is_mana_draw + mana_draw_legal (C3 BCE term)
# ---------------------------------------------------------------------------


def test_mana_draw_rows_carry_head_fields(tmp_path):
    """A mana_draw row carries is_mana_draw=True + mana_draw_legal=True (for the
    C3 BCE term); target_tcode=None (-1); old_log_prob=0; value populated."""
    # hand=2 (< HAND_CAP=4) + mana=5 (>= mana_draw_cost(0)=2) -> mana_draw LEGAL.
    snap, live = _minimal_state_snapshot(p1_hand=2, p1_mana=5, owner=1)
    actor_uid = int(snap["p1"]["user_id"])
    action_native = ManaDrawAction().to_dict()
    rows = [{
        "seq": 1, "battle_id": "b_md", "turn_number": int(snap["turn_number"]),
        "actor_user_id": actor_uid, "actor_player": 1, "decision_source": "human",
        "legal_action_index": None, "action_type": "mana_draw",
        "action_json": action_native, "action_native": action_native,
        "source_card": None, "target_card": None, "legal_actions": [action_native],
        "legal_action_count": 1, "pre_state": snap, "post_state": snap,
        "deltas": None, "accepted": True, "error": None, "timestamp_ms": 0,
        "visibility": "omniscient_offline_only",
    }]
    rec = _write_battle_index(tmp_path, "b_md", meta_status="p2_win",
                              v5_trace_ok=True, rows=rows)
    _write_manifest(tmp_path, [rec])

    out = build_offline_replay_batch(_fake_policy, group_dirs=tmp_path)
    assert out.num_games == 1
    assert out.num_rows == 1
    assert out.mana_draw_row_count == 1, "exactly one mana_draw row"
    assert bool(out.is_mana_draw[0, 0]) is True, "mana_draw -> is_mana_draw=True"
    assert bool(out.mana_draw_legal[0, 0]) is True, (
        "mana_draw was legal on this pre_state -> mana_draw_legal=True"
    )
    assert out.target_tcodes[0, 0] == -1, "mana_draw -> target_tcode=None (-1)"
    assert out.batch.log_probs[0, 0] == 0.0, "mana_draw old_log_prob=0 (no 601 action)"
    # value STILL populated (GAE needs V(s_t) for every step, D-C10).
    assert out.batch.values[0, 0] == pytest.approx(
        float(out.batch.observations[0, 0, 0]), abs=1e-6
    )


# ---------------------------------------------------------------------------
# (11) checkpoint loader skip-gate (A2 pattern; MLX-free gate test)
# ---------------------------------------------------------------------------


def test_checkpoint_loader_skip_gate(tmp_path):
    """make_policy_fn_from_checkpoint gates on the CHECKPOINT FILE's existence
    (A2 pattern, bc_train.py:37-39): None or an absent file raises
    FileNotFoundError so the C3 driver can skip the bridge run for this group
    (no crash, no partial batch). This test never imports MLX -- it only
    exercises the file-existence gate (the lazy MLX import is downstream of the
    gate and thus never reached on a missing file)."""
    # None -> FileNotFoundError (no checkpoint path provided).
    with pytest.raises(FileNotFoundError):
        make_policy_fn_from_checkpoint(None)
    # Absent file -> FileNotFoundError (A2 skip-gate; MLX NOT imported).
    bogus = tmp_path / "does_not_exist.npz"
    with pytest.raises(FileNotFoundError):
        make_policy_fn_from_checkpoint(bogus)
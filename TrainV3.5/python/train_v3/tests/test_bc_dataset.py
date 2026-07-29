"""Block A component A1 — bc_dataset tests (TRACKED ``test_bc_dataset``).

Five gates for ``TrainV3.5/python/train_v3/bc_dataset.py`` (the recorded v5/
trace -> BC training-tuple builder + V5 601-tcode resolver + human filter):

  (1) ``test_tcode_matches_recorded_action`` — SOURCE-VS-SOURCE gate
      (verifier finding 4a): drive a REAL ``ClassicRLEnv`` battle exercising
      end_turn + play_card (warrior+potion) + attack + mana_draw; record rows
      with ``action_native = legal_raw[idx].to_dict()`` (the ENGINE's
      ``BaseAction``, same source as ``v5_trace.py:481`` — NOT
      ``decode_action(...).to_dict()``); build the BC dataset (strict=True);
      assert ``decode_action(pre_state, actor, resolved_tcode).to_dict() ==
      action_native`` for every non-mana_draw row. This is a TRUE source-vs-
      source check (codec vs engine): a ``decode_action`` regression that
      diverges from the engine's ``BaseAction`` emission is caught (no candidate
      matches -> strict raises), unlike the legacy
      ``test_train_v2_offline_bridge.py:_write_real_trace`` which sources
      ``action_native`` from ``decode_action`` itself (self-referential
      decode_action-vs-decode_action, cannot detect a codec-vs-engine
      regression).
  (2) ``test_mana_draw_row_targets_head`` — a row where the human took
      ``ManaDrawAction`` yields ``is_mana_draw=True``, ``target_tcode=None``.
  (3) ``test_legal_mask_matches_action_features`` — ``BCTransition.legal_mask``
      consistency with ``action_features``: the append_only legal-mask nonzero
      rows exactly equal the action_features nonzero rows (encoding invariant),
      and the source-card channel is populated for legal play/attack actions.
  (4) ``test_orphan_and_terminal_skip`` — orphan (``meta.status='ongoing'``)
      battle excluded by the loader; a surrender-terminal row emits a
      ``BCTransition`` with ``terminal=True``, ``target_tcode=None``,
      ``is_mana_draw=False``.
  (5) ``test_decision_source_human_filter`` — a synthetic battle with mixed
      ``decision_source`` rows (human + bot + rl) -> only ``decision_source==
      'human'`` rows emit ``BCTransition``; bot/rl rows EXCLUDED (verifier
      finding 4b).

Oracle strategy: a REAL ``ClassicRLEnv`` battle driven via
``env.step_core_action`` with the engine's OWN ``BaseAction`` objects
(``env._env.get_legal_actions`` -> ``core/engine.py:1193`` -> append_only
warrior placement at ``position=len(player.board)``). ``action_native`` is
sourced from the engine's ``BaseAction.to_dict()`` (CRITICAL FIX A) — the
``StubRecorder`` (REAL ``V5TraceRecorder._snapshot_state`` serializer bound to
the live state) produces the v5_trace-schema snapshots the loader consumes.
Synthetic data only; live ``ClassicRLEnv`` is the oracle.

Run: ``PYTHONPATH=.:TrainV3.5/python python3 -m pytest \
TrainV3.5/python/train_v3/tests/test_bc_dataset.py``
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from ai.train_v2.classic_actions_v1 import build_action_mask, decode_action
from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.offline_dataset_loader import (
    iter_offline_transitions,
    reconstruct_gamestate,
)
from core.actions import ManaDrawAction
from core.state import GameStatus
from rlhf_env.components.arena_engine import RlhfBattleEngine
from rlhf_env.components.v5_trace import V5TraceRecorder
from train_v3.bc_dataset import (
    BCTransition,
    TcodeResolutionError,
    build_bc_dataset,
    load_bc_dataset,
    resolve_v5_tcode,
)
from train_v3.contracts import AssistModeV5, InfoModeV5
from train_v3.mana_draw_head_v5 import mana_draw_legal_mask


# ---------------------------------------------------------------------------
# Oracle: drive the REAL v5_trace serializer against a live ClassicRLEnv state
# (same StubRecorder pattern as tests/test_train_v2_offline_bridge.py — the
# recorder is coupled to RlhfBattleEngine via engine._arena.state +
# engine._snapshot_card; the stub wires them to the live ClassicRLEnv state).
# ---------------------------------------------------------------------------


class StubRecorder:
    """Drive the REAL ``V5TraceRecorder`` serializer methods against a live
    state. See ``tests/test_train_v2_offline_bridge.py:StubRecorder``."""

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
    """The trace is omniscient; use an omniscient info mode for the oracle obs."""
    return InfoModeV5(
        own_hand_identity_known=True,
        own_deck_known=True,
        enemy_hand_known=True,
        enemy_deck_known=True,
        enemy_deck_order_known=True,
    )


# ---------------------------------------------------------------------------
# CRITICAL FIX A — forked helper: action_native sourced from the ENGINE's
# BaseAction (legal_raw[idx].to_dict()), NOT decode_action(...).to_dict().
# ---------------------------------------------------------------------------


def _pick_action_nonend(legal_raw):
    """Pick the first non-end_turn action (to exercise play/attack/mana_draw),
    falling back to end_turn. Returns (action, index) into legal_raw."""
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
    return the recorded rows.

    CRITICAL FIX A (verifier finding 4a): ``action_native`` is sourced from
    the ENGINE's ``BaseAction`` (``legal_raw[idx].to_dict()`` where
    ``legal_raw = env._env.get_legal_actions(actor)`` — the engine's own
    ``BaseAction`` list, INDEPENDENT of ``decode_action``), mirroring
    ``v5_trace.py:481`` ``action_native = legal[legal_index].to_dict()``. This
    is the ENGINE oracle. The BC resolver then decodes 601-candidates via the
    FROZEN ``decode_action`` and value-matches against this engine-sourced
    dict — a TRUE source-vs-source check. (The legacy
    ``test_train_v2_offline_bridge.py:_write_real_trace`` sources
    ``action_native = decode_action(...).to_dict()`` — self-referential; this
    forked helper does NOT.)

    The battle is driven via ``env.step_core_action`` with the engine's own
    ``BaseAction`` objects (engine-legal append_only actions), so every recorded
    action is one the engine actually emitted — the resolver's append_only mask
    candidate set corresponds EXACTLY to these engine actions.

    ``meta_status`` is set to a terminal value so the loader does NOT skip the
    battle as an orphan (the loader filters on ``meta.status``); the recorded
    rows themselves are ongoing (``post_state.status='ongoing'``) so they are
    non-terminal replay rows.
    """
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
        legal_raw = env._env.get_legal_actions(actor)  # engine BaseActions
        if not legal_raw:
            break
        action, idx = _pick_action_nonend(legal_raw)
        # CRITICAL FIX A: ENGINE-sourced action_native (NOT decode_action).
        action_native = action.to_dict()
        action_type = action_native["type"]
        actor_player = 1 if actor == live.p1.user_id else 2
        # Step the engine with its own BaseAction (engine-legal append_only).
        env.step_core_action(action)
        post_snap = stub.snapshot_state()
        seq += 1
        rows.append({
            "seq": seq,
            "battle_id": battle_id,
            "turn_number": int(pre_snap["turn_number"]),
            "actor_user_id": int(actor),
            "actor_player": actor_player,
            "decision_source": decision_source,
            "legal_action_index": idx,
            "action_type": action_type,
            "action_json": action_native,
            "action_native": action_native,
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

    _write_battle_files(group_dir, battle_id, rows, live, meta_status)
    return rows


def _write_battle_files(
    group_dir: Path,
    battle_id: str,
    rows: list[dict],
    live,
    meta_status: str,
) -> None:
    """Write actions.jsonl + meta.json + manifest.json for a battle."""
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


def _minimal_state_snapshot(*, turn=1, owner=1, p1_hand=2, p1_mana=5,
                             status="ongoing"):
    """Hand-build a minimal v5_trace-schema state snapshot (real schema via
    StubRecorder on a hand-built live ClassicRLEnv state). Mirrors
    ``tests/test_train_v2_offline_bridge.py:_minimal_state_snapshot``."""
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
    """Write a battle with the given meta.status + optional action rows
    (mirrors ``tests/test_train_v2_offline_bridge.py:_write_battle_index``)."""
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


# ---------------------------------------------------------------------------
# (1) SOURCE-VS-SOURCE: tcode round-trips to the ENGINE-sourced action_native
# ---------------------------------------------------------------------------


def test_tcode_matches_recorded_action(tmp_path):
    """drive a REAL ClassicRLEnv battle (end_turn + play_card warrior+potion +
    attack + mana_draw); record action_native = legal_raw[idx].to_dict()
    (ENGINE oracle, CRITICAL FIX A); build the BC dataset (strict=True);
    assert decode_action(pre_state, actor, resolved_tcode).to_dict() ==
    action_native for every non-mana_draw row.

    True source-vs-source: action_native is the engine's BaseAction; the
    resolver decodes 601-candidates via the FROZEN decode_action and matches.
    A decode_action-vs-engine regression makes no candidate match -> strict
    raises -> test fails. (The legacy self-referential helper would mask this.)
    """
    rows = _write_real_trace_bc(tmp_path, "b_tcode", n_steps=40, seed=21,
                                meta_status="p2_win")
    assert len(rows) >= 10, "fixture: battle should produce >=10 rows"

    bc = load_bc_dataset(tmp_path, info_mode=_make_omniscient_info_mode(),
                         strict=True)
    assert len(bc) >= 1

    # The battle exercised the spec's required action types.
    types_emitted = {t.meta["action_type"] for t in bc}
    assert "end_turn" in types_emitted, "fixture must exercise end_turn"
    assert "play_card" in types_emitted, "fixture must exercise play_card"
    assert "attack" in types_emitted, "fixture must exercise attack"
    assert "mana_draw" in types_emitted, "fixture must exercise mana_draw"

    bc_by_seq = {t.meta["seq"]: t for t in bc}
    checked = 0
    mana_draw_rows = 0
    # Lock in BOTH card_kinds among play_card rows (spec acceptance
    # BLOCK_A_PLAN.md:177-178 requires play_card warrior+potion). The
    # card_type is derived from the pre_state snapshot's actor hand at
    # hand_index (engine-faithful snapshot, NOT bc_dataset output), so this
    # pins warrior+potion coverage to the fixture rather than relying on
    # seed-dependent behavior alone.
    play_card_kinds = set()
    for row in rows:
        seq = row["seq"]
        t = bc_by_seq.get(seq)
        if t is None:
            # All rows are decision_source='human' here, so none should be
            # filtered; a missing seq is a bug.
            pytest.fail(f"seq={seq} missing from BC dataset (no filter expected)")
        # Shape + type sanity.
        assert isinstance(t, BCTransition)
        assert t.obs.shape == (7128,)
        assert t.action_features.shape == (601, 171)
        assert t.legal_mask.shape == (601,)
        assert isinstance(t.mana_draw_legal, bool)
        assert t.meta["decision_source"] == "human"
        if t.is_mana_draw:
            assert t.target_tcode is None, (
                f"seq={seq}: mana_draw row must have target_tcode=None"
            )
            mana_draw_rows += 1
            continue
        # Non-mana_draw row: target_tcode MUST be resolved (strict=True raised
        # otherwise). Assert the SOURCE-VS-SOURCE round-trip: decode_action at
        # the resolved tcode reproduces the ENGINE-sourced action_native.
        assert t.target_tcode is not None, (
            f"seq={seq}: non-mana_draw row has target_tcode=None (resolution failed)"
        )
        action_native = row["action_native"]  # ENGINE-sourced (legal_raw[idx].to_dict())
        pre_state = reconstruct_gamestate(row["pre_state"])
        actor = row["actor_user_id"]
        # Collect the played card's kind (warrior vs potion) from the
        # engine-faithful pre_state snapshot hand to lock in warrior+potion.
        if action_native["type"] == "play_card":
            hi = action_native["hand_index"]
            for p in ("p1", "p2"):
                if row["pre_state"][p]["user_id"] == int(actor):
                    play_card_kinds.add(
                        row["pre_state"][p]["hand"][hi]["card_type"]
                    )
                    break
        decoded = decode_action(pre_state, actor, t.target_tcode)
        assert decoded is not None, (
            f"seq={seq}: decode_action returned None for resolved tcode={t.target_tcode}"
        )
        assert decoded.to_dict() == action_native, (
            f"seq={seq} SOURCE-VS-SOURCE FAILURE: "
            f"decode_action(tcode={t.target_tcode}).to_dict()={decoded.to_dict()} "
            f"!= ENGINE action_native={action_native}"
        )
        checked += 1
    assert checked >= 1, "must round-trip at least one non-mana_draw row"
    assert mana_draw_rows >= 1, "must have at least one mana_draw row"
    # Spec acceptance: play_card must exercise BOTH warrior and potion.
    assert "warrior" in play_card_kinds, (
        f"fixture must play a warrior; kinds={play_card_kinds}"
    )
    assert "potion" in play_card_kinds, (
        f"fixture must play a potion; kinds={play_card_kinds}"
    )
    # Every BCTransition came from this single battle.
    assert all(t.meta["battle_id"] == "b_tcode" for t in bc)


def test_resolve_v5_tcode_unit():
    """Direct unit gate on resolve_v5_tcode: an engine-sourced EndTurnAction
    resolves to tcode 0; an engine-sourced action that is NOT legal raises in
    strict mode (no candidate match) and returns None in non-strict."""
    snap, live = _minimal_state_snapshot(p1_hand=2, p1_mana=5, owner=1)
    pre = reconstruct_gamestate(snap)
    actor = snap["p1"]["user_id"]
    # EndTurnAction is always legal on the actor's turn -> resolves to tcode 0.
    tcode = resolve_v5_tcode(pre, actor, {"type": "end_turn"}, strict=True)
    assert tcode == 0
    # An action_native that no legal candidate decodes to: strict raises.
    bogus = {"type": "play_card", "hand_index": 99, "target_id": None,
             "position": None}
    with pytest.raises(TcodeResolutionError):
        resolve_v5_tcode(pre, actor, bogus, strict=True)
    # Non-strict: returns None (skip) instead of raising.
    assert resolve_v5_tcode(pre, actor, bogus, strict=False) is None
    # None action_native -> None (defensive; terminal synthetic rows).
    assert resolve_v5_tcode(pre, actor, None) is None


# ---------------------------------------------------------------------------
# (2) mana_draw row targets the parallel head, not a 601 slot
# ---------------------------------------------------------------------------


def test_mana_draw_row_targets_head(tmp_path):
    """A row where the human took ManaDrawAction yields is_mana_draw=True,
    target_tcode=None (BC targets the mana_draw head, not a 601 slot)."""
    # hand=2 (< HAND_CAP=4) + mana=5 (>= mana_draw_cost(0)=2) -> mana_draw LEGAL.
    snap, live = _minimal_state_snapshot(p1_hand=2, p1_mana=5, owner=1)
    actor_uid = int(snap["p1"]["user_id"])
    # Engine-faithful action_native: ManaDrawAction.to_dict() == {"type":"mana_draw"}
    # (the engine emits ManaDrawAction at core/engine.py:1347; its to_dict is
    # deterministic {"type":"mana_draw"} — same source as v5_trace.py:481).
    action_native = ManaDrawAction().to_dict()
    # Independent oracle: mana_draw_legal_mask on the live + reconstructed state.
    assert mana_draw_legal_mask(live, live.current_turn_owner_id) is True, \
        "fixture: hand=2,mana=5 must be mana_draw-legal"
    assert mana_draw_legal_mask(reconstruct_gamestate(snap), actor_uid) is True

    rows = [{
        "seq": 1, "battle_id": "b_md", "turn_number": int(snap["turn_number"]),
        "actor_user_id": actor_uid, "actor_player": 1,
        "decision_source": "human",
        "legal_action_index": None,
        "action_type": "mana_draw",
        "action_json": action_native, "action_native": action_native,
        "source_card": None, "target_card": None,
        "legal_actions": [action_native], "legal_action_count": 1,
        "pre_state": snap, "post_state": snap,
        "deltas": None, "accepted": True, "error": None,
        "timestamp_ms": 0, "visibility": "omniscient_offline_only",
    }]
    rec = _write_battle_index(tmp_path, "b_md", meta_status="p2_win",
                              v5_trace_ok=True, rows=rows)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "manifest_version": "rlhf_manifest_v1", "group_id": "g",
        "battles_results": [rec], "battle_ids": ["b_md"],
    }), encoding="utf-8")

    bc = load_bc_dataset(tmp_path)
    assert len(bc) == 1, "exactly one human mana_draw row -> one BCTransition"
    t = bc[0]
    assert t.is_mana_draw is True, "mana_draw row must set is_mana_draw=True"
    assert t.target_tcode is None, "mana_draw row must have target_tcode=None"
    assert t.mana_draw_legal is True, "mana_draw was legal on this pre_state"
    assert t.meta["action_type"] == "mana_draw"
    # legal_mask still the append_only 601 mask (mana_draw is OUTSIDE the 601
    # space; the mask reflects the normal 601 candidates, not mana_draw).
    assert t.legal_mask.shape == (601,)


# ---------------------------------------------------------------------------
# (3) legal_mask consistency with action_features
# ---------------------------------------------------------------------------


def test_legal_mask_matches_action_features(tmp_path):
    """BCTransition.legal_mask consistency with action_features: the append_only
    legal-mask nonzero rows exactly equal the action_features nonzero rows
    (encode_action_features fills exactly the masked rows), and the source-card
    channel is populated for legal play/attack actions (end_turn has no source
    card — its nonzero channel is the action-type one-hot at index 128)."""
    _write_real_trace_bc(tmp_path, "b_mask", n_steps=30, seed=21, meta_status="p2_win")
    bc = load_bc_dataset(tmp_path, info_mode=_make_omniscient_info_mode(),
                         strict=True)
    assert len(bc) >= 1
    for t in bc:
        # Strong encoding invariant: legal_mask nonzero == action_features
        # nonzero rows (encode_action_features encodes exactly masked rows).
        legal_ids = set(np.flatnonzero(t.legal_mask == 1.0).tolist())
        nz_rows = set(
            np.flatnonzero(np.any(np.abs(t.action_features) > 0, axis=1)).tolist()
        )
        assert legal_ids == nz_rows, (
            f"seq={t.meta['seq']}: legal_mask nonzero rows {legal_ids} != "
            f"action_features nonzero rows {nz_rows}"
        )
        # Source-card channel (out[:64]) populated for legal play/attack
        # (action_id 1..600); zero for non-legal. end_turn (0) has no source
        # card (its nonzero channel is the action-type one-hot at index 128).
        for aid in range(1, 601):
            src_nonzero = bool(np.any(t.action_features[aid, :64] != 0))
            if t.legal_mask[aid] == 1.0:
                assert src_nonzero, (
                    f"seq={t.meta['seq']} aid={aid}: legal play/attack has zero "
                    f"source-card channel"
                )
            else:
                assert not src_nonzero, (
                    f"seq={t.meta['seq']} aid={aid}: non-legal action has nonzero "
                    f"source-card channel"
                )
        # end_turn (action 0): legal -> no source card, but action-type one-hot.
        if t.legal_mask[0] == 1.0:
            assert not np.any(t.action_features[0, :64] != 0), (
                "end_turn must have zero source-card channel (no source card)"
            )
            assert t.action_features[0, 128] != 0.0, (
                "end_turn action-type one-hot (index 128) must be set"
            )


# ---------------------------------------------------------------------------
# (4) orphan excluded by the loader + surrender-terminal row handled
# ---------------------------------------------------------------------------


def test_orphan_and_terminal_skip(tmp_path):
    """Orphan (meta.status='ongoing') battle excluded by the loader (0
    BCTransitions); each terminal action_type in _TERMINAL_ACTION_TYPES
    {surrender, draw, stalemate} emits a BCTransition with terminal=True,
    target_tcode=None, is_mana_draw=False (locks in ALL three terminal
    variants, not just surrender)."""
    # Orphan battle (ongoing) with a real human row.
    snap_orphan, _ = _minimal_state_snapshot(p1_hand=2, p1_mana=5, status="ongoing")
    orphan_rows = [{
        "seq": 1, "battle_id": "b_orphan", "turn_number": 1,
        "actor_user_id": int(snap_orphan["p1"]["user_id"]), "actor_player": 1,
        "decision_source": "human",
        "legal_action_index": 0, "action_type": "end_turn",
        "action_json": {"type": "end_turn"}, "action_native": {"type": "end_turn"},
        "source_card": None, "target_card": None, "legal_actions": [],
        "legal_action_count": 0, "pre_state": snap_orphan,
        "post_state": snap_orphan, "deltas": None, "accepted": True,
        "error": None, "timestamp_ms": 0, "visibility": "omniscient_offline_only",
    }]
    rec_orphan = _write_battle_index(tmp_path, "b_orphan", meta_status="ongoing",
                                     v5_trace_ok=True, rows=orphan_rows)
    # One terminal battle per action_type in _TERMINAL_ACTION_TYPES. Each has a
    # single HUMAN terminal row (decision_source='human' so it passes the BC
    # human filter; mirrors v5_trace.record_terminal which defaults
    # decision_source='human').
    terminal_types = ("surrender", "draw", "stalemate")
    term_recs = []
    for term_type in terminal_types:
        snap_t, _ = _minimal_state_snapshot(p1_hand=2, p1_mana=5, status="ongoing")
        rows = [{
            "seq": 1, "battle_id": f"b_{term_type}", "turn_number": 1,
            "actor_user_id": int(snap_t["p2"]["user_id"]), "actor_player": 2,
            "decision_source": "human",
            "legal_action_index": None, "action_type": term_type,
            "action_json": {"type": term_type},
            "action_native": None,
            "source_card": None, "target_card": None, "legal_actions": [],
            "legal_action_count": 0, "pre_state": snap_t,
            "post_state": snap_t, "deltas": None, "accepted": True,
            "error": None, "timestamp_ms": 0, "visibility": "omniscient_offline_only",
        }]
        term_recs.append(_write_battle_index(tmp_path, f"b_{term_type}",
                                             meta_status="p1_win",
                                             v5_trace_ok=True, rows=rows))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "manifest_version": "rlhf_manifest_v1", "group_id": "g",
        "battles_results": [rec_orphan, *term_recs],
        "battle_ids": ["b_orphan", *[f"b_{tt}" for tt in terminal_types]],
    }), encoding="utf-8")

    bc = load_bc_dataset(tmp_path)
    battle_ids = {t.meta["battle_id"] for t in bc}
    assert "b_orphan" not in battle_ids, "orphan (ongoing) battle must be skipped"
    # Each terminal action_type emits exactly one BCTransition with the
    # required terminal shape (target_tcode=None, terminal=True,
    # is_mana_draw=False).
    for term_type in terminal_types:
        bid = f"b_{term_type}"
        assert bid in battle_ids, f"{term_type} terminal battle must be included"
        term_bc = [t for t in bc if t.meta["battle_id"] == bid]
        assert len(term_bc) == 1, (
            f"{term_type}: expected 1 BCTransition, got {len(term_bc)}"
        )
        t = term_bc[0]
        assert t.meta["action_type"] == term_type
        assert t.terminal is True, f"{term_type} row must be terminal"
        assert t.target_tcode is None, (
            f"{term_type}: terminal synthetic row carries no 601 target"
        )
        assert t.is_mana_draw is False, f"{term_type} is not a mana_draw row"
    # Sanity: the loader itself skips the orphan (no OfflineTransition from it).
    loader_transitions = list(iter_offline_transitions(tmp_path))
    assert all(tt.meta["battle_id"] != "b_orphan" for tt in loader_transitions), \
        "loader must skip orphan battle"


# ---------------------------------------------------------------------------
# (5) decision_source=='human' filter (verifier finding 4b)
# ---------------------------------------------------------------------------


def test_decision_source_and_accepted_filter(tmp_path):
    """Only accepted human rows emit BCTransition.

    Bot/RL and rejected-human rows remain available in the raw loader output,
    but none may become a behavior-cloning target.
    """
    snap, live = _minimal_state_snapshot(p1_hand=2, p1_mana=5, owner=1)
    p1_uid = int(snap["p1"]["user_id"])
    p2_uid = int(snap["p2"]["user_id"])

    def _row(seq, decision_source, action_type, action_native, actor_uid,
             actor_player):
        return {
            "seq": seq, "battle_id": "b_ds", "turn_number": int(snap["turn_number"]),
            "actor_user_id": actor_uid, "actor_player": actor_player,
            "decision_source": decision_source,
            "legal_action_index": 0 if action_type == "end_turn" else None,
            "action_type": action_type,
            "action_json": action_native, "action_native": action_native,
            "source_card": None, "target_card": None, "legal_actions": [],
            "legal_action_count": 0, "pre_state": snap, "post_state": snap,
            "deltas": None, "accepted": True, "error": None,
            "timestamp_ms": 0, "visibility": "omniscient_offline_only",
        }

    rows = [
        # human end_turn (resolves to tcode 0).
        _row(1, "human", "end_turn", {"type": "end_turn"}, p1_uid, 1),
        # bot end_turn -> EXCLUDED.
        _row(2, "bot", "end_turn", {"type": "end_turn"}, p2_uid, 2),
        # rl attack -> EXCLUDED (action_native need not resolve; it's filtered).
        _row(3, "rl", "attack",
             {"type": "attack", "attacker_id": "x", "target_id": "y",
              "target_is_hero": False}, p2_uid, 2),
        # rejected human action -> audit row, never a policy target.
        _row(4, "human", "end_turn", {"type": "end_turn"}, p1_uid, 1),
    ]
    rows[-1]["accepted"] = False
    rows[-1]["error"] = "not_your_turn"
    rec = _write_battle_index(tmp_path, "b_ds", meta_status="p2_win",
                              v5_trace_ok=True, rows=rows)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "manifest_version": "rlhf_manifest_v1", "group_id": "g",
        "battles_results": [rec], "battle_ids": ["b_ds"],
    }), encoding="utf-8")

    bc = load_bc_dataset(tmp_path)
    # ONLY the human row emits a BCTransition.
    assert len(bc) == 1, f"expected 1 human BCTransition, got {len(bc)}"
    t = bc[0]
    assert t.meta["decision_source"] == "human"
    assert t.meta["seq"] == 1
    # The human end_turn row resolves to tcode 0 (source-vs-source on the
    # engine-faithful {"type":"end_turn"} action_native).
    assert t.target_tcode == 0, (
        f"human end_turn must resolve to tcode 0, got {t.target_tcode}"
    )
    assert t.is_mana_draw is False
    # bot/rl rows (seq 2,3) and rejected human row (seq 4) are EXCLUDED.
    emitted_seqs = {tt.meta["seq"] for tt in bc}
    assert emitted_seqs == {1}, f"only seq=1 (human) must emit; got {emitted_seqs}"

    # Cross-check: the loader emits ALL 4 rows (it does not filter
    # decision_source); BC filters to 1.
    loader_transitions = list(iter_offline_transitions(tmp_path))
    assert len(loader_transitions) == 4, "loader emits all rows (no filter)"
    assert {tt.meta["decision_source"] for tt in loader_transitions} == \
        {"human", "bot", "rl"}
    assert [tt.meta["accepted"] for tt in loader_transitions] == [
        True,
        True,
        True,
        False,
    ]

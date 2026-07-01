"""Block-C component C0 — loader ``placement_mode='append_only'`` regression.

The loader (``ai/train_v2/offline_dataset_loader.py:744``) builds
``OfflineTransition.action_features`` via ``encode_action_features``. Before
C0 it used the frozen default ``placement_mode='full'`` which, for a WARRIOR
card in hand, emits PlayCard candidates at every board position
``0..len(board)`` (``_mask_play_actions`` ``num_positions =
min(len(board)+1, 8)``). The engine (``core/engine.py:1260``) only ever
emits warriors at ``position=len(player.board)``. The 'full' mask therefore
OVER-INCLUDES warrior candidates the engine does NOT offer, breaking the
consistency invariant (``action_features`` nonzero rows vs
``get_legal_actions`` count).

C0 flips the loader to ``placement_mode='append_only'`` so ALL consumers
(BC ``bc_dataset.py`` already rebuilt with append_only; the new C3
offline-replay path consumes the loader field directly) get one
engine-faithful source.

This test SYNTHETICALLY builds the bug-trigger (a warrior in hand AND >=1
warrior already on board — the state where 'full' and 'append_only'
diverge), runs the loader, and asserts on the LOADER output:

  (a) ``action_features`` nonzero rows == ``build_action_mask(...,
      placement_mode='append_only')`` nonzero rows (loader honours the
      kwarg);
  (b) that count == ``len(engine.get_legal_actions(actor))`` (the
      consistency invariant — engine is the ORACLE);
  (c) a warrior PlayCard candidate appears ONLY at the append position
      ``pos == len(board)`` and NOT at any other board position (the
      assertion that FAILS under ``placement_mode='full'``);
  (d) ``placement_mode='full'`` on the SAME state emits strictly MORE
      nonzero rows than the loader (proving the flip changed the output
      and that the test would have FAILED under 'full' — the bug-trigger
      is genuinely exercised).

No real MLX/Rust/ONNX/rlhf_env infra — uses ``ClassicRLEnv`` + the REAL
``V5TraceRecorder`` serializer via ``StubRecorder`` (same oracle strategy
as ``test_train_v2_offline_bridge.py``) + the frozen
``reconstruct_gamestate`` / ``build_action_mask`` / ``encode_action_features``.

Run: ``PYTHONPATH=.:TrainV3.5/python python3 -m pytest tests/test_train_v2_loader_append_only.py -q``
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest

from ai.train_v2.classic_actions_v1 import (
    _NUM_PLAY_POS,
    _NUM_PLAY_TARGETS,
    _PLAY_BASE,
    _PLAY_STRIDE,
    build_action_mask,
    encode_action_features,
)
from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.offline_dataset_loader import (
    load_offline_dataset,
    reconstruct_gamestate,
)
from core.state import CardInstance, CardType, GameStatus
from rlhf_env.components.arena_engine import RlhfBattleEngine
from rlhf_env.components.v5_trace import V5TraceRecorder
from train_v3.contracts import AssistModeV5, InfoModeV5


# Re-use the StubRecorder oracle strategy from the bridge suite (drive the
# REAL V5TraceRecorder serializer against a live, hand-mutated state).
class _StubRecorder:
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


def _bug_trigger_state():
    """A live state with a WARRIOR in hand AND >=1 warrior on board.

    This is the state where 'full' and 'append_only' diverge: 'full' emits
    PlayCard candidates at every position ``0..len(board)``, 'append_only'
    only at ``pos=len(board)``. The hand warrior has NO mechanics and
    affordable mana, so the engine emits exactly ONE PlayCardAction for it
    (no-target, ``position=len(board)``). The board warrior is NOT ready
    (``is_ready=False``) so it contributes no attack candidates — keeping
    the legal set minimal and the count invariant unambiguous.
    """
    env = ClassicRLEnv(seed=7)
    env.reset()
    live = env._env.state

    # Wipe both boards + hands to a clean, controlled config.
    live.p1.board = []
    live.p2.board = []

    # One NOT-ready warrior on p1's board (len(board) == 1 -> append pos = 1).
    board_warrior = CardInstance(
        instance_id=UUID(int=9001), card_id=9001, name="board_warrior",
        card_type=CardType.WARRIOR, mana_cost=1, attack=2, hp=3, max_hp=3,
        is_ready=False,
    )
    live.p1.board.append(board_warrior)

    # One affordable no-mechanics WARRIOR in p1's hand at hand_index 0.
    hand_warrior = CardInstance(
        instance_id=UUID(int=9002), card_id=9002, name="hand_warrior",
        card_type=CardType.WARRIOR, mana_cost=1, attack=1, hp=1, max_hp=1,
        is_ready=False,
    )
    live.p1.hand = [hand_warrior]
    live.p1.mana = 5
    live.p1.max_mana = 5

    # p1 is the actor.
    live.current_turn_owner_id = live.p1.user_id
    live.turn_number = 3
    live.status = GameStatus.ONGOING
    return env, live


def _write_single_row_trace(group_dir, battle_id, snap, *, actor_uid,
                            meta_status="p2_win"):
    """Write a one-row v5 trace (manifest + meta + actions.jsonl)."""
    row = {
        "seq": 1, "battle_id": battle_id, "turn_number": int(snap["turn_number"]),
        "actor_user_id": int(actor_uid), "actor_player": 1,
        "decision_source": "test", "legal_action_index": None,
        "action_type": "play_card",
        "action_json": {"type": "play_card"}, "action_native": None,
        "source_card": None, "target_card": None, "legal_actions": [],
        "legal_action_count": 0, "pre_state": snap, "post_state": snap,
        "deltas": None, "accepted": True, "error": None, "timestamp_ms": 0,
        "visibility": "omniscient_offline_only",
    }
    v5_dir = group_dir / "battles" / battle_id / "v5"
    v5_dir.mkdir(parents=True, exist_ok=True)
    with (v5_dir / "actions.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "schema_version": "rlhf_v5_storage_v1", "group_id": "g",
        "battle_id": battle_id, "status": meta_status, "winner_user_id": None,
        "turns": int(snap["turn_number"]),
        "p1_user_id": int(snap["p1"]["user_id"]),
        "p2_user_id": int(snap["p2"]["user_id"]),
        "p1_actor_type": "human", "p2_actor_type": "bot",
        "v5_trace_present": True,
    }
    (v5_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    manifest = {
        "manifest_version": "rlhf_manifest_v1", "group_id": "g",
        "battles_results": [{
            "battle_id": battle_id,
            "battle_log_path": f"b_{battle_id}.json",
            "winner_user_id": None, "loser_user_id": None,
            "status": meta_status.upper(), "turns": int(snap["turn_number"]),
            "duration_seconds": 0.0,
            "v5_dir": f"battles/{battle_id}/v5",
            "v5_meta_path": f"battles/{battle_id}/v5/meta.json",
            "v5_trace_ok": True,
        }],
        "battle_ids": [battle_id],
    }
    (group_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _nonzero_rows(feats_or_mask):
    """Rows that are not entirely zero."""
    if feats_or_mask.ndim == 1:
        return int(np.count_nonzero(feats_or_mask))
    return int(np.count_nonzero(np.any(feats_or_mask != 0.0, axis=1)))


def test_loader_action_features_append_only(tmp_path):
    """C0: the loader's ``action_features`` is engine-faithful (append_only).

    Bug-trigger: a WARRIOR in hand with >=1 warrior on board. 'full' would
    emit PlayCard candidates at every board position; 'append_only' only at
    ``pos=len(board)``. Asserted on the LOADER output (not a rebuild).
    """
    env, live = _bug_trigger_state()
    actor = live.current_turn_owner_id
    assert actor == live.p1.user_id

    # Sanity: the bug-trigger is genuinely exercisable.
    assert any(c.card_type == CardType.WARRIOR for c in live.p1.hand), "need a warrior in hand"
    assert len(live.p1.board) >= 1, "need >=1 unit on board (append pos != 0)"
    append_pos = len(live.p1.board)

    # Engine ORACLE: raw legal-action list on the LIVE state. The 601-
    # candidate codec (end_turn + play_card + attack) does NOT model
    # mana_draw — that lives in a separate head (``mana_draw_head_v5``).
    # So the consistency invariant compares the codec-shaped subset:
    # engine legal actions MINUS ManaDrawAction.
    from core.actions import ManaDrawAction
    engine_legal = env._env.get_legal_actions(actor)
    engine_count = sum(1 for a in engine_legal
                       if not isinstance(a, ManaDrawAction))

    # Codec masks on the reconstructed state (what the loader sees).
    snap = _StubRecorder(live).snapshot_state()
    recon = reconstruct_gamestate(snap)
    mask_append = build_action_mask(recon, actor, verify_mask=False,
                                    placement_mode="append_only")
    mask_full = build_action_mask(recon, actor, verify_mask=False,
                                  placement_mode="full")

    # The bug-trigger must produce a divergence: 'full' has strictly more
    # nonzero candidates than 'append_only' (the extra warrior positions).
    assert _nonzero_rows(mask_full) > _nonzero_rows(mask_append), (
        "fixture does not exercise the bug: 'full' and 'append_only' agree "
        f"(full={_nonzero_rows(mask_full)}, append={_nonzero_rows(mask_append)})"
    )

    # Consistency invariant: append_only mask nonzero count == engine count.
    assert _nonzero_rows(mask_append) == engine_count, (
        f"append_only mask count={_nonzero_rows(mask_append)} != "
        f"engine get_legal_actions count={engine_count}"
    )

    # Run the LOADER on a one-row trace built from this snapshot.
    _write_single_row_trace(tmp_path, "b_c0", snap, actor_uid=actor)
    info = InfoModeV5(
        own_hand_identity_known=True, own_deck_known=True,
        enemy_hand_known=True, enemy_deck_known=True,
        enemy_deck_order_known=True,
    )
    transitions = load_offline_dataset(tmp_path, info_mode=info)
    assert len(transitions) == 1, f"expected 1 transition, got {len(transitions)}"
    t = transitions[0]
    af = t.action_features
    assert af.shape == (601, 171)

    # (a) loader action_features nonzero rows == append_only mask nonzero rows.
    af_nz = _nonzero_rows(af)
    mask_append_nz = _nonzero_rows(mask_append)
    assert af_nz == mask_append_nz, (
        f"loader action_features nonzero rows={af_nz} != "
        f"append_only mask nonzero rows={mask_append_nz} "
        "(loader did NOT honour placement_mode='append_only')"
    )

    # (b) loader action_features nonzero rows == engine legal-action count.
    assert af_nz == engine_count, (
        f"loader action_features nonzero rows={af_nz} != "
        f"engine get_legal_actions count={engine_count} "
        "(consistency invariant broken)"
    )

    # (c) warrior PlayCard candidate appears ONLY at the append position.
    # hand_index 0 is the warrior; for a no-target warrior the candidate is
    # at tcode=0 within each position block.
    hand_idx = 0
    nz_positions = []
    for pos_idx in range(_NUM_PLAY_POS):
        base = _PLAY_BASE + hand_idx * _PLAY_STRIDE + pos_idx * _NUM_PLAY_TARGETS
        # The whole 17-target block for this (hand_idx, pos_idx) — a
        # no-mechanics warrior only sets tcode=0, but check the block to be
        # robust: a position is "emitted" if ANY row in its block is nonzero.
        block = af[base:base + _NUM_PLAY_TARGETS]
        if np.any(block != 0.0):
            nz_positions.append(pos_idx)
    assert nz_positions == [append_pos], (
        f"warrior PlayCard candidates emitted at positions {nz_positions}; "
        f"expected ONLY [{append_pos}] (the append position). "
        "Under placement_mode='full' this would be [0, 1, ...] — the bug."
    )

    # Stronger: the non-append position blocks are entirely zero in the
    # loader output (this is the row-level assertion that FAILS under 'full').
    for pos_idx in range(_NUM_PLAY_POS):
        if pos_idx == append_pos:
            continue
        base = _PLAY_BASE + hand_idx * _PLAY_STRIDE + pos_idx * _NUM_PLAY_TARGETS
        block = af[base:base + _NUM_PLAY_TARGETS]
        assert not np.any(block != 0.0), (
            f"loader emitted warrior PlayCard at non-append pos={pos_idx} "
            "(should be zero under append_only)"
        )

    # Cross-check: the direct encode_action_features call with 'full' on the
    # SAME reconstructed state DOES emit at non-append positions — proving
    # the test would FAIL if the loader still used 'full'. (This is the
    # falsification guard for assertion (c).)
    af_full = encode_action_features(recon, actor, verify_mask=False,
                                     include_preview=False,
                                     placement_mode="full")
    full_positions = []
    for pos_idx in range(_NUM_PLAY_POS):
        base = _PLAY_BASE + hand_idx * _PLAY_STRIDE + pos_idx * _NUM_PLAY_TARGETS
        if np.any(af_full[base:base + _NUM_PLAY_TARGETS] != 0.0):
            full_positions.append(pos_idx)
    assert append_pos in full_positions and len(full_positions) > 1, (
        "'full' control did not emit warrior at multiple positions; "
        f"full_positions={full_positions} — the bug-trigger is not exercising "
        "the divergence, so assertion (c) is not a meaningful regression guard"
    )
    assert _nonzero_rows(af_full) == _nonzero_rows(mask_full), (
        "control: encode_action_features('full') nonzero rows must equal "
        "build_action_mask('full') nonzero rows"
    )
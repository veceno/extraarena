"""Re-encode the action_features + mask hashes of the three attack-bearing
golden fixtures from the PYTHON ORACLE so they become true Rust-vs-Python
checks instead of self-referential Rust-derived hashes.

CONTEXT (kernel-fix agent): kernel.rs:1711 now uses `/6.0` for `att_pos`
(mirroring ai/train_v2/classic_actions_v1.py:589
``out[141] = (att_pos + 1) / (_NUM_BOARD + 1)`` where ``_NUM_BOARD = 5``).
The three fixtures that exercise a minion attack
(golden_trace_scripted_basic / taunt_attack / attack_cleanup) embed
``action_features_sha256_f32_le`` hashes that were STALE — they were derived
from the OLD Rust `/8.0` encoder and never validated against the Python real
game (the modality gap). This script closes that gap.

METHOD: like regen_obs5_fixtures.py, this is a RE-ENCODE, not a re-simulate.
Each recorded ``state`` payload is rebuilt into a Python ``GameState`` and the
action_features / mask are recomputed with the SAME Python encoders the
``ClassicRLEnv`` uses (ai/train_v2/classic_actions_v1.build_action_mask /
encode_action_features — the fns behind ClassicRLEnv.action_mask /
.action_features at classic_rl_env.py:210 / :221), using the env_config's
``placement_mode`` / ``verify_mask`` / ``include_preview`` and the snapshot's
``current_turn_owner_id`` as the perspective player (the same player
RolloutKernel::encode_snapshot_with_history uses — kernel.rs:1531 /
golden_kernel.rs:1605). ONLY ``action_features_sha256_f32_le`` +
``mask_sha256_f32_le`` are overwritten; state / obs / obs_v5 / legal_ids /
history / rewards / state_sha256 stay byte-identical. Mask is recomputed too
even though the kernel fix does not affect it (mask has no att_pos channel):
this promotes the mask hash from self-referential Rust-derived to a true
Python-vs-Rust check as well.

The mask does not depend on att_pos, so its recomputed hash is expected to
match the stale value; action_features changes because att_pos channel 141
moves from /8.0 to /6.0 for every attack action_id (545..600).

Re-encodes:
  golden_trace_scripted_basic.json
  golden_trace_taunt_attack.json
  golden_trace_attack_cleanup.json

Usage (from the worktree TrainV3.5 dir):
  PYTHONPATH=<worktree-root>:<worktree-root>/TrainV3.5/python \
      python3 -m train_v3.regen_action_fixtures
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai.train_v2.classic_actions_v1 import (
    build_action_mask,
    encode_action_features,
)

# Reuse the recorded-state -> GameState reconstruction from regen_obs5_fixtures.
from train_v3.regen_obs5_fixtures import _state_from_payload
from train_v3.golden_trace import _hash_f32

FIXTURE_NAMES = (
    "scripted_basic",
    "taunt_attack",
    "attack_cleanup",
)

FIXTURE_DIR = Path("rust/trainv3_core/tests/fixtures")


def _reencode_snapshot(snap: dict[str, Any], ec: dict[str, Any]) -> tuple[str, str]:
    state = _state_from_payload(snap["state"])
    player_id = state.current_turn_owner_id
    placement_mode = str(ec.get("placement_mode", "append_only"))
    verify_mask = bool(ec.get("verify_mask", False))
    include_preview = bool(ec.get("include_preview", False))

    mask = build_action_mask(
        state,
        player_id,
        verify_mask=verify_mask,
        placement_mode=placement_mode,
    )
    features = encode_action_features(
        state,
        player_id,
        include_preview=include_preview,
        verify_mask=verify_mask,
        placement_mode=placement_mode,
        mask=mask,
    )
    new_mask_hash = _hash_f32(mask)
    new_af_hash = _hash_f32(features)
    snap["mask_sha256_f32_le"] = new_mask_hash
    snap["action_features_sha256_f32_le"] = new_af_hash
    return new_af_hash, new_mask_hash


def reencode_one(name: str) -> None:
    path = FIXTURE_DIR / f"golden_trace_{name}.json"
    with open(path) as f:
        trace = json.load(f)
    ec = trace["env_config"]

    n_changed_af = 0
    n_changed_mask = 0

    def _re(snap: dict[str, Any], label: str) -> None:
        nonlocal n_changed_af, n_changed_mask
        old_af = snap.get("action_features_sha256_f32_le")
        old_mask = snap.get("mask_sha256_f32_le")
        new_af, new_mask = _reencode_snapshot(snap, ec)
        if new_af != old_af:
            n_changed_af += 1
            print(f"  {label} action_features: {old_af} -> {new_af}")
        if new_mask != old_mask:
            n_changed_mask += 1
            print(f"  {label} mask: {old_mask} -> {new_mask}")

    _re(trace["initial"], "initial")
    for i, step in enumerate(trace["steps"]):
        _re(step["pre"], f"step{i} pre ")
        _re(step["post"], f"step{i} post")

    with open(path, "w") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
    print(
        f"  [{name}] re-encoded action_features/mask ({len(trace['steps'])} steps):"
        f" {n_changed_af} af hashes changed, {n_changed_mask} mask hashes changed"
    )


def main() -> None:
    for name in FIXTURE_NAMES:
        print(f"[{name}] re-encoding action_features + mask hashes from Python oracle...")
        reencode_one(name)
    print("[action-regen] done")


if __name__ == "__main__":
    main()
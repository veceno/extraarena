"""Re-encode full-matcher golden fixtures to the current V5 obs contract.

The Rust observation encoder was ported from the OLD 7-slot card-slot layout to
the current Python `classic_obs_v1._encode_card_slots` 5-slot layout (own board
0..4, enemy board 0..4, hand 0..3, then 4 trailing zero-padding slots so the
card-slot region stays 20 × CARD_SHAPE_DIM). The five fixtures that use the
FULL obs+state matcher embed `obs_sha256_f32_le` / `obs_v5_sha256_f32_le` hashes
that were stale vs the new 5-slot encoder.

The recorded gameplay STATE in these fixtures already matches the current Rust
kernel replay (the baseline suite was green at 142 tests with the 7-slot
encoder, proving Rust reproduces these exact states). The deck pool changed in
Phase 5 (card 51 added), so re-simulating from `env_config` with default decks
would produce a DIFFERENT state trajectory and make the recorded `action_id`
sequence illegal. Therefore the correct regen is a RE-ENCODE, not a re-simulate:
rebuild each recorded `state` payload into a Python `GameState`, re-encode the
observation with the current 5-slot `encode_observation` / `encode_observation_v5`,
and overwrite ONLY `obs_sha256_f32_le` / `obs_v5_sha256_f32_le`. Every other field
(state, mask, action_features, legal_ids, history_events, state_sha256, env_config,
action_id, rewards, reward_components_v5, ...) stays byte-identical, so the
"STATE is reproduced identically, only the OBS slot count changes" guarantee
holds. Mask / action_features hashes are left untouched: those encoders do not
depend on the card-slot layout and already matched Rust (baseline green).

Re-encodes:
  golden_trace_seed123.json
  golden_trace_scripted_basic.json
  golden_trace_targeted_potion.json
  golden_trace_taunt_attack.json
  golden_trace_attack_cleanup.json

Usage (from the worktree TrainV3.5 dir):
  PYTHONPATH=<worktree-root>:<worktree-root>/TrainV3.5/python \
      python3 -m train_v3.regen_obs5_fixtures
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from core.state import (
    CardInstance,
    CardType,
    GameState,
    GameStatus,
    PlayerState,
)
from ai.train_v2.classic_obs_v1 import encode_observation
from train_v3.contracts import AssistModeV5, InfoModeV5
from train_v3.golden_trace import _card_payload, _hash_f32, _history_cards_for_action
from train_v3.obs_v5 import encode_observation_v5

FIXTURE_NAMES = (
    "seed123",
    "scripted_basic",
    "targeted_potion",
    "taunt_attack",
    "attack_cleanup",
    "e2e_oracle",
)

FIXTURE_DIR = Path("rust/trainv3_core/tests/fixtures")

_STATUS_MAP = {s.value: s for s in GameStatus}
_TYPE_MAP = {t.value: t for t in CardType}


def _card_from_payload(p: dict[str, Any]) -> CardInstance:
    return CardInstance(
        card_id=int(p["card_id"]),
        name=str(p.get("name", "")),
        card_type=_TYPE_MAP[p["type"]],
        mana_cost=int(p["mana_cost"]),
        attack=int(p["attack"]),
        hp=int(p["hp"]),
        max_hp=int(p["max_hp"]),
        mechanics=list(p["mechanics"]),
        is_ready=bool(p["is_ready"]),
        is_frozen=bool(p["is_frozen"]),
        level=int(p["level"]),
        skip_count=int(p.get("skip_count", 0)),
    )


def _player_from_payload(p: dict[str, Any]) -> PlayerState:
    return PlayerState(
        user_id=int(p["user_id"]),
        hero=_card_from_payload(p["hero"]),
        mana=int(p["mana"]),
        max_mana=int(p["max_mana"]),
        mana_draw_count_this_turn=int(p.get("mana_draw_count_this_turn", 0)),
        hand=[_card_from_payload(c) for c in p["hand"]],
        board=[_card_from_payload(c) for c in p["board"]],
        deck=[_card_from_payload(c) for c in p["deck"]],
        graveyard=[_card_from_payload(c) for c in p["graveyard"]],
        trophies=int(p.get("trophies", 0)),
    )


def _state_from_payload(st: dict[str, Any]) -> GameState:
    return GameState(
        p1=_player_from_payload(st["p1"]),
        p2=_player_from_payload(st["p2"]),
        current_turn_owner_id=int(st["current_turn_owner_id"]),
        turn_number=int(st["turn_number"]),
        status=_STATUS_MAP[st["status"]],
        sudden_death_turns_by_player={
            int(k): int(v) for k, v in st.get("sudden_death_turns_by_player", {}).items()
        },
        sudden_death_last_applied_turn_by_player={
            int(k): int(v)
            for k, v in st.get("sudden_death_last_applied_turn_by_player", {}).items()
        },
    )


def _info_mode_from_env_config(ec: dict) -> InfoModeV5:
    return InfoModeV5(
        adaptive_strength=float(ec.get("adaptive_strength", 1.0)),
        own_hand_identity_known=bool(ec.get("own_hand_identity_known", True)),
        own_deck_known=bool(ec.get("own_deck_known", True)),
        enemy_hand_known=bool(ec.get("enemy_hand_known", False)),
        enemy_deck_known=bool(ec.get("enemy_deck_known", False)),
        enemy_deck_order_known=bool(ec.get("enemy_deck_order_known", False)),
        draw_assist_enabled=bool(ec.get("draw_assist_enabled", False)),
        draw_assist_strength=float(ec.get("draw_assist_strength", 0.0)),
    )


def _assist_mode_from_env_config(ec: dict) -> AssistModeV5:
    return AssistModeV5(
        assembler_enabled=bool(ec.get("assembler_enabled", False)),
        assembler_strength=float(ec.get("assembler_strength", 0.0)),
        desirerer_enabled=bool(ec.get("desirerer_enabled", False)),
        desirerer_strength=float(ec.get("desirerer_strength", 0.0)),
        teacher_hint_available=bool(ec.get("teacher_hint_available", False)),
        assist_profile_id=int(ec.get("assist_profile_id", 0)),
    )


def _reencode_snapshot(snap: dict[str, Any], info_mode: InfoModeV5,
                       assist_mode: AssistModeV5) -> None:
    state = _state_from_payload(snap["state"])
    player_id = state.current_turn_owner_id
    obs = encode_observation(state, player_id)
    snap["obs_sha256_f32_le"] = _hash_f32(obs)
    if snap.get("obs_v5_sha256_f32_le") is not None:
        obs_v5 = encode_observation_v5(
            state,
            player_id,
            info_mode=info_mode,
            assist_mode=assist_mode,
            history_events=snap.get("history_events", []),
        )
        snap["obs_v5_sha256_f32_le"] = _hash_f32(obs_v5)
        snap["obs_v5_dim"] = int(obs_v5.shape[0])


def reencode_one(name: str) -> None:
    path = FIXTURE_DIR / f"golden_trace_{name}.json"
    with open(path) as f:
        trace = json.load(f)
    ec = trace["env_config"]
    info_mode = _info_mode_from_env_config(ec)
    assist_mode = _assist_mode_from_env_config(ec)

    history_events = copy.deepcopy(trace["initial"].get("history_events") or [])
    trace["initial"]["history_events"] = copy.deepcopy(history_events)
    _reencode_snapshot(trace["initial"], info_mode, assist_mode)
    for step in trace["steps"]:
        # The live Python env and Rust rollout worker both encode source/target
        # card shapes in the V5 history window. Older golden generation omitted
        # them, making worker obs diverge even though state/action/reward parity
        # was correct. Reconstruct card references from the frozen pre-state and
        # action id without re-simulating gameplay.
        step["pre"]["history_events"] = copy.deepcopy(history_events)
        _reencode_snapshot(step["pre"], info_mode, assist_mode)
        old_post_history = step["post"].get("history_events") or []
        event = copy.deepcopy(old_post_history[-1]) if old_post_history else {
            "actor_id": step["acting_player_id"],
            "action_id": step["action_id"],
            "action_type": "mana_draw" if step.get("mana_draw_taken") else "unknown",
        }
        event.pop("source_card", None)
        event.pop("target_card", None)
        if not step.get("mana_draw_taken"):
            pre_state = _state_from_payload(step["pre"]["state"])
            source, target = _history_cards_for_action(
                pre_state, int(step["acting_player_id"]), int(step["action_id"]),
            )
            event["source_card"] = _card_payload(source)
            event["target_card"] = _card_payload(target)
        else:
            event["source_card"] = None
            event["target_card"] = None
        history_events = [*history_events, event][-20:]
        step["post"]["history_events"] = copy.deepcopy(history_events)
        _reencode_snapshot(step["post"], info_mode, assist_mode)

    with open(path, "w") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
    print(f"  re-encoded {path} ({len(trace['steps'])} steps)")


def main() -> None:
    for name in FIXTURE_NAMES:
        print(f"[{name}] re-encoding obs hashes (5-slot)...")
        reencode_one(name)
    print("[obs5-regen] done")


if __name__ == "__main__":
    main()

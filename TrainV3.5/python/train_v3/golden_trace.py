"""Golden trace generator for future TrainV3/Rust parity checks.

This module intentionally uses the current Python TrainV2 environment as the oracle.
It is training tooling, not production bot runtime code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import numpy as np

import core.engine as _core_engine
from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.classic_actions_v1 import decode_action
from core.actions import ManaDrawAction
from core.state import GameStatus

from .contracts import AssistModeV5, HISTORY_EVENTS, InfoModeV5
from .obs_v5 import encode_observation_v5
from .reward_v5 import (
    compute_history_outcome_deltas_v5,
    compute_reward_components_v5,
    compute_weighted_reward_v5,
    reward_snapshot_v5,
)

SCHEMA = "trainv3-golden-trace-v1"


def _hash_f32(arr: np.ndarray) -> str:
    canonical = np.ascontiguousarray(arr.astype("<f4", copy=False))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _card_payload(card) -> dict[str, Any] | None:
    if card is None:
        return None
    base_snapshot_set = card.base_attack is not None
    return {
        "card_id": int(card.card_id),
        "type": card.card_type.value,
        "mana_cost": int(card.mana_cost),
        "attack": int(card.attack),
        "hp": int(card.hp),
        "max_hp": int(card.max_hp),
        "mechanics": list(card.mechanics),
        "is_ready": bool(card.is_ready),
        "is_frozen": bool(card.is_frozen),
        "level": int(card.level),
        "skip_count": int(card.skip_count),
        "base_attack": int(card.base_attack or 0),
        "base_hp": int(card.base_hp or 0),
        "base_max_hp": int(card.base_max_hp or 0),
        "base_mana_cost": int(card.base_mana_cost or 0),
        "base_mechanics": list(card.base_mechanics or []),
        "base_snapshot_set": bool(base_snapshot_set),
    }


def _player_payload(player) -> dict[str, Any]:
    return {
        "user_id": int(player.user_id),
        "mana": int(player.mana),
        "max_mana": int(player.max_mana),
        "hero": _card_payload(player.hero),
        "hand": [_card_payload(c) for c in player.hand],
        "board": [_card_payload(c) for c in player.board],
        "deck": [_card_payload(c) for c in player.deck],
        "graveyard": [_card_payload(c) for c in player.graveyard],
        "mana_draw_count_this_turn": int(player.mana_draw_count_this_turn),
    }


def _state_payload(env: ClassicRLEnv) -> dict[str, Any]:
    st = env._env.state
    return {
        "current_turn_owner_id": int(st.current_turn_owner_id),
        "turn_number": int(st.turn_number),
        "status": st.status.value,
        "p1": _player_payload(st.p1),
        "p2": _player_payload(st.p2),
        "sudden_death_turns_by_player": {
            int(k): int(v) for k, v in st.sudden_death_turns_by_player.items()
        },
        "sudden_death_last_applied_turn_by_player": {
            int(k): int(v)
            for k, v in st.sudden_death_last_applied_turn_by_player.items()
        },
        # pending_mana_drain_by_player (core/state.py:183) — two-stage mana_drain
        # scheduled amount per opponent user_id, applied at the opponent's next
        # turn start (core/engine.py:700-703). Included so the Rust
        # state-transition matcher can replay the pending drain ACROSS steps
        # (the pending field is internal state that crosses step boundaries).
        "pending_mana_drain_by_player": {
            int(k): int(v) for k, v in st.pending_mana_drain_by_player.items()
        },
    }


def _history_event_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "actor_id": int(event.get("actor_id", 0) or 0),
        "action_id": int(event.get("action_id", 0) or 0),
        "action_type": str(event.get("action_type") or ""),
        "enemy_hero_hp_delta": int(event.get("enemy_hero_hp_delta", 0) or 0),
        "own_hero_hp_delta": int(event.get("own_hero_hp_delta", 0) or 0),
        "my_board_count_delta": int(event.get("my_board_count_delta", 0) or 0),
        "enemy_board_count_delta": int(event.get("enemy_board_count_delta", 0) or 0),
        "board_power_delta": float(event.get("board_power_delta", 0.0) or 0.0),
        "turn_number": int(event.get("turn_number", 0) or 0),
    }
    if "source_card" in event:
        payload["source_card"] = _card_payload(event.get("source_card"))
    if "target_card" in event:
        payload["target_card"] = _card_payload(event.get("target_card"))
    return payload


def _hash_state(env: ClassicRLEnv) -> str:
    raw = json.dumps(_state_payload(env), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _snapshot_hashes(
    env: ClassicRLEnv,
    *,
    include_preview: bool,
    include_v5: bool,
    info_mode: InfoModeV5,
    assist_mode: AssistModeV5,
    history_events: list[dict[str, Any]],
) -> dict[str, Any]:
    cp = env.current_player_id()
    obs = env.observe(cp)
    mask = env.action_mask(cp)
    features = env.action_features(cp, include_preview=include_preview)
    payload = {
        "state": _state_payload(env),
        "history_events": [_history_event_payload(e) for e in history_events[-HISTORY_EVENTS:]],
        "state_sha256": _hash_state(env),
        "obs_sha256_f32_le": _hash_f32(obs),
        "mask_sha256_f32_le": _hash_f32(mask),
        "action_features_sha256_f32_le": _hash_f32(features),
        "legal_ids": [int(i) for i in np.flatnonzero(mask == 1.0)],
    }
    if include_v5:
        obs_v5 = encode_observation_v5(
            env._env.state,
            cp,
            info_mode=info_mode,
            assist_mode=assist_mode,
            history_events=history_events,
        )
        payload["obs_v5_sha256_f32_le"] = _hash_f32(obs_v5)
        payload["obs_v5_dim"] = int(obs_v5.shape[0])
    return payload


class _RecordingRng:
    """Proxy RNG that records graveyard→deck reshuffle orders.

    Replaces `env._env._rng` so every `rng.shuffle(x)` call inside
    `core.engine.draw_one_from_deck` is captured. `x` is the list of
    CardInstance being shuffled into the deck; after the inner RNG shuffles
    it in place we record `[c.card_id for c in x]` — the post-shuffle deck
    card_id sequence that Rust replays directly (outcome-based, no MT19937
    primitive replay). All other RNG primitives (`random()`, `getrandbits`,
    ...) are forwarded to the inner `random.Random` so the seeded MT19937
    stream — and therefore the recorded draw picks — stays deterministic.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.shuffle_orders: list[list[int]] = []

    def shuffle(self, x: list) -> None:
        self._inner.shuffle(x)
        self.shuffle_orders.append([int(c.card_id) for c in x])

    def random(self) -> float:
        return self._inner.random()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _DrawRecorder:
    """Records the outcome streams for the recorded-outcome RNG protocol.

    `picks`  : the chosen deck index (usize) for every `_weighted_choice_idx`
               call — i.e. every weighted draw (clean draw + overdraw-discard).
    `orders` : the post-shuffle deck card_id sequence for every
               graveyard→deck reshuffle.
    `randint_rolls` : the result of every module-level `random.randint(a, b)`
               call during a step (Phase 4: `armor_X_Y` range roll in
               `core/effects.py::apply_damage_modifiers`). Rust replays
               these via `roll_range` in the SAME call order — Rust must
               call `roll_range` at exactly the Python `random.randint`
               call sites so the streams stay aligned.
    `choice_rolls` : the 0-based index chosen by every module-level
               `random.choice(seq)` call during a step (Phase 6: card15
               `battlecry_damage_X_random` random-target pick in
               `core/effects.py::_apply_random_battlecry_damage`, where
               `targets = list(opponent.board) + [opponent.hero]`). Rust
               replays these via `roll_choice` in the SAME call order —
               Rust must call `roll_choice` at exactly the Python
               `random.choice` call sites so the streams stay aligned.

    All lists are appended to in call order across the whole step loop; the
    per-step slice is computed by snapshotting lengths before each step.
    """

    def __init__(self) -> None:
        self.picks: list[int] = []
        self.orders: list[list[int]] = []
        self.randint_rolls: list[int] = []
        self.choice_rolls: list[int] = []
        self.sample_rolls: list[list[int]] = []

    def install(self, env: ClassicRLEnv) -> callable:
        """Install the recording instrumentation on `env`.

        Must be called AFTER `env.reset(...)` (so reset-time RNG / initial
        deck shuffle is NOT recorded — Rust loads the recorded initial state
        directly) and BEFORE the step loop. Returns a restore callable that
        removes the patches (call in a `finally` block).
        """
        original_wci = _core_engine._weighted_choice_idx
        recorder = self

        def _recording_wci(weights, rng):
            idx = original_wci(weights, rng)
            recorder.picks.append(int(idx))
            return idx

        _core_engine._weighted_choice_idx = _recording_wci

        original_rng = env._env._rng
        recording_rng = _RecordingRng(original_rng)
        recording_rng.shuffle_orders = self.orders
        env._env._rng = recording_rng
        # The inner RNG used by `_weighted_choice_idx` (via rng.random()) is
        # now the recording proxy; this is intentional — it forwards to the
        # original MT19937 so picks stay deterministic.

        # Phase 4: record module-level `random.randint` outcomes (armor_X_Y
        # range roll, future card15 spell choice). `core/effects.py` calls
        # `random.randint(...)` (the module-level function bound to
        # `random._inst`, which ClassicRLEnv.reset seeds via
        # `rand_mod.seed(seed)`). Patching `random.randint` captures every
        # call site that routes through the module-level random — Rust mirrors
        # those exact call sites with `roll_range`.
        import random as _random_module

        original_randint = _random_module.randint

        def _recording_randint(a: int, b: int) -> int:
            result = original_randint(a, b)
            recorder.randint_rolls.append(int(result))
            return result

        _random_module.randint = _recording_randint

        # Phase 6: record module-level `random.choice` outcomes (card15
        # `battlecry_damage_X_random` random-target pick). `core/effects.py`
        # calls `random.choice(...)` (module-level). We record the 0-based
        # index of the chosen element within the input sequence (matched by
        # identity via `id()` so duplicate-valued cards with unique
        # instance_ids still resolve correctly). Rust mirrors the exact
        # call sites with `roll_choice`.
        original_choice = _random_module.choice

        def _recording_choice(seq):
            result = original_choice(seq)
            seq_list = list(seq)
            result_id = id(result)
            idx = next(
                (i for i, x in enumerate(seq_list) if id(x) == result_id),
                0,
            )
            recorder.choice_rolls.append(int(idx))
            return result

        _random_module.choice = _recording_choice

        # Phase 9: record module-level `random.sample` outcomes (card26
        # `cast_random_spell` Blackwhip freeze-target pick). `core/effects.py`
        # calls `random.sample(population, k)` (module-level). We record the
        # 0-based indices of the chosen elements within the input population,
        # in selection order (matched by identity via `id()` so duplicate-
        # valued cards with unique instance_ids still resolve correctly). Rust
        # mirrors the exact call sites with `roll_sample`.
        original_sample = _random_module.sample

        def _recording_sample(population, k):
            result = original_sample(population, k)
            pop_list = list(population)
            result_ids = [id(x) for x in result]
            indices: list[int] = []
            for rid in result_ids:
                idx = next(
                    (i for i, x in enumerate(pop_list) if id(x) == rid),
                    0,
                )
                indices.append(int(idx))
            recorder.sample_rolls.append(indices)
            return result

        _random_module.sample = _recording_sample

        def _restore() -> None:
            _core_engine._weighted_choice_idx = original_wci
            env._env._rng = original_rng
            _random_module.randint = original_randint
            _random_module.choice = original_choice
            _random_module.sample = original_sample

        return _restore


def _forced_engine_step(env: ClassicRLEnv, player_id: int, action_id: int):
    """Force-apply a (possibly mask-illegal) `action_id` via the engine apply
    path (`env._env.step`), bypassing the TrainV2 action_mask legality check.

    Used by `build_golden_trace` for `force_steps`: the consume_ally play at a
    full board is masked OUT by the frozen `classic_actions_v1` mask (no
    consume_ally exemption), but `core/engine.py:1228` exempts consume_ally in
    the apply path, so the play succeeds when forced. Mirrors the
    `step_core_action` reward/snapshot/terminate logic but decodes the
    TrainV2 `action_id` to a production `BaseAction` first.
    """
    st = env._env.state
    if st.status != GameStatus.ONGOING:
        return env.observe(), 0.0, True, False, env._make_info(
            action_id=action_id, success=False, error="game_over", invalid=True,
            acting_player_id=player_id,
        )
    action = decode_action(st, player_id, action_id)
    if action is None:
        raise ValueError(f"forced action_id {action_id} did not decode for player {player_id}")
    pre_snapshot = env._snapshot(player_id)
    success, error = env._env.step(player_id, action)
    if not success:
        raise ValueError(f"forced action_id {action_id} failed: {error}")
    env._steps += 1
    post_snapshot = env._snapshot(player_id)
    reward = env._compute_reward(player_id, pre_snapshot, post_snapshot, success)
    env._add_reward(player_id, reward)
    terminated = st.status != GameStatus.ONGOING
    truncated = st.turn_number > env._max_turns
    if env._cache is not None:
        env._cache.set_state(env._env.state, env.current_player_id())
    info = env._make_info(
        action_id=action_id, success=True, error="",
        acting_reward=reward, acting_player_id=player_id, action=action,
    )
    return env.observe(), reward, terminated, truncated, info


def build_golden_trace(
    *,
    seed: int = 42,
    steps: int = 16,
    placement_mode: str = "append_only",
    verify_mask: bool = True,
    include_preview: bool = False,
    include_v5: bool = True,
    info_mode: InfoModeV5 | None = None,
    assist_mode: AssistModeV5 | None = None,
    v5_weighted_reward: bool = True,
    choose: str = "first",
    p1_deck_ids: list[int] | None = None,
    p2_deck_ids: list[int] | None = None,
    p1_level: int | None = None,
    p2_level: int | None = None,
    p1_levels: dict[int, int] | None = None,
    p2_levels: dict[int, int] | None = None,
    action_ids: list[int] | None = None,
    mana_draw_flags: list[bool] | None = None,
    force_steps: set[int] | None = None,
    mana_per_turn: int = 1,
    post_reset_setup: "callable | None" = None,
    sudden_death_enabled: bool = False,
    sudden_death_damage_start: int = 1,
    sudden_death_damage_step: int = 1,
    max_turns: int = 80,
) -> dict[str, Any]:
    """Build a deterministic TrainV3 parity trace from Python TrainV2."""
    if choose not in {"first", "last"}:
        raise ValueError("choose must be 'first' or 'last'")
    if action_ids is not None and len(action_ids) > steps:
        raise ValueError("action_ids length must be <= steps")
    if mana_draw_flags is not None and len(mana_draw_flags) > steps:
        raise ValueError("mana_draw_flags length must be <= steps")

    info_mode = info_mode or InfoModeV5(enemy_hand_known=True, enemy_deck_known=True)
    assist_mode = assist_mode or AssistModeV5()
    history_events: list[dict[str, Any]] = []

    classic_params = None
    if sudden_death_enabled:
        from infrastructure.match_modes import ClassicParams

        classic_params = ClassicParams(
            sudden_death_enabled=True,
            sudden_death_damage_start=sudden_death_damage_start,
            sudden_death_damage_step=sudden_death_damage_step,
            mana_per_turn=mana_per_turn,
        )
    env = ClassicRLEnv(
        seed=seed,
        verify_mask=verify_mask,
        placement_mode=placement_mode,
        include_legal_actions_in_info=True,
        mana_per_turn=mana_per_turn,
        max_turns=max_turns,
        classic_params=classic_params,
    )
    if p1_levels is None and p1_level is not None:
        p1_levels = _all_card_levels(env, p1_level)
    if p2_levels is None and p2_level is not None:
        p2_levels = _all_card_levels(env, p2_level)
    env.reset(
        seed=seed,
        p1_deck_ids=p1_deck_ids,
        p2_deck_ids=p2_deck_ids,
        p1_levels=p1_levels,
        p2_levels=p2_levels,
    )
    # Optional post-reset state mutation hook (Phase 4: the armor_X_Y fixture
    # injects `armor_1_3` directly into the hero's mechanics, because
    # `core/converter._normalize_mechanic` collapses `armor_X_Y` → `armor_X`
    # at deck-construction and the engine would otherwise never see the
    # range form. This is a TEST-only injection to exercise the engine's
    # `random.randint` armor path; prod decks always carry the collapsed
    # `armor_X` form. Runs AFTER reset, BEFORE recorder install + step loop.
    if post_reset_setup is not None:
        post_reset_setup(env)
    level_handicap = _level_handicap_payload(p1_level=p1_level, p2_level=p2_level)

    trace: dict[str, Any] = {
        "schema": SCHEMA,
        "env_config": {
            "seed": seed,
            "verify_mask": verify_mask,
            "placement_mode": placement_mode,
            "include_preview": include_preview,
            "include_v5": include_v5,
            "adaptive_strength": info_mode.clipped_strength(),
            "own_hand_identity_known": info_mode.own_hand_identity_known,
            "own_deck_known": info_mode.own_deck_known,
            "enemy_hand_known": info_mode.enemy_hand_known,
            "enemy_deck_known": info_mode.enemy_deck_known,
            "enemy_deck_order_known": info_mode.enemy_deck_order_known,
            "draw_assist_enabled": info_mode.draw_assist_enabled,
            "draw_assist_strength": info_mode.clipped_draw_assist_strength(),
            **assist_mode.to_dict(),
            "level_handicap": level_handicap,
            "v5_weighted_reward": bool(v5_weighted_reward),
            "mana_per_turn": env._mana_per_turn,
            "overdraw_to_discard": bool(
                getattr(env._env.classic_params, "overdraw_to_discard", False)
            ),
            "sudden_death_enabled": bool(sudden_death_enabled),
            "sudden_death_damage_start": int(sudden_death_damage_start),
            "sudden_death_damage_step": int(sudden_death_damage_step),
            "max_turns": int(env._max_turns),
        },
        "initial": _snapshot_hashes(
            env,
            include_preview=include_preview,
            include_v5=include_v5,
            info_mode=info_mode,
            assist_mode=assist_mode,
            history_events=history_events,
        ),
        "steps": [],
    }

    # Recorded-outcome RNG protocol (task #14 / DW-7): install the draw
    # instrumentation AFTER reset (reset-time RNG / initial deck shuffle is
    # NOT recorded — Rust loads the recorded initial state directly) and
    # BEFORE the step loop. The recorder captures two outcome streams in
    # call order across all steps; per-step slices are snapshotted below.
    recorder = _DrawRecorder()
    restore_recorder = recorder.install(env)

    try:
        for t in range(steps):
            # Snapshot recorder lengths BEFORE the step so the per-step slices
            # (draw_picks / reshuffle_orders / randint_rolls) capture only this
            # step's outcomes.
            picks_before = len(recorder.picks)
            orders_before = len(recorder.orders)
            randint_before = len(recorder.randint_rolls)
            choice_before = len(recorder.choice_rolls)
            sample_before = len(recorder.sample_rolls)
            pre = _snapshot_hashes(
                env,
                include_preview=include_preview,
                include_v5=include_v5,
                info_mode=info_mode,
                assist_mode=assist_mode,
                history_events=history_events,
            )
            legal_ids = pre["legal_ids"]
            forced = bool(force_steps is not None and t in force_steps)
            if action_ids is not None and t < len(action_ids):
                action_id = int(action_ids[t])
                if (not forced) and (action_id not in legal_ids):
                    raise ValueError(f"action_id {action_id} at step {t} is not legal; legal_ids={legal_ids}")
            else:
                action_id = legal_ids[0] if choose == "first" else legal_ids[-1]
            acting_player_id = env.current_player_id()
            # Parallel binary mana_draw head (Phase 2: MD-3). mana_draw is a
            # standalone core action NOT in the 601 action space; legality is
            # reported separately. When mana_draw_flags[t] is set, the step takes
            # a ManaDrawAction via step_core_action instead of env.step(action_id).
            engine_legal = env._env.get_legal_actions(acting_player_id)
            mana_draw_legal = any(isinstance(a, ManaDrawAction) for a in engine_legal)
            mana_draw_taken = bool(mana_draw_flags is not None and t < len(mana_draw_flags) and mana_draw_flags[t])
            if mana_draw_taken and not mana_draw_legal:
                _st = env._env.state
                _pl = _st.p1 if _st.p1.user_id == acting_player_id else _st.p2
                raise ValueError(
                    f"mana_draw_flags[{t}] set but mana_draw is not legal; "
                    f"hand={len(_pl.hand)} mana={_pl.mana} "
                    f"count={_pl.mana_draw_count_this_turn}"
                )
            reward_pre = reward_snapshot_v5(env._env.state, acting_player_id)
            if mana_draw_taken:
                _, reward, terminated, truncated, info = env.step_core_action(ManaDrawAction())
            elif forced:
                # Force-apply a (possibly mask-illegal) action_id via the
                # engine apply path (env._env.step), bypassing the TrainV2
                # action_mask check. Used for consume_ally at a full board:
                # the frozen classic_actions_v1 mask masks the consume play
                # OUT (no exemption), but core/engine.py:1228 exempts
                # consume_ally in the apply path, so the play succeeds.
                _, reward, terminated, truncated, info = _forced_engine_step(
                    env, acting_player_id, action_id
                )
            else:
                _, reward, terminated, truncated, info = env.step(action_id)
            reward_post = reward_snapshot_v5(env._env.state, acting_player_id)
            reward_components = compute_reward_components_v5(reward_pre, reward_post)
            weighted_reward = (
                compute_weighted_reward_v5(reward, reward_components, info_mode=info_mode)
                if v5_weighted_reward
                else float(reward)
            )
            event = {
                "actor_id": acting_player_id,
                "action_id": int(action_id),
                "action_type": (info.get("action") or {}).get("type", "unknown"),
                "turn_number": info["turn_number"],
            }
            event.update(compute_history_outcome_deltas_v5(reward_pre, reward_post))
            history_events.append(event)
            post = _snapshot_hashes(
                env,
                include_preview=include_preview,
                include_v5=include_v5,
                info_mode=info_mode,
                assist_mode=assist_mode,
                history_events=history_events,
            )
            draw_picks = [int(i) for i in recorder.picks[picks_before:]]
            reshuffle_orders = [
                [int(cid) for cid in order]
                for order in recorder.orders[orders_before:]
            ]
            randint_rolls = [int(r) for r in recorder.randint_rolls[randint_before:]]
            choice_rolls = [int(i) for i in recorder.choice_rolls[choice_before:]]
            sample_rolls = [
                [int(i) for i in idxs]
                for idxs in recorder.sample_rolls[sample_before:]
            ]
            trace["steps"].append(
                {
                    "t": t,
                    "acting_player_id": acting_player_id,
                    "action_id": int(action_id),
                    "mana_draw_legal": bool(mana_draw_legal),
                    "mana_draw_taken": bool(mana_draw_taken),
                    "pre": pre,
                    "base_reward": float(reward),
                    "reward": float(weighted_reward),
                    "reward_components_v5": reward_components,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": {
                        "current_player_id": info["current_player_id"],
                        "legal_actions": info["legal_actions"],
                        "status": info["status"],
                        "p1_hp": info["p1_hp"],
                        "p2_hp": info["p2_hp"],
                    },
                    "draw_picks": draw_picks,
                    "reshuffle_orders": reshuffle_orders,
                    "randint_rolls": randint_rolls,
                    "choice_rolls": choice_rolls,
                    "sample_rolls": sample_rolls,
                    "post": post,
                }
            )
            if terminated or truncated:
                break
    finally:
        restore_recorder()

    return trace


def _all_card_levels(env: ClassicRLEnv, level: int) -> dict[int, int]:
    safe_level = max(1, min(10, int(level)))
    return {int(card_id): safe_level for card_id in env._cards_data.keys()}


def _level_handicap_payload(*, p1_level: int | None, p2_level: int | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if p1_level is not None:
        payload["p1_level"] = max(1, min(10, int(p1_level)))
    if p2_level is not None:
        payload["p2_level"] = max(1, min(10, int(p2_level)))
    if payload:
        payload["label"] = _level_handicap_label(payload.get("p1_level"), payload.get("p2_level"))
    return payload


def _level_handicap_label(p1_level: int | None, p2_level: int | None) -> str:
    p1 = "default" if p1_level is None else f"l{int(p1_level)}"
    p2 = "default" if p2_level is None else f"l{int(p2_level)}"
    return f"p1_{p1}_vs_p2_{p2}"


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate TrainV3 Python-oracle golden trace")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--placement-mode", default="append_only", choices=["append_only", "full"])
    parser.add_argument("--verify-mask", default=True, type=lambda x: x.lower() in ("true", "1", "yes"))
    parser.add_argument("--include-preview", action="store_true")
    parser.add_argument("--no-v5", action="store_true")
    parser.add_argument("--adaptive-strength", type=float, default=1.0)
    parser.add_argument("--own-hand-known", default=True, type=lambda x: x.lower() in ("true", "1", "yes"))
    parser.add_argument("--own-deck-known", default=True, type=lambda x: x.lower() in ("true", "1", "yes"))
    parser.add_argument("--enemy-hand-known", default=True, type=lambda x: x.lower() in ("true", "1", "yes"))
    parser.add_argument("--enemy-deck-known", default=True, type=lambda x: x.lower() in ("true", "1", "yes"))
    parser.add_argument("--enemy-deck-order-known", default=False, type=lambda x: x.lower() in ("true", "1", "yes"))
    parser.add_argument("--draw-assist-enabled", default=False, type=lambda x: x.lower() in ("true", "1", "yes"))
    parser.add_argument("--draw-assist-strength", type=float, default=0.0)
    parser.add_argument("--v5-weighted-reward", default=True, type=lambda x: x.lower() in ("true", "1", "yes"))
    parser.add_argument("--choose", default="first", choices=["first", "last"])
    parser.add_argument("--p1-deck-ids", default=None, help="Comma-separated card ids for player 1 deck")
    parser.add_argument("--p2-deck-ids", default=None, help="Comma-separated card ids for player 2 deck")
    parser.add_argument("--p1-level", type=int, default=None)
    parser.add_argument("--p2-level", type=int, default=None)
    parser.add_argument("--action-ids", default=None, help="Comma-separated action ids to force for initial steps")
    parser.add_argument(
        "--mana-draw-steps",
        default=None,
        help="Comma-separated step indices (0-based) at which to take a mana_draw action",
    )
    parser.add_argument(
        "--mana-per-turn",
        type=int,
        default=1,
        help="Mana gained per turn (default 1); raise to enable mana_draw scenarios",
    )
    parser.add_argument(
        "--sudden-death-enabled",
        action="store_true",
        help="Enable the sudden-death modifier (hero takes escalating damage each own turn)",
    )
    parser.add_argument(
        "--sudden-death-damage-start",
        type=int,
        default=1,
        help="Sudden-death base damage on a player's first tick (default 1)",
    )
    parser.add_argument(
        "--sudden-death-damage-step",
        type=int,
        default=1,
        help="Sudden-death per-tick damage escalation step (default 1)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=80,
        help="Truncation turn limit: truncated when turn_number > max_turns (default 80)",
    )
    args = parser.parse_args()
    info_mode = InfoModeV5(
        adaptive_strength=args.adaptive_strength,
        own_hand_identity_known=args.own_hand_known,
        own_deck_known=args.own_deck_known,
        enemy_hand_known=args.enemy_hand_known,
        enemy_deck_known=args.enemy_deck_known,
        enemy_deck_order_known=args.enemy_deck_order_known,
        draw_assist_enabled=args.draw_assist_enabled,
        draw_assist_strength=args.draw_assist_strength,
    )
    print(
        json.dumps(
            build_golden_trace(
                seed=args.seed,
                steps=args.steps,
                placement_mode=args.placement_mode,
                verify_mask=args.verify_mask,
                include_preview=args.include_preview,
                include_v5=not args.no_v5,
                info_mode=info_mode,
                v5_weighted_reward=args.v5_weighted_reward,
                choose=args.choose,
                p1_deck_ids=_parse_int_list(args.p1_deck_ids),
                p2_deck_ids=_parse_int_list(args.p2_deck_ids),
                p1_level=args.p1_level,
                p2_level=args.p2_level,
                action_ids=_parse_int_list(args.action_ids),
                mana_draw_flags=_parse_mana_draw_steps(args.mana_draw_steps, args.steps),
                mana_per_turn=args.mana_per_turn,
                sudden_death_enabled=args.sudden_death_enabled,
                sudden_death_damage_start=args.sudden_death_damage_start,
                sudden_death_damage_step=args.sudden_death_damage_step,
                max_turns=args.max_turns,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_mana_draw_steps(value: str | None, steps: int) -> list[bool] | None:
    """Parse `--mana-draw-steps` (comma-separated 0-based step indices)."""
    if value is None or value == "":
        return None
    indices = {int(part.strip()) for part in value.split(",") if part.strip()}
    return [t in indices for t in range(steps)]


if __name__ == "__main__":
    _main()

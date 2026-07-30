"""Block C component C2 -- offline replay bridge (v5_trace -> RustTransitionBatch).

Consumes fresh HUMAN v5_trace rows (from C1 collection via
``offline_dataset_loader.iter_offline_transitions`` OR a pre-materialized
iterable of ``OfflineTransition``) and emits a ``RustTransitionBatch``-shaped
DENSE batch for the C3 AWAC/CRR offline-PPO replay (which feeds
``prepare_rust_ppo_batch`` -> ``RustPPOBatch`` ->
``train_dense_rust_ppo_minibatch`` + ``evaluate_dense_rust_ppo_batch``).

Design (BLOCK_C_PLAN.md section C2 + decisions D-C7 / D-C10 / D11):

  * D11 OMNISCIENT: the bridge calls ``iter_offline_transitions`` with an
    OMNISCIENT ``InfoModeV5(enemy_hand_known=True, enemy_deck_known=True,
    enemy_deck_order_known=True)`` so obs/next_obs match the omniscient deploy
    encoder (design.md:46) AND the omniscient pre_state v5_trace records. The
    loader defaults to self-visible ``InfoModeV5()`` (offline_dataset_loader.py
    :685); the bridge overrides it.
  * D-C7 HUMAN-ONLY: only ``decision_source=='human'`` rows are replayed (reuse
    A1 filter pattern bc_dataset.py:313). bot/rl/llm rows are EXCLUDED -- the
    bot's actions are NOT replayed (we replay the human policy, scored by the
    current V5 policy at bridge time).
  * 601-tcode: per human row, reconstruct ``pre_state`` from the loader's
    ``pre_state_snapshot``; build the append_only mask
    (``build_action_mask(..., placement_mode='append_only')`` -- the C0
    engine-faithful mask, frozen classic_actions_v1:188); resolve the V5
    601-tcode via ``resolve_v5_tcode(pre_state, actor, t.action_native, mask=mask,
    strict=False)`` (A1, bc_dataset.py:172-250). ``action_native`` is the
    loader's ENGINE-sourced field (``t.action_native``), NOT a ``decode_action``
    output (the self-referential trap, bc_dataset.py:34-37). mana_draw rows ->
    ``target_tcode=None``, ``is_mana_draw=True``; terminal/surrender rows ->
    ``target_tcode=None``. Unresolvable rows are skipped (strict=False, A1
    production-robust pattern).
  * D-C10 OLD_LOG_PROB + VALUE AT BRIDGE TIME: run the CURRENT V5 policy forward
    on the batch obs + action_features -> ``(logits, values, mana_draw_logit)``.
    For each NON-mana_draw row with a target_tcode:
    ``old_log_prob = log(softmax(where(mask==1, logits, -1e9))[target_tcode] +
    1e-10)`` -- mirroring the dense evaluator (rust_ppo.py:755-767: masked
    softmax over the 601 candidates, log of the selected-action prob). For
    mana_draw / terminal rows: ``old_log_prob = 0.0`` (no 601 action -- excluded
    from the PPO surrogate; the mana_draw head is targeted by the C3 BCE term,
    not the PPO ratio). ``value = values[i]`` for EVERY row (the per-obs values
    array, rust_ppo.py:756 _out[1]) -- GAE needs V(s_t) at every step. These are
    computed ONCE at bridge time and frozen into the batch (standard PPO old;
    the ratio corrects drift bridge-time -> update-time; D-C10: "old_log_prob =
    current policy at bridge time, NOT the human behavior policy; no
    pi_behavior needed = we need only the human's chosen action").
  * GAE EPISODE BOUNDARIES (LOAD-BEARING): the offline batch concatenates
    disjoint human rows across MANY games. GAE (``_compute_python_gae_returns``,
    rust_ppo.py:581-609) uses ``values[step+1]`` as the per-step next-state value
    and resets the GAE recursion only when terminated OR truncated is True. The
    bridge organizes the batch as ``(steps, env_count=num_games)`` -- each GAME
    is one env, padded to max human-actions-per-game -- NOT a flattened
    ``(total_rows, env_count=1)`` sequence (which would leak value across game
    boundaries because ``bootstrap_values`` is shape ``(env_count,)`` and a
    single env cannot give per-game bootstraps). Per game: ``terminated=True``
    on the game's LAST real human action (closes the episode); padded steps past
    the game's end: ``terminated=True``, ``reward=0``, ``values=0``,
    ``old_log_prob=0``, ``action_mask=zeros``, ``action_features=zeros``,
    ``actions=0`` (dummy). ``bootstrap_values = V(next_obs)`` of each game's
    FINAL transition, shape ``(num_games,)`` (rust_ppo.py:485-490; computed from
    the current policy forward on each game's last real next_obs). Intermediate
    next-values come from the per-obs values array itself (``values[step+1]``
    within a game).

The batch flows through ``prepare_rust_ppo_batch`` (gamma=0.99,
gae_lambda=0.95, bootstrap_values=per-game tail) -> ``RustPPOBatch`` ->
``train_dense_rust_ppo_minibatch`` WITHOUT a shape error.

No edit to frozen-classic / A1-A5 / B1-B8 / rust_ppo.py / rust_collector.py /
core / obs_v5.py / contracts.py / v5_policy.py / offline_dataset_loader.py.
Reuses A1's ``resolve_v5_tcode`` (CONSUMED, READ-ONLY import) and the C0
append_only loader ``action_features`` field.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ai.train_v2.classic_actions_v1 import (
    ACTION_FEATURE_DIM,
    MAX_CANDIDATE_ACTIONS,
    build_action_mask,
)
from ai.train_v2.offline_dataset_loader import (
    OfflineTransition,
    compute_offline_reward,
    iter_offline_transitions,
    reconstruct_gamestate,
    reward_view_from_snapshot,
)
from ai.train_v2.obs_v5 import encode_observation_v5
from train_v3.bc_dataset import resolve_v5_tcode
from train_v3.contracts import AssistModeV5, InfoModeV5, OBS_V5_DIM
from train_v3.rust_collector import RustTransitionBatch

logger = logging.getLogger(__name__)

# Mirrors bc_dataset.py:95 _PLACEMENT_MODE -- the engine's sole legal warrior
# placement is position=len(player.board) (core/engine.py:1260), so the
# append_only mask candidate set corresponds EXACTLY to the engine's emitted
# legal actions (source-vs-source parity for resolve_v5_tcode).
_PLACEMENT_MODE = "append_only"

# mana_draw action_type -- ManaDrawAction is OUTSIDE the 601 candidate space
# (bc_dataset.py:99; mana_draw_head_v5.py:4-6). BC/C3 targets the parallel
# mana_draw binary head for these rows, not a 601 slot.
_MANA_DRAW_ACTION_TYPE = "mana_draw"

# Terminal synthetic action_types -- mirror bc_dataset.py:86
# (surrender/draw/stalemate). action_native is None for these rows; no 601
# target.
_TERMINAL_ACTION_TYPES = frozenset({"surrender", "draw", "stalemate"})


def _omniscient_info_mode() -> InfoModeV5:
    """One explicit A/B/C/production private-information contract."""
    return InfoModeV5(
        enemy_hand_known=True,
        enemy_deck_known=True,
        enemy_deck_order_known=True,
    )


# Policy signature: policy_fn(obs_batch, action_features_batch) ->
# (logits, values, mana_draw_logit), mirroring the dense evaluator
# model(obs, action_features) at rust_ppo.py:755-759 (V5 returns a 3-tuple).
PolicyFn = Callable[
    [np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]


@dataclass(frozen=True)
class _ResolvedRow:
    """One resolved human replay row (after D-C7 filter + 601-tcode resolution)."""

    obs: np.ndarray
    next_obs: np.ndarray
    action_features: np.ndarray
    mask: np.ndarray
    target_tcode: int | None
    is_mana_draw: bool
    mana_draw_legal: bool
    reward: float
    terminal: bool
    # action id used for the RustTransitionBatch.actions array: target_tcode for
    # normal rows; the first legal id (dummy) for mana_draw/terminal rows so
    # prepare_rust_ppo_batch's _selected_local_indices finds the action in the
    # legal ids tape (old_log_prob=0 excludes these rows from the PPO ratio via
    # the is_mana_draw flag in C3).
    action_id: int
    battle_id: Any


@dataclass(frozen=True)
class OfflineReplayBatch:
    """Result of ``build_offline_replay_batch`` -- a RustTransitionBatch-shaped
    dense batch + metadata for the C3 AWAC/CRR replay + the C4 loop driver.

    Fields:
        batch: the ``RustTransitionBatch`` (DENSE action_mask + action_features)
            ready for ``prepare_rust_ppo_batch``.
        bootstrap_values: per-game V(next_obs) of each game's FINAL real
            transition, shape ``(num_games,)`` (rust_ppo.py:485-490).
        is_mana_draw: ``(steps, env_count)`` bool -- True on mana_draw rows (C3
            targets the mana_draw BCE head; these rows are excluded from the PPO
            ratio surrogate).
        mana_draw_legal: ``(steps, env_count)`` bool -- the loader's
            mana_draw_legal_mask per row (parity with the parallel binary head's
            legal predicate; C3 BCE target mask).
        target_tcodes: ``(steps, env_count)`` int32 -- the resolved V5 601-tcode
            for normal rows; -1 for mana_draw / terminal / padded rows.
        num_games: number of games (envs) in the batch.
        num_rows: total real human rows across games (before padding).
        mana_draw_row_count: number of real mana_draw rows.
        skipped_rows: rows skipped (unresolvable tcode or missing snapshot).
    """

    batch: RustTransitionBatch
    bootstrap_values: np.ndarray
    is_mana_draw: np.ndarray
    mana_draw_legal: np.ndarray
    target_tcodes: np.ndarray
    num_games: int
    num_rows: int
    mana_draw_row_count: int
    skipped_rows: int


def _normalize_group_dirs(
    group_dirs: "Path | str | Sequence[Path | str] | None",
) -> list[Path]:
    if group_dirs is None:
        return []
    if isinstance(group_dirs, (str, Path)):
        return [Path(group_dirs)]
    return [Path(g) for g in group_dirs]


def _resolve_row(
    t: OfflineTransition, *, strict: bool, accepted_sources: frozenset[str]
) -> _ResolvedRow | None:
    """Apply the D-C7 human filter + 601-tcode resolution to one
    ``OfflineTransition``. Returns ``None`` if the row is filtered (non-human)
    or skipped (missing snapshot / unresolvable tcode)."""
    # D-C7 remains human-only by default.  Other decision sources are an
    # explicit opt-in and must be quality-gated before this bridge is called.
    if t.meta.get("decision_source") not in accepted_sources:
        return None
    # A rejected UI/MCP attempt is present in the trace for auditability, but
    # it did not change the environment and must never become a policy target.
    # Structural v5 validation intentionally permits such rows; the replay
    # bridge is the authoritative training-ingestion filter.
    if t.meta.get("accepted") is not True:
        return None
    # The loader carries the raw pre_state snapshot (additive Block-A field);
    # reconstruct the GameState for the append_only mask + tcode resolution.
    if t.pre_state_snapshot is None:
        logger.warning(
            "offline_replay_bridge: row seq=%s battle=%s skipped (no pre_state_snapshot)",
            t.meta.get("seq"), t.meta.get("battle_id"),
        )
        return None
    pre_state = reconstruct_gamestate(t.pre_state_snapshot)
    actor = t.meta.get("actor_user_id")
    action_type = t.meta.get("action_type")
    is_mana_draw = action_type == _MANA_DRAW_ACTION_TYPE

    # Build the append_only legal mask ONCE (engine-faithful, C0).
    mask = build_action_mask(
        pre_state, actor, verify_mask=False, placement_mode=_PLACEMENT_MODE,
    )

    if is_mana_draw:
        target_tcode: int | None = None
    elif action_type in _TERMINAL_ACTION_TYPES:
        target_tcode = None
    else:
        # Normal OR natural-lethal: resolve the V5 601-tcode against the
        # ENGINE-sourced action_native (t.action_native -- NOT decode_action;
        # bc_dataset.py:34-37 self-referential trap). strict=False skips
        # unresolvable rows (A1 production-robust pattern).
        target_tcode = resolve_v5_tcode(
            pre_state, actor, t.action_native, mask=mask, strict=strict,
        )
        if target_tcode is None:
            return None  # skipped (resolve_v5_tcode already logged)

    # action_id for the RustTransitionBatch.actions array.
    legal_ids = np.flatnonzero(mask == 1.0)
    if target_tcode is not None:
        action_id = int(target_tcode)
    elif legal_ids.size > 0:
        # mana_draw / terminal row: dummy legal action so prepare's
        # _selected_local_indices finds the action in the legal ids tape.
        action_id = int(legal_ids[0])
    else:
        # Degenerate: no legal 601 action (shouldn't happen for a real actor's
        # turn -- end_turn is always offered). Use 0 defensively; the legal ids
        # tape for this row will carry a [0] dummy below.
        action_id = 0

    return _ResolvedRow(
        obs=t.obs,
        next_obs=t.next_obs,
        action_features=t.action_features,  # C0 append_only loader field
        mask=mask,
        target_tcode=target_tcode,
        is_mana_draw=is_mana_draw,
        mana_draw_legal=bool(t.mana_draw_legal),
        reward=float(t.reward),
        terminal=bool(t.terminal),
        action_id=action_id,
        battle_id=t.meta.get("trajectory_id", t.meta.get("battle_id")),
    )


def _aggregate_actor_macro_transitions(
    raw: list[OfflineTransition],
    *,
    accepted_sources: frozenset[str],
    info_mode: InfoModeV5,
    assist_mode: AssistModeV5,
) -> list[OfflineTransition]:
    """Collapse environment steps into actor-decision macro transitions.

    A Phase-C row represents the consequence of one accepted human/LLM
    decision through all intervening opponent actions, up to that same actor's
    next decision (or the terminal state).  This prevents human-only filtering
    from discarding opponent damage and losses that occur on the bot's turn.
    Separate actors in the same battle become separate trajectories.
    """
    by_battle: dict[Any, list[OfflineTransition]] = {}
    for t in raw:
        by_battle.setdefault(t.meta.get("battle_id"), []).append(t)

    aggregated: list[OfflineTransition] = []
    for battle_id, rows in by_battle.items():
        accepted_positions = [
            i for i, t in enumerate(rows)
            if (
                t.meta.get("decision_source") in accepted_sources
                and t.meta.get("accepted") is True
            )
        ]
        for pos in accepted_positions:
            current = rows[pos]
            actor_id = current.meta.get("actor_user_id")
            actor_player = int(current.meta.get("actor_player") or 1)
            next_pos = next(
                (
                    i for i in accepted_positions
                    if i > pos and rows[i].meta.get("actor_user_id") == actor_id
                ),
                None,
            )
            endpoint_pos = (next_pos - 1) if next_pos is not None else len(rows) - 1
            endpoint = rows[endpoint_pos]
            terminal = any(t.terminal for t in rows[pos : endpoint_pos + 1])

            if next_pos is not None and not terminal:
                next_obs = rows[next_pos].obs
            elif endpoint.post_state_snapshot is not None:
                final_state = reconstruct_gamestate(endpoint.post_state_snapshot)
                next_obs = encode_observation_v5(
                    final_state,
                    int(actor_id),
                    info_mode=info_mode,
                    assist_mode=assist_mode,
                    history_events=endpoint.post_state_snapshot.get("v5_history_events") or [],
                )
            else:
                logger.warning(
                    "offline_replay_bridge: battle=%s actor=%s lacks final post snapshot; "
                    "falling back to immediate next_obs",
                    battle_id,
                    actor_id,
                )
                next_obs = current.next_obs

            reward = float(current.reward)
            if current.pre_state_snapshot is not None and endpoint.post_state_snapshot is not None:
                status = str(endpoint.meta.get("status") or "ongoing")
                pre_view = reward_view_from_snapshot(current.pre_state_snapshot, actor_player)
                post_view = reward_view_from_snapshot(endpoint.post_state_snapshot, actor_player)
                reward = compute_offline_reward(
                    int(actor_id),
                    pre_view,
                    post_view,
                    current.meta.get("accepted"),
                    status,
                    is_mana_draw=current.meta.get("action_type") == _MANA_DRAW_ACTION_TYPE,
                )

            aggregated.append(
                replace(
                    current,
                    reward=reward,
                    next_obs=next_obs,
                    terminal=terminal,
                    meta={
                        **current.meta,
                        "trajectory_id": (battle_id, actor_id),
                        "macro_endpoint_seq": endpoint.meta.get("seq"),
                    },
                )
            )
    return aggregated


def _materialize_transitions(
    *,
    group_dirs: "Path | str | Sequence[Path | str] | None",
    transitions: Iterable[OfflineTransition] | None,
    info_mode: InfoModeV5 | None,
    assist_mode: AssistModeV5 | None,
    max_battles: int | None,
) -> list[OfflineTransition]:
    """Materialize the raw ``OfflineTransition`` stream from either group_dirs
    (calling ``iter_offline_transitions`` with the omniscient info_mode, D11) or
    a pre-materialized iterable."""
    if transitions is not None:
        return list(transitions)
    dirs = _normalize_group_dirs(group_dirs)
    if not dirs:
        raise ValueError(
            "build_offline_replay_batch: provide either group_dirs or transitions"
        )
    # D11: omniscient InfoModeV5 so obs/next_obs match the omniscient deploy
    # encoder. assist_mode defaults to AssistModeV5() (parity with the loader).
    info = info_mode or _omniscient_info_mode()
    assist = assist_mode or AssistModeV5()
    out: list[OfflineTransition] = []
    for g in dirs:
        out.extend(
            iter_offline_transitions(
                g, info_mode=info, assist_mode=assist, max_battles=max_battles,
            )
        )
    return out


def _masked_softmax_probs(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mirror the dense evaluator (rust_ppo.py:760-761):
    ``masked = where(mask, logits, -1e9); probs = softmax(masked, axis=-1)``."""
    masked = np.where(mask == 1.0, logits, np.float32(-1.0e9))
    m = masked.max(axis=-1, keepdims=True)
    ex = np.exp(masked - m)
    return ex / ex.sum(axis=-1, keepdims=True)


def build_offline_replay_batch(
    policy_fn: PolicyFn,
    *,
    group_dirs: "Path | str | Sequence[Path | str] | None" = None,
    transitions: Iterable[OfflineTransition] | None = None,
    info_mode: InfoModeV5 | None = None,
    assist_mode: AssistModeV5 | None = None,
    max_battles: int | None = None,
    strict: bool = False,
    accepted_decision_sources: Sequence[str] = ("human",),
) -> OfflineReplayBatch:
    """Build a ``RustTransitionBatch``-shaped DENSE offline-replay batch from
    fresh HUMAN v5_trace rows for the C3 AWAC/CRR offline-PPO replay.

    Args:
        policy_fn: the CURRENT V5 policy at bridge time --
            ``policy_fn(obs_batch, action_features_batch) -> (logits, values,
            mana_draw_logit)`` (mirrors the dense evaluator model(obs,
            action_features), rust_ppo.py:755-759). For D-C10 old_log_prob +
            value computation. For synthetic tests this is a FAKE callable
            returning canned numpy arrays (no real MLX/ONNX).
        group_dirs: a single group_dir path or a list of group_dir paths (v5_trace
            group directories from C1 collection). The bridge calls
            ``iter_offline_transitions`` on each with the omniscient InfoModeV5
            (D11).
        transitions: a pre-materialized iterable of ``OfflineTransition`` (used
            when the caller already loaded rows with the desired info_mode).
            Mutually exclusive with ``group_dirs``; at least one must be given.
        info_mode: override the omniscient InfoModeV5 default (D11). Rarely
            needed -- only for tests that spy on the loader call.
        assist_mode: forwarded to ``iter_offline_transitions`` (default
            ``AssistModeV5()``).
        max_battles: forwarded to ``iter_offline_transitions``.
        strict: ``False`` (default) skips a row whose ``action_native`` fails to
            resolve (logged warning, A1 production-robust pattern); ``True``
            propagates ``TcodeResolutionError``.
        accepted_decision_sources: decision sources eligible for replay.
            Defaults to the frozen Phase-C contract ``("human",)``.  Passing
            ``("llm",)`` is an explicit semi-synthetic opt-in and is only safe
            after the campaign quality gate has passed.

    Returns:
        an ``OfflineReplayBatch`` carrying the ``RustTransitionBatch`` +
        per-game ``bootstrap_values`` + ``is_mana_draw`` / ``mana_draw_legal`` /
        ``target_tcodes`` parallel arrays + metadata (num_games, num_rows,
        mana_draw_row_count, skipped_rows).
    """
    raw = _materialize_transitions(
        group_dirs=group_dirs,
        transitions=transitions,
        info_mode=info_mode,
        assist_mode=assist_mode,
        max_battles=max_battles,
    )
    accepted_sources = frozenset(str(s) for s in accepted_decision_sources)
    if not accepted_sources:
        raise ValueError("accepted_decision_sources must not be empty")
    effective_info = info_mode or _omniscient_info_mode()
    effective_assist = assist_mode or AssistModeV5()
    raw = _aggregate_actor_macro_transitions(
        raw,
        accepted_sources=accepted_sources,
        info_mode=effective_info,
        assist_mode=effective_assist,
    )

    # Step 1+2: D-C7 human filter + 601-tcode resolution. Group by battle_id
    # (preserving first-appearance order) -- each battle is one game (one env).
    games: list[list[_ResolvedRow]] = []
    game_index: dict[Any, int] = {}
    skipped = 0
    for t in raw:
        row = _resolve_row(t, strict=strict, accepted_sources=accepted_sources)
        if row is None:
            # Distinguish skipped (unresolvable / missing snapshot) from filtered
            # (wrong source or rejected attempt). Only an accepted eligible row
            # that cannot be resolved counts as skipped.
            if (
                t.meta.get("decision_source") in accepted_sources
                and t.meta.get("accepted") is True
            ):
                skipped += 1
            continue
        bid = row.battle_id
        if bid not in game_index:
            game_index[bid] = len(games)
            games.append([])
        games[game_index[bid]].append(row)

    num_games = len(games)
    num_rows = sum(len(g) for g in games)
    mana_draw_row_count = sum(
        1 for g in games for r in g if r.is_mana_draw
    )

    # Empty-collection no-crash (D-C7 / test 9): 0 human rows -> empty batch.
    if num_games == 0:
        empty = _build_empty_batch()
        return OfflineReplayBatch(
            batch=empty,
            bootstrap_values=np.zeros((0,), dtype=np.float32),
            is_mana_draw=np.zeros((0, 0), dtype=np.bool_),
            mana_draw_legal=np.zeros((0, 0), dtype=np.bool_),
            target_tcodes=np.full((0, 0), -1, dtype=np.int32),
            num_games=0,
            num_rows=0,
            mana_draw_row_count=0,
            skipped_rows=skipped,
        )

    steps = max(len(g) for g in games)

    # Step 3 (D-C10): run the current policy forward on ALL real obs +
    # action_features (concatenated in game order) -> (logits, values,
    # mana_draw_logit). Compute old_log_prob + value per row.
    obs_all = np.stack([r.obs for g in games for r in g], axis=0).astype(
        np.float32, copy=False
    )
    feats_all = np.stack(
        [r.action_features for g in games for r in g], axis=0
    ).astype(np.float32, copy=False)
    logits_all, values_all, mdl_all = policy_fn(obs_all, feats_all)
    logits_all = np.asarray(logits_all, dtype=np.float32)
    values_all = np.asarray(values_all, dtype=np.float32).reshape(-1)
    draw_logits_all = np.asarray(mdl_all, dtype=np.float32).reshape(-1)

    # Masked softmax over the 601 candidates (mirror the dense evaluator,
    # rust_ppo.py:760-761). The mask is the per-row append_only legal mask
    # (NOT action_features); stack it in the same game-order as obs_all.
    masks_all = _masks_from_rows(games)
    probs_all = _masked_softmax_probs(logits_all, masks_all)

    # Exact factorized online contract:
    #   draw: log P(draw)
    #   card: log(1-P(draw)) + log P(card | no draw)
    flat_rows = [r for g in games for r in g]
    old_log_probs_flat = np.zeros(len(flat_rows), dtype=np.float32)
    for i, r in enumerate(flat_rows):
        draw_p = 0.0
        if r.mana_draw_legal:
            draw_p = float(1.0 / (1.0 + np.exp(-np.clip(draw_logits_all[i], -60.0, 60.0))))
        if r.is_mana_draw:
            old_log_probs_flat[i] = float(np.log(draw_p + 1.0e-10))
        elif r.target_tcode is not None:
            old_log_probs_flat[i] = float(
                np.log(1.0 - draw_p + 1.0e-10)
                + np.log(probs_all[i, r.target_tcode] + 1.0e-10)
            )

    # bootstrap_values (Step 4): V(next_obs) of each game's FINAL real
    # transition, shape (num_games,). Run the current policy on each game's last
    # real next_obs + action_features.
    next_obs_last = np.stack(
        [g[-1].next_obs for g in games], axis=0
    ).astype(np.float32, copy=False)
    feats_last = np.stack(
        [g[-1].action_features for g in games], axis=0
    ).astype(np.float32, copy=False)
    _bl, bootstrap_values, _bmdl = policy_fn(next_obs_last, feats_last)
    bootstrap_values = np.asarray(bootstrap_values, dtype=np.float32).reshape(-1)
    if bootstrap_values.shape != (num_games,):
        raise ValueError(
            f"policy_fn must return values with shape ({num_games},) for the "
            f"bootstrap call, got {bootstrap_values.shape}"
        )

    # Step 4+5: assemble the (steps, env_count=num_games) padded dense batch.
    batch = _assemble_batch(
        games=games,
        steps=steps,
        env_count=num_games,
        values_all=values_all,
        old_log_probs_flat=old_log_probs_flat,
        bootstrap_values=bootstrap_values,
    )

    # Parallel metadata arrays for C3 (is_mana_draw + mana_draw_legal +
    # target_tcodes).
    is_mana_draw_arr = np.zeros((steps, num_games), dtype=np.bool_)
    mana_draw_legal_arr = np.zeros((steps, num_games), dtype=np.bool_)
    target_tcodes_arr = np.full((steps, num_games), -1, dtype=np.int32)
    for env, g in enumerate(games):
        for step, r in enumerate(g):
            is_mana_draw_arr[step, env] = r.is_mana_draw
            mana_draw_legal_arr[step, env] = r.mana_draw_legal
            target_tcodes_arr[step, env] = (
                r.target_tcode if r.target_tcode is not None else -1
            )

    return OfflineReplayBatch(
        batch=batch,
        bootstrap_values=bootstrap_values,
        is_mana_draw=is_mana_draw_arr,
        mana_draw_legal=mana_draw_legal_arr,
        target_tcodes=target_tcodes_arr,
        num_games=num_games,
        num_rows=num_rows,
        mana_draw_row_count=mana_draw_row_count,
        skipped_rows=skipped,
    )


def _masks_from_rows(games: list[list[_ResolvedRow]]) -> np.ndarray:
    """Stack the per-row append_only masks for the policy-forward rows (in game
    order) -- shape (N, 601). Used for the masked softmax (D-C10)."""
    return np.stack([r.mask for g in games for r in g], axis=0).astype(
        np.float32, copy=False
    )


def _assemble_batch(
    *,
    games: list[list[_ResolvedRow]],
    steps: int,
    env_count: int,
    values_all: np.ndarray,
    old_log_probs_flat: np.ndarray,
    bootstrap_values: np.ndarray,
) -> RustTransitionBatch:
    """Assemble the padded ``(steps, env_count)`` RustTransitionBatch + the
    flat legal-action tape (counts / offsets / ids / features)."""
    observations = np.zeros((steps, env_count, OBS_V5_DIM), dtype=np.float32)
    next_observations = np.zeros((steps, env_count, OBS_V5_DIM), dtype=np.float32)
    action_mask = np.zeros((steps, env_count, MAX_CANDIDATE_ACTIONS), dtype=np.float32)
    action_features = np.zeros(
        (steps, env_count, MAX_CANDIDATE_ACTIONS, ACTION_FEATURE_DIM), dtype=np.float32
    )
    actions = np.zeros((steps, env_count), dtype=np.uintp)
    rewards = np.zeros((steps, env_count), dtype=np.float32)
    terminated = np.zeros((steps, env_count), dtype=np.bool_)
    values = np.zeros((steps, env_count), dtype=np.float32)
    log_probs = np.zeros((steps, env_count), dtype=np.float32)

    # Per (step, env) legal ids + features for the flat tape (step-major order,
    # matching RustPPOBatch.flatten / _selected_local_indices).
    legal_ids_per: list[list[np.ndarray]] = [
        [None] * env_count for _ in range(steps)
    ]
    legal_feats_per: list[list[np.ndarray]] = [
        [None] * env_count for _ in range(steps)
    ]

    flat_idx = 0
    for env, g in enumerate(games):
        for step, r in enumerate(g):
            observations[step, env] = r.obs
            next_observations[step, env] = r.next_obs
            action_mask[step, env] = r.mask
            action_features[step, env] = r.action_features
            actions[step, env] = r.action_id
            rewards[step, env] = r.reward
            values[step, env] = float(values_all[flat_idx])
            log_probs[step, env] = float(old_log_probs_flat[flat_idx])
            # terminated: True on the game's LAST real human action (episode
            # boundary for GAE); use t.terminal for intermediate real steps.
            if step == len(g) - 1:
                terminated[step, env] = True
            else:
                terminated[step, env] = r.terminal
            legal_ids = np.flatnonzero(r.mask == 1.0).astype(np.uintp)
            if legal_ids.size == 0:
                # Degenerate real row with no legal 601 action -- carry a [0]
                # dummy so prepare's _selected_local_indices (count>=1, action in
                # ids) does not raise. action_id is 0 (set above defensively).
                legal_ids = np.array([0], dtype=np.uintp)
                legal_feats_per[step][env] = np.zeros(
                    (1, ACTION_FEATURE_DIM), dtype=np.float32
                )
            else:
                legal_feats_per[step][env] = r.action_features[legal_ids]
            legal_ids_per[step][env] = legal_ids
            flat_idx += 1
        # Padded steps past the game's end: terminated=True, reward=0, values=0,
        # action_mask=zeros, action_features=zeros, old_log_prob=0, action=0
        # (dummy). The legal ids tape carries a DECOUPLED [0] dummy (count=1,
        # action=0) so prepare_rust_ppo_batch's _selected_local_indices -- which
        # requires count>=1 and the action present in legal ids -- does not
        # raise; the dense evaluator ignores legal_action_ids and reads
        # action_mask (zeros -> uniform softmax -> ratio~0, advantage=0 -> zero
        # policy-loss contribution). This decoupling is the minimal fix per the
        # C2 spec ("FIX the batch shape if prepare raises").
        for step in range(len(g), steps):
            terminated[step, env] = True
            actions[step, env] = 0
            legal_ids_per[step][env] = np.array([0], dtype=np.uintp)
            legal_feats_per[step][env] = np.zeros(
                (1, ACTION_FEATURE_DIM), dtype=np.float32
            )

    # Build the flat legal-action tape in step-major (C) order --
    # index = step * env_count + env -- matching RustPPOBatch.flatten().
    flat_counts = np.zeros((steps * env_count,), dtype=np.uintp)
    flat_offsets = np.zeros((steps * env_count,), dtype=np.uintp)
    ids_chunks: list[np.ndarray] = []
    feats_chunks: list[np.ndarray] = []
    running = 0
    for step in range(steps):
        for env in range(env_count):
            idx = step * env_count + env
            ids = legal_ids_per[step][env]
            cnt = int(ids.shape[0])
            flat_counts[idx] = cnt
            flat_offsets[idx] = running
            ids_chunks.append(ids)
            feats_chunks.append(legal_feats_per[step][env])
            running += cnt
    legal_action_ids = (
        np.concatenate(ids_chunks) if ids_chunks else np.zeros((0,), dtype=np.uintp)
    )
    legal_action_features = (
        np.concatenate(feats_chunks, axis=0)
        if feats_chunks
        else np.zeros((0, ACTION_FEATURE_DIM), dtype=np.float32)
    )
    legal_action_counts = flat_counts.reshape((steps, env_count))
    legal_action_offsets = flat_offsets.reshape((steps, env_count))

    return RustTransitionBatch(
        observations=observations,
        next_observations=next_observations,
        action_mask=action_mask,
        action_features=action_features,
        legal_action_counts=legal_action_counts,
        legal_action_offsets=legal_action_offsets,
        legal_action_ids=legal_action_ids,
        legal_action_features=legal_action_features,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=None,
        reset_flags=None,
        terminal_observations=None,
        terminal_observation_valid=None,
        episode_returns=None,
        episode_lengths=None,
        infos=None,
        values=values,
        log_probs=log_probs,
        selected_local_indices=None,
    )


def make_policy_fn_from_checkpoint(
    checkpoint_path: "str | Path | None",
    *,
    hidden_dim: int = 256,
    action_feature_dim: int = ACTION_FEATURE_DIM,
) -> PolicyFn:
    """Load a V5 MLX policy checkpoint and wrap it as a ``policy_fn`` returning
    NUMPY ``(logits, values, mana_draw_logit)`` (the bridge consumes numpy, not
    MLX arrays).

    A2 skip-gate pattern (bc_train.py:37-39, warm_start_v5.resolve_v4_max_npz_path
    :111): gate on the CHECKPOINT FILE's existence -- NOT on the ``mlx`` import.
    If ``checkpoint_path`` is ``None`` or the file is absent, raise
    ``FileNotFoundError`` so the caller (C3 loop driver) can skip the bridge run
    for this group (no crash, no partial batch). MLX itself is assumed present
    (do NOT gate on the mlx import; gate on the checkpoint file only).

    Lazy MLX / v5_policy / model_mlx imports keep the bridge module MLX-free at
    import time so the SYNTHETIC tests (fake policy_fn, no MLX) never touch MLX.
    """
    import os

    if checkpoint_path is None:
        raise FileNotFoundError("checkpoint_path is None (no V5 policy checkpoint)")
    resolved = Path(checkpoint_path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"V5 policy checkpoint not found: {resolved} (skip-gated, A2 pattern)"
        )

    import mlx.core as mx
    from ai.train_v2 import model_mlx
    from train_v3.v5_policy import create_v5_policy

    # Phase C resumes a learned V5 checkpoint.  Current post-B checkpoints use
    # a 256-wide state fuser, while older fixtures may still be 128-wide.
    # Instantiating the default width and relying on a permissive checkpoint
    # loader leaves a shape-incoherent model that only fails on its first
    # forward pass.  Infer both learned widths from the archive, matching the
    # authoritative C3 loader in awac_crr_replay._resolve_model.
    with np.load(resolved, allow_pickle=False) as archive:
        fuser_key = next(
            (
                key
                for key in archive.files
                if key.endswith("state_fuser.layers.0.weight")
            ),
            None,
        )
        inferred_hidden = (
            int(archive[fuser_key].shape[0])
            if fuser_key is not None
            else int(hidden_dim)
        )
        action_key = next(
            (
                key
                for key in archive.files
                if key.endswith("action_encoder.weight")
            ),
            None,
        )
        inferred_action_hidden = (
            int(archive[action_key].shape[0])
            if action_key is not None
            else 128
        )

    policy = create_v5_policy(
        policy_kind="v5_split_encoder",
        hidden_dim=inferred_hidden,
        action_hidden_dim=inferred_action_hidden,
        action_feature_dim=action_feature_dim,
    )
    model_mlx.load_checkpoint(str(resolved), policy)

    def _policy_fn(obs_batch: np.ndarray, action_features_batch: np.ndarray):
        obs_mx = mx.array(np.asarray(obs_batch, dtype=np.float32))
        feats_mx = mx.array(np.asarray(action_features_batch, dtype=np.float32))
        logits_mx, values_mx, mana_draw_logit_mx = policy(obs_mx, feats_mx)
        return (
            np.asarray(logits_mx, dtype=np.float32),
            np.asarray(values_mx, dtype=np.float32).reshape(-1),
            np.asarray(mana_draw_logit_mx, dtype=np.float32).reshape(-1),
        )

    return _policy_fn


def _build_empty_batch() -> RustTransitionBatch:
    """Empty (0 games) RustTransitionBatch -- no-crash on empty collection."""
    return RustTransitionBatch(
        observations=np.zeros((0, 0, OBS_V5_DIM), dtype=np.float32),
        next_observations=np.zeros((0, 0, OBS_V5_DIM), dtype=np.float32),
        action_mask=np.zeros((0, 0, MAX_CANDIDATE_ACTIONS), dtype=np.float32),
        action_features=np.zeros(
            (0, 0, MAX_CANDIDATE_ACTIONS, ACTION_FEATURE_DIM), dtype=np.float32
        ),
        legal_action_counts=np.zeros((0, 0), dtype=np.uintp),
        legal_action_offsets=np.zeros((0, 0), dtype=np.uintp),
        legal_action_ids=np.zeros((0,), dtype=np.uintp),
        legal_action_features=np.zeros((0, ACTION_FEATURE_DIM), dtype=np.float32),
        actions=np.zeros((0, 0), dtype=np.uintp),
        rewards=np.zeros((0, 0), dtype=np.float32),
        terminated=np.zeros((0, 0), dtype=np.bool_),
        truncated=None,
        reset_flags=None,
        terminal_observations=None,
        terminal_observation_valid=None,
        episode_returns=None,
        episode_lengths=None,
        infos=None,
        values=np.zeros((0, 0), dtype=np.float32),
        log_probs=np.zeros((0, 0), dtype=np.float32),
        selected_local_indices=None,
    )


__all__ = [
    "OfflineReplayBatch",
    "build_offline_replay_batch",
    "make_policy_fn_from_checkpoint",
]

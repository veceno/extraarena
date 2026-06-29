"""Policy-driven rollout collection for the training-only Rust VecEnv."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .rust_vec_env import RustVecEnv


PolicyFn = Callable[[np.ndarray, np.ndarray | None, Any], Any]


@dataclass(frozen=True)
class RustLegalActionFeatures:
    counts: np.ndarray
    offsets: np.ndarray
    ids: np.ndarray
    features: np.ndarray


@dataclass(frozen=True)
class RustTransitionBatch:
    observations: np.ndarray
    next_observations: np.ndarray | None
    action_mask: np.ndarray | None
    action_features: np.ndarray | None
    legal_action_counts: np.ndarray
    legal_action_offsets: np.ndarray
    legal_action_ids: np.ndarray
    legal_action_features: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray | None
    reset_flags: np.ndarray | None
    terminal_observations: np.ndarray | None
    terminal_observation_valid: np.ndarray | None
    episode_returns: np.ndarray | None
    episode_lengths: np.ndarray | None
    infos: list[list[dict[str, Any]]] | None
    values: np.ndarray | None = None
    log_probs: np.ndarray | None = None
    selected_local_indices: np.ndarray | None = None
    policy_seconds: float = 0.0
    env_step_seconds: float = 0.0
    policy_profile: dict[str, float] | None = None


def collect_rust_vec_rollout(
    env: RustVecEnv,
    policy: PolicyFn,
    *,
    steps: int,
    use_compact_action_features: bool = False,
    store_dense_action_features: bool = True,
    store_dense_action_mask: bool = True,
    store_terminal_observations: bool = True,
    store_next_observations: bool = True,
    store_infos: bool = True,
    store_truncated: bool = True,
    store_reset_flags: bool = True,
    store_episode_stats: bool = True,
) -> RustTransitionBatch:
    """Collect a fixed-length vectorized rollout from `RustVecEnv`.

    `policy` receives `(observations, action_mask, action_features)` and may
    return either action ids directly, `(actions, values, log_probs)`, or a dict
    containing `actions` plus optional `values`, `log_probs`, and
    `selected_local_indices`.
    """
    steps = int(steps)
    if steps <= 0:
        raise ValueError("steps must be positive")

    reset = env.reset(copy=False, include_infos=store_infos)
    env_count = int(env.env_count)
    obs_shape = reset.observations.shape[1:]
    if store_dense_action_mask and reset.action_mask is None:
        raise ValueError(
            "dense action_mask is unavailable; disable store_dense_action_mask "
            "or create the RustVecEnv with action_mask_mode='dense'"
        )
    mask_shape = reset.action_mask.shape[1:] if reset.action_mask is not None else ()
    if store_dense_action_features and reset.action_features is None:
        raise ValueError(
            "dense action_features are unavailable; disable store_dense_action_features "
            "or create the RustVecEnv with action_features_mode='dense_and_legal'"
        )
    feature_shape = reset.action_features.shape[1:] if reset.action_features is not None else ()

    observations = np.empty((steps, env_count, *obs_shape), dtype=reset.observations.dtype)
    next_observations = np.empty_like(observations) if store_next_observations else None
    action_mask = (
        np.empty((steps, env_count, *mask_shape), dtype=reset.action_mask.dtype)
        if store_dense_action_mask and reset.action_mask is not None
        else None
    )
    action_features = (
        np.empty((steps, env_count, *feature_shape), dtype=reset.action_features.dtype)
        if store_dense_action_features and reset.action_features is not None
        else None
    )
    legal_action_counts = np.empty((steps, env_count), dtype=reset.legal_action_counts.dtype)
    legal_action_offsets = np.empty((steps, env_count), dtype=reset.legal_action_offsets.dtype)
    legal_tape = _LegalActionTapeBuilder(
        ids_dtype=reset.legal_action_ids.dtype,
        features_dtype=reset.legal_action_features.dtype,
        feature_shape=reset.legal_action_features.shape[1:],
        initial_capacity=int(reset.legal_action_ids.shape[0]) * steps,
    )
    actions = np.empty((steps, env_count), dtype=np.uintp)
    rewards = np.empty((steps, env_count), dtype=np.float32)
    terminated = np.empty((steps, env_count), dtype=np.bool_)
    truncated = np.empty((steps, env_count), dtype=np.bool_) if store_truncated else None
    reset_flags = np.empty((steps, env_count), dtype=np.bool_) if store_reset_flags else None
    terminal_observations = np.empty_like(observations) if store_terminal_observations else None
    terminal_observation_valid = np.empty((steps, env_count), dtype=np.bool_) if store_episode_stats else None
    episode_returns = np.empty((steps, env_count), dtype=np.float32) if store_episode_stats else None
    episode_lengths = np.empty((steps, env_count), dtype=np.int32) if store_episode_stats else None
    infos: list[list[dict[str, Any]]] | None = [] if store_infos else None

    current_obs = reset.observations
    current_mask = reset.action_mask
    current_features = reset.action_features
    current_legal_counts = reset.legal_action_counts
    current_legal_offsets = reset.legal_action_offsets
    current_legal_ids = reset.legal_action_ids
    current_legal_features = reset.legal_action_features
    values: np.ndarray | None = None
    log_probs: np.ndarray | None = None
    selected_local_indices: np.ndarray | None = None
    policy_seconds = 0.0
    env_step_seconds = 0.0
    policy_profile: dict[str, float] = {}

    for step_idx in range(steps):
        observations[step_idx] = current_obs
        if action_mask is not None:
            if current_mask is None:
                raise ValueError("dense action_mask disappeared during rollout")
            action_mask[step_idx] = current_mask
        if action_features is not None:
            if current_features is None:
                raise ValueError("dense action_features disappeared during rollout")
            action_features[step_idx] = current_features

        legal_action_counts[step_idx] = current_legal_counts
        legal_action_offsets[step_idx] = current_legal_offsets + legal_tape.size
        legal_tape.append(current_legal_ids, current_legal_features)

        if use_compact_action_features:
            policy_features = RustLegalActionFeatures(
                counts=current_legal_counts,
                offsets=current_legal_offsets,
                ids=current_legal_ids,
                features=current_legal_features,
            )
        else:
            if current_features is None:
                raise ValueError(
                    "dense action_features are unavailable; pass use_compact_action_features=True"
                )
            if current_mask is None:
                raise ValueError("dense action_mask is unavailable for dense policy path")
            policy_features = current_features
        policy_start = time.perf_counter()
        policy_out = policy(current_obs, current_mask, policy_features)
        policy_seconds += time.perf_counter() - policy_start
        (
            action_ids,
            step_values,
            step_log_probs,
            step_selected_local,
            step_policy_profile,
        ) = _parse_policy_output(policy_out)
        action_ids = _validate_action_ids(action_ids, env_count)
        actions[step_idx] = action_ids
        if step_policy_profile is not None:
            for key, value in step_policy_profile.items():
                policy_profile[str(key)] = policy_profile.get(str(key), 0.0) + float(value)

        if step_values is not None:
            if values is None:
                values = np.empty((steps, env_count), dtype=np.float32)
                if step_idx > 0:
                    values.fill(np.nan)
            values[step_idx] = _validate_float_vector(step_values, env_count, "values")
        if step_log_probs is not None:
            if log_probs is None:
                log_probs = np.empty((steps, env_count), dtype=np.float32)
                if step_idx > 0:
                    log_probs.fill(np.nan)
            log_probs[step_idx] = _validate_float_vector(step_log_probs, env_count, "log_probs")
        if step_selected_local is not None:
            if selected_local_indices is None:
                selected_local_indices = np.empty((steps, env_count), dtype=np.int32)
                if step_idx > 0:
                    selected_local_indices.fill(-1)
            selected_local_indices[step_idx] = _validate_selected_local_indices(
                step_selected_local,
                current_legal_counts,
                env_count,
            )

        env_step_start = time.perf_counter()
        step = env.step(
            action_ids,
            copy=False,
            include_infos=store_infos,
            include_truncated=store_truncated,
        )
        env_step_seconds += time.perf_counter() - env_step_start
        if next_observations is not None:
            next_observations[step_idx] = step.observations
        rewards[step_idx] = step.rewards
        terminated[step_idx] = step.terminated
        if truncated is not None:
            if step.truncated is None:
                raise ValueError("step truncated flags are unavailable while store_truncated=True")
            truncated[step_idx] = step.truncated
        if reset_flags is not None:
            if step.reset_flags is None:
                raise ValueError("step reset flags are unavailable while store_reset_flags=True")
            reset_flags[step_idx] = step.reset_flags
        if terminal_observations is not None:
            terminal_raw = step.raw["terminal_" + env.observation_key]
            if terminal_raw is None:
                raise ValueError("terminal observations are unavailable; disable store_terminal_observations")
            terminal_observations[step_idx] = terminal_raw
        if terminal_observation_valid is not None:
            if step.raw["terminal_observation_valid"] is None:
                raise ValueError("terminal observation flags are unavailable while store_episode_stats=True")
            terminal_observation_valid[step_idx] = step.raw["terminal_observation_valid"]
        if episode_returns is not None:
            if step.raw["episode_returns"] is None:
                raise ValueError("episode returns are unavailable while store_episode_stats=True")
            episode_returns[step_idx] = step.raw["episode_returns"]
        if episode_lengths is not None:
            if step.raw["episode_lengths"] is None:
                raise ValueError("episode lengths are unavailable while store_episode_stats=True")
            episode_lengths[step_idx] = step.raw["episode_lengths"]
        if infos is not None:
            if step.infos is None:
                raise ValueError("step infos are unavailable; disable store_infos")
            infos.append([_copy_info(info) for info in step.infos])

        current_obs = step.observations
        current_mask = step.action_mask
        current_features = step.action_features
        current_legal_counts = step.legal_action_counts
        current_legal_offsets = step.legal_action_offsets
        current_legal_ids = step.legal_action_ids
        current_legal_features = step.legal_action_features

    legal_action_ids, legal_action_features = legal_tape.finish()

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
        truncated=truncated,
        reset_flags=reset_flags,
        terminal_observations=terminal_observations,
        terminal_observation_valid=terminal_observation_valid,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        infos=infos,
        values=values,
        log_probs=log_probs,
        selected_local_indices=selected_local_indices,
        policy_seconds=policy_seconds,
        env_step_seconds=env_step_seconds,
        policy_profile=None if not policy_profile else policy_profile,
    )


def transition_batch_from_action_tape_rollout(
    rollout: dict[str, np.ndarray | None],
    action_ids: np.ndarray,
    *,
    observation_key: str = "observation_v5",
    values: np.ndarray | None = None,
    log_probs: np.ndarray | None = None,
    selected_local_indices: np.ndarray | None = None,
    store_truncated: bool = False,
    store_reset_flags: bool = True,
    store_episode_stats: bool = True,
) -> RustTransitionBatch:
    """Build a PPO-compatible transition batch from a pre-step action-tape rollout."""
    if observation_key not in {"observation_v1", "observation_v5"}:
        raise ValueError("observation_key must be observation_v1 or observation_v5")
    observations = rollout.get(observation_key)
    if observations is None:
        raise ValueError(f"{observation_key} is unavailable in action-tape rollout")
    rewards = np.asarray(rollout["rewards"], dtype=np.float32)
    actions = np.asarray(action_ids, dtype=np.uintp)
    if actions.ndim == 1 and actions.shape == (rewards.shape[0],):
        actions = np.broadcast_to(actions[:, None], rewards.shape)
    elif actions.shape != rewards.shape:
        raise ValueError(f"action_ids shape {actions.shape} must match rewards shape {rewards.shape}")
    if observations.shape[:2] != rewards.shape:
        raise ValueError(
            f"{observation_key} leading shape {observations.shape[:2]} must match rewards shape {rewards.shape}"
        )

    legal_action_counts = np.asarray(rollout["legal_action_counts"])
    if legal_action_counts.shape != rewards.shape:
        raise ValueError("legal_action_counts must match rewards shape")
    legal_action_offsets = rollout.get("legal_action_offsets")
    if legal_action_offsets is None:
        legal_action_offsets = _legal_offsets_from_counts(legal_action_counts)
    else:
        legal_action_offsets = np.asarray(legal_action_offsets, dtype=np.uintp)
        if legal_action_offsets.shape != rewards.shape:
            raise ValueError("legal_action_offsets must match rewards shape")

    terminal_observations = rollout.get("terminal_" + observation_key)
    if terminal_observations is not None and terminal_observations.shape[:2] != rewards.shape:
        raise ValueError(f"terminal_{observation_key} leading shape must match rewards shape")

    transition_values = None if values is None else _validate_rollout_stat(values, rewards.shape, "values")
    transition_log_probs = None if log_probs is None else _validate_rollout_stat(log_probs, rewards.shape, "log_probs")
    rollout_selected_local = (
        None
        if selected_local_indices is None
        else selected_local_indices
    )
    if rollout_selected_local is None:
        rollout_selected_local = rollout.get("selected_local_indices")
    transition_selected_local = (
        None
        if rollout_selected_local is None
        else _validate_rollout_selected_local_indices(rollout_selected_local, legal_action_counts, rewards.shape)
    )

    return RustTransitionBatch(
        observations=np.asarray(observations),
        next_observations=None,
        action_mask=rollout.get("action_mask"),
        action_features=rollout.get("action_features"),
        legal_action_counts=legal_action_counts,
        legal_action_offsets=legal_action_offsets,
        legal_action_ids=np.asarray(rollout["legal_action_ids"], dtype=np.uintp),
        legal_action_features=np.asarray(rollout["legal_action_features"]),
        actions=actions,
        rewards=rewards,
        terminated=np.asarray(rollout["terminated"], dtype=np.bool_),
        truncated=np.zeros_like(rewards, dtype=np.bool_) if store_truncated else None,
        reset_flags=None if not store_reset_flags else np.asarray(rollout["reset_flags"], dtype=np.bool_),
        terminal_observations=terminal_observations,
        terminal_observation_valid=(
            None
            if not store_episode_stats
            else np.asarray(rollout["terminal_observation_valid"], dtype=np.bool_)
        ),
        episode_returns=(
            None
            if not store_episode_stats
            else np.asarray(rollout["episode_returns"], dtype=np.float32)
        ),
        episode_lengths=(
            None
            if not store_episode_stats
            else np.asarray(rollout["episode_lengths"], dtype=np.int32)
        ),
        infos=None,
        values=transition_values,
        log_probs=transition_log_probs,
        selected_local_indices=transition_selected_local,
    )


def _parse_policy_output(policy_out: Any) -> tuple[Any, Any | None, Any | None, Any | None, dict[str, Any] | None]:
    if isinstance(policy_out, dict):
        if "actions" not in policy_out:
            raise ValueError("policy output dict must include actions")
        return (
            policy_out["actions"],
            policy_out.get("values"),
            policy_out.get("log_probs"),
            policy_out.get("selected_local_indices"),
            policy_out.get("policy_profile"),
        )
    if isinstance(policy_out, tuple):
        if len(policy_out) == 5:
            return policy_out[0], policy_out[1], policy_out[2], policy_out[3], policy_out[4]
        if len(policy_out) == 4:
            return policy_out[0], policy_out[1], policy_out[2], policy_out[3], None
        if len(policy_out) == 3:
            return policy_out[0], policy_out[1], policy_out[2], None, None
        if len(policy_out) == 1:
            return policy_out[0], None, None, None, None
        raise ValueError(
            "policy output tuple must be (actions, values, log_probs) "
            "or (actions, values, log_probs, selected_local_indices)"
        )
    return policy_out, None, None, None, None


def _validate_action_ids(action_ids: Any, env_count: int) -> np.ndarray:
    out = np.asarray(action_ids, dtype=np.uintp)
    if out.shape != (env_count,):
        raise ValueError(f"policy must return action ids with shape ({env_count},), got {out.shape}")
    return out


def _validate_float_vector(values: Any, env_count: int, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32)
    if out.shape != (env_count,):
        raise ValueError(f"policy must return {name} with shape ({env_count},), got {out.shape}")
    return out


def _validate_selected_local_indices(values: Any, counts: np.ndarray, env_count: int) -> np.ndarray:
    out = np.asarray(values, dtype=np.int32)
    if out.shape != (env_count,):
        raise ValueError(
            f"policy must return selected_local_indices with shape ({env_count},), got {out.shape}"
        )
    counts_arr = np.asarray(counts)
    if counts_arr.shape != (env_count,):
        raise ValueError(f"legal_action_counts must have shape ({env_count},), got {counts_arr.shape}")
    if np.any(out < 0) or np.any(out >= counts_arr):
        raise ValueError("policy selected_local_indices must be within each row's legal action count")
    return out


def _validate_rollout_stat(values: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32)
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
    return out


def _validate_rollout_selected_local_indices(
    values: Any,
    counts: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    out = np.asarray(values, dtype=np.int32)
    if out.shape != shape:
        raise ValueError(f"selected_local_indices must have shape {shape}, got {out.shape}")
    counts_arr = np.asarray(counts)
    if counts_arr.shape != shape:
        raise ValueError(f"legal_action_counts must have shape {shape}, got {counts_arr.shape}")
    if np.any(out < 0) or np.any(out >= counts_arr):
        raise ValueError("selected_local_indices must be within each row's legal action count")
    return out


class _LegalActionTapeBuilder:
    def __init__(
        self,
        *,
        ids_dtype: np.dtype,
        features_dtype: np.dtype,
        feature_shape: tuple[int, ...],
        initial_capacity: int,
    ):
        self._ids_dtype = np.dtype(ids_dtype)
        self._features_dtype = np.dtype(features_dtype)
        self._feature_shape = tuple(feature_shape)
        capacity = max(0, int(initial_capacity))
        self._ids = np.empty((capacity,), dtype=self._ids_dtype)
        self._features = np.empty((capacity, *self._feature_shape), dtype=self._features_dtype)
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def append(self, ids: Any, features: Any) -> None:
        ids_arr = np.asarray(ids, dtype=self._ids_dtype)
        features_arr = np.asarray(features, dtype=self._features_dtype)
        if ids_arr.ndim != 1:
            raise ValueError(f"legal_action_ids must be 1D, got shape {ids_arr.shape}")
        expected_shape = (ids_arr.shape[0], *self._feature_shape)
        if features_arr.shape != expected_shape:
            raise ValueError(
                f"legal_action_features must have shape {expected_shape}, got {features_arr.shape}"
            )
        end = self._size + int(ids_arr.shape[0])
        self._ensure_capacity(end)
        self._ids[self._size:end] = ids_arr
        self._features[self._size:end] = features_arr
        self._size = end

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        return self._ids[: self._size], self._features[: self._size]

    def _ensure_capacity(self, required: int) -> None:
        current = int(self._ids.shape[0])
        if required <= current:
            return
        new_capacity = max(required, current * 2 if current > 0 else 1)
        new_ids = np.empty((new_capacity,), dtype=self._ids_dtype)
        new_features = np.empty((new_capacity, *self._feature_shape), dtype=self._features_dtype)
        if self._size:
            new_ids[: self._size] = self._ids[: self._size]
            new_features[: self._size] = self._features[: self._size]
        self._ids = new_ids
        self._features = new_features


def _legal_offsets_from_counts(counts: np.ndarray) -> np.ndarray:
    flat_counts = np.asarray(counts, dtype=np.uintp).reshape(-1)
    flat_offsets = np.empty_like(flat_counts, dtype=np.uintp)
    if flat_counts.size:
        flat_offsets[0] = 0
        if flat_counts.size > 1:
            flat_offsets[1:] = np.cumsum(flat_counts[:-1], dtype=np.uintp)
    return flat_offsets.reshape(counts.shape)


def _copy_info(info: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        elif isinstance(value, dict):
            copied[key] = dict(value)
        else:
            copied[key] = value
    return copied


__all__ = [
    "RustLegalActionFeatures",
    "RustTransitionBatch",
    "collect_rust_vec_rollout",
    "transition_batch_from_action_tape_rollout",
]

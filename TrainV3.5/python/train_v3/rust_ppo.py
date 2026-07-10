"""PPO batch preparation for Rust TrainV3 rollout batches."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .rust_collector import RustTransitionBatch
from .rust_ffi import (
    compute_rust_gae_returns,
    compute_rust_normalized_legal_offsets,
    compute_rust_pad_legal_actions,
    compute_rust_pack_legal_action_rows,
    compute_rust_prepare_ppo_batch,
    compute_rust_selected_local_indices,
)
from .rust_policy import (
    PaddedLegalActionInputs,
    score_padded_legal_action_inputs,
    score_padded_legal_actions,
)


@dataclass(frozen=True)
class RustPPOBatch:
    observations: np.ndarray
    action_mask: np.ndarray | None
    action_features: np.ndarray | None
    legal_action_counts: np.ndarray
    legal_action_offsets: np.ndarray
    legal_action_ids: np.ndarray
    legal_action_features: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray | None
    advantages: np.ndarray
    returns: np.ndarray
    selected_local_indices: np.ndarray | None = None
    # Parallel action channel outside the frozen 601-candidate codec. When
    # present, PPO evaluates a binary mana-draw gate followed by the
    # conditional candidate-action distribution.
    mana_draw_legal: np.ndarray | None = None
    mana_draw_taken: np.ndarray | None = None

    def flatten(self) -> dict[str, np.ndarray | None]:
        steps, env_count = self.actions.shape
        return {
            "obs": self.observations.reshape((steps * env_count, *self.observations.shape[2:])),
            "action_mask": (
                None
                if self.action_mask is None
                else self.action_mask.reshape((steps * env_count, *self.action_mask.shape[2:]))
            ),
            "action_features": (
                None
                if self.action_features is None
                else self.action_features.reshape((steps * env_count, *self.action_features.shape[2:]))
            ),
            "legal_action_counts": self.legal_action_counts.reshape((steps * env_count,)),
            "legal_action_offsets": self.legal_action_offsets.reshape((steps * env_count,)),
            "legal_action_ids": self.legal_action_ids,
            "legal_action_features": self.legal_action_features,
            "actions": self.actions.reshape((steps * env_count,)),
            "old_log_probs": self.old_log_probs.reshape((steps * env_count,)),
            "values": self.values.reshape((steps * env_count,)),
            "rewards": self.rewards.reshape((steps * env_count,)),
            "terminated": self.terminated.reshape((steps * env_count,)),
            "truncated": None if self.truncated is None else self.truncated.reshape((steps * env_count,)),
            "advantages": self.advantages.reshape((steps * env_count,)),
            "returns": self.returns.reshape((steps * env_count,)),
            "selected_local_indices": (
                None
                if self.selected_local_indices is None
                else self.selected_local_indices.reshape((steps * env_count,))
            ),
            "mana_draw_legal": (
                None
                if self.mana_draw_legal is None
                else self.mana_draw_legal.reshape((steps * env_count,))
            ),
            "mana_draw_taken": (
                None
                if self.mana_draw_taken is None
                else self.mana_draw_taken.reshape((steps * env_count,))
            ),
        }


@dataclass(frozen=True)
class RustPPOEvaluation:
    loss: Any
    policy_loss: Any
    value_loss: Any
    entropy: Any
    clip_fraction: Any
    approx_kl: Any
    new_log_probs: Any
    values: Any
    ratios: Any


@dataclass(frozen=True)
class _ContiguousMinibatchPlan:
    batches: tuple[RustPPOBatch, ...]
    row_count: int
    minibatch_size: int
    kind: str
    planned_legal_action_rows: int
    planned_padded_action_rows: int
    planned_padding_waste_rows: int
    planned_padding_expansion_ratio: float
    planned_padded_feature_bytes: int
    planned_padded_mask_bytes: int
    planned_padded_id_bytes: int
    planned_padded_total_bytes: int

    @property
    def minibatch_count(self) -> int:
        return len(self.batches)


def train_rust_ppo_minibatch(
    model: Any,
    optimizer: Any,
    batch: RustPPOBatch,
    *,
    epochs: int = 1,
    minibatch_size: int = 256,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float | None = None,
    target_kl: float | None = None,
    shuffle: bool = True,
    seed: int | None = None,
    legal_row_pack_backend: str = "auto",
    full_batch_eval: bool = True,
    minibatch_plan: str = "contiguous",
    library_path: str | Path | None = None,
) -> dict[str, float | int | bool | str | None]:
    """Run PPO optimizer steps directly from a compact legal-action Rust batch."""
    return _train_rust_ppo_minibatch_with_evaluator(
        model,
        optimizer,
        batch,
        evaluator=evaluate_rust_ppo_batch,
        epochs=epochs,
        minibatch_size=minibatch_size,
        clip_epsilon=clip_epsilon,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        shuffle=shuffle,
        seed=seed,
        legal_row_pack_backend=legal_row_pack_backend,
        full_batch_eval=full_batch_eval,
        minibatch_plan=minibatch_plan,
        library_path=library_path,
    )


def train_dense_rust_ppo_minibatch(
    model: Any,
    optimizer: Any,
    batch: RustPPOBatch,
    *,
    epochs: int = 1,
    minibatch_size: int = 256,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float | None = None,
    target_kl: float | None = None,
    shuffle: bool = True,
    seed: int | None = None,
    legal_row_pack_backend: str = "auto",
    full_batch_eval: bool = True,
    minibatch_plan: str = "contiguous",
    library_path: str | Path | None = None,
) -> dict[str, float | int | bool | str | None]:
    """Run PPO optimizer steps through the dense 601-action model path."""
    if batch.action_features is None:
        raise ValueError("dense PPO updates require batch.action_features")
    return _train_rust_ppo_minibatch_with_evaluator(
        model,
        optimizer,
        batch,
        evaluator=evaluate_dense_rust_ppo_batch,
        epochs=epochs,
        minibatch_size=minibatch_size,
        clip_epsilon=clip_epsilon,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        shuffle=shuffle,
        seed=seed,
        legal_row_pack_backend=legal_row_pack_backend,
        full_batch_eval=full_batch_eval,
        minibatch_plan=minibatch_plan,
        library_path=library_path,
    )


def _train_rust_ppo_minibatch_with_evaluator(
    model: Any,
    optimizer: Any,
    batch: RustPPOBatch,
    *,
    evaluator: Any,
    epochs: int,
    minibatch_size: int,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float | None,
    target_kl: float | None,
    shuffle: bool,
    seed: int | None,
    legal_row_pack_backend: str,
    full_batch_eval: bool,
    minibatch_plan: str = "contiguous",
    library_path: str | Path | None,
) -> dict[str, float | int | bool | str | None]:
    import mlx.core as mx
    import mlx.nn as nn

    epochs = int(epochs)
    minibatch_size = int(minibatch_size)
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")
    if legal_row_pack_backend not in {"python", "rust", "auto"}:
        raise ValueError("legal_row_pack_backend must be python, rust, or auto")
    if minibatch_plan not in {"contiguous", "legal_count_sorted"}:
        raise ValueError("minibatch_plan must be contiguous or legal_count_sorted")
    if target_kl is not None and float(target_kl) <= 0.0:
        raise ValueError("target_kl must be positive when provided")

    flat = batch.flatten()
    row_count = int(flat["actions"].shape[0])
    if row_count <= 0:
        raise ValueError("RustPPOBatch must contain at least one row")

    before = None
    if full_batch_eval:
        before = evaluator(
            model,
            batch,
            clip_epsilon=clip_epsilon,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
        )
        mx.eval(
            before.loss,
            before.policy_loss,
            before.value_loss,
            before.entropy,
            before.clip_fraction,
            before.approx_kl,
        )

    current_batch: list[RustPPOBatch | None] = [None]
    current_padded_cache: list[PaddedLegalActionInputs | None] = [None]
    evaluator_accepts_padded_cache = _accepts_padded_legal_action_cache(evaluator)

    def loss_fn(model: Any):
        mini = current_batch[0]
        if mini is None:
            raise RuntimeError("internal error: minibatch not set")
        evaluation_kwargs = {
            "clip_epsilon": clip_epsilon,
            "value_coef": value_coef,
            "entropy_coef": entropy_coef,
        }
        if evaluator_accepts_padded_cache:
            evaluation_kwargs["padded_legal_action_cache"] = current_padded_cache[0]
        evaluation = evaluator(model, mini, **evaluation_kwargs)
        return evaluation.loss, {
            "policy_loss": evaluation.policy_loss,
            "value_loss": evaluation.value_loss,
            "entropy": evaluation.entropy,
            "clip_fraction": evaluation.clip_fraction,
            "approx_kl": evaluation.approx_kl,
        }

    value_and_grad = nn.value_and_grad(model, loss_fn)
    contiguous_plan = None
    if not shuffle and legal_row_pack_backend == "auto":
        contiguous_plan = (
            _plan_legal_count_sorted_minibatches(
                batch,
                flat=flat,
                row_count=row_count,
                minibatch_size=minibatch_size,
                legal_row_pack_backend="auto",
                library_path=library_path,
            )
            if minibatch_plan == "legal_count_sorted"
            else _plan_contiguous_minibatches(
                batch,
                flat=flat,
                row_count=row_count,
                minibatch_size=minibatch_size,
                legal_offset_backend="rust",
                library_path=library_path,
            )
        )
    elif minibatch_plan != "contiguous":
        raise ValueError("minibatch_plan requires shuffle=False and legal_row_pack_backend='auto'")
    padded_cache_plan = (
        tuple(
            _build_padded_legal_action_cache(
                mini,
                library_path=library_path,
            )
            for mini in contiguous_plan.batches
        )
        if contiguous_plan is not None and evaluator_accepts_padded_cache and epochs > 1
        else None
    )
    rng = np.random.default_rng(seed) if shuffle else None
    indices = None if contiguous_plan is not None else np.arange(row_count, dtype=np.int64)
    metric_values: dict[str, list[float]] = {
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "clip_fraction": [],
        "approx_kl": [],
    }

    def apply_minibatch(
        mini: RustPPOBatch,
        padded_cache: PaddedLegalActionInputs | None = None,
    ) -> float:
        current_batch[0] = mini
        current_padded_cache[0] = padded_cache
        (loss_value, aux), grads = value_and_grad(model)
        if max_grad_norm is not None:
            grads = _clip_grads(grads, max_grad_norm)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        metric_values["loss"].append(float(loss_value.item()))
        for name, value in aux.items():
            metric_values[name].append(float(value.item()))
        return float(aux["approx_kl"].item())

    early_stopped_by_kl = False
    epochs_completed = 0
    for _epoch in range(epochs):
        if shuffle:
            if indices is None:
                raise RuntimeError("internal error: shuffled indices not initialized")
            rng.shuffle(indices)
        if contiguous_plan is not None:
            if padded_cache_plan is None:
                for mini in contiguous_plan.batches:
                    latest_kl = apply_minibatch(mini)
                    if target_kl is not None and latest_kl > float(target_kl):
                        early_stopped_by_kl = True
                        break
            else:
                for mini, padded_cache in zip(contiguous_plan.batches, padded_cache_plan, strict=True):
                    latest_kl = apply_minibatch(mini, padded_cache)
                    if target_kl is not None and latest_kl > float(target_kl):
                        early_stopped_by_kl = True
                        break
            epochs_completed += 1
            if early_stopped_by_kl:
                break
            continue

        if indices is None:
            raise RuntimeError("internal error: minibatch indices not initialized")
        for start in range(0, row_count, minibatch_size):
            end = min(start + minibatch_size, row_count)
            current_batch[0] = _take_flat_rows(
                batch,
                indices[start:end],
                flat=flat,
                legal_row_pack_backend=legal_row_pack_backend,
                library_path=library_path,
            )
            latest_kl = apply_minibatch(current_batch[0])
            if target_kl is not None and latest_kl > float(target_kl):
                early_stopped_by_kl = True
                break
        epochs_completed += 1
        if early_stopped_by_kl:
            break

    current_batch[0] = None
    current_padded_cache[0] = None
    after = None
    if full_batch_eval:
        after = evaluator(
            model,
            batch,
            clip_epsilon=clip_epsilon,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
        )
        mx.eval(
            after.loss,
            after.policy_loss,
            after.value_loss,
            after.entropy,
            after.clip_fraction,
            after.approx_kl,
        )

    metrics: dict[str, float | int | bool | str | None] = {
        "loss_before": None if before is None else float(before.loss.item()),
        "loss_after": None if after is None else float(after.loss.item()),
        "updates": len(metric_values["loss"]),
        "rows": row_count,
        "epochs": epochs,
        "epochs_completed": epochs_completed,
        "target_kl": None if target_kl is None else float(target_kl),
        "early_stopped_by_kl": early_stopped_by_kl,
        "minibatch_size": minibatch_size,
        "dense_action_features": batch.action_features is not None,
        "legal_row_pack_backend": legal_row_pack_backend,
        "full_batch_eval": full_batch_eval,
        "minibatch_plan": minibatch_plan,
        "minibatch_plan_kind": None if contiguous_plan is None else contiguous_plan.kind,
        "contiguous_minibatch_plan": contiguous_plan is not None,
        "planned_minibatches": 0 if contiguous_plan is None else contiguous_plan.minibatch_count,
        "planned_minibatch_reuses": 0 if contiguous_plan is None else contiguous_plan.minibatch_count * epochs,
        "planned_legal_action_rows": 0 if contiguous_plan is None else contiguous_plan.planned_legal_action_rows,
        "planned_padded_action_rows": 0 if contiguous_plan is None else contiguous_plan.planned_padded_action_rows,
        "planned_padding_waste_rows": 0 if contiguous_plan is None else contiguous_plan.planned_padding_waste_rows,
        "planned_padding_expansion_ratio": (
            0.0 if contiguous_plan is None else contiguous_plan.planned_padding_expansion_ratio
        ),
        "planned_padded_feature_bytes": 0 if contiguous_plan is None else contiguous_plan.planned_padded_feature_bytes,
        "planned_padded_mask_bytes": 0 if contiguous_plan is None else contiguous_plan.planned_padded_mask_bytes,
        "planned_padded_id_bytes": 0 if contiguous_plan is None else contiguous_plan.planned_padded_id_bytes,
        "planned_padded_total_bytes": 0 if contiguous_plan is None else contiguous_plan.planned_padded_total_bytes,
        "planned_reused_padded_action_rows": (
            0 if contiguous_plan is None else contiguous_plan.planned_padded_action_rows * epochs
        ),
        "planned_recomputed_padded_total_bytes": (
            0 if contiguous_plan is None else contiguous_plan.planned_padded_total_bytes * epochs
        ),
        "padded_cache_enabled": padded_cache_plan is not None,
        "padded_cache_builds": 0 if padded_cache_plan is None else len(padded_cache_plan),
        "padded_cache_hits": 0 if padded_cache_plan is None else contiguous_plan.minibatch_count * epochs,
        "padded_cache_reuses": (
            0
            if padded_cache_plan is None
            else contiguous_plan.minibatch_count * max(epochs - 1, 0)
        ),
        "padded_cache_bytes": 0 if contiguous_plan is None or padded_cache_plan is None else contiguous_plan.planned_padded_total_bytes,
        "padded_cache_saved_builds": (
            0
            if contiguous_plan is None or padded_cache_plan is None
            else contiguous_plan.minibatch_count * max(epochs - 1, 0)
        ),
        "padded_cache_saved_padded_total_bytes": (
            0
            if contiguous_plan is None or padded_cache_plan is None
            else contiguous_plan.planned_padded_total_bytes * max(epochs - 1, 0)
        ),
    }
    for name, values in metric_values.items():
        if not values:
            raise ValueError(f"train_rust_ppo_minibatch: no metric values collected for {name}")
        _assert_finite(name, values)
        metrics[name] = float(np.mean(values))
    if full_batch_eval:
        _assert_finite("loss_before", [metrics["loss_before"]])
        _assert_finite("loss_after", [metrics["loss_after"]])
    return metrics


def prepare_rust_ppo_batch(
    transitions: RustTransitionBatch,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    bootstrap_values: np.ndarray | None = None,
    normalize_advantages: bool = True,
    advantage_backend: str = "python",
    selected_local_backend: str = "python",
    prepare_backend: str = "separate",
    mana_draw_legal: np.ndarray | None = None,
    mana_draw_taken: np.ndarray | None = None,
    library_path: str | Path | None = None,
) -> RustPPOBatch:
    """Compute GAE/returns for a Rust vectorized rollout batch."""
    if transitions.values is None:
        raise ValueError("RustTransitionBatch.values is required for PPO batch preparation")
    if transitions.log_probs is None:
        raise ValueError("RustTransitionBatch.log_probs is required for PPO batch preparation")

    rewards = np.asarray(transitions.rewards, dtype=np.float32)
    values = np.asarray(transitions.values, dtype=np.float32)
    old_log_probs = np.asarray(transitions.log_probs, dtype=np.float32)
    terminated = np.asarray(transitions.terminated, dtype=np.bool_)
    truncated = None if transitions.truncated is None else np.asarray(transitions.truncated, dtype=np.bool_)
    if rewards.shape != values.shape or rewards.shape != old_log_probs.shape:
        raise ValueError("rewards, values, and log_probs must have matching (steps, env_count) shapes")
    if terminated.shape != rewards.shape:
        raise ValueError("terminated must match rewards shape")
    if truncated is not None and truncated.shape != rewards.shape:
        raise ValueError("truncated must match rewards shape")

    steps, env_count = rewards.shape
    if (mana_draw_legal is None) != (mana_draw_taken is None):
        raise ValueError("mana_draw_legal and mana_draw_taken must be provided together")
    if mana_draw_legal is not None:
        mana_draw_legal = np.asarray(mana_draw_legal, dtype=np.bool_)
        mana_draw_taken = np.asarray(mana_draw_taken, dtype=np.bool_)
        if mana_draw_legal.shape != rewards.shape or mana_draw_taken.shape != rewards.shape:
            raise ValueError("mana-draw arrays must match rollout (steps, env_count) shape")
        if np.any(mana_draw_taken & ~mana_draw_legal):
            raise ValueError("mana_draw_taken requires mana_draw_legal")
    if prepare_backend not in {"separate", "rust_fused"}:
        raise ValueError("prepare_backend must be separate or rust_fused")
    if advantage_backend not in {"python", "rust"}:
        raise ValueError("advantage_backend must be python or rust")
    if selected_local_backend not in {"python", "rust", "provided"}:
        raise ValueError("selected_local_backend must be python, rust, or provided")
    if prepare_backend == "rust_fused" and (
        advantage_backend != "rust" or selected_local_backend != "rust"
    ):
        raise ValueError("prepare_backend='rust_fused' requires rust advantage and selected-local backends")
    if bootstrap_values is None:
        bootstrap = None if advantage_backend == "rust" else np.zeros(env_count, dtype=np.float32)
    else:
        bootstrap = np.asarray(bootstrap_values, dtype=np.float32)
        if bootstrap.shape != (env_count,):
            raise ValueError(f"bootstrap_values must have shape ({env_count},), got {bootstrap.shape}")

    if prepare_backend == "rust_fused":
        fused = compute_rust_prepare_ppo_batch(
            rewards,
            values,
            terminated,
            truncated,
            transitions.actions,
            transitions.legal_action_counts,
            transitions.legal_action_offsets,
            transitions.legal_action_ids,
            bootstrap_values=bootstrap,
            gamma=float(gamma),
            gae_lambda=float(gae_lambda),
            normalize_advantages=normalize_advantages,
            library_path=library_path,
        )
        advantages = fused.advantages
        returns = fused.returns
        selected_local_indices = fused.selected_local_indices
    else:
        if advantage_backend == "python":
            advantages, returns = _compute_python_gae_returns(
                rewards,
                values,
                terminated,
                truncated,
                bootstrap,
                gamma=float(gamma),
                gae_lambda=float(gae_lambda),
                normalize_advantages=normalize_advantages,
            )
        elif advantage_backend == "rust":
            advantages, returns = compute_rust_gae_returns(
                rewards,
                values,
                terminated,
                truncated,
                bootstrap_values=bootstrap,
                gamma=float(gamma),
                gae_lambda=float(gae_lambda),
                normalize_advantages=normalize_advantages,
                library_path=library_path,
            )

        flat_counts = transitions.legal_action_counts.reshape((steps * env_count,))
        flat_offsets = transitions.legal_action_offsets.reshape((steps * env_count,))
        flat_actions = transitions.actions.reshape((steps * env_count,))
        if selected_local_backend == "python":
            selected_local_indices = _selected_local_indices(
                np.asarray(flat_actions, dtype=np.uintp),
                np.asarray(flat_counts, dtype=np.int64),
                np.asarray(flat_offsets, dtype=np.int64),
                np.asarray(transitions.legal_action_ids, dtype=np.uintp),
            ).reshape((steps, env_count))
        elif selected_local_backend == "rust":
            selected_local_indices = compute_rust_selected_local_indices(
                flat_actions,
                flat_counts,
                flat_offsets,
                transitions.legal_action_ids,
                library_path=library_path,
            ).reshape((steps, env_count))
        elif selected_local_backend == "provided":
            selected_local_indices = _provided_selected_local_indices(
                transitions.selected_local_indices,
                transitions.legal_action_counts,
                (steps, env_count),
            )

    return RustPPOBatch(
        observations=transitions.observations,
        action_mask=transitions.action_mask,
        action_features=transitions.action_features,
        legal_action_counts=transitions.legal_action_counts,
        legal_action_offsets=transitions.legal_action_offsets,
        legal_action_ids=transitions.legal_action_ids,
        legal_action_features=transitions.legal_action_features,
        actions=np.asarray(transitions.actions, dtype=np.uintp),
        old_log_probs=old_log_probs,
        values=values,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        advantages=advantages.astype(np.float32, copy=False),
        returns=returns.astype(np.float32, copy=False),
        selected_local_indices=selected_local_indices.astype(np.int32, copy=False),
        mana_draw_legal=mana_draw_legal,
        mana_draw_taken=mana_draw_taken,
    )


def _compute_python_gae_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray | None,
    bootstrap: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
    normalize_advantages: bool,
) -> tuple[np.ndarray, np.ndarray]:
    steps, env_count = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)
    for env_idx in range(env_count):
        last_gae = 0.0
        for step_idx in range(steps - 1, -1, -1):
            done = bool(
                terminated[step_idx, env_idx]
                or (truncated is not None and truncated[step_idx, env_idx])
            )
            nonterminal = 0.0 if done else 1.0
            next_value = (
                float(values[step_idx + 1, env_idx])
                if step_idx + 1 < steps
                else float(bootstrap[env_idx])
            )
            delta = rewards[step_idx, env_idx] + gamma * next_value * nonterminal - values[step_idx, env_idx]
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
            advantages[step_idx, env_idx] = last_gae

    returns = advantages + values
    if normalize_advantages:
        mean = float(np.mean(advantages))
        std = float(np.std(advantages))
        advantages = advantages - mean
        if std > 1.0e-8:
            advantages = advantages / (std + 1.0e-8)
    return advantages.astype(np.float32, copy=False), returns.astype(np.float32, copy=False)


def evaluate_rust_ppo_batch(
    model: Any,
    batch: RustPPOBatch,
    *,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    padded_legal_action_cache: PaddedLegalActionInputs | None = None,
) -> RustPPOEvaluation:
    """Evaluate PPO loss from a legal-only Rust batch without dense action features."""
    import mlx.core as mx
    import mlx.nn as nn

    flat = batch.flatten()
    counts = np.asarray(flat["legal_action_counts"], dtype=np.uintp)
    offsets = np.asarray(flat["legal_action_offsets"], dtype=np.uintp)
    ids = np.asarray(flat["legal_action_ids"], dtype=np.uintp)
    actions = np.asarray(flat["actions"], dtype=np.uintp)
    if flat["selected_local_indices"] is None:
        selected_local = compute_rust_selected_local_indices(actions, counts, offsets, ids)
    else:
        selected_local = np.asarray(flat["selected_local_indices"], dtype=np.int32)

    if padded_legal_action_cache is None:
        scores = score_padded_legal_actions(
            model,
            flat["obs"],
            counts,
            flat["legal_action_features"],
            legal_action_offsets=offsets,
            legal_action_ids=ids,
            padding_backend="rust",
        )
    else:
        scores = score_padded_legal_action_inputs(
            model,
            flat["obs"],
            padded_legal_action_cache,
        )
    mana_draw_legal = flat["mana_draw_legal"]
    mana_draw_taken = flat["mana_draw_taken"]
    if (mana_draw_legal is None) != (mana_draw_taken is None):
        raise ValueError("mana-draw rollout fields must be present together")
    candidate_probs = nn.softmax(scores.padded_logits, axis=-1)
    selected = mx.array(selected_local, dtype=mx.int32)
    selected_candidate_probs = _gather_selected_action_probs(candidate_probs, selected)
    candidate_log_probs = mx.log(selected_candidate_probs + 1.0e-10)
    candidate_entropy = -mx.sum(
        candidate_probs * mx.log(candidate_probs + 1.0e-10), axis=-1
    )
    if mana_draw_legal is None:
        new_log_probs = candidate_log_probs
        entropy_per_row = candidate_entropy
    else:
        md_legal_np = np.asarray(mana_draw_legal, dtype=np.bool_)
        md_taken_np = np.asarray(mana_draw_taken, dtype=np.bool_)
        if scores.mana_draw_logits is None:
            if bool(np.any(md_taken_np)):
                raise ValueError("mana-draw transition requires model.mana_draw_head")
            new_log_probs = candidate_log_probs
            entropy_per_row = candidate_entropy
        else:
            # Factorized policy: P(draw)=sigmoid(gate), and P(card_i) is the
            # conditional candidate softmax.  Appending the gate to the card
            # vector makes deterministic argmax compare a draw option to each
            # individual card while sampling compares it to their aggregate.
            md_legal = mx.array(md_legal_np)
            md_taken = mx.array(md_taken_np)
            raw_draw_probability = mx.sigmoid(scores.mana_draw_logits)
            draw_probability = mx.where(
                md_legal,
                raw_draw_probability,
                mx.zeros_like(raw_draw_probability),
            )
            draw_log_prob = mx.log(draw_probability + 1.0e-10)
            no_draw_probability = 1.0 - draw_probability
            no_draw_log_prob = mx.log(no_draw_probability + 1.0e-10)
            gate_log_probs = mx.where(md_taken, draw_log_prob, no_draw_log_prob)
            new_log_probs = gate_log_probs + mx.where(
                md_taken,
                mx.zeros_like(candidate_log_probs),
                candidate_log_probs,
            )
            gate_entropy = -(
                draw_probability * draw_log_prob
                + no_draw_probability * no_draw_log_prob
            )
            # H(draw, card) = H(draw) + P(no draw) * H(card | no draw).
            entropy_per_row = gate_entropy + no_draw_probability * candidate_entropy

    old_log_probs = mx.array(flat["old_log_probs"])
    advantages = mx.array(flat["advantages"])
    returns = mx.array(flat["returns"])
    ratios = mx.exp(new_log_probs - old_log_probs)

    surr1 = ratios * advantages
    surr2 = mx.clip(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -mx.mean(mx.minimum(surr1, surr2))
    value_loss = value_coef * mx.mean((returns - scores.values) ** 2)
    entropy = mx.mean(entropy_per_row)
    clip_fraction = mx.mean(
        mx.where(
            ratios < 1.0 - clip_epsilon,
            mx.ones_like(ratios),
            mx.where(ratios > 1.0 + clip_epsilon, mx.ones_like(ratios), mx.zeros_like(ratios)),
        )
    )
    approx_kl = mx.mean(old_log_probs - new_log_probs)
    loss = policy_loss + value_loss - entropy_coef * entropy

    return RustPPOEvaluation(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        clip_fraction=clip_fraction,
        approx_kl=approx_kl,
        new_log_probs=new_log_probs,
        values=scores.values,
        ratios=ratios,
    )


def _accepts_padded_legal_action_cache(evaluator: Any) -> bool:
    try:
        parameters = inspect.signature(evaluator).parameters
    except (TypeError, ValueError):
        return False
    return "padded_legal_action_cache" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _build_padded_legal_action_cache(
    batch: RustPPOBatch,
    *,
    library_path: str | Path | None = None,
) -> PaddedLegalActionInputs:
    import mlx.core as mx

    flat = batch.flatten()
    counts = np.asarray(flat["legal_action_counts"], dtype=np.uintp)
    offsets = np.asarray(flat["legal_action_offsets"], dtype=np.uintp)
    ids = np.asarray(flat["legal_action_ids"], dtype=np.uintp)
    features = np.asarray(flat["legal_action_features"])
    padded = compute_rust_pad_legal_actions(
        counts,
        offsets,
        ids,
        features,
        library_path=library_path,
    )
    return PaddedLegalActionInputs(
        padded_features=mx.array(padded.features),
        legal_mask=mx.array(padded.mask),
    )


def evaluate_dense_rust_ppo_batch(
    model: Any,
    batch: RustPPOBatch,
    *,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> RustPPOEvaluation:
    """Evaluate PPO loss through the dense 601-action model path."""
    import mlx.core as mx
    import mlx.nn as nn

    flat = batch.flatten()
    if flat["action_features"] is None:
        raise ValueError("dense PPO evaluation requires action_features")
    if flat["action_mask"] is None:
        raise ValueError("dense PPO evaluation requires action_mask")

    obs = mx.array(flat["obs"])
    action_features = mx.array(flat["action_features"])
    mask = mx.array(flat["action_mask"])
    _out = model(obs, action_features)
    # V5 returns (logits, value, mana_draw_logit); baseline returns (logits,
    # value). This dense-PPO eval path scores the 601 candidates only, so drop
    # any 3rd element. Indexing is robust to both arities.
    logits, values = _out[0], _out[1]
    masked = mx.where(mask.astype(mx.bool_), logits, mx.array(-1.0e9, dtype=logits.dtype))
    probs = nn.softmax(masked, axis=-1)

    actions = mx.array(flat["actions"], dtype=mx.int32)
    action_probs = _gather_selected_action_probs(probs, actions)
    new_log_probs = mx.log(action_probs + 1.0e-10)

    old_log_probs = mx.array(flat["old_log_probs"])
    advantages = mx.array(flat["advantages"])
    returns = mx.array(flat["returns"])
    ratios = mx.exp(new_log_probs - old_log_probs)

    surr1 = ratios * advantages
    surr2 = mx.clip(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -mx.mean(mx.minimum(surr1, surr2))
    value_loss = value_coef * mx.mean((returns - values) ** 2)
    entropy = mx.mean(-mx.sum(probs * mx.log(probs + 1.0e-10), axis=-1))
    clip_fraction = mx.mean(
        mx.where(
            ratios < 1.0 - clip_epsilon,
            mx.ones_like(ratios),
            mx.where(ratios > 1.0 + clip_epsilon, mx.ones_like(ratios), mx.zeros_like(ratios)),
        )
    )
    approx_kl = mx.mean(old_log_probs - new_log_probs)
    loss = policy_loss + value_loss - entropy_coef * entropy

    return RustPPOEvaluation(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        clip_fraction=clip_fraction,
        approx_kl=approx_kl,
        new_log_probs=new_log_probs,
        values=values,
        ratios=ratios,
    )


def _selected_local_indices(
    actions: np.ndarray,
    counts: np.ndarray,
    offsets: np.ndarray,
    ids: np.ndarray,
) -> np.ndarray:
    if actions.shape != counts.shape or actions.shape != offsets.shape:
        raise ValueError("actions, legal_action_counts, and legal_action_offsets must have matching flat shapes")

    selected = np.empty(actions.shape[0], dtype=np.int32)
    for row_idx, (action_id, count, offset) in enumerate(zip(actions.tolist(), counts.tolist(), offsets.tolist())):
        if count <= 0:
            raise ValueError(f"row {row_idx} has no legal actions")
        legal_ids = ids[offset : offset + count]
        matches = np.flatnonzero(legal_ids == action_id)
        if matches.size != 1:
            raise ValueError(f"action {action_id} is not uniquely present in legal ids for row {row_idx}")
        selected[row_idx] = int(matches[0])
    return selected


def _provided_selected_local_indices(
    selected_local_indices: np.ndarray | None,
    legal_action_counts: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    if selected_local_indices is None:
        raise ValueError("selected_local_backend='provided' requires transition selected_local_indices")
    selected = np.asarray(selected_local_indices, dtype=np.int32)
    if selected.shape != shape:
        raise ValueError(f"transition selected_local_indices must have shape {shape}, got {selected.shape}")
    counts = np.asarray(legal_action_counts)
    if counts.shape != shape:
        raise ValueError(f"legal_action_counts must have shape {shape}, got {counts.shape}")
    if np.any(selected < 0) or np.any(selected >= counts):
        raise ValueError("transition selected_local_indices must be within each row's legal action count")
    return selected


def _gather_selected_action_probs(probs: Any, selected: Any) -> Any:
    import mlx.core as mx

    if len(probs.shape) != 2:
        raise ValueError(f"probs must be 2D, got shape {probs.shape}")
    if len(selected.shape) != 1:
        raise ValueError(f"selected indices must be 1D, got shape {selected.shape}")
    if probs.shape[0] != selected.shape[0]:
        raise ValueError(f"probs rows {probs.shape[0]} must match selected rows {selected.shape[0]}")
    return probs[mx.arange(selected.shape[0]), selected]


def _take_flat_rows(
    batch: RustPPOBatch,
    indices: np.ndarray,
    *,
    flat: dict[str, np.ndarray | None] | None = None,
    legal_row_pack_backend: str = "auto",
    legal_offset_backend: str = "python",
    library_path: str | Path | None = None,
) -> RustPPOBatch:
    if flat is None:
        flat = batch.flatten()
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError(f"indices must be 1D, got shape {idx.shape}")
    if idx.size <= 0:
        raise ValueError("indices must contain at least one row")
    if legal_row_pack_backend not in {"python", "rust", "auto"}:
        raise ValueError("legal_row_pack_backend must be python, rust, or auto")
    if legal_offset_backend not in {"python", "rust"}:
        raise ValueError("legal_offset_backend must be python or rust")

    row_count = int(flat["actions"].shape[0])
    if int(idx.min()) < 0 or int(idx.max()) >= row_count:
        raise IndexError(f"indices out of range for {row_count} PPO rows")

    original_counts = np.asarray(flat["legal_action_counts"], dtype=np.uintp)
    original_offsets = np.asarray(flat["legal_action_offsets"], dtype=np.uintp)
    original_ids = np.asarray(flat["legal_action_ids"], dtype=np.uintp)
    original_features = np.asarray(flat["legal_action_features"])
    contiguous_range = _contiguous_index_range(idx)
    if legal_row_pack_backend == "auto" and contiguous_range is not None:
        counts, new_offsets, legal_ids, legal_features = _take_contiguous_legal_rows(
            original_counts,
            original_offsets,
            original_ids,
            original_features,
            *contiguous_range,
            legal_offset_backend=legal_offset_backend,
            library_path=library_path,
        )
    elif legal_row_pack_backend == "python":
        counts = original_counts[idx]
        selected_offsets = original_offsets[idx]
        new_offsets = np.empty(idx.shape[0], dtype=np.uintp)
        ids_chunks: list[np.ndarray] = []
        feature_chunks: list[np.ndarray] = []
        legal_total = 0
        for out_idx, (count, original_offset) in enumerate(zip(counts.tolist(), selected_offsets.tolist())):
            count = int(count)
            original_offset = int(original_offset)
            if count <= 0:
                raise ValueError(f"row {int(idx[out_idx])} has no legal actions")
            new_offsets[out_idx] = legal_total
            ids_chunks.append(original_ids[original_offset : original_offset + count])
            feature_chunks.append(original_features[original_offset : original_offset + count])
            legal_total += count

        legal_ids = np.concatenate(ids_chunks) if ids_chunks else np.empty((0,), dtype=original_ids.dtype)
        legal_features = (
            np.concatenate(feature_chunks, axis=0)
            if feature_chunks
            else np.empty((0, original_features.shape[1]), dtype=original_features.dtype)
        )
    else:
        packed = compute_rust_pack_legal_action_rows(
            idx,
            original_counts,
            original_offsets,
            original_ids,
            original_features,
            library_path=library_path,
        )
        counts = packed.counts
        new_offsets = packed.offsets
        legal_ids = packed.ids
        legal_features = packed.features

    def selected_rows(name: str) -> np.ndarray:
        source = np.asarray(flat[name])
        rows = source[contiguous_range[0] : contiguous_range[1]] if contiguous_range is not None else source[idx]
        return rows.reshape((idx.shape[0], 1, *rows.shape[1:]))

    def selected_scalars(name: str) -> np.ndarray:
        source = np.asarray(flat[name])
        rows = source[contiguous_range[0] : contiguous_range[1]] if contiguous_range is not None else source[idx]
        return rows.reshape((idx.shape[0], 1))

    def selected_optional_scalars(name: str) -> np.ndarray | None:
        source = flat[name]
        if source is None:
            return None
        rows = source[contiguous_range[0] : contiguous_range[1]] if contiguous_range is not None else source[idx]
        return rows.reshape((idx.shape[0], 1))

    action_features = flat["action_features"]
    if action_features is None:
        selected_action_features = None
    else:
        dense_source = np.asarray(action_features)
        dense_rows = (
            dense_source[contiguous_range[0] : contiguous_range[1]]
            if contiguous_range is not None
            else dense_source[idx]
        )
        selected_action_features = dense_rows.reshape((idx.shape[0], 1, *dense_rows.shape[1:]))

    action_mask = flat["action_mask"]
    selected_action_mask = None
    if action_mask is not None:
        mask_source = np.asarray(action_mask)
        mask_rows = (
            mask_source[contiguous_range[0] : contiguous_range[1]]
            if contiguous_range is not None
            else mask_source[idx]
        )
        selected_action_mask = mask_rows.reshape((idx.shape[0], 1, *mask_rows.shape[1:]))

    selected_local_indices = flat["selected_local_indices"]
    selected_local = None
    if selected_local_indices is not None:
        selected_source = np.asarray(selected_local_indices, dtype=np.int32)
        selected_local_rows = (
            selected_source[contiguous_range[0] : contiguous_range[1]]
            if contiguous_range is not None
            else selected_source[idx]
        )
        selected_local = selected_local_rows.reshape((idx.shape[0], 1))

    selected_truncated = selected_optional_scalars("truncated")
    return RustPPOBatch(
        observations=selected_rows("obs"),
        action_mask=selected_action_mask,
        action_features=selected_action_features,
        legal_action_counts=counts.reshape((idx.shape[0], 1)),
        legal_action_offsets=new_offsets.reshape((idx.shape[0], 1)),
        legal_action_ids=legal_ids,
        legal_action_features=legal_features,
        actions=selected_scalars("actions").astype(batch.actions.dtype, copy=False),
        old_log_probs=selected_scalars("old_log_probs").astype(batch.old_log_probs.dtype, copy=False),
        values=selected_scalars("values").astype(batch.values.dtype, copy=False),
        rewards=selected_scalars("rewards").astype(batch.rewards.dtype, copy=False),
        terminated=selected_scalars("terminated").astype(batch.terminated.dtype, copy=False),
        truncated=None if selected_truncated is None else selected_truncated.astype(selected_truncated.dtype, copy=False),
        advantages=selected_scalars("advantages").astype(batch.advantages.dtype, copy=False),
        returns=selected_scalars("returns").astype(batch.returns.dtype, copy=False),
        selected_local_indices=selected_local,
        mana_draw_legal=selected_optional_scalars("mana_draw_legal"),
        mana_draw_taken=selected_optional_scalars("mana_draw_taken"),
    )


def _take_flat_row_range(
    batch: RustPPOBatch,
    start: int,
    end: int,
    *,
    flat: dict[str, np.ndarray | None] | None = None,
    legal_offset_backend: str = "python",
    library_path: str | Path | None = None,
) -> RustPPOBatch:
    if flat is None:
        flat = batch.flatten()
    start = int(start)
    end = int(end)
    row_count = int(flat["actions"].shape[0])
    if start < 0 or end <= start or end > row_count:
        raise IndexError(f"row range [{start}, {end}) is invalid for {row_count} PPO rows")
    if legal_offset_backend not in {"python", "rust"}:
        raise ValueError("legal_offset_backend must be python or rust")

    original_counts = np.asarray(flat["legal_action_counts"], dtype=np.uintp)
    original_offsets = np.asarray(flat["legal_action_offsets"], dtype=np.uintp)
    original_ids = np.asarray(flat["legal_action_ids"], dtype=np.uintp)
    original_features = np.asarray(flat["legal_action_features"])
    counts, new_offsets, legal_ids, legal_features = _take_contiguous_legal_rows(
        original_counts,
        original_offsets,
        original_ids,
        original_features,
        start,
        end,
        legal_offset_backend=legal_offset_backend,
        library_path=library_path,
    )
    size = end - start

    def selected_rows(name: str) -> np.ndarray:
        source = np.asarray(flat[name])
        rows = source[start:end]
        return rows.reshape((size, 1, *rows.shape[1:]))

    def selected_scalars(name: str) -> np.ndarray:
        source = np.asarray(flat[name])
        rows = source[start:end]
        return rows.reshape((size, 1))

    def selected_optional_scalars(name: str) -> np.ndarray | None:
        source = flat[name]
        if source is None:
            return None
        rows = source[start:end]
        return rows.reshape((size, 1))

    action_features = flat["action_features"]
    selected_action_features = None
    if action_features is not None:
        dense_rows = np.asarray(action_features)[start:end]
        selected_action_features = dense_rows.reshape((size, 1, *dense_rows.shape[1:]))

    action_mask = flat["action_mask"]
    selected_action_mask = None
    if action_mask is not None:
        mask_rows = np.asarray(action_mask)[start:end]
        selected_action_mask = mask_rows.reshape((size, 1, *mask_rows.shape[1:]))

    selected_local_indices = flat["selected_local_indices"]
    selected_local = None
    if selected_local_indices is not None:
        selected_local = np.asarray(selected_local_indices, dtype=np.int32)[start:end].reshape((size, 1))

    selected_truncated = selected_optional_scalars("truncated")
    return RustPPOBatch(
        observations=selected_rows("obs"),
        action_mask=selected_action_mask,
        action_features=selected_action_features,
        legal_action_counts=counts.reshape((size, 1)),
        legal_action_offsets=new_offsets.reshape((size, 1)),
        legal_action_ids=legal_ids,
        legal_action_features=legal_features,
        actions=selected_scalars("actions").astype(batch.actions.dtype, copy=False),
        old_log_probs=selected_scalars("old_log_probs").astype(batch.old_log_probs.dtype, copy=False),
        values=selected_scalars("values").astype(batch.values.dtype, copy=False),
        rewards=selected_scalars("rewards").astype(batch.rewards.dtype, copy=False),
        terminated=selected_scalars("terminated").astype(batch.terminated.dtype, copy=False),
        truncated=None if selected_truncated is None else selected_truncated.astype(selected_truncated.dtype, copy=False),
        advantages=selected_scalars("advantages").astype(batch.advantages.dtype, copy=False),
        returns=selected_scalars("returns").astype(batch.returns.dtype, copy=False),
        selected_local_indices=selected_local,
        mana_draw_legal=selected_optional_scalars("mana_draw_legal"),
        mana_draw_taken=selected_optional_scalars("mana_draw_taken"),
    )


def _plan_contiguous_minibatches(
    batch: RustPPOBatch,
    *,
    flat: dict[str, np.ndarray | None],
    row_count: int,
    minibatch_size: int,
    legal_offset_backend: str = "rust",
    library_path: str | Path | None = None,
) -> _ContiguousMinibatchPlan:
    row_count = int(row_count)
    minibatch_size = int(minibatch_size)
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")
    batches = tuple(
        _take_flat_row_range(
            batch,
            start,
            min(start + minibatch_size, row_count),
            flat=flat,
            legal_offset_backend=legal_offset_backend,
            library_path=library_path,
        )
        for start in range(0, row_count, minibatch_size)
    )
    planned_legal_action_rows = sum(
        int(mini.legal_action_counts.reshape(-1).sum(dtype=np.uintp))
        for mini in batches
    )
    planned_padded_action_rows = sum(
        int(mini.actions.size) * int(mini.legal_action_counts.reshape(-1).max(initial=0))
        for mini in batches
    )
    planned_padding_waste_rows = planned_padded_action_rows - planned_legal_action_rows
    planned_padding_expansion_ratio = (
        0.0
        if planned_legal_action_rows <= 0
        else planned_padded_action_rows / planned_legal_action_rows
    )
    action_feature_dim = int(batch.legal_action_features.shape[1])
    planned_padded_feature_bytes = (
        planned_padded_action_rows
        * action_feature_dim
        * int(batch.legal_action_features.dtype.itemsize)
    )
    planned_padded_mask_bytes = planned_padded_action_rows * int(np.dtype(np.bool_).itemsize)
    planned_padded_id_bytes = planned_padded_action_rows * int(np.dtype(np.uintp).itemsize)
    planned_padded_total_bytes = (
        planned_padded_feature_bytes
        + planned_padded_mask_bytes
        + planned_padded_id_bytes
    )
    return _ContiguousMinibatchPlan(
        batches=batches,
        row_count=row_count,
        minibatch_size=minibatch_size,
        kind="contiguous",
        planned_legal_action_rows=planned_legal_action_rows,
        planned_padded_action_rows=planned_padded_action_rows,
        planned_padding_waste_rows=planned_padding_waste_rows,
        planned_padding_expansion_ratio=planned_padding_expansion_ratio,
        planned_padded_feature_bytes=planned_padded_feature_bytes,
        planned_padded_mask_bytes=planned_padded_mask_bytes,
        planned_padded_id_bytes=planned_padded_id_bytes,
        planned_padded_total_bytes=planned_padded_total_bytes,
    )


def _plan_legal_count_sorted_minibatches(
    batch: RustPPOBatch,
    *,
    flat: dict[str, np.ndarray | None],
    row_count: int,
    minibatch_size: int,
    legal_row_pack_backend: str = "auto",
    library_path: str | Path | None = None,
) -> _ContiguousMinibatchPlan:
    row_count = int(row_count)
    minibatch_size = int(minibatch_size)
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")
    if legal_row_pack_backend not in {"python", "rust", "auto"}:
        raise ValueError("legal_row_pack_backend must be python, rust, or auto")
    counts = np.asarray(flat["legal_action_counts"], dtype=np.uintp)
    if counts.shape != (row_count,):
        raise ValueError(f"flat legal_action_counts must have shape ({row_count},), got {counts.shape}")
    order = np.argsort(counts, kind="stable").astype(np.int64, copy=False)
    batches = tuple(
        _take_flat_rows(
            batch,
            order[start : min(start + minibatch_size, row_count)],
            flat=flat,
            legal_row_pack_backend=legal_row_pack_backend,
            library_path=library_path,
        )
        for start in range(0, row_count, minibatch_size)
    )
    return _minibatch_plan_from_batches(
        batches,
        row_count=row_count,
        minibatch_size=minibatch_size,
        kind="legal_count_sorted",
        action_feature_dim=int(batch.legal_action_features.shape[1]),
        action_feature_itemsize=int(batch.legal_action_features.dtype.itemsize),
    )


def _minibatch_plan_from_batches(
    batches: tuple[RustPPOBatch, ...],
    *,
    row_count: int,
    minibatch_size: int,
    kind: str,
    action_feature_dim: int,
    action_feature_itemsize: int,
) -> _ContiguousMinibatchPlan:
    planned_legal_action_rows = sum(
        int(mini.legal_action_counts.reshape(-1).sum(dtype=np.uintp))
        for mini in batches
    )
    planned_padded_action_rows = sum(
        int(mini.actions.size) * int(mini.legal_action_counts.reshape(-1).max(initial=0))
        for mini in batches
    )
    planned_padding_waste_rows = planned_padded_action_rows - planned_legal_action_rows
    planned_padding_expansion_ratio = (
        0.0
        if planned_legal_action_rows <= 0
        else planned_padded_action_rows / planned_legal_action_rows
    )
    planned_padded_feature_bytes = (
        planned_padded_action_rows
        * int(action_feature_dim)
        * int(action_feature_itemsize)
    )
    planned_padded_mask_bytes = planned_padded_action_rows * int(np.dtype(np.bool_).itemsize)
    planned_padded_id_bytes = planned_padded_action_rows * int(np.dtype(np.uintp).itemsize)
    planned_padded_total_bytes = (
        planned_padded_feature_bytes
        + planned_padded_mask_bytes
        + planned_padded_id_bytes
    )
    return _ContiguousMinibatchPlan(
        batches=batches,
        row_count=row_count,
        minibatch_size=minibatch_size,
        kind=kind,
        planned_legal_action_rows=planned_legal_action_rows,
        planned_padded_action_rows=planned_padded_action_rows,
        planned_padding_waste_rows=planned_padding_waste_rows,
        planned_padding_expansion_ratio=planned_padding_expansion_ratio,
        planned_padded_feature_bytes=planned_padded_feature_bytes,
        planned_padded_mask_bytes=planned_padded_mask_bytes,
        planned_padded_id_bytes=planned_padded_id_bytes,
        planned_padded_total_bytes=planned_padded_total_bytes,
    )


def _contiguous_index_range(indices: np.ndarray) -> tuple[int, int] | None:
    if indices.size <= 0:
        return None
    start = int(indices[0])
    end = start + int(indices.size)
    if indices.size == 1 or bool(np.all(indices[1:] - indices[:-1] == 1)):
        return start, end
    return None


def _take_contiguous_legal_rows(
    original_counts: np.ndarray,
    original_offsets: np.ndarray,
    original_ids: np.ndarray,
    original_features: np.ndarray,
    start: int,
    end: int,
    *,
    legal_offset_backend: str = "python",
    library_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = original_counts[start:end]
    if np.any(counts == 0):
        raise ValueError("selected legal-action rows must contain at least one action")
    offsets = original_offsets[start:end]
    first_offset = int(offsets[0])
    legal_total = int(counts.sum(dtype=np.uintp))
    if legal_offset_backend == "rust":
        new_offsets = compute_rust_normalized_legal_offsets(
            counts,
            offsets,
            library_path=library_path,
        )
    elif legal_offset_backend == "python":
        new_offsets = offsets - np.asarray(first_offset, dtype=np.uintp)
    else:
        raise ValueError("legal_offset_backend must be python or rust")
    last_offset = first_offset + legal_total
    if legal_offset_backend == "python" and offsets.shape[0] > 1:
        expected_offsets = first_offset + np.cumsum(counts[:-1], dtype=np.uintp)
        if not np.array_equal(offsets[1:], expected_offsets):
            raise ValueError("contiguous row indices do not map to a contiguous legal-action tape")
    legal_ids = original_ids[first_offset:last_offset]
    legal_features = original_features[first_offset:last_offset]
    return counts, new_offsets, legal_ids, legal_features


def _clip_grads(grads: Any, max_norm: float):
    import mlx.core as mx
    import mlx.nn as nn

    flat = nn.utils.tree_flatten(grads)
    total_norm = mx.sqrt(sum(mx.sum(value**2) for _, value in flat))
    total_norm = mx.maximum(total_norm, mx.array(1.0e-6, dtype=mx.float32))
    scale = mx.minimum(mx.array(float(max_norm), dtype=mx.float32), total_norm) / total_norm
    return nn.utils.tree_unflatten([(key, value * scale) for key, value in flat])


def _assert_finite(name: str, values: Any) -> None:
    arr = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")


__all__ = [
    "RustPPOBatch",
    "RustPPOEvaluation",
    "evaluate_dense_rust_ppo_batch",
    "evaluate_rust_ppo_batch",
    "prepare_rust_ppo_batch",
    "train_dense_rust_ppo_minibatch",
    "train_rust_ppo_minibatch",
]

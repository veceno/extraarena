"""Block C component C3 -- AWAC/CRR offline-PPO replay loss + trainer (D-C1,
D-C2, D-C3, D-C9, D-C10 + the BLOCKER loss-math fix).

Turns a C2 fresh-human ``OfflineReplayBatch`` (``offline_replay_bridge.py`` --
COMMITTED C2 output, consumed READ-ONLY) into a V5 policy update via the
**Hybrid AWAC x PPO-clip** loss (D-C1). Three layers:

  (A1) PURE-NUMPY loss core ``awac_crr_loss`` -- the SYNTHETICALLY-TESTABLE
       core (NO mlx import; the PRIMARY regression guard). Inputs are numpy
       arrays; outputs ``(total_loss, metrics_dict)``.
  (A2) MLX evaluator ``evaluate_awac_dense_batch`` -- mirrors the dense
       evaluator ``rust_ppo.py:734-797`` BUT RETAINS the V5 3rd output
       ``mana_draw_logit = _out[2]`` (the dense template ``:759`` DROPS it),
       applies the AWAC weight ``w`` + valid_policy_mask, adds the mana_draw
       BCE term, and masks padded rows out of ``value_loss``.
  (A3) Trainer ``train_awac_crr_replay`` -- mirrors
       ``_train_rust_ppo_minibatch_with_evaluator`` (``rust_ppo.py:193-441``):
       minibatch loop, ``nn.value_and_grad``, ``_zero_frozen_grads`` (A2
       freeze_faithful byte-identity), ``mlx.optimizers.Adam.update``,
       ``mx.eval``, ``_clip_grads``, full-batch eval before/after. A
       C3-SPECIFIC loop slices BOTH the ``RustPPOBatch`` minibatch AND the C2
       parallel arrays (``is_mana_draw`` / ``mana_draw_legal`` /
       ``target_tcodes`` / ``is_padded``) in lockstep -- ``RustPPOBatch`` does
       NOT carry these, and ``rust_ppo.py`` is READ-ONLY (NOT modified to
       thread extra args).
  (A4) Driver ``AwacCrrReplay.run`` -> ``AwacCrrMetrics`` -- MONITORING-ONLY
       (NO promote/score field; B6 external-bench owns promotion, C4 owns the
       promote decision). Skip-gated.

LOSS MATH (EXACT -- the BLOCKER sign fix; ``log pi`` appears ONLY inside the
ratio via ``new_log_prob``, NEVER as a standalone multiplier; the ADVANTAGE
``A`` is the surrogate multiplier):

    masked = where(action_mask==1, logits, -1e9);  probs = softmax(masked, -1)
    new_log_probs = log(gather(probs, actions) + 1e-10)
    ratios = exp(new_log_probs - old_log_probs)
    A = advantages                                # GAE advantage (the multiplier)
    w = exp(clamp(A / lambda_awac, -awac_clamp, awac_clamp))   # AWAC weight
    surr1 = ratios * A
    surr2 = clip(ratios, 1-eps, 1+eps) * A
    valid_policy_mask = (target_tcodes >= 0) AND (NOT is_padded)
    policy_loss = -sum(w * minimum(surr1, surr2) * valid_policy_mask)
                  / max(sum(valid_policy_mask), 1)
    valid_value_mask = (NOT is_padded)
    value_loss = value_coef * sum((returns - values)^2 * valid_value_mask)
                 / max(sum(valid_value_mask), 1)
    mana_draw BCE (RETAINS mana_draw_logit): over mana_draw_legal rows only.
    entropy = sum(-sum(probs*log(probs+1e-10), -1) * valid_policy_mask)
              / max(sum(valid_policy_mask), 1)
    total = policy_loss + value_loss - entropy_coef*entropy
            + mana_draw_bce_weight * mana_draw_bce

The incoherent prior draft ``(1+log pi)*w*log pi`` DECREASED ``log pi`` for
high-advantage actions whenever ``log pi < -1`` (the common 601-way softmax
regime); the ``A``-multiplier form INCREASES ``log pi`` for high-advantage
actions (the correct AWAC direction -- verified by the sign test).

NO edit to frozen-classic / A1-A5 / B1-B8 / rust_ppo.py / rust_collector.py /
bc_train.py / v5_policy.py / warm_start_v5.py / core / obs_v5.py / contracts.py
/ offline_replay_bridge.py (C2). All consumed READ-ONLY.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from .bc_train import (
    DEFAULT_MANA_DRAW_BCE_WEIGHT,
    _zero_frozen_grads,
    assert_frozen_preserved,
    frozen_param_names,
    snapshot_frozen_params,
    trainable_param_names,
)
from .offline_replay_bridge import OfflineReplayBatch
from .rust_ppo import (
    RustPPOBatch,
    RustPPOEvaluation,
    _assert_finite,
    _clip_grads,
    _gather_selected_action_probs,
    _take_flat_rows,
    prepare_rust_ppo_batch,
)

logger = logging.getLogger(__name__)

# Stable -inf equivalent for illegal-candidate masking (mirrors
# rust_ppo.py:760 + bc_train.py:128).
_NEG_INF_STABLE = -1.0e9
_LOG_EPS = 1.0e-10


# ---------------------------------------------------------------------------
# (A1) PURE-NUMPY loss core -- the PRIMARY regression guard (NO mlx import).
# ---------------------------------------------------------------------------

def _numpy_softmax_masked(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mirror rust_ppo.py:760-761: masked = where(mask, logits, -1e9);
    probs = softmax(masked, axis=-1). Numerically stable (max-subtract)."""
    masked = np.where(mask == 1.0, logits, np.float32(_NEG_INF_STABLE))
    m = masked.max(axis=-1, keepdims=True)
    ex = np.exp(masked - m)
    return ex / ex.sum(axis=-1, keepdims=True)


def awac_crr_loss(
    model_outputs: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_fields: Dict[str, np.ndarray],
    *,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    lambda_awac: float = 1.0,
    awac_clamp: float = 4.0,
    mana_draw_bce_weight: float = DEFAULT_MANA_DRAW_BCE_WEIGHT,
) -> tuple[float, Dict[str, float]]:
    """Pure-numpy Hybrid AWAC x PPO-clip loss (NO mlx -- the PRIMARY regression
    guard).

    Args:
        model_outputs: ``(logits[B,601], values[B], mana_draw_logit[B])`` --
            the V5 3-tuple. ``mana_draw_logit`` is RETAINED (the dense template
            ``rust_ppo.py:759`` drops ``_out[2]``; C3 must NOT).
        batch_fields: dict of numpy arrays --
            ``old_log_probs[B]``, ``advantages[B]`` (GAE A), ``returns[B]``,
            ``actions[B]`` (target_tcode, int32), ``action_mask[B,601]`` (bool
            / 0-1), ``is_mana_draw[B]`` (float 0/1), ``mana_draw_legal[B]``
            (bool), ``target_tcodes[B]`` (int32, -1 for mana_draw/terminal/
            padded -- the C2 parallel array), ``is_padded[B]`` (bool).

    Returns:
        ``(total_loss, metrics_dict)``. ``metrics_dict`` keys: ``policy_loss``,
        ``value_loss``, ``mana_draw_bce``, ``entropy``, ``total``,
        ``valid_policy_rows``, ``valid_value_rows``, ``mana_draw_legal_rows``,
        ``mean_ratio``, ``clip_fraction``, ``approx_kl`` (all MONITORING-ONLY).
    """
    logits, values, mana_draw_logit = model_outputs
    logits = np.asarray(logits, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    mana_draw_logit = np.asarray(mana_draw_logit, dtype=np.float32).reshape(-1)

    old_log_probs = np.asarray(batch_fields["old_log_probs"], dtype=np.float32).reshape(-1)
    advantages = np.asarray(batch_fields["advantages"], dtype=np.float32).reshape(-1)
    returns = np.asarray(batch_fields["returns"], dtype=np.float32).reshape(-1)
    actions = np.asarray(batch_fields["actions"], dtype=np.int64).reshape(-1)
    action_mask = np.asarray(batch_fields["action_mask"], dtype=np.float32)
    is_mana_draw = np.asarray(batch_fields["is_mana_draw"], dtype=np.float32).reshape(-1)
    mana_draw_legal = np.asarray(batch_fields["mana_draw_legal"], dtype=np.bool_).reshape(-1)
    target_tcodes = np.asarray(batch_fields["target_tcodes"], dtype=np.int64).reshape(-1)
    is_padded = np.asarray(batch_fields["is_padded"], dtype=np.bool_).reshape(-1)

    B = logits.shape[0]
    if values.shape[0] != B or mana_draw_logit.shape[0] != B:
        raise ValueError(
            f"model_outputs rows must match logits rows ({B}); got values="
            f"{values.shape[0]}, mana_draw_logit={mana_draw_logit.shape[0]}"
        )

    # Masked softmax + new_log_probs (mirror rust_ppo.py:760-765).
    probs = _numpy_softmax_masked(logits, action_mask)
    rows = np.arange(B)
    action_probs = probs[rows, actions]
    new_log_probs = np.log(action_probs + _LOG_EPS)

    # PPO ratio (rust_ppo.py:770).
    ratios = np.exp(new_log_probs - old_log_probs)

    # Advantage IS the surrogate multiplier (the BLOCKER fix -- NOT log pi).
    A = advantages
    # AWAC weight: clamp the EXPONENT before exp for stability.
    w = awac_weight(A, lambda_awac=lambda_awac, awac_clamp=awac_clamp)

    surr1 = ratios * A
    surr2 = np.clip(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * A

    # valid_policy_mask: EXCLUDES mana_draw (target_tcode=-1), terminal (-1),
    # and padded rows from the PPO surrogate.
    valid_policy_mask = (target_tcodes >= 0) & (~is_padded)
    vpf = valid_policy_mask.astype(np.float32)
    vpf_count = float(vpf.sum())
    vpf_denom = max(vpf_count, 1.0)

    policy_loss = float(
        -np.sum(w * np.minimum(surr1, surr2) * vpf) / vpf_denom
    )

    # value_loss over REAL rows only (D-C3; padded rows masked -- a padded row
    # with values=0 returns=0 would otherwise contribute a spurious 0 term,
    # but a padded row with perturbed values MUST NOT contribute).
    valid_value_mask = ~is_padded
    vvf = valid_value_mask.astype(np.float32)
    vvf_count = float(vvf.sum())
    vvf_denom = max(vvf_count, 1.0)
    value_loss = float(value_coef * np.sum((returns - values) ** 2 * vvf) / vvf_denom)

    # mana_draw BCE (retain mana_draw_logit; A2 pattern bc_train.py:332-338).
    md_p = np.clip(_numpy_sigmoid(mana_draw_logit), 1e-7, 1.0 - 1e-7)
    bce_per_row = -(is_mana_draw * np.log(md_p) + (1.0 - is_mana_draw) * np.log(1.0 - md_p))
    mdf = mana_draw_legal.astype(np.float32)
    mdf_count = float(mdf.sum())
    mdf_denom = max(mdf_count, 1.0)
    mana_draw_bce = float(np.sum(bce_per_row * mdf) / mdf_denom)

    # entropy over valid rows (rust_ppo.py:776).
    ent_per_row = -np.sum(probs * np.log(probs + _LOG_EPS), axis=-1)
    entropy = float(np.sum(ent_per_row * vpf) / vpf_denom)

    total = (
        policy_loss + value_loss - entropy_coef * entropy
        + mana_draw_bce_weight * mana_draw_bce
    )

    # Monitoring metrics.
    ratios_v = ratios[valid_policy_mask]
    if ratios_v.size > 0:
        mean_ratio = float(ratios_v.mean())
        clip_fraction = float(
            np.mean((ratios_v < 1.0 - clip_epsilon) | (ratios_v > 1.0 + clip_epsilon))
        )
        approx_kl = float(np.mean(old_log_probs[valid_policy_mask] - new_log_probs[valid_policy_mask]))
    else:
        mean_ratio = 0.0
        clip_fraction = 0.0
        approx_kl = 0.0

    metrics: Dict[str, float] = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "mana_draw_bce": mana_draw_bce,
        "entropy": entropy,
        "total": float(total),
        "valid_policy_rows": vpf_count,
        "valid_value_rows": vvf_count,
        "mana_draw_legal_rows": mdf_count,
        "mean_ratio": mean_ratio,
        "clip_fraction": clip_fraction,
        "approx_kl": approx_kl,
    }
    return float(total), metrics


def _numpy_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def awac_weight(
    advantages: np.ndarray,
    *,
    lambda_awac: float = 1.0,
    awac_clamp: float = 4.0,
) -> np.ndarray:
    """AWAC advantage weight ``w = exp(clamp(A / lambda_awac, -C, C))``.

    The EXPONENT is clamped BEFORE ``exp`` for numerical stability (GAE
    advantages can be large; an unclamped ``exp(A/lambda)`` overflows/
    underflows and a few outlier weights dominate the loss). For ``A >>
    lambda*C`` ``w`` saturates at ``exp(C)``; for ``A=0`` ``w=1``; for ``A<0``
    ``w<1``.
    """
    A = np.asarray(advantages, dtype=np.float32)
    return np.exp(
        np.clip(A / float(lambda_awac), -float(awac_clamp), float(awac_clamp))
    )


# ---------------------------------------------------------------------------
# (A2) MLX evaluator -- mirrors evaluate_dense_rust_ppo_batch:734-797 BUT
# retains mana_draw_logit + AWAC weight + valid_policy_mask + mana_draw BCE +
# padded-row masking on value_loss.
# ---------------------------------------------------------------------------

def evaluate_awac_dense_batch(
    model: Any,
    batch: RustPPOBatch,
    *,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    lambda_awac: float = 1.0,
    awac_clamp: float = 4.0,
    mana_draw_bce_weight: float = DEFAULT_MANA_DRAW_BCE_WEIGHT,
    is_mana_draw: "Any",
    mana_draw_legal: "Any",
    target_tcodes: "Any",
    is_padded: "Any",
) -> RustPPOEvaluation:
    """MLX AWAC x PPO-clip evaluator (dense 601-action path).

    Mirrors ``evaluate_dense_rust_ppo_batch`` (``rust_ppo.py:734-797``) BUT:

      (a) ``_out = model(obs, action_features, mana_draw_legal=None)`` returns
          the V5 3-tuple and C3 RETAINS ``mana_draw_logit = _out[2]`` (the
          dense template ``:759`` drops it).
      (b) computes ``new_log_probs`` / ``ratios`` / ``surr`` identically.
      (c) applies the AWAC weight ``w`` + ``valid_policy_mask``.
      (d) adds the mana_draw BCE term (over ``mana_draw_legal`` rows only).
      (e) masks padded rows out of ``value_loss``.

    The model forward passes ``mana_draw_legal=None`` so the RAW
    ``mana_draw_logit`` is returned (not -inf-gated), mirroring
    ``bc_train.py:306-308``.
    """
    import mlx.core as mx
    import mlx.nn as nn

    flat = batch.flatten()
    if flat["action_features"] is None:
        raise ValueError("dense AWAC evaluation requires action_features")
    if flat["action_mask"] is None:
        raise ValueError("dense AWAC evaluation requires action_mask")

    obs = mx.array(flat["obs"])
    action_features = mx.array(flat["action_features"])
    mask = mx.array(flat["action_mask"])
    # V5 3-tuple -- RETAIN mana_draw_logit (_out[2]); raw head logit
    # (mana_draw_legal=None -> not -inf-gated).
    _out = model(obs, action_features, mana_draw_legal=None)
    logits, values, mana_draw_logit = _out[0], _out[1], _out[2]

    masked = mx.where(mask.astype(mx.bool_), logits, mx.array(_NEG_INF_STABLE, dtype=logits.dtype))
    probs = nn.softmax(masked, axis=-1)

    actions = mx.array(flat["actions"], dtype=mx.int32)
    action_probs = _gather_selected_action_probs(probs, actions)
    new_log_probs = mx.log(action_probs + _LOG_EPS)

    old_log_probs = mx.array(flat["old_log_probs"])
    advantages = mx.array(flat["advantages"])
    returns = mx.array(flat["returns"])
    ratios = mx.exp(new_log_probs - old_log_probs)

    A = advantages
    # AWAC weight: clamp the EXPONENT before exp (stability).
    w = mx.exp(mx.clip(A / float(lambda_awac), a_min=-float(awac_clamp), a_max=float(awac_clamp)))

    surr1 = ratios * A
    surr2 = mx.clip(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * A

    target_tcodes_mx = mx.array(target_tcodes)
    is_padded_mx = mx.array(is_padded)
    valid_policy_mask = (target_tcodes_mx >= 0) & (~is_padded_mx)
    vpf = valid_policy_mask.astype(advantages.dtype)
    vpf_sum = mx.sum(vpf)
    vpf_denom = mx.maximum(vpf_sum, mx.array(1.0, dtype=vpf_sum.dtype))

    policy_loss = -mx.sum(w * mx.minimum(surr1, surr2) * vpf) / vpf_denom

    valid_value_mask = ~is_padded_mx
    vvf = valid_value_mask.astype(advantages.dtype)
    vvf_sum = mx.sum(vvf)
    vvf_denom = mx.maximum(vvf_sum, mx.array(1.0, dtype=vvf_sum.dtype))
    value_loss = value_coef * mx.sum((returns - values) ** 2 * vvf) / vvf_denom

    # mana_draw BCE (retain mana_draw_logit; A2 pattern bc_train.py:332-338).
    md_p = mx.clip(mx.sigmoid(mana_draw_logit), 1e-7, 1.0 - 1e-7)
    is_md_mx = mx.array(is_mana_draw).astype(md_p.dtype)
    bce_per_row = -(is_md_mx * mx.log(md_p) + (1.0 - is_md_mx) * mx.log(1.0 - md_p))
    mdf = mx.array(mana_draw_legal).astype(md_p.dtype)
    mdf_sum = mx.sum(mdf)
    mdf_denom = mx.maximum(mdf_sum, mx.array(1.0, dtype=mdf_sum.dtype))
    mana_draw_bce = mx.sum(bce_per_row * mdf) / mdf_denom

    # entropy over valid rows (rust_ppo.py:776).
    ent_per_row = -mx.sum(probs * mx.log(probs + _LOG_EPS), axis=-1)
    entropy = mx.sum(ent_per_row * vpf) / vpf_denom

    # clip_fraction + approx_kl over valid rows (mirror rust_ppo.py:777-784).
    clipped = mx.where(
        ratios < 1.0 - clip_epsilon,
        mx.ones_like(ratios),
        mx.where(ratios > 1.0 + clip_epsilon, mx.ones_like(ratios), mx.zeros_like(ratios)),
    )
    clip_fraction = mx.sum(clipped * vpf) / vpf_denom
    approx_kl = mx.sum((old_log_probs - new_log_probs) * vpf) / vpf_denom

    loss = (
        policy_loss + value_loss - entropy_coef * entropy
        + mana_draw_bce_weight * mana_draw_bce
    )

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


# ---------------------------------------------------------------------------
# Helpers for the C3-specific minibatch loop (parallel-array lockstep).
# ---------------------------------------------------------------------------

def _flatten_parallel_arrays(
    offline_replay_batch: OfflineReplayBatch,
    *,
    ppo_batch: RustPPOBatch,
    padded_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Flatten the C2 parallel arrays to ``(steps*env_count,)`` step-major
    (matching ``RustPPOBatch.flatten()`` which is row-major C order
    ``step*env_count + env``).

    ``is_padded`` is derived from ``action_mask.sum(-1) == 0`` (the C2 bridge
    sets padded rows' action_mask to zeros; real rows always have >=1 legal
    candidate -- end_turn is always offered, and a terminal/surrender row's
    pre_state still has legal actions so its mask is non-zero) unless an
    explicit ``padded_mask`` override is given.
    """
    is_mana_draw = np.asarray(offline_replay_batch.is_mana_draw, dtype=np.bool_)
    mana_draw_legal = np.asarray(offline_replay_batch.mana_draw_legal, dtype=np.bool_)
    target_tcodes = np.asarray(offline_replay_batch.target_tcodes, dtype=np.int64)
    steps, env_count = is_mana_draw.shape

    if padded_mask is not None:
        is_padded = np.asarray(padded_mask, dtype=np.bool_)
    else:
        am = np.asarray(ppo_batch.action_mask, dtype=np.float32)
        # am shape (steps, env_count, 601)
        is_padded = am.sum(axis=-1) == 0

    return {
        "is_mana_draw": is_mana_draw.reshape((steps * env_count,)).astype(np.float32),
        "mana_draw_legal": mana_draw_legal.reshape((steps * env_count,)).astype(np.bool_),
        "target_tcodes": target_tcodes.reshape((steps * env_count,)).astype(np.int64),
        "is_padded": is_padded.reshape((steps * env_count,)).astype(np.bool_),
    }


def _resolve_model(
    checkpoint_path_or_model: Any,
    *,
    hidden_dim: int,
) -> tuple[Any, str]:
    """Resolve the ``checkpoint_path_or_model`` arg to a V5 model.

    If a model instance (has ``parameters``) is passed, use it directly (no
    warm-start -- e.g. test 10 random init). If a path is passed, gate on file
    existence + mlx, then warm-start via ``load_v4_max_into_v5`` (Q3 PARTIAL,
    strict=False).

    Returns ``(model, source)`` where ``source`` is ``'model'`` or ``'npz'``.
    Raises ``FileNotFoundError`` if a path is given but the file is absent
    (caller skip-gates on this).
    """
    from .v5_policy import V5ActionConditionedPolicy

    if hasattr(checkpoint_path_or_model, "parameters") and hasattr(
        checkpoint_path_or_model, "encode_state"
    ):
        return checkpoint_path_or_model, "model"

    path = Path(str(checkpoint_path_or_model))
    if not path.is_file():
        raise FileNotFoundError(
            f"V5 policy checkpoint not found: {path} (skip-gated, A2 pattern)"
        )

    from .warm_start_v5 import load_v4_max_into_v5

    model = V5ActionConditionedPolicy(hidden_dim=hidden_dim)
    # Q3 PARTIAL warm-start (strict=False): only mapped params are overwritten.
    load_v4_max_into_v5(model, npz_path=str(path))
    return model, "npz"


# ---------------------------------------------------------------------------
# (A3) Trainer.
# ---------------------------------------------------------------------------

def train_awac_crr_replay(
    checkpoint_path_or_model: Any,
    offline_replay_batch: OfflineReplayBatch,
    *,
    epochs: int = 2,
    minibatch_size: int = 256,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    lambda_awac: float = 1.0,
    awac_clamp: float = 4.0,
    mana_draw_bce_weight: float = DEFAULT_MANA_DRAW_BCE_WEIGHT,
    max_grad_norm: Optional[float] = None,
    lr: float = 1e-3,
    freeze_faithful: bool = True,
    train_value_head: bool = True,
    seed: int = 0,
    hidden_dim: int = 128,
    save_checkpoint_path: Optional[Union[str, Path]] = None,
    padded_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run the Hybrid AWAC x PPO-clip offline-PPO replay on a C2
    ``OfflineReplayBatch``.

    Mirrors ``_train_rust_ppo_minibatch_with_evaluator`` (``rust_ppo.py:193``):
    minibatch loop, ``nn.value_and_grad``, ``_zero_frozen_grads`` (A2
    freeze_faithful byte-identity), ``mlx.optimizers.Adam.update``, ``mx.eval``,
    ``_clip_grads``, full-batch eval before/after.

    CRITICAL INTEGRATION DETAIL: the AWAC evaluator additionally needs the C2
    parallel arrays (``is_mana_draw``, ``mana_draw_legal``, ``target_tcodes``,
    ``is_padded``) that ``RustPPOBatch.flatten()`` does NOT carry. The trainer
    is a C3-SPECIFIC loop that slices BOTH the ``RustPPOBatch`` minibatch AND
    the parallel arrays in lockstep (``rust_ppo.py`` is READ-ONLY -- NOT
    modified to thread extra args).

    Skip-gate (A2 pattern): returns ``{status: 'skipped', reason, ...}`` (NO
    crash) when mlx is absent OR a path is given but the npz is absent. When a
    MODEL instance is passed, no warm-start gate applies (the model is used
    directly -- e.g. random-init tests).

    D-C3: ``train_value_head=True`` (default) -> ``value_head`` is TRAINABLE;
    FAITHFUL + SHAPE-COMPAT params frozen byte-identical (A2 freeze).
    """
    # ---- Skip-gate: mlx absent -> skip (NO crash, no mlx import at top). ----
    try:
        import mlx.core as mx  # noqa: F401
        import mlx.nn as nn  # noqa: F401
        import mlx.optimizers as optim  # noqa: F401
        _mlx_available = True
    except ImportError:
        _mlx_available = False

    if not _mlx_available:
        return {
            "status": "skipped",
            "reason": "mlx not available",
            "policy_loss": None,
            "value_loss": None,
            "mana_draw_bce": None,
            "entropy": None,
            "approx_kl": None,
            "clip_fraction": None,
            "num_updates": 0,
            "new_checkpoint_path": None,
        }

    # ---- Skip-gate: path given but npz absent -> skip (A2 pattern). ----------
    try:
        model, source = _resolve_model(checkpoint_path_or_model, hidden_dim=hidden_dim)
    except FileNotFoundError as exc:
        return {
            "status": "skipped",
            "reason": str(exc),
            "policy_loss": None,
            "value_loss": None,
            "mana_draw_bce": None,
            "entropy": None,
            "approx_kl": None,
            "clip_fraction": None,
            "num_updates": 0,
            "new_checkpoint_path": None,
        }

    # ---- GAE (D-C2): prepare_rust_ppo_batch on the C2 RustTransitionBatch. ----
    ppo_batch = prepare_rust_ppo_batch(
        offline_replay_batch.batch,
        gamma=0.99,
        gae_lambda=0.95,
        bootstrap_values=offline_replay_batch.bootstrap_values,
    )

    # ---- Flatten the C2 parallel arrays in lockstep with flatten(). ----
    par = _flatten_parallel_arrays(
        offline_replay_batch, ppo_batch=ppo_batch, padded_mask=padded_mask
    )

    # ---- A2 freeze (D-C3: value_head TRAINABLE). ----
    frozen_names = frozen_param_names(
        freeze_faithful=freeze_faithful, train_value_head=train_value_head
    )
    train_names = trainable_param_names(
        freeze_faithful=freeze_faithful, train_value_head=train_value_head
    )
    frozen_snapshot = snapshot_frozen_params(model, frozen_names)

    optimizer = optim.Adam(learning_rate=float(lr))
    rng = np.random.default_rng(seed)

    flat = ppo_batch.flatten()
    row_count = int(flat["actions"].shape[0])
    if row_count <= 0:
        return {
            "status": "skipped",
            "reason": "empty offline replay batch (0 rows)",
            "policy_loss": None,
            "value_loss": None,
            "mana_draw_bce": None,
            "entropy": None,
            "approx_kl": None,
            "clip_fraction": None,
            "num_updates": 0,
            "new_checkpoint_path": None,
        }

    epochs = int(epochs)
    minibatch_size = int(minibatch_size)
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")

    eval_kwargs = dict(
        clip_epsilon=clip_epsilon,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        lambda_awac=lambda_awac,
        awac_clamp=awac_clamp,
        mana_draw_bce_weight=mana_draw_bce_weight,
    )

    def _full_eval():
        return evaluate_awac_dense_batch(
            model,
            ppo_batch,
            is_mana_draw=mx.array(par["is_mana_draw"]),
            mana_draw_legal=mx.array(par["mana_draw_legal"]),
            target_tcodes=mx.array(par["target_tcodes"]),
            is_padded=mx.array(par["is_padded"]),
            **eval_kwargs,
        )

    before = _full_eval()
    mx.eval(
        before.loss,
        before.policy_loss,
        before.value_loss,
        before.entropy,
        before.clip_fraction,
        before.approx_kl,
    )

    # ---- C3-SPECIFIC minibatch loop (slices BOTH batch AND parallel arrays). ----
    current: list[Any] = [None, None]  # (mini RustPPOBatch, parallel slice dict)

    def loss_fn(model_arg: Any):
        mini, par_slice = current[0], current[1]
        if mini is None:
            raise RuntimeError("internal error: minibatch not set")
        evaluation = evaluate_awac_dense_batch(
            model_arg,
            mini,
            is_mana_draw=mx.array(par_slice["is_mana_draw"]),
            mana_draw_legal=mx.array(par_slice["mana_draw_legal"]),
            target_tcodes=mx.array(par_slice["target_tcodes"]),
            is_padded=mx.array(par_slice["is_padded"]),
            **eval_kwargs,
        )
        return evaluation.loss, {
            "policy_loss": evaluation.policy_loss,
            "value_loss": evaluation.value_loss,
            "entropy": evaluation.entropy,
            "clip_fraction": evaluation.clip_fraction,
            "approx_kl": evaluation.approx_kl,
        }

    value_and_grad = nn.value_and_grad(model, loss_fn)

    metric_values: Dict[str, list[float]] = {
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "clip_fraction": [],
        "approx_kl": [],
    }

    indices = np.arange(row_count, dtype=np.int64)

    def _slice_parallel(idx: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "is_mana_draw": par["is_mana_draw"][idx],
            "mana_draw_legal": par["mana_draw_legal"][idx],
            "target_tcodes": par["target_tcodes"][idx],
            "is_padded": par["is_padded"][idx],
        }

    for _epoch in range(epochs):
        rng.shuffle(indices)
        for start in range(0, row_count, minibatch_size):
            end = min(start + minibatch_size, row_count)
            idx = indices[start:end]
            if idx.size <= 0:
                continue
            mini = _take_flat_rows(
                ppo_batch, idx, flat=flat, legal_row_pack_backend="python"
            )
            current[0] = mini
            current[1] = _slice_parallel(idx)
            (loss_value, aux), grads = value_and_grad(model)
            grads = _zero_frozen_grads(grads, frozen_names)
            if max_grad_norm is not None:
                grads = _clip_grads(grads, float(max_grad_norm))
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            metric_values["loss"].append(float(loss_value.item()))
            for name, value in aux.items():
                metric_values[name].append(float(value.item()))

    current[0] = None
    current[1] = None

    after = _full_eval()
    mx.eval(
        after.loss,
        after.policy_loss,
        after.value_loss,
        after.entropy,
        after.clip_fraction,
        after.approx_kl,
    )

    # ---- A2 freeze byte-identity assertion (defensive). ----
    assert_frozen_preserved(model, frozen_snapshot)

    # ---- Numpy after-eval (populates mana_draw_bce -- not carried by
    # RustPPOEvaluation -- AND cross-checks the MLX evaluator against the
    # pure-numpy loss core on the post-training model). ----
    after_np = _numpy_full_eval(model, ppo_batch, par)

    # ---- Optionally write a checkpoint (same format as rust_trainer). ----
    new_checkpoint_path: Optional[str] = None
    if save_checkpoint_path is not None:
        from ai.train_v2.model_mlx import save_checkpoint

        save_checkpoint(
            str(save_checkpoint_path),
            model,
            optimizer,
            metadata={
                "kind": "awac_crr_replay",
                "lambda_awac": float(lambda_awac),
                "awac_clamp": float(awac_clamp),
                "epochs": int(epochs),
                "minibatch_size": int(minibatch_size),
                "freeze_faithful": bool(freeze_faithful),
                "train_value_head": bool(train_value_head),
            },
        )
        new_checkpoint_path = str(save_checkpoint_path)

    metrics: Dict[str, Any] = {
        "status": "trained",
        "source": source,
        "loss_before": float(before.loss.item()),
        "loss_after": float(after.loss.item()),
        "policy_loss": float(after.policy_loss.item()),
        "value_loss": float(after.value_loss.item()),
        "mana_draw_bce": after_np["mana_draw_bce"],
        "entropy": float(after.entropy.item()),
        "approx_kl": float(after.approx_kl.item()),
        "clip_fraction": float(after.clip_fraction.item()),
        "num_updates": len(metric_values["loss"]),
        "rows": row_count,
        "epochs": epochs,
        "minibatch_size": minibatch_size,
        "lambda_awac": float(lambda_awac),
        "awac_clamp": float(awac_clamp),
        "frozen_param_names": sorted(frozen_names),
        "trainable_param_names": sorted(train_names),
        "frozen_preserved": True,
        "new_checkpoint_path": new_checkpoint_path,
    }

    # mean per-step metrics (monitoring).
    for name in ("loss", "policy_loss", "value_loss", "entropy", "clip_fraction", "approx_kl"):
        if metric_values[name]:
            _assert_finite(name, metric_values[name])
            metrics[f"mean_{name}"] = float(np.mean(metric_values[name]))

    return metrics


def _numpy_full_eval(
    model: Any,
    ppo_batch: RustPPOBatch,
    par: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Run the model forward in NUMPY on the full flat PPO batch and call
    ``awac_crr_loss`` -- populates ``mana_draw_bce`` (not carried by
    ``RustPPOEvaluation``) and cross-checks the MLX evaluator against the
    pure-numpy loss core."""
    import mlx.core as mx

    flat = ppo_batch.flatten()
    obs = mx.array(flat["obs"])
    action_features = mx.array(flat["action_features"])
    _out = model(obs, action_features, mana_draw_legal=None)
    logits_np = np.asarray(_out[0], dtype=np.float32)
    values_np = np.asarray(_out[1], dtype=np.float32).reshape(-1)
    md_logit_np = np.asarray(_out[2], dtype=np.float32).reshape(-1)

    batch_fields = {
        "old_log_probs": flat["old_log_probs"],
        "advantages": flat["advantages"],
        "returns": flat["returns"],
        "actions": flat["actions"].astype(np.int64, copy=False),
        "action_mask": flat["action_mask"],
        "is_mana_draw": par["is_mana_draw"],
        "mana_draw_legal": par["mana_draw_legal"],
        "target_tcodes": par["target_tcodes"],
        "is_padded": par["is_padded"],
    }
    _total, m = awac_crr_loss(
        (logits_np, values_np, md_logit_np),
        batch_fields,
    )
    return m


def _mlx_to_float(x: Any) -> float:
    return float(x.item())


# ---------------------------------------------------------------------------
# (A4) Driver interface for C4 -- MONITORING-ONLY (NO promote decision).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AwacCrrMetrics:
    """Monitoring-only metrics from one AWAC/CRR replay iteration.

    B6 external-bench owns the promote signal; C4 owns the promote decision.
    This object carries NO ``promote`` / ``score`` field (by design).
    """

    status: str
    policy_loss: Optional[float]
    value_loss: Optional[float]
    mana_draw_bce: Optional[float]
    entropy: Optional[float]
    approx_kl: Optional[float]
    clip_fraction: Optional[float]
    num_updates: int
    new_checkpoint_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class AwacCrrReplay:
    """C4-facing driver for one AWAC/CRR offline-PPO replay iteration.

    ``run`` is MONITORING-ONLY: it returns ``AwacCrrMetrics`` with no
    promote/score field. B6 (external bench) is the promote signal; C4 owns
    promotion. Skip-gated (mlx absent / npz absent -> ``status='skipped'``).
    """

    def run(
        self,
        offline_replay_batch: OfflineReplayBatch,
        *,
        checkpoint_path: Any,
        epochs: int = 2,
        minibatch_size: int = 256,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        lambda_awac: float = 1.0,
        awac_clamp: float = 4.0,
        mana_draw_bce_weight: float = DEFAULT_MANA_DRAW_BCE_WEIGHT,
        max_grad_norm: Optional[float] = None,
        lr: float = 1e-3,
        freeze_faithful: bool = True,
        train_value_head: bool = True,
        seed: int = 0,
        hidden_dim: int = 128,
        save_checkpoint_path: Optional[Union[str, Path]] = None,
        padded_mask: Optional[np.ndarray] = None,
    ) -> AwacCrrMetrics:
        raw = train_awac_crr_replay(
            checkpoint_path,
            offline_replay_batch,
            epochs=epochs,
            minibatch_size=minibatch_size,
            clip_epsilon=clip_epsilon,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            lambda_awac=lambda_awac,
            awac_clamp=awac_clamp,
            mana_draw_bce_weight=mana_draw_bce_weight,
            max_grad_norm=max_grad_norm,
            lr=lr,
            freeze_faithful=freeze_faithful,
            train_value_head=train_value_head,
            seed=seed,
            hidden_dim=hidden_dim,
            save_checkpoint_path=save_checkpoint_path,
            padded_mask=padded_mask,
        )
        extra = {
            k: v
            for k, v in raw.items()
            if k
            not in {
                "status",
                "policy_loss",
                "value_loss",
                "mana_draw_bce",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "num_updates",
                "new_checkpoint_path",
            }
        }
        return AwacCrrMetrics(
            status=raw.get("status", "skipped"),
            policy_loss=raw.get("policy_loss"),
            value_loss=raw.get("value_loss"),
            mana_draw_bce=raw.get("mana_draw_bce"),
            entropy=raw.get("entropy"),
            approx_kl=raw.get("approx_kl"),
            clip_fraction=raw.get("clip_fraction"),
            num_updates=int(raw.get("num_updates", 0)),
            new_checkpoint_path=raw.get("new_checkpoint_path"),
            extra=extra,
        )


__all__ = [
    "awac_crr_loss",
    "awac_weight",
    "evaluate_awac_dense_batch",
    "train_awac_crr_replay",
    "AwacCrrReplay",
    "AwacCrrMetrics",
]
"""Block A component A2 — BC (behavior-cloning) training loop.

Sequence: load ``V5ActionConditionedPolicy`` (``v5_policy.py:29``) ->
``warm_start_v5.load_v4_max_into_v5`` (``warm_start_v5.py:176``, Block 0
component 4 — consumed READ-ONLY) -> BC fine-tune on the A1 ``BCTransition``
dataset (``bc_dataset.py`` — A1, DONE).

Loss = (a) SUPERVISED CROSS-ENTROPY on the 601-candidate logits MASKED TO LEGAL
(target = ``BCTransition.target_tcode``; illegal candidates masked out before
softmax/CE; CE over legal candidates only) PLUS (b) BCE on the mana_draw head
(target = 1.0 if ``is_mana_draw`` else 0.0; BCE only on rows where
``mana_draw_legal`` is True — mirroring ``mana_draw_head_v5.select_includes_mana_draw``
at ``mana_draw_head_v5.py:116``, where the head only matters when mana_draw is
legal). Total = candidate_CE + mana_draw_BCE (plain sum, weight 1.0 — D-A9
spec-literal PLAIN CE BC; AWAC/CRR reserved for Block C, NOT here). Optional
value-head fine-tune is OFF by default (BC targets the policy head + mana_draw
head only; ``train_value_head=False``).

freeze_faithful=True (DEFAULT): the Q3 FAITHFUL layers
(``base_encoder.layers.0`` + ``action_encoder`` — ``warm_start_v5.py:67-73``)
are frozen so BC does NOT destroy the warm start. Additionally the
SHAPE-COMPAT-DISCONNECTED ``state_fuser.layers.2`` (loaded from V4 but NOT in
the spec's BC "move set") and ``value_head`` (value fine-tune OFF) are frozen.
Only ``candidate_scorer`` + ``mana_draw_head`` + the FRESH layers
(``global_encoder.layers.0`` / ``private_encoder.layers.0`` /
``history_encoder.layers.0`` / ``state_fuser.layers.0``) move during BC.
freeze_faithful=False (ablation) unfreezes the FAITHFUL + ``state_fuser.layers.2``
layers (``value_head`` stays controlled by ``train_value_head``).

Freeze is enforced by ZEROING the frozen params' gradients each step before
``optimizer.update`` (the spec's "zero grads" idiom; MLX
``optim.Adam.update`` with a zero gradient produces an EXACTLY zero update, so
the frozen params are BYTE-IDENTICAL across all BC steps — verified in the
companion tests). This is simpler + more robust than rebuilding a filtered
param subtree (which breaks ``nn.utils.tree_unflatten`` structure matching).

Skip-gate: if the V4-Max npz is absent (``resolve_v4_max_npz_path`` raises), the
warm-start step is SKIPPED and BC runs on a fresh-init policy (documented) — no
crash. MLX itself is present (do NOT gate on mlx import; gate on npz only).

Emits a BC-seed checkpoint via ``ai.train_v2.model_mlx.save_checkpoint`` (SAME
format as ``rust_trainer._save_checkpoint`` at ``rust_trainer.py:838``, which
calls ``save_checkpoint(path, model, optimizer, metadata)``). PPO phase A4
resumes via ``model_mlx.load_checkpoint(path, policy)`` — the SAME loader, so
the BC checkpoint round-trips back into ``V5ActionConditionedPolicy``.

FROZEN-CLASSIC GUARD: ``classic_*`` / ``classic_rl_env`` / ``reward_v5.py``
byte-locked (read-only); ``v5_trace.py`` NOT imported; ``core/state.py`` NOT
modified; no TrainV3.5 import into prod. ``warm_start_v5.py`` consumed
READ-ONLY (NOT modified).
"""
from __future__ import annotations

import logging
from dataclasses import is_dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .bc_dataset import BCTransition
from .contracts import ACTION_FEATURE_DIM, MAX_CANDIDATE_ACTIONS, OBS_V5_DIM
from .v5_policy import V5ActionConditionedPolicy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Param-name sets (derived from the V5 policy param tree — confirmed via
# nn.utils.tree_flatten on a real V5ActionConditionedPolicy; see
# ``_audit_param_names``). These are the EXACT dotted names tree_flatten emits.
# ---------------------------------------------------------------------------
# Q3 FAITHFUL layers (warm_start_v5.py:67-73) — frozen by default so BC does
# not destroy the warm start.
FAITHFUL_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "base_encoder.layers.0.weight",
        "base_encoder.layers.0.bias",
        "action_encoder.weight",
        "action_encoder.bias",
    }
)

# SHAPE-COMPAT-DISCONNECTED layer NOT in the spec's BC "move set"
# (warm_start_v5.py:70 — loaded from V4 state_encoder.layers.2 but the fused
# input differs). Frozen by default; unfrozen only under freeze_faithful=False.
_SHAPE_COMPAT_FROZEN_NAMES: frozenset[str] = frozenset(
    {
        "state_fuser.layers.2.weight",
        "state_fuser.layers.2.bias",
    }
)

# value_head — frozen unless train_value_head=True (value fine-tune OFF by
# default; BC targets the policy + mana_draw heads only).
_VALUE_HEAD_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "value_head.weight",
        "value_head.bias",
    }
)

# The BC "move set" (spec A2): candidate_scorer + mana_draw_head + FRESH layers
# (global/private/history encoders + state_fuser.layers.0). These move when
# freeze_faithful=True (default).
TRAINABLE_BC_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "candidate_scorer.weight",
        "candidate_scorer.bias",
        "state_action_query.weight",
        "state_action_query.bias",
        "state_action_gate.weight",
        "mana_draw_head.weight",
        "mana_draw_head.bias",
        "global_encoder.layers.0.weight",
        "global_encoder.layers.0.bias",
        "private_encoder.layers.0.weight",
        "private_encoder.layers.0.bias",
        "history_encoder.layers.0.weight",
        "history_encoder.layers.0.bias",
        "state_fuser.layers.0.weight",
        "state_fuser.layers.0.bias",
    }
)

# Numerically-stable -inf equivalent for illegal-candidate masking.
# exp(-1e9) underflows to 0 so softmax over legal candidates is UNAFFECTED, and
# log_softmax stays finite even for all-False legal-mask rows (a real -inf would
# produce NaN via logsumexp(all -inf)). The BC target is always a LEGAL
# candidate (A1 guarantees target_tcode is in legal_mask), so the CE at the
# target is computed over a valid distribution. This is the standard masking
# idiom and is behaviorally identical to -inf for the legal candidates.
_NEG_INF_STABLE = -1e9

# Default plain-sum weighting (D-A9 spec-literal: plain CE BC, no fancy
# weighting). Exposed as a knob so Block C can re-tune; A2 uses 1.0.
DEFAULT_MANA_DRAW_BCE_WEIGHT = 1.0


def frozen_param_names(
    *, freeze_faithful: bool, train_value_head: bool
) -> frozenset[str]:
    """Return the set of dotted param names that must NOT move during BC.

    ``freeze_faithful=True`` (default): FAITHFUL (base_encoder.layers.0 +
    action_encoder) + state_fuser.layers.2 are frozen. ``value_head`` is frozen
    unless ``train_value_head=True``.

    ``freeze_faithful=False`` (ablation): FAITHFUL + state_fuser.layers.2 are
    unfrozen; only ``value_head`` remains frozen when
    ``train_value_head=False`` (it has no loss term, so it would not move
    anyway, but it is kept out of the trainable set for clarity).
    """
    frozen: set[str] = set()
    if freeze_faithful:
        frozen |= FAITHFUL_PARAM_NAMES
        frozen |= _SHAPE_COMPAT_FROZEN_NAMES
    if not train_value_head:
        frozen |= _VALUE_HEAD_PARAM_NAMES
    return frozenset(frozen)


def trainable_param_names(
    *, freeze_faithful: bool, train_value_head: bool
) -> frozenset[str]:
    """Complement of ``frozen_param_names`` over the full V5 param tree."""
    all_names = FAITHFUL_PARAM_NAMES | _SHAPE_COMPAT_FROZEN_NAMES | _VALUE_HEAD_PARAM_NAMES | TRAINABLE_BC_PARAM_NAMES
    return all_names - frozen_param_names(
        freeze_faithful=freeze_faithful, train_value_head=train_value_head
    )


# ---------------------------------------------------------------------------
# Batching — convert BCTransition tuples (A1) into MLX tensors.
# ---------------------------------------------------------------------------
def _bc_field(t: Any, name: str) -> Any:
    """Read ``name`` from a BCTransition dataclass OR a dict (test fixtures)."""
    if is_dataclass(t) and not isinstance(t, type):
        return getattr(t, name)
    if isinstance(t, dict):
        return t[name]
    return getattr(t, name)


def collate_bc_batch(transitions: Sequence[Any]) -> Dict[str, "Any"]:
    """Stack a sequence of ``BCTransition`` (or dict) into MLX batch tensors.

    Returns a dict with keys:
      ``obs``               — (B, OBS_V5_DIM) float32
      ``action_features``   — (B, 601, ACTION_FEATURE_DIM) float32
      ``target_tcode``      — (B,) int32, -1 where target is None (mana_draw /
                              terminal rows — these do not contribute to the
                              candidate-CE term)
      ``legal_mask``        — (B, 601) bool
      ``is_mana_draw``      — (B,) float32 (1.0 / 0.0)
      ``mana_draw_legal``   — (B,) bool
      ``terminal``          — (B,) bool (carried for provenance; not in the loss)
    """
    import mlx.core as mx

    n = len(transitions)
    if n == 0:
        raise ValueError("collate_bc_batch: empty transition list")

    obs_list: list[np.ndarray] = []
    af_list: list[np.ndarray] = []
    target_list: list[int] = []
    legal_mask_list: list[np.ndarray] = []
    is_md_list: list[float] = []
    md_legal_list: list[bool] = []
    terminal_list: list[bool] = []

    for t in transitions:
        obs_arr = np.asarray(_bc_field(t, "obs"), dtype=np.float32)
        if obs_arr.shape != (OBS_V5_DIM,):
            raise ValueError(
                f"collate_bc_batch: obs shape {obs_arr.shape} != ({OBS_V5_DIM},)"
            )
        obs_list.append(obs_arr)
        af_arr = np.asarray(_bc_field(t, "action_features"), dtype=np.float32)
        if af_arr.shape != (MAX_CANDIDATE_ACTIONS, ACTION_FEATURE_DIM):
            raise ValueError(
                f"collate_bc_batch: action_features shape {af_arr.shape} != "
                f"({MAX_CANDIDATE_ACTIONS}, {ACTION_FEATURE_DIM})"
            )
        af_list.append(af_arr)
        tgt = _bc_field(t, "target_tcode")
        target_list.append(-1 if tgt is None else int(tgt))
        lm = np.asarray(_bc_field(t, "legal_mask"))
        if lm.shape != (MAX_CANDIDATE_ACTIONS,):
            raise ValueError(
                f"collate_bc_batch: legal_mask shape {lm.shape} != "
                f"({MAX_CANDIDATE_ACTIONS},)"
            )
        legal_mask_list.append(lm.astype(bool))
        is_md_list.append(1.0 if bool(_bc_field(t, "is_mana_draw")) else 0.0)
        md_legal_list.append(bool(_bc_field(t, "mana_draw_legal")))
        terminal_list.append(bool(_bc_field(t, "terminal")))

    return {
        "obs": mx.array(np.stack(obs_list)),
        "action_features": mx.array(np.stack(af_list)),
        "target_tcode": mx.array(np.array(target_list, dtype=np.int32)),
        "legal_mask": mx.array(np.stack(legal_mask_list)),
        "is_mana_draw": mx.array(np.array(is_md_list, dtype=np.float32)),
        "mana_draw_legal": mx.array(np.array(md_legal_list, dtype=bool)),
        "terminal": mx.array(np.array(terminal_list, dtype=bool)),
    }


# ---------------------------------------------------------------------------
# BC loss — masked candidate CE + mana_draw BCE (plain sum, D-A9).
# ---------------------------------------------------------------------------
def compute_bc_loss(
    model: V5ActionConditionedPolicy,
    batch: Dict[str, "Any"],
    *,
    mana_draw_bce_weight: float = DEFAULT_MANA_DRAW_BCE_WEIGHT,
) -> "Any":
    """Compute the BC loss for one batch.

    Loss = candidate_CE + ``mana_draw_bce_weight`` * mana_draw_BCE (plain sum,
    D-A9). The candidate CE is the mean negative log-likelihood of
    ``target_tcode`` over the LEGAL-candidate softmax (illegal candidates
    masked to ``_NEG_INF_STABLE`` before softmax); rows where ``target_tcode``
    is None (mana_draw / terminal) do NOT contribute. The mana_draw BCE is the
    mean binary cross-entropy of the raw mana_draw-head logit (sigmoid) against
    the ``is_mana_draw`` target, computed ONLY on rows where
    ``mana_draw_legal`` is True (mirrors
    ``mana_draw_head_v5.select_includes_mana_draw:116`` — the head only matters
    when mana_draw is legal).

    The forward pass is called with ``mana_draw_legal=None`` so the
    mana_draw head returns its RAW logit (BCE on a -inf-gated logit would be
    NaN); the BCE legality mask is applied in the loss, not the forward.

    Returns a scalar ``mx.array`` (the total loss). Callers extract per-term
    metrics via ``compute_bc_loss_terms``.
    """
    import mlx.core as mx

    total, _ = compute_bc_loss_terms(
        model, batch, mana_draw_bce_weight=mana_draw_bce_weight
    )
    return total


def compute_bc_loss_terms(
    model: V5ActionConditionedPolicy,
    batch: Dict[str, "Any"],
    *,
    mana_draw_bce_weight: float = DEFAULT_MANA_DRAW_BCE_WEIGHT,
) -> tuple["Any", Dict[str, "Any"]]:
    """Compute (total_loss, metrics) for one BC batch.

    ``metrics`` keys: ``candidate_ce``, ``mana_draw_bce``, ``total``,
    ``valid_rows`` (count of rows contributing to candidate CE),
    ``mana_draw_legal_rows`` (count of rows contributing to mana_draw BCE).
    """
    import mlx.core as mx

    obs = batch["obs"]
    action_features = batch["action_features"]
    target = batch["target_tcode"]  # (B,) int32, -1 for None rows
    legal_mask = batch["legal_mask"]  # (B, 601) bool
    is_mana_draw = batch["is_mana_draw"]  # (B,) float32
    mana_draw_legal = batch["mana_draw_legal"]  # (B,) bool

    # Forward: raw mana_draw_logit (NOT -inf-gated) so BCE is finite on all
    # rows; the BCE legality mask is applied in the loss.
    candidate_logits, _value, mana_draw_logit = model(
        obs, action_features, mana_draw_legal=None
    )

    # ---- (a) Masked candidate cross-entropy -------------------------------
    # valid = target >= 0 : rows with a real 601 target (normal / natural-
    # lethal). mana_draw rows (target=None -> -1) and terminal rows do not
    # contribute.
    valid = target >= 0  # (B,) bool
    # Mask illegal candidates to a large negative (numerically-stable -inf).
    neg = mx.full(candidate_logits.shape, _NEG_INF_STABLE)
    masked_logits = mx.where(legal_mask, candidate_logits, neg)
    # log_softmax via logsumexp (stable).
    log_probs = masked_logits - mx.logsumexp(masked_logits, axis=-1, keepdims=True)
    # Clamp target to >= 0 for the gather (None rows -> 0, masked out by valid).
    target_clamped = mx.maximum(target, mx.array(0, dtype=target.dtype))
    target_logp = mx.take_along_axis(
        log_probs, target_clamped[:, None], axis=1
    )[:, 0]  # (B,)
    ce_per_row = -target_logp  # (B,)
    valid_count = mx.sum(valid)
    # mean over valid rows (guard divide-by-zero with max(count, 1)).
    candidate_ce = mx.sum(ce_per_row * valid) / mx.maximum(
        valid_count, mx.array(1, dtype=valid_count.dtype)
    )

    # ---- (b) mana_draw BCE (only on mana_draw-legal rows) -----------------
    md_p = mx.clip(mx.sigmoid(mana_draw_logit), 1e-7, 1.0 - 1e-7)
    bce_per_row = -(is_mana_draw * mx.log(md_p) + (1.0 - is_mana_draw) * mx.log(1.0 - md_p))
    md_legal_count = mx.sum(mana_draw_legal)
    mana_draw_bce = mx.sum(bce_per_row * mana_draw_legal) / mx.maximum(
        md_legal_count, mx.array(1, dtype=md_legal_count.dtype)
    )

    total = candidate_ce + mx.array(mana_draw_bce_weight, dtype=mana_draw_bce.dtype) * mana_draw_bce

    metrics = {
        "candidate_ce": candidate_ce,
        "mana_draw_bce": mana_draw_bce,
        "total": total,
        "valid_rows": valid_count,
        "mana_draw_legal_rows": md_legal_count,
    }
    return total, metrics


# ---------------------------------------------------------------------------
# Freeze enforcement — zero frozen params' grads each step (byte-identical).
# ---------------------------------------------------------------------------
def _zero_frozen_grads(grads: Any, frozen_names: frozenset[str]) -> Any:
    """Return a grad tree with the frozen params' gradients replaced by zeros.

    MLX ``optim.Adam.update`` with a zero gradient produces an EXACTLY zero
    update (m=beta1*m+(1-beta1)*0 with m0=0 -> 0; v likewise; the update is
    lr * 0 / (sqrt(0)+eps) = 0), so frozen params are BYTE-IDENTICAL across
    all steps. This is the spec's documented "zero grads" idiom and is simpler
    + more robust than rebuilding a filtered param subtree (which breaks
    ``tree_unflatten`` structure matching).
    """
    import mlx.core as mx
    import mlx.nn as nn

    flat = nn.utils.tree_flatten(grads)  # list[(name, val)]
    new_flat = [
        (name, mx.zeros_like(val) if name in frozen_names else val)
        for name, val in flat
    ]
    return nn.utils.tree_unflatten(new_flat)


def snapshot_frozen_params(
    policy: V5ActionConditionedPolicy, frozen_names: frozenset[str]
) -> Dict[str, np.ndarray]:
    """Snapshot the frozen params as numpy arrays (for the byte-identity test).

    The zero-grad idiom already guarantees byte-identity; this snapshot is a
    defensive assertion aid so tests can verify frozen params did not move
    without relying on optimizer internals.
    """
    import mlx.nn as nn

    flat = nn.utils.tree_flatten(policy.trainable_parameters())
    return {
        name: np.array(val) for name, val in flat if name in frozen_names
    }


def assert_frozen_preserved(
    policy: V5ActionConditionedPolicy,
    snapshot: Dict[str, np.ndarray],
) -> None:
    """Assert every frozen param in ``snapshot`` is byte-identical now."""
    import mlx.nn as nn

    flat = dict(nn.utils.tree_flatten(policy.trainable_parameters()))
    for name, before in snapshot.items():
        after = np.array(flat[name])
        if not np.array_equal(before, after):
            raise AssertionError(
                f"Frozen param {name!r} changed during BC: max abs diff = "
                f"{float(np.max(np.abs(before - after)))}"
            )


# ---------------------------------------------------------------------------
# Core BC training loop.
# ---------------------------------------------------------------------------
def train_bc(
    policy: V5ActionConditionedPolicy,
    dataset: Sequence[Any],
    *,
    freeze_faithful: bool = True,
    train_value_head: bool = False,
    learning_rate: float = 1e-3,
    steps: int = 20,
    batch_size: Optional[int] = None,
    mana_draw_bce_weight: float = DEFAULT_MANA_DRAW_BCE_WEIGHT,
    optimizer: Any | None = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """BC fine-tune a ``V5ActionConditionedPolicy`` on ``BCTransition`` tuples.

    Args:
        policy: a ``V5ActionConditionedPolicy`` (optionally warm-started).
        dataset: a sequence of ``BCTransition`` (A1) or dicts with the same
            fields. Re-used each ``step`` (full-batch by default, or
            minibatched when ``batch_size`` is set).
        freeze_faithful: freeze the FAITHFUL + ``state_fuser.layers.2`` layers
            (default True — preserve the warm start).
        train_value_head: also train ``value_head`` (default False — value
            fine-tune is OFF in A2; BC targets the policy + mana_draw heads).
        learning_rate: Adam learning rate.
        steps: number of BC gradient steps.
        batch_size: minibatch size (None = full-batch each step).
        mana_draw_bce_weight: scalar weight on the mana_draw BCE term (default
            1.0 = plain sum, D-A9 spec-literal).
        optimizer: an MLX optimizer (default ``mlx.optimizers.Adam``); pass a
            custom optimizer to resume Adam state.
        rng: numpy Generator for shuffling (default a fresh seeded one).

    Returns:
        A report dict with keys: ``losses`` (per-step total loss),
        ``candidate_ce`` / ``mana_draw_bce`` (per-step terms), ``step_count``,
        ``frozen_param_names``, ``trainable_param_names``, ``frozen_snapshot``
        (pre-BC frozen param values for byte-identity verification), and
        ``config`` (the effective training config).
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    if len(dataset) == 0:
        raise ValueError("train_bc: empty dataset")
    if steps < 1:
        raise ValueError("train_bc: steps must be >= 1")

    frozen_names = frozen_param_names(
        freeze_faithful=freeze_faithful, train_value_head=train_value_head
    )
    train_names = trainable_param_names(
        freeze_faithful=freeze_faithful, train_value_head=train_value_head
    )

    # Snapshot frozen params (pre-BC) for byte-identity verification.
    frozen_snapshot = snapshot_frozen_params(policy, frozen_names)

    if optimizer is None:
        optimizer = optim.Adam(learning_rate=float(learning_rate))

    rng = rng or np.random.default_rng(0)

    # Pre-collate the full dataset once (full-batch or minibatch indices into
    # the collated tensors — avoids re-stacking each step).
    full_batch = collate_bc_batch(dataset)
    n = len(dataset)
    bs = batch_size if batch_size is not None else n
    bs = max(1, min(bs, n))

    def _slice_batch(batch: Dict[str, Any], idx: np.ndarray) -> Dict[str, Any]:
        out = {}
        for k, v in batch.items():
            arr = np.array(v)
            out[k] = mx.array(arr[idx])
        return out

    # Loss fn for value_and_grad: fn(model, batch) -> scalar loss.
    def _loss_fn(model_arg, batch_arg):
        total, _ = compute_bc_loss_terms(
            model_arg, batch_arg, mana_draw_bce_weight=mana_draw_bce_weight
        )
        return total

    value_and_grad = nn.value_and_grad(policy, _loss_fn)

    losses: list[float] = []
    ce_terms: list[float] = []
    bce_terms: list[float] = []

    for step in range(steps):
        if bs < n:
            idx = rng.choice(n, size=bs, replace=False)
        else:
            idx = np.arange(n)
        batch = _slice_batch(full_batch, idx) if bs < n else full_batch

        loss, grads = value_and_grad(policy, batch)
        # Freeze: zero the frozen params' gradients before the update.
        grads = _zero_frozen_grads(grads, frozen_names)
        optimizer.update(policy, grads)
        mx.eval(policy.parameters(), optimizer.state)

        # Per-term metrics (re-evaluate on the SAME batch for reporting; cheap
        # and avoids holding extra graph outputs through the grad path).
        _, metrics = compute_bc_loss_terms(
            policy, batch, mana_draw_bce_weight=mana_draw_bce_weight
        )
        losses.append(float(loss))
        ce_terms.append(float(metrics["candidate_ce"]))
        bce_terms.append(float(metrics["mana_draw_bce"]))

    return {
        "losses": losses,
        "candidate_ce": ce_terms,
        "mana_draw_bce": bce_terms,
        "step_count": steps,
        "frozen_param_names": sorted(frozen_names),
        "trainable_param_names": sorted(train_names),
        "frozen_snapshot": frozen_snapshot,
        "config": {
            "freeze_faithful": freeze_faithful,
            "train_value_head": train_value_head,
            "learning_rate": float(learning_rate),
            "steps": int(steps),
            "batch_size": int(bs),
            "mana_draw_bce_weight": float(mana_draw_bce_weight),
            "dataset_size": int(n),
        },
    }


# ---------------------------------------------------------------------------
# Full BC pipeline: load -> warm-start (skip-gated) -> train -> save checkpoint.
# ---------------------------------------------------------------------------
def prepare_bc_policy(
    npz_path: Optional[str] = None,
    *,
    freeze_faithful: bool = True,
) -> tuple[V5ActionConditionedPolicy, Dict[str, Any]]:
    """Load a fresh ``V5ActionConditionedPolicy`` and warm-start it from V4-Max.

    Skip-gate: if the V4-Max npz is absent (``resolve_v4_max_npz_path`` raises
    ``RuntimeError``), the warm-start is SKIPPED and BC will run on a fresh-init
    policy. The returned ``warm_report`` carries ``warm_started=False`` and the
    skip reason so callers (and the checkpoint metadata) can record it. No crash.

    Returns ``(policy, warm_report)``.
    """
    from .warm_start_v5 import load_v4_max_into_v5, resolve_v4_max_npz_path

    policy = V5ActionConditionedPolicy()
    try:
        resolved = resolve_v4_max_npz_path(npz_path)
    except RuntimeError as exc:
        logger.warning(
            "prepare_bc_policy: V4-Max npz unavailable — skipping warm-start; "
            "BC will run on a fresh-init policy. Reason: %s", exc
        )
        return policy, {
            "warm_started": False,
            "skip_reason": str(exc),
            "npz_path": None,
            "transferred": [],
            "dropped": [],
            "fresh": [],
        }

    report = load_v4_max_into_v5(policy, npz_path=resolved)
    report["warm_started"] = True
    report["skip_reason"] = None
    return policy, report


def run_bc_training(
    dataset: Sequence[Any],
    *,
    npz_path: Optional[str] = None,
    freeze_faithful: bool = True,
    train_value_head: bool = False,
    learning_rate: float = 1e-3,
    steps: int = 20,
    batch_size: Optional[int] = None,
    mana_draw_bce_weight: float = DEFAULT_MANA_DRAW_BCE_WEIGHT,
    checkpoint_path: Optional[str] = None,
    checkpoint_step: Optional[int] = None,
    optimizer: Any | None = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[V5ActionConditionedPolicy, Dict[str, Any]]:
    """Full BC pipeline: load -> warm-start (skip-gated) -> train -> save.

    Args:
        dataset: sequence of ``BCTransition`` (A1) or dicts.
        npz_path: optional explicit V4-Max npz path (else resolved by
            ``resolve_v4_max_npz_path``; skip-gated if absent).
        checkpoint_path: if set, write a BC-seed checkpoint via
            ``ai.train_v2.model_mlx.save_checkpoint`` (consumable by PPO A4 via
            ``load_checkpoint``).
        checkpoint_step: the BC step number to record in the checkpoint
            metadata (defaults to ``steps``).

    Returns ``(policy, report)`` where ``report`` merges the warm-start report
    and the ``train_bc`` report, plus ``checkpoint_path`` / ``checkpoint_meta``
    when a checkpoint was written.
    """
    from ai.train_v2.model_mlx import save_checkpoint

    policy, warm_report = prepare_bc_policy(
        npz_path=npz_path, freeze_faithful=freeze_faithful
    )
    train_report = train_bc(
        policy,
        dataset,
        freeze_faithful=freeze_faithful,
        train_value_head=train_value_head,
        learning_rate=learning_rate,
        steps=steps,
        batch_size=batch_size,
        mana_draw_bce_weight=mana_draw_bce_weight,
        optimizer=optimizer,
        rng=rng,
    )

    report: Dict[str, Any] = {
        "warm_start": warm_report,
        "training": train_report,
        "checkpoint_path": None,
        "checkpoint_meta": None,
    }

    if checkpoint_path is not None:
        step = int(checkpoint_step if checkpoint_step is not None else steps)
        meta = {
            "kind": "bc_seed",
            "policy_kind": V5ActionConditionedPolicy.policy_kind,
            "obs_dim": int(OBS_V5_DIM),
            "action_feature_dim": int(ACTION_FEATURE_DIM),
            "max_candidate_actions": int(MAX_CANDIDATE_ACTIONS),
            "freeze_faithful": bool(freeze_faithful),
            "train_value_head": bool(train_value_head),
            "warm_started": bool(warm_report.get("warm_started", False)),
            "source_npz": warm_report.get("npz_path"),
            "step": step,
            "bc_steps": int(steps),
            "mana_draw_bce_weight": float(mana_draw_bce_weight),
            "frozen_param_names": train_report["frozen_param_names"],
            "trainable_param_names": train_report["trainable_param_names"],
        }
        save_checkpoint(checkpoint_path, policy, optimizer=optimizer, metadata=meta)
        report["checkpoint_path"] = str(checkpoint_path)
        report["checkpoint_meta"] = meta

    return policy, report


def _audit_param_names() -> None:
    """Dev/debug helper: print the V5 policy param tree (dotted names + shapes).

    Run via ``python -c "from train_v3.bc_train import _audit_param_names;
    _audit_param_names()"`` to confirm the FROZEN/TRAINABLE name sets above
    match the live param tree (regression guard against layout changes).
    """
    import mlx.nn as nn

    p = V5ActionConditionedPolicy()
    flat = nn.utils.tree_flatten(p.trainable_parameters())
    for name, val in flat:
        print(f"{name}\t{tuple(val.shape)}")


__all__ = [
    "DEFAULT_MANA_DRAW_BCE_WEIGHT",
    "FAITHFUL_PARAM_NAMES",
    "TRAINABLE_BC_PARAM_NAMES",
    "assert_frozen_preserved",
    "collate_bc_batch",
    "compute_bc_loss",
    "compute_bc_loss_terms",
    "frozen_param_names",
    "prepare_bc_policy",
    "run_bc_training",
    "snapshot_frozen_params",
    "train_bc",
    "trainable_param_names",
]
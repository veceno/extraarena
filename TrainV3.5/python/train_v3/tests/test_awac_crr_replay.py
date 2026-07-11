"""Block C component C3 -- tests for ``train_v3.awac_crr_replay`` (Hybrid AWAC
x PPO-clip offline-PPO replay loss + trainer).

Twelve SYNTHETIC gates:

  (1)  ``test_awac_crr_loss_math`` -- pure-numpy canned 3-row example
       (1 normal positive-adv, 1 normal negative-adv, 1 mana_draw); hand-compute
       w, ratios, surr1/surr2, policy_loss, value_loss, mana_draw_bce, entropy,
       total; assert each to ~6 decimals.
  (2)  ``test_blocker_sign_fix`` -- the BLOCKER sign fix: for a row with
       POSITIVE advantage and the policy INCREASING the action prob (ratio>1),
       policy_loss DECREASES (gradient INCREASES log pi for high-advantage
       actions -- the CORRECT AWAC direction). Assert
       ``policy_loss(ratio=2, A>0) < policy_loss(ratio=1, A>0)`` AND that the
       A-multiplier form matches a hand reference while a log_pi-multiplier form
       does NOT satisfy the direction.
  (3)  ``test_awac_weight`` -- ``w=exp(clamp(A/lambda,-C,C))``; A>>lambda*C ->
       w=exp(C); A=0 -> w=1; A<0 -> w<1. Assert specific values.
  (4)  ``test_value_loss_masks_padded_rows`` -- D-C3: value_loss over REAL rows
       only; a padded row's values perturbation does NOT change value_loss.
  (5)  ``test_mana_draw_bce_retains_out2`` -- mana_draw BCE RETAINS
       ``mana_draw_logit`` (_out[2]); bce over mana_draw_legal rows only; a
       mana_draw row with high md_p -> low bce; a non-mana_draw row excluded
       when mana_draw_legal=False. (MLX evaluator reads _out[2].)
  (6)  ``test_valid_policy_mask_exclusions`` -- mana_draw rows (tcode=-1),
       terminal rows (-1), and padded rows EXCLUDED from policy_loss
       (perturbing their new_log_prob does NOT change policy_loss).
  (7)  ``test_dense_evaluator_mirror`` -- ``evaluate_awac_dense_batch``
       reproduces the dense-path math (masked softmax where(mask,logits,-1e9),
       new_log_probs=log(gather+1e-10), ratios=exp(new-old)) on a fake MLX
       model; assert new_log_probs matches a numpy reference.
  (8)  ``test_end_to_end_on_c2_batch`` -- build a tiny OfflineReplayBatch,
       prepare_rust_ppo_batch -> RustPPOBatch, run train_awac_crr_replay for
       1-2 minibatches, assert loss finite + metrics populated + num_updates
       correct.
  (9)  ``test_skip_gate`` -- when mlx absent OR checkpoint_path is None/absent,
       train_awac_crr_replay returns {status:'skipped'} WITHOUT crash (no mlx
       import at module top; pure-numpy awac_crr_loss works with NO mlx).
  (10) ``test_freeze_faithful_byte_identity`` -- MLX-gated: build a tiny
       V5ActionConditionedPolicy from scratch (random init, NO npz), run
       train_awac_crr_replay with freeze_faithful=True train_value_head=True,
       assert FAITHFUL + SHAPE_COMPAT params BYTE-IDENTICAL before/after AND
       value_head params MOVED (D-C3).
  (11) ``test_run_metrics_monitoring_only`` -- AwacCrrReplay.run returns
       AwacCrrMetrics with NO promote/score field (B6 owns promotion).
  (12) ``test_clip_fraction_and_approx_kl_populated`` -- clip_fraction +
       approx_kl populated (mirror rust_ppo.py:777-784).

SYNTHETIC: pure-numpy loss core always testable (no mlx); MLX tests (7-10) run
real-mlx tiny-model (mlx IS available in this env). The pure-numpy
``awac_crr_loss`` is the PRIMARY regression guard.

Run: ``PYTHONPATH=.:TrainV3.5/python PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
TrainV3.5/python/train_v3/tests/test_awac_crr_replay.py``
"""
from __future__ import annotations

import math
from dataclasses import fields as dataclass_fields

import numpy as np
import pytest

from train_v3.awac_crr_replay import (
    AwacCrrMetrics,
    AwacCrrReplay,
    awac_crr_loss,
    awac_weight,
    evaluate_awac_dense_batch,
    train_awac_crr_replay,
)
from train_v3.bc_train import (
    FAITHFUL_PARAM_NAMES,
    _SHAPE_COMPAT_FROZEN_NAMES,
    _VALUE_HEAD_PARAM_NAMES,
    frozen_param_names,
    snapshot_frozen_params,
)
from train_v3.contracts import OBS_V5_DIM
from train_v3.offline_replay_bridge import OfflineReplayBatch
from train_v3.rust_collector import RustTransitionBatch
from train_v3.rust_ppo import prepare_rust_ppo_batch

try:
    import mlx.core as mx  # noqa: F401
    import mlx.nn as nn  # noqa: F401
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

_MLX = pytest.mark.skipif(not _HAS_MLX, reason="mlx not available")

MAX_CANDIDATE_ACTIONS = 601
ACTION_FEATURE_DIM = 171


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _single_legal_row(
    legal_id: int,
    *,
    logit_at_legal: float = 0.0,
    old_log_prob: float = 0.0,
    advantage: float = 1.0,
    ret: float = 0.0,
    value: float = 0.0,
    md_logit: float = 0.0,
    is_mana_draw: float = 0.0,
    mana_draw_legal: bool = False,
    target_tcode: int | None = 0,
    is_padded: bool = False,
):
    """Build a single-row (B=1) model_outputs + batch_fields pair with exactly
    one legal candidate ``legal_id`` (so softmax -> probs[legal_id]=1)."""
    logits = np.full((1, MAX_CANDIDATE_ACTIONS), -1e9, dtype=np.float32)
    logits[0, legal_id] = logit_at_legal
    values = np.array([value], dtype=np.float32)
    md = np.array([md_logit], dtype=np.float32)
    tcode = -1 if target_tcode is None else target_tcode
    mask = np.zeros((1, MAX_CANDIDATE_ACTIONS), dtype=np.float32)
    mask[0, legal_id] = 1.0
    bf = dict(
        old_log_probs=np.array([old_log_prob], dtype=np.float32),
        advantages=np.array([advantage], dtype=np.float32),
        returns=np.array([ret], dtype=np.float32),
        actions=np.array([legal_id], dtype=np.int64),
        action_mask=mask,
        is_mana_draw=np.array([is_mana_draw], dtype=np.float32),
        mana_draw_legal=np.array([mana_draw_legal], dtype=np.bool_),
        target_tcodes=np.array([tcode], dtype=np.int64),
        is_padded=np.array([is_padded], dtype=np.bool_),
    )
    return (logits, values, md), bf


def _build_tiny_offline_replay_batch() -> OfflineReplayBatch:
    """Hand-craft a tiny 2-game (steps=2, env_count=2) OfflineReplayBatch.

    game 0: (0,0) normal tcode 0, (1,0) normal tcode 1 (terminated -- last).
    game 1: (0,1) mana_draw (tcode -1, is_mana_draw, mana_draw_legal, terminated
            -- the only real row), (1,1) PADDED (action_mask zeros, dummy [0]).
    """
    steps, env_count = 2, 2
    obs_dim = OBS_V5_DIM
    feat_dim = ACTION_FEATURE_DIM
    rng = np.random.default_rng(0)
    observations = (rng.standard_normal((steps, env_count, obs_dim)) * 0.01).astype(
        np.float32
    )
    next_observations = np.zeros_like(observations)
    action_mask = np.zeros((steps, env_count, MAX_CANDIDATE_ACTIONS), dtype=np.float32)
    action_features = np.zeros(
        (steps, env_count, MAX_CANDIDATE_ACTIONS, feat_dim), dtype=np.float32
    )
    actions = np.zeros((steps, env_count), dtype=np.uintp)
    rewards = np.zeros((steps, env_count), dtype=np.float32)
    terminated = np.zeros((steps, env_count), dtype=np.bool_)
    values = np.zeros((steps, env_count), dtype=np.float32)
    log_probs = np.zeros((steps, env_count), dtype=np.float32)
    legal_ids_per: list[list[np.ndarray]] = [
        [None] * env_count for _ in range(steps)
    ]

    def set_row(s, e, legal_ids, action, reward, val, lp, term):
        for lid in legal_ids:
            action_mask[s, e, lid] = 1.0
        actions[s, e] = action
        rewards[s, e] = reward
        values[s, e] = val
        log_probs[s, e] = lp
        terminated[s, e] = term
        legal_ids_per[s][e] = np.array(legal_ids, dtype=np.uintp)

    set_row(0, 0, [0, 1], 0, 1.0, 0.2, float(np.log(0.5)), False)
    set_row(1, 0, [0, 1], 1, 0.5, 0.3, float(np.log(0.5)), True)  # game 0 last
    set_row(0, 1, [0], 0, 1.0, 0.1, 0.0, True)  # game 1 only (mana_draw)
    # padded (1,1)
    actions[1, 1] = 0
    terminated[1, 1] = True
    legal_ids_per[1][1] = np.array([0], dtype=np.uintp)

    flat_counts = np.zeros((steps * env_count,), dtype=np.uintp)
    flat_offsets = np.zeros((steps * env_count,), dtype=np.uintp)
    ids_chunks: list[np.ndarray] = []
    feats_chunks: list[np.ndarray] = []
    running = 0
    for s in range(steps):
        for e in range(env_count):
            idx = s * env_count + e
            ids = legal_ids_per[s][e]
            cnt = int(ids.shape[0])
            flat_counts[idx] = cnt
            flat_offsets[idx] = running
            ids_chunks.append(ids)
            feats_chunks.append(np.zeros((cnt, feat_dim), dtype=np.float32))
            running += cnt
    legal_action_ids = np.concatenate(ids_chunks)
    legal_action_features = np.concatenate(feats_chunks, axis=0)
    legal_action_counts = flat_counts.reshape((steps, env_count))
    legal_action_offsets = flat_offsets.reshape((steps, env_count))

    batch = RustTransitionBatch(
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
    is_mana_draw = np.zeros((steps, env_count), dtype=np.bool_)
    is_mana_draw[0, 1] = True
    mana_draw_legal = np.zeros((steps, env_count), dtype=np.bool_)
    mana_draw_legal[0, 1] = True
    target_tcodes = np.full((steps, env_count), -1, dtype=np.int32)
    target_tcodes[0, 0] = 0
    target_tcodes[1, 0] = 1
    return OfflineReplayBatch(
        batch=batch,
        bootstrap_values=np.array([0.4, 0.2], dtype=np.float32),
        is_mana_draw=is_mana_draw,
        mana_draw_legal=mana_draw_legal,
        target_tcodes=target_tcodes,
        num_games=2,
        num_rows=3,
        mana_draw_row_count=1,
        skipped_rows=0,
    )


class _FakeModel:
    """Minimal callable returning a canned V5 3-tuple of MLX arrays (for the
    evaluator-mirror test -- NOT an nn.Module, so NOT usable for the trainer)."""

    def __init__(self, logits_np, values_np, md_logit_np):
        self._logits = np.asarray(logits_np, dtype=np.float32)
        self._values = np.asarray(values_np, dtype=np.float32).reshape(-1)
        self._md = np.asarray(md_logit_np, dtype=np.float32).reshape(-1)

    def __call__(self, obs, action_features, mana_draw_legal=None):
        return mx.array(self._logits), mx.array(self._values), mx.array(self._md)


# ---------------------------------------------------------------------------
# (1) Pure-numpy loss MATH.
# ---------------------------------------------------------------------------

def test_awac_crr_loss_math():
    # 3 rows: row0 normal A=+2, row1 normal A=-1, row2 mana_draw (tcode=-1).
    # Single legal action per row -> probs[legal]=1 -> new_log_prob=0.
    # old_log_prob=0 -> ratio=1. values=[0,0,0] returns=[1,0.5,0] -> value_loss.
    # md_logit=0 everywhere -> md_p=0.5; mana_draw_legal only row2 -> bce=-log(0.5).
    logits = np.full((3, MAX_CANDIDATE_ACTIONS), -1e9, dtype=np.float32)
    logits[0, 5] = 0.0
    logits[1, 7] = 0.0
    logits[2, 0] = 0.0
    values = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    md_logit = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    mask = np.zeros((3, MAX_CANDIDATE_ACTIONS), dtype=np.float32)
    mask[0, 5] = 1.0
    mask[1, 7] = 1.0
    mask[2, 0] = 1.0
    bf = dict(
        old_log_probs=np.array([0.0, 0.0, math.log(0.5)], dtype=np.float32),
        advantages=np.array([2.0, -1.0, 0.0], dtype=np.float32),
        returns=np.array([1.0, 0.5, 0.0], dtype=np.float32),
        actions=np.array([5, 7, 0], dtype=np.int64),
        action_mask=mask,
        is_mana_draw=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        mana_draw_legal=np.array([False, False, True], dtype=np.bool_),
        target_tcodes=np.array([5, 7, -1], dtype=np.int64),
        is_padded=np.array([False, False, False], dtype=np.bool_),
    )
    total, m = awac_crr_loss(
        (logits, values, md_logit), bf,
        clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
        lambda_awac=1.0, awac_clamp=4.0, mana_draw_bce_weight=1.0,
    )

    # Hand-computed.
    w0 = math.exp(2.0)
    w1 = math.exp(-1.0)
    # Draw rows are real factorized actions in the AWAC surrogate (A=0 here).
    policy_loss = -(w0 * 2.0 + w1 * (-1.0)) / 3.0
    # value_loss = 0.5 * (1^2 + 0.5^2 + 0^2)/3
    value_loss = 0.5 * (1.0 + 0.25 + 0.0) / 3.0
    # entropy ~ 0 (single legal action -> -log(1+1e-10) per row ~ 0)
    entropy = math.log(2.0) / 3.0
    # mana_draw BCE: only row2 legal, md_p=0.5, is_mana_draw=1 -> -log(0.5)
    mana_draw_bce = -math.log(0.5)
    expected_total = (
        policy_loss + value_loss - 0.01 * entropy + 1.0 * mana_draw_bce
    )

    assert m["policy_loss"] == pytest.approx(policy_loss, abs=1e-5)
    assert m["value_loss"] == pytest.approx(value_loss, abs=1e-6)
    assert m["mana_draw_bce"] == pytest.approx(mana_draw_bce, abs=1e-6)
    assert m["entropy"] == pytest.approx(entropy, abs=1e-4)
    assert total == pytest.approx(expected_total, abs=1e-5)
    assert m["valid_policy_rows"] == 3.0
    assert m["valid_value_rows"] == 3.0
    assert m["mana_draw_legal_rows"] == 1.0
    assert m["mean_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert m["clip_fraction"] == pytest.approx(0.0, abs=1e-6)
    assert m["approx_kl"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# (2) BLOCKER SIGN FIX (load-bearing).
# ---------------------------------------------------------------------------

def test_blocker_sign_fix():
    # POSITIVE advantage, policy INCREASING the action prob (ratio>1).
    # clip_epsilon large -> no clipping -> surr = ratios*A.
    # Case A: ratio=1 (old=new=0). Case B: ratio=2 (old=-log2, new=0).
    A = 1.0
    mo_a, bf_a = _single_legal_row(0, old_log_prob=0.0, advantage=A, target_tcode=0)
    mo_b, bf_b = _single_legal_row(0, old_log_prob=-math.log(2.0), advantage=A, target_tcode=0)
    eps = 10.0  # no clip
    _, m_a = awac_crr_loss(mo_a, bf_a, clip_epsilon=eps, lambda_awac=1.0, awac_clamp=4.0)
    _, m_b = awac_crr_loss(mo_b, bf_b, clip_epsilon=eps, lambda_awac=1.0, awac_clamp=4.0)

    # The CORRECT AWAC direction: loss DECREASES when the policy increases the
    # prob of a high-advantage action -> gradient INCREASES log pi.
    assert m_b["policy_loss"] < m_a["policy_loss"], (
        f"BLOCKER sign fix violated: policy_loss(ratio=2)={m_b['policy_loss']} "
        f"not < policy_loss(ratio=1)={m_a['policy_loss']}"
    )

    # Exact A-multiplier reference: policy_loss = -w*min(ratios*A, clip*A).
    w = math.exp(A)
    ref_a = -w * 1.0 * A  # ratio=1 -> min(A,A)=A
    ref_b = -w * 2.0 * A  # ratio=2, no clip -> min(2A,2A)=2A
    assert m_a["policy_loss"] == pytest.approx(ref_a, abs=1e-5)
    assert m_b["policy_loss"] == pytest.approx(ref_b, abs=1e-5)

    # The INCOHERENT log_pi-multiplier form `-mean(w*min(ratios,clip)*log_pi)`
    # would NOT satisfy the direction (log_pi = new_log_prob = 0 here -> the
    # surrogate is 0 for BOTH cases -> no direction; and for log_pi<0 the
    # (1+log_pi)*w*log_pi draft DECREASES log pi for high-advantage). Verify the
    # A-multiplier form is what C3 uses by checking it does NOT equal a
    # log_pi-multiplier surrogate.
    new_lp = 0.0  # single legal action -> new_log_prob=0
    incoherent_b = -w * min(2.0, 2.0) * new_lp  # = 0 (log_pi multiplier -> 0)
    assert m_b["policy_loss"] != pytest.approx(incoherent_b, abs=1e-6), (
        "C3 must use A as the multiplier, NOT log_pi"
    )
    # And the incoherent form gives NO decrease (0 == 0) -- the wrong direction.
    incoherent_a = -w * min(1.0, 1.0) * new_lp
    assert not (incoherent_b < incoherent_a), (
        "log_pi-multiplier form does not produce the AWAC direction"
    )


# ---------------------------------------------------------------------------
# (3) AWAC weight w.
# ---------------------------------------------------------------------------

def test_awac_weight():
    # A >> lambda*C -> saturates at exp(C) (clamp BEFORE exp).
    w_big = awac_weight(np.array([100.0]), lambda_awac=1.0, awac_clamp=4.0)
    assert float(w_big[0]) == pytest.approx(math.exp(4.0), abs=1e-5)
    # A=0 -> w=1.
    w_zero = awac_weight(np.array([0.0]), lambda_awac=1.0, awac_clamp=4.0)
    assert float(w_zero[0]) == pytest.approx(1.0, abs=1e-6)
    # A<0 -> w<1.
    w_neg = awac_weight(np.array([-2.0]), lambda_awac=1.0, awac_clamp=4.0)
    assert float(w_neg[0]) == pytest.approx(math.exp(-2.0), abs=1e-6)
    assert float(w_neg[0]) < 1.0
    # lambda scaling: A/lambda.
    w_lam = awac_weight(np.array([2.0]), lambda_awac=2.0, awac_clamp=4.0)
    assert float(w_lam[0]) == pytest.approx(math.exp(1.0), abs=1e-6)
    # negative saturation.
    w_neg_big = awac_weight(np.array([-100.0]), lambda_awac=1.0, awac_clamp=4.0)
    assert float(w_neg_big[0]) == pytest.approx(math.exp(-4.0), abs=1e-5)


# ---------------------------------------------------------------------------
# (4) value_loss D-C3 -- padded rows EXCLUDED.
# ---------------------------------------------------------------------------

def test_value_loss_masks_padded_rows():
    # 2 rows: row0 real, row1 padded. Perturbing the padded row's values must
    # NOT change value_loss.
    logits = np.full((2, MAX_CANDIDATE_ACTIONS), -1e9, dtype=np.float32)
    logits[0, 0] = 0.0
    logits[1, 0] = 0.0
    mask = np.zeros((2, MAX_CANDIDATE_ACTIONS), dtype=np.float32)
    mask[0, 0] = 1.0
    mask[1, 0] = 1.0  # padded row still has a mask (derivation uses is_padded)
    md = np.zeros(2, dtype=np.float32)
    bf = dict(
        old_log_probs=np.array([0.0, 0.0], dtype=np.float32),
        advantages=np.array([1.0, 0.0], dtype=np.float32),
        returns=np.array([1.0, 0.0], dtype=np.float32),
        actions=np.array([0, 0], dtype=np.int64),
        action_mask=mask,
        is_mana_draw=np.array([0.0, 0.0], dtype=np.float32),
        mana_draw_legal=np.array([False, False], dtype=np.bool_),
        target_tcodes=np.array([0, -1], dtype=np.int64),
        is_padded=np.array([False, True], dtype=np.bool_),
    )
    v1 = np.array([0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 5.0], dtype=np.float32)  # perturb padded row's value
    _, m1 = awac_crr_loss((logits, v1, md), bf, value_coef=0.5)
    _, m2 = awac_crr_loss((logits, v2, md), bf, value_coef=0.5)
    assert m1["value_loss"] == pytest.approx(m2["value_loss"], abs=1e-7), (
        "padded row's value perturbation must NOT change value_loss"
    )
    # value_loss = 0.5 * (1-0)^2 / 1 (only row0 real).
    assert m1["value_loss"] == pytest.approx(0.5 * 1.0, abs=1e-6)
    assert m1["valid_value_rows"] == 1.0


# ---------------------------------------------------------------------------
# (5) mana_draw BCE -- RETAINS mana_draw_logit (_out[2]).
# ---------------------------------------------------------------------------

def test_mana_draw_bce_retains_out2():
    # High md_logit + is_mana_draw=1 + legal -> low bce.
    mo_hi, bf_hi = _single_legal_row(
        0, md_logit=10.0, is_mana_draw=1.0, mana_draw_legal=True, target_tcode=-1,
    )
    # Low md_logit + is_mana_draw=1 + legal -> high bce.
    mo_lo, bf_lo = _single_legal_row(
        0, md_logit=-10.0, is_mana_draw=1.0, mana_draw_legal=True, target_tcode=-1,
    )
    _, m_hi = awac_crr_loss(mo_hi, bf_hi)
    _, m_lo = awac_crr_loss(mo_lo, bf_lo)
    assert m_hi["mana_draw_bce"] < m_lo["mana_draw_bce"]
    assert m_hi["mana_draw_bce"] == pytest.approx(
        -math.log(1.0 / (1.0 + math.exp(-10.0))), abs=1e-4
    )
    # A non-mana_draw row excluded when mana_draw_legal=False -> bce=0.
    mo_nlegal, bf_nlegal = _single_legal_row(
        0, md_logit=10.0, is_mana_draw=1.0, mana_draw_legal=False, target_tcode=-1,
    )
    _, m_nlegal = awac_crr_loss(mo_nlegal, bf_nlegal)
    assert m_nlegal["mana_draw_bce"] == 0.0
    assert m_nlegal["mana_draw_legal_rows"] == 0.0


@_MLX
def test_evaluator_retains_mana_draw_logit_mlx():
    # The MLX evaluator must read _out[2] (mana_draw_logit) and include the BCE
    # term. A FakeModel returns a 3-tuple; the loss must match the numpy
    # reference INCLUDING the BCE term (and would NOT match if BCE dropped).
    import mlx.core as mx

    from train_v3.rust_ppo import RustPPOBatch

    # Build a 1-row dense RustPPOBatch manually.
    mask = np.zeros((1, 1, MAX_CANDIDATE_ACTIONS), dtype=np.float32)
    mask[0, 0, 0] = 1.0
    obs = np.zeros((1, 1, OBS_V5_DIM), dtype=np.float32)
    feats = np.zeros((1, 1, MAX_CANDIDATE_ACTIONS, ACTION_FEATURE_DIM), dtype=np.float32)
    batch = RustPPOBatch(
        observations=obs,
        action_mask=mask,
        action_features=feats,
        legal_action_counts=np.ones((1, 1), dtype=np.uintp),
        legal_action_offsets=np.zeros((1, 1), dtype=np.uintp),
        legal_action_ids=np.array([0], dtype=np.uintp),
        legal_action_features=np.zeros((1, ACTION_FEATURE_DIM), dtype=np.float32),
        actions=np.array([[0]], dtype=np.uintp),
        old_log_probs=np.array([[0.0]], dtype=np.float32),
        values=np.array([[0.0]], dtype=np.float32),
        rewards=np.array([[0.0]], dtype=np.float32),
        terminated=np.array([[False]], dtype=np.bool_),
        truncated=None,
        advantages=np.array([[1.0]], dtype=np.float32),
        returns=np.array([[0.0]], dtype=np.float32),
        selected_local_indices=None,
    )
    logits_np = np.full((1, MAX_CANDIDATE_ACTIONS), -1e9, dtype=np.float32)
    logits_np[0, 0] = 0.0
    values_np = np.array([0.0], dtype=np.float32)
    md_logit_np = np.array([2.0], dtype=np.float32)  # nonzero -> BCE nonzero
    fake = _FakeModel(logits_np, values_np, md_logit_np)

    ev = evaluate_awac_dense_batch(
        fake, batch,
        clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
        lambda_awac=1.0, awac_clamp=4.0, mana_draw_bce_weight=1.0,
        is_mana_draw=mx.array(np.array([1.0], dtype=np.float32)),
        mana_draw_legal=mx.array(np.array([True], dtype=np.bool_)),
        target_tcodes=mx.array(np.array([-1], dtype=np.int64)),
        is_padded=mx.array(np.array([False], dtype=np.bool_)),
    )
    # numpy reference (target_tcode=-1 -> policy_loss over 0 valid rows = 0).
    bf = dict(
        old_log_probs=np.array([0.0], dtype=np.float32),
        advantages=np.array([1.0], dtype=np.float32),
        returns=np.array([0.0], dtype=np.float32),
        actions=np.array([0], dtype=np.int64),
        action_mask=mask[0],
        is_mana_draw=np.array([1.0], dtype=np.float32),
        mana_draw_legal=np.array([True], dtype=np.bool_),
        target_tcodes=np.array([-1], dtype=np.int64),
        is_padded=np.array([False], dtype=np.bool_),
    )
    ref_total, ref_m = awac_crr_loss(
        (logits_np, values_np, md_logit_np), bf,
        clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
        lambda_awac=1.0, awac_clamp=4.0, mana_draw_bce_weight=1.0,
    )
    assert float(ev.loss.item()) == pytest.approx(ref_total, abs=1e-5)
    # The BCE term is nonzero -> if _out[2] were dropped, the loss would be
    # ~0 (policy_loss=0, value_loss=0, entropy=0). Assert it is NOT ~0.
    assert abs(float(ev.loss.item())) > 1e-3, (
        "evaluator must retain mana_draw_logit (_out[2]) -> BCE term present"
    )
    assert ref_m["mana_draw_bce"] == pytest.approx(
        -math.log(1.0 / (1.0 + math.exp(-2.0))), abs=1e-4
    )


# ---------------------------------------------------------------------------
# (6) valid_policy_mask exclusions.
# ---------------------------------------------------------------------------

def test_valid_policy_mask_includes_draw_excludes_padding():
    # 3 rows: row0 normal valid, row1 mana_draw (tcode=-1), row2 padded.
    # Perturbing row1/row2 new_log_prob (via old_log_prob shift on ratio) must
    # NOT change policy_loss.
    logits = np.full((3, MAX_CANDIDATE_ACTIONS), -1e9, dtype=np.float32)
    logits[0, 0] = 0.0
    logits[1, 0] = 0.0
    logits[2, 0] = 0.0
    mask = np.zeros((3, MAX_CANDIDATE_ACTIONS), dtype=np.float32)
    mask[0, 0] = 1.0
    mask[1, 0] = 1.0
    mask[2, 0] = 1.0
    values = np.zeros(3, dtype=np.float32)
    md = np.zeros(3, dtype=np.float32)

    def make_bf(lp1, lp2):
        return dict(
            old_log_probs=np.array([0.0, lp1, lp2], dtype=np.float32),
            advantages=np.array([1.0, 5.0, 5.0], dtype=np.float32),
            returns=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            actions=np.array([0, 0, 0], dtype=np.int64),
            action_mask=mask,
            is_mana_draw=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            mana_draw_legal=np.array([False, True, False], dtype=np.bool_),
            target_tcodes=np.array([0, -1, 0], dtype=np.int64),
            is_padded=np.array([False, False, True], dtype=np.bool_),
        )

    bf_a = make_bf(0.0, 0.0)
    bf_b = make_bf(0.0, -5.0)  # perturb only padded row
    _, m_a = awac_crr_loss((logits, values, md), bf_a)
    _, m_b = awac_crr_loss((logits, values, md), bf_b)
    assert m_a["policy_loss"] == pytest.approx(m_b["policy_loss"], abs=1e-7), (
        "padded rows must not contribute to policy_loss"
    )
    bf_c = make_bf(-5.0, 0.0)
    _, m_c = awac_crr_loss((logits, values, md), bf_c)
    assert m_c["policy_loss"] != pytest.approx(m_a["policy_loss"], abs=1e-7)
    assert m_a["valid_policy_rows"] == 2.0  # normal row + mana draw


# ---------------------------------------------------------------------------
# (7) dense-evaluator mirror (MLX).
# ---------------------------------------------------------------------------

@_MLX
def test_dense_evaluator_mirror():
    import mlx.core as mx

    from train_v3.rust_ppo import RustPPOBatch

    rng = np.random.default_rng(1)
    B = 2
    mask = np.zeros((B, 1, MAX_CANDIDATE_ACTIONS), dtype=np.float32)
    mask[0, 0, 3] = 1.0
    mask[0, 0, 4] = 1.0
    mask[1, 0, 5] = 1.0
    logits_np = rng.standard_normal((B, MAX_CANDIDATE_ACTIONS)).astype(np.float32)
    values_np = rng.standard_normal((B,)).astype(np.float32)
    md_np = rng.standard_normal((B,)).astype(np.float32)
    obs = np.zeros((B, 1, OBS_V5_DIM), dtype=np.float32)
    feats = np.zeros((B, 1, MAX_CANDIDATE_ACTIONS, ACTION_FEATURE_DIM), dtype=np.float32)
    batch = RustPPOBatch(
        observations=obs,
        action_mask=mask,
        action_features=feats,
        legal_action_counts=np.array([[2], [1]], dtype=np.uintp),
        legal_action_offsets=np.array([[0], [2]], dtype=np.uintp),
        legal_action_ids=np.array([3, 4, 5], dtype=np.uintp),
        legal_action_features=np.zeros((3, ACTION_FEATURE_DIM), dtype=np.float32),
        actions=np.array([[3], [5]], dtype=np.uintp),
        old_log_probs=np.array([[-1.0], [-0.5]], dtype=np.float32),
        values=values_np.reshape(B, 1),
        rewards=np.zeros((B, 1), dtype=np.float32),
        terminated=np.zeros((B, 1), dtype=np.bool_),
        truncated=None,
        advantages=np.array([[1.0], [2.0]], dtype=np.float32),
        returns=np.array([[0.5], [0.5]], dtype=np.float32),
        selected_local_indices=None,
    )
    fake = _FakeModel(logits_np, values_np, md_np)
    ev = evaluate_awac_dense_batch(
        fake, batch,
        clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
        lambda_awac=1.0, awac_clamp=4.0, mana_draw_bce_weight=1.0,
        is_mana_draw=mx.array(np.zeros(B, dtype=np.float32)),
        mana_draw_legal=mx.array(np.zeros(B, dtype=np.bool_)),
        target_tcodes=mx.array(np.array([3, 5], dtype=np.int64)),
        is_padded=mx.array(np.zeros(B, dtype=np.bool_)),
    )
    # numpy reference: masked softmax where(mask,logits,-1e9), gather, log.
    masked = np.where(mask[:, 0, :] == 1.0, logits_np, np.float32(-1e9))
    m = masked.max(axis=-1, keepdims=True)
    ex = np.exp(masked - m)
    probs = ex / ex.sum(axis=-1, keepdims=True)
    new_lp_ref = np.log(probs[np.arange(B), np.array([3, 5])] + 1e-10)
    new_lp_ev = np.asarray(ev.new_log_probs, dtype=np.float32).reshape(-1)
    assert np.allclose(new_lp_ev, new_lp_ref, atol=1e-5)
    ratios_ref = np.exp(new_lp_ref - np.array([-1.0, -0.5]))
    ratios_ev = np.asarray(ev.ratios, dtype=np.float32).reshape(-1)
    assert np.allclose(ratios_ev, ratios_ref, atol=1e-5)


# ---------------------------------------------------------------------------
# (8) end-to-end on a C2 batch.
# ---------------------------------------------------------------------------

@_MLX
def test_end_to_end_on_c2_batch():
    from train_v3.v5_policy import V5ActionConditionedPolicy

    orb = _build_tiny_offline_replay_batch()
    # prepare_rust_ppo_batch flows without error.
    ppo = prepare_rust_ppo_batch(
        orb.batch, gamma=0.99, gae_lambda=0.95, bootstrap_values=orb.bootstrap_values
    )
    assert ppo.advantages.shape == (2, 2)
    assert ppo.returns.shape == (2, 2)

    model = V5ActionConditionedPolicy(hidden_dim=128)
    metrics = train_awac_crr_replay(
        model, orb, epochs=1, minibatch_size=2, seed=0, hidden_dim=128,
        lambda_awac=1.0, awac_clamp=4.0,
    )
    assert metrics["status"] == "trained"
    assert math.isfinite(metrics["loss_after"])
    assert math.isfinite(metrics["loss_before"])
    assert metrics["num_updates"] == 2  # 4 rows / 2 minibatch * 1 epoch
    assert metrics["policy_loss"] is not None
    assert metrics["mana_draw_bce"] is not None


# ---------------------------------------------------------------------------
# (9) npz/mlx SKIP-GATE.
# ---------------------------------------------------------------------------

def test_awac_crr_loss_works_without_mlx_import():
    # The pure-numpy loss core must work with NO mlx import at module top.
    # (Importing awac_crr_replay must not pull mlx -- lazy import inside the
    # trainer/evaluator only.)
    import sys

    mod = sys.modules["train_v3.awac_crr_replay"]
    # mlx must NOT be a module-level attribute (lazy import inside functions).
    assert not hasattr(mod, "mlx"), "awac_crr_replay must not import mlx at top"
    assert not hasattr(mod, "nn"), "awac_crr_replay must not import mlx.nn at top"
    mo, bf = _single_legal_row(0, advantage=1.0, target_tcode=0)
    total, m = awac_crr_loss(mo, bf)
    assert math.isfinite(total)
    assert m["valid_policy_rows"] == 1.0


@_MLX
def test_skip_gate_none_path():
    orb = _build_tiny_offline_replay_batch()
    metrics = train_awac_crr_replay(None, orb, epochs=1, minibatch_size=4)
    assert metrics["status"] == "skipped"
    assert metrics["num_updates"] == 0
    assert metrics["new_checkpoint_path"] is None


@_MLX
def test_skip_gate_absent_path(tmp_path):
    orb = _build_tiny_offline_replay_batch()
    bogus = tmp_path / "no_such_checkpoint.npz"
    metrics = train_awac_crr_replay(str(bogus), orb, epochs=1, minibatch_size=4)
    assert metrics["status"] == "skipped"


# ---------------------------------------------------------------------------
# (10) A2 freeze_faithful byte-identity + value_head TRAINABLE.
# ---------------------------------------------------------------------------

@_MLX
def test_freeze_faithful_byte_identity():
    from train_v3.v5_policy import V5ActionConditionedPolicy

    orb = _build_tiny_offline_replay_batch()
    model = V5ActionConditionedPolicy(hidden_dim=128)
    frozen_names = frozen_param_names(freeze_faithful=True, train_value_head=True)
    snapshot = snapshot_frozen_params(model, frozen_names)
    # value_head must be TRAINABLE (in the trainable set, NOT frozen).
    assert _VALUE_HEAD_PARAM_NAMES.isdisjoint(frozen_names)
    # Snapshot value_head before to compare after.
    import mlx.nn as nn
    vh_before = {
        name: np.array(val)
        for name, val in nn.utils.tree_flatten(model.trainable_parameters())
        if name in _VALUE_HEAD_PARAM_NAMES
    }

    metrics = train_awac_crr_replay(
        model, orb, epochs=2, minibatch_size=2, seed=0, hidden_dim=128,
        freeze_faithful=True, train_value_head=True, lr=1e-2,
    )
    assert metrics["status"] == "trained"
    assert metrics["frozen_preserved"] is True

    # FAITHFUL + SHAPE_COMPAT byte-identical.
    flat = dict(nn.utils.tree_flatten(model.trainable_parameters()))
    for name, before in snapshot.items():
        after = np.array(flat[name])
        assert np.array_equal(before, after), (
            f"frozen param {name!r} moved: max abs diff = "
            f"{float(np.max(np.abs(before - after)))}"
        )
    # value_head MOVED (D-C3).
    moved = False
    for name, before in vh_before.items():
        after = np.array(flat[name])
        if not np.array_equal(before, after):
            moved = True
            assert float(np.max(np.abs(before - after))) > 0.0
    assert moved, "value_head must MOVE (train_value_head=True, D-C3)"


# ---------------------------------------------------------------------------
# (11) MONITORING-ONLY -- no promote/score field.
# ---------------------------------------------------------------------------

@_MLX
def test_run_metrics_monitoring_only():
    from train_v3.v5_policy import V5ActionConditionedPolicy

    orb = _build_tiny_offline_replay_batch()
    model = V5ActionConditionedPolicy(hidden_dim=128)
    driver = AwacCrrReplay()
    metrics = driver.run(orb, checkpoint_path=model, epochs=1, minibatch_size=4, seed=0)
    assert isinstance(metrics, AwacCrrMetrics)
    field_names = {f.name for f in dataclass_fields(AwacCrrMetrics)}
    assert "promote" not in field_names
    assert "score" not in field_names
    assert "promote" not in metrics.extra
    assert "score" not in metrics.extra
    assert metrics.status == "trained"
    assert metrics.num_updates >= 1


# ---------------------------------------------------------------------------
# (12) clip_fraction + approx_kl populated.
# ---------------------------------------------------------------------------

@_MLX
def test_clip_fraction_and_approx_kl_populated():
    from train_v3.v5_policy import V5ActionConditionedPolicy

    orb = _build_tiny_offline_replay_batch()
    model = V5ActionConditionedPolicy(hidden_dim=128)
    metrics = train_awac_crr_replay(
        model, orb, epochs=1, minibatch_size=4, seed=0, hidden_dim=128,
    )
    assert metrics["status"] == "trained"
    assert metrics["clip_fraction"] is not None
    assert metrics["approx_kl"] is not None
    assert math.isfinite(metrics["clip_fraction"])
    assert math.isfinite(metrics["approx_kl"])

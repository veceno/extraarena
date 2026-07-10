"""Regression tests for the V5 factorized candidate/mana-draw policy."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "TrainV3.5" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_phaseA_random_bootstrap import _select_from_joint_padded  # noqa: E402


def test_binary_gate_argmax_selects_legal_mana_draw_and_uses_gate_log_prob() -> None:
    actions, log_probs, selected, mana_draw = _select_from_joint_padded(
        logits=np.asarray([[1.0, 2.0]], dtype=np.float32),
        counts=np.asarray([2], dtype=np.uintp),
        ids=np.asarray([17, 23], dtype=np.uintp),
        mana_draw_logits=np.asarray([5.0], dtype=np.float32),
        mana_draw_legal=np.asarray([True]),
        rng=None,
    )
    assert actions.tolist() == [23]  # valid ignored placeholder for Rust FFI
    assert selected.tolist() == [1]
    assert mana_draw.tolist() == [True]
    expected = np.log(1.0 / (1.0 + np.exp(-5.0)))
    assert log_probs[0] == np.float32(expected)


def test_binary_gate_argmax_masks_illegal_mana_draw_even_at_huge_logit() -> None:
    actions, _log_probs, selected, mana_draw = _select_from_joint_padded(
        logits=np.asarray([[1.0, 2.0]], dtype=np.float32),
        counts=np.asarray([2], dtype=np.uintp),
        ids=np.asarray([17, 23], dtype=np.uintp),
        mana_draw_logits=np.asarray([100.0], dtype=np.float32),
        mana_draw_legal=np.asarray([False]),
        rng=None,
    )
    assert actions.tolist() == [23]
    assert selected.tolist() == [1]
    assert mana_draw.tolist() == [False]


def test_binary_gate_does_not_compare_gate_logit_to_single_card_logit() -> None:
    """A negative gate must choose a card even if it beats every card scalar."""
    actions, log_probs, selected, mana_draw = _select_from_joint_padded(
        logits=np.asarray([[-4.0, -3.0, -2.0]], dtype=np.float32),
        counts=np.asarray([3], dtype=np.uintp),
        ids=np.asarray([17, 23, 42], dtype=np.uintp),
        mana_draw_logits=np.asarray([-0.25], dtype=np.float32),
        mana_draw_legal=np.asarray([True]),
        rng=None,
    )
    assert actions.tolist() == [42]
    assert selected.tolist() == [2]
    assert mana_draw.tolist() == [False]
    candidate_probability = np.exp(-2.0) / (np.exp(-4.0) + np.exp(-3.0) + np.exp(-2.0))
    expected = np.log1p(-1.0 / (1.0 + np.exp(0.25))) + np.log(candidate_probability)
    assert log_probs[0] == np.float32(expected)


@pytest.mark.parametrize("gate_logit, expected_draw", [(1.0, True), (-1.0, False)])
def test_factorized_ppo_log_prob_matches_sampler_for_both_gate_branches(
    gate_logit: float,
    expected_draw: bool,
) -> None:
    """The rollout selector and PPO evaluator must use identical factorization."""
    pytest.importorskip("mlx")
    import mlx.core as mx

    from train_v3.contracts import ACTION_FEATURE_DIM, OBS_V5_DIM
    from train_v3.rust_policy import PaddedLegalActionInputs, score_padded_legal_action_inputs
    from train_v3.rust_ppo import RustPPOBatch, evaluate_rust_ppo_batch
    from train_v3.v5_policy import create_v5_policy

    model = create_v5_policy(policy_kind="v5_split_encoder", hidden_dim=32, action_hidden_dim=16)
    model.mana_draw_head.weight = mx.zeros_like(model.mana_draw_head.weight)
    model.mana_draw_head.bias = mx.array([gate_logit], dtype=mx.float32)
    obs = np.zeros((1, OBS_V5_DIM), dtype=np.float32)
    padded_features = np.zeros((1, 2, ACTION_FEATURE_DIM), dtype=np.float32)
    padded_features[0, 1, 0] = 1.0
    padded_cache = PaddedLegalActionInputs(
        padded_features=padded_features,
        legal_mask=np.asarray([[True, True]], dtype=np.bool_),
    )
    scores = score_padded_legal_action_inputs(model, obs, padded_cache)
    mx.eval(scores.padded_logits, scores.values, scores.mana_draw_logits)
    actions, old_log_probs, selected, mana_draw_taken = _select_from_joint_padded(
        logits=np.asarray(scores.padded_logits),
        counts=np.asarray([2], dtype=np.uintp),
        ids=np.asarray([11, 12], dtype=np.uintp),
        mana_draw_logits=np.asarray(scores.mana_draw_logits),
        mana_draw_legal=np.asarray([True]),
        rng=None,
    )
    assert bool(mana_draw_taken[0]) is expected_draw
    batch = RustPPOBatch(
        observations=obs.reshape((1, 1, OBS_V5_DIM)),
        action_mask=None,
        action_features=None,
        legal_action_counts=np.asarray([[2]], dtype=np.uintp),
        legal_action_offsets=np.asarray([[0]], dtype=np.uintp),
        legal_action_ids=np.asarray([11, 12], dtype=np.uintp),
        legal_action_features=padded_features.reshape((2, ACTION_FEATURE_DIM)),
        actions=actions.reshape((1, 1)),
        old_log_probs=old_log_probs.reshape((1, 1)),
        values=np.asarray(scores.values).reshape((1, 1)),
        rewards=np.zeros((1, 1), dtype=np.float32),
        terminated=np.zeros((1, 1), dtype=np.bool_),
        truncated=None,
        advantages=np.zeros((1, 1), dtype=np.float32),
        returns=np.zeros((1, 1), dtype=np.float32),
        selected_local_indices=selected.reshape((1, 1)),
        mana_draw_legal=np.asarray([[True]], dtype=np.bool_),
        mana_draw_taken=mana_draw_taken.reshape((1, 1)),
    )
    evaluation = evaluate_rust_ppo_batch(model, batch, padded_legal_action_cache=padded_cache)
    mx.eval(evaluation.new_log_probs)
    np.testing.assert_allclose(
        np.asarray(evaluation.new_log_probs),
        old_log_probs,
        rtol=0.0,
        atol=2.0e-6,
    )

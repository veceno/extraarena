"""Regression tests for V5 state-conditioned candidate ranking."""
from __future__ import annotations

import numpy as np
import pytest


mx = pytest.importorskip("mlx.core")

from train_v3.contracts import ACTION_FEATURE_DIM, OBS_V5_DIM  # noqa: E402
from train_v3.rust_policy import (  # noqa: E402
    score_compact_legal_actions,
    score_padded_legal_actions,
)
from train_v3.v5_policy import V5ActionConditionedPolicy  # noqa: E402


def _center(logits):
    return logits - mx.mean(logits, axis=-1, keepdims=True)


def _make_inputs(*, batch: int = 2):
    rng = np.random.default_rng(7305)
    obs = rng.normal(size=(batch, OBS_V5_DIM)).astype(np.float32) * 0.1
    features = rng.normal(size=(batch, 601, ACTION_FEATURE_DIM)).astype(np.float32) * 0.1
    return mx.array(obs), mx.array(features)


def test_interaction_is_zero_initialized_for_legacy_logit_parity():
    """Adding the branch must not perturb a pre-interaction source policy."""
    mx.random.seed(11)
    model = V5ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
    obs, features = _make_inputs()

    logits, _, _ = model(obs, features)
    state = model.encode_state(obs)
    batch = int(obs.shape[0])
    action = model.action_encoder(
        mx.reshape(features, (batch * 601, ACTION_FEATURE_DIM))
    )
    action = mx.reshape(action, (batch, 601, model.action_hidden_dim))
    state_rows = mx.broadcast_to(
        mx.expand_dims(state, axis=1), (batch, 601, model.hidden_dim)
    )
    joint = mx.reshape(
        mx.concatenate([state_rows, action], axis=-1),
        (batch * 601, model.hidden_dim + model.action_hidden_dim),
    )
    legacy = mx.reshape(model.candidate_scorer(joint), (batch, 601))
    mx.eval(logits, legacy, model.state_action_gate.weight)

    np.testing.assert_array_equal(np.asarray(model.state_action_gate.weight), 0.0)
    np.testing.assert_array_equal(np.asarray(logits), np.asarray(legacy))


def test_private_info_and_history_can_change_relative_candidate_logits():
    """Counterfactual private/history observations must affect action ranking."""
    mx.random.seed(12)
    model = V5ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
    # Activate the compatibility branch deterministically. Training performs
    # the same transition away from its zero-compatible initialization.
    model.state_action_query.weight = mx.ones_like(model.state_action_query.weight) * 0.05
    model.state_action_gate.weight = mx.ones_like(model.state_action_gate.weight) * 0.5

    rng = np.random.default_rng(991)
    features = mx.array(
        rng.normal(size=(1, 601, ACTION_FEATURE_DIM)).astype(np.float32) * 0.1
    )
    obs_a = np.zeros((1, OBS_V5_DIM), dtype=np.float32)
    obs_b = obs_a.copy()
    # Keep the classic/public prefix fixed. Change only V5 private information
    # and the 20-event history tape.
    obs_b[:, 1488:] = rng.normal(size=(1, OBS_V5_DIM - 1488)).astype(np.float32)

    logits_a, _, _ = model(mx.array(obs_a), features)
    logits_b, _, _ = model(mx.array(obs_b), features)
    delta = mx.max(mx.abs(_center(logits_a) - _center(logits_b)))
    mx.eval(delta)
    assert float(delta.item()) > 1e-5


def test_zero_initialized_interaction_receives_first_step_policy_gradient():
    """Compatibility initialization must not create a dead branch."""
    import mlx.nn as nn

    mx.random.seed(14)
    model = V5ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
    obs, features = _make_inputs(batch=2)

    def policy_loss(current_model, observations, action_features):
        logits, _, _ = current_model(observations, action_features)
        return -mx.mean(logits[:, 0] - mx.logsumexp(logits, axis=-1))

    _, gradients = nn.value_and_grad(model, policy_loss)(model, obs, features)
    gate_grad = gradients["state_action_gate"]["weight"]
    mx.eval(gate_grad)
    assert float(mx.max(mx.abs(gate_grad)).item()) > 0.0

    # Once the scalar gate has opened, the query receives policy gradients.
    model.state_action_gate.weight = mx.ones_like(model.state_action_gate.weight) * 0.01
    _, gradients = nn.value_and_grad(model, policy_loss)(model, obs, features)
    query_grad = gradients["state_action_query"]
    mx.eval(query_grad["weight"], query_grad["bias"])
    assert float(mx.max(mx.abs(query_grad["weight"])).item()) > 0.0
    assert float(mx.max(mx.abs(query_grad["bias"])).item()) > 0.0


def test_pre_interaction_checkpoint_load_keeps_zero_residual(tmp_path):
    """Old Phase A/B checkpoints remain loadable and behavior-compatible."""
    import mlx.nn as nn

    from ai.train_v2.model_mlx import flatten_params, load_checkpoint

    mx.random.seed(15)
    source = V5ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
    legacy_weights = {
        key: value
        for key, value in flatten_params(source).items()
        if not key.startswith(("state_action_query.", "state_action_gate."))
    }
    checkpoint = tmp_path / "legacy_v5.npz"
    np.savez(checkpoint, **legacy_weights)

    restored = V5ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
    load_checkpoint(str(checkpoint), restored)
    flat = dict(nn.utils.tree_flatten(restored.trainable_parameters()))
    np.testing.assert_array_equal(np.asarray(flat["state_action_gate.weight"]), 0.0)

    obs, features = _make_inputs(batch=1)
    logits, _, _ = restored(obs, features)
    state = restored.encode_state(obs)
    action = restored.action_encoder(mx.reshape(features, (601, ACTION_FEATURE_DIM)))
    state_rows = mx.broadcast_to(mx.expand_dims(state, 1), (1, 601, restored.hidden_dim))
    joint = mx.reshape(
        mx.concatenate([state_rows, mx.reshape(action, (1, 601, restored.action_hidden_dim))], -1),
        (601, restored.hidden_dim + restored.action_hidden_dim),
    )
    legacy_logits = mx.reshape(restored.candidate_scorer(joint), (1, 601))
    mx.eval(logits, legacy_logits)
    np.testing.assert_array_equal(np.asarray(logits), np.asarray(legacy_logits))


def test_compact_and_padded_scorers_include_interaction_residual():
    """The optimized rollout scorers must match the dense policy forward."""
    mx.random.seed(13)
    model = V5ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
    model.state_action_query.weight = mx.ones_like(model.state_action_query.weight) * 0.03
    model.state_action_gate.weight = mx.ones_like(model.state_action_gate.weight) * 0.4
    obs, dense_features = _make_inputs(batch=2)
    counts = np.asarray([3, 2], dtype=np.uintp)
    legal_ids = [[5, 17, 400], [2, 300]]
    compact_np = np.concatenate(
        [
            np.asarray(dense_features[env, ids], dtype=np.float32)
            for env, ids in enumerate(legal_ids)
        ],
        axis=0,
    )

    compact = score_compact_legal_actions(
        model, obs, counts, compact_np, row_index_backend="python"
    )
    padded = score_padded_legal_actions(
        model,
        obs,
        counts,
        compact_np,
        legal_action_ids=np.arange(int(counts.sum()), dtype=np.uintp),
        padding_backend="python",
    )

    # Build a dense tensor whose selected slots carry the exact compact rows.
    dense_np = np.zeros((2, 601, ACTION_FEATURE_DIM), dtype=np.float32)
    offset = 0
    expected = []
    for env, ids in enumerate(legal_ids):
        for action_id in ids:
            dense_np[env, action_id] = compact_np[offset]
            offset += 1
        logits, _, _ = model(obs[env : env + 1], mx.array(dense_np[env : env + 1]))
        expected.extend(np.asarray(logits[0, ids], dtype=np.float32).tolist())

    mx.eval(compact.legal_logits, padded.padded_logits)
    np.testing.assert_allclose(np.asarray(compact.legal_logits), expected, atol=1e-5, rtol=1e-5)
    padded_flat = np.concatenate(
        [
            np.asarray(padded.padded_logits[env, :count], dtype=np.float32)
            for env, count in enumerate(counts.tolist())
        ]
    )
    np.testing.assert_allclose(padded_flat, expected, atol=1e-5, rtol=1e-5)

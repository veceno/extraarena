"""
Tests for training stability guards (Task 15).
"""
import numpy as np
import pytest

from ai.train_v2.train_ppo import (
    PPOConfig,
    train,
    _prepare_batch,
    _assert_finite_array,
    _validate_batch,
)


class TestAssertFiniteArray:
    def test_assert_finite_array_rejects_nan(self):
        with pytest.raises(ValueError, match="contains non-finite values"):
            _assert_finite_array("test", np.array([1.0, np.nan]))

    def test_assert_finite_array_accepts_finite(self):
        _assert_finite_array("test", np.array([1.0, 2.0, -0.5]))

    def test_assert_finite_array_rejects_inf(self):
        with pytest.raises(ValueError, match="contains non-finite values"):
            _assert_finite_array("test", np.array([np.inf]))


class TestValidateBatch:
    def test_validate_batch_rejects_illegal_action(self):
        N, A = 3, 601
        batch = {
            "obs": np.zeros((N, 1456), dtype=np.float32),
            "action_features": np.zeros((N, A, 171), dtype=np.float32),
            "mask": np.ones((N, A), dtype=np.float32),
            "action_ids": np.array([0, 1, 2], dtype=np.int32),
            "log_probs": np.zeros(N, dtype=np.float32),
            "advantages": np.zeros(N, dtype=np.float32),
            "returns": np.zeros(N, dtype=np.float32),
        }
        batch["mask"][1, 1] = 0.0

        with pytest.raises(ValueError, match="illegal action"):
            _validate_batch(batch, PPOConfig(min_batch_transitions=2))

    def test_validate_batch_rejects_action_out_of_range(self):
        N, A = 3, 601
        batch = {
            "obs": np.zeros((N, 1456), dtype=np.float32),
            "action_features": np.zeros((N, A, 171), dtype=np.float32),
            "mask": np.ones((N, A), dtype=np.float32),
            "action_ids": np.array([0, 999, 2], dtype=np.int32),
            "log_probs": np.zeros(N, dtype=np.float32),
            "advantages": np.zeros(N, dtype=np.float32),
            "returns": np.zeros(N, dtype=np.float32),
        }

        with pytest.raises(ValueError, match="out of range"):
            _validate_batch(batch, PPOConfig(min_batch_transitions=2))

    def test_validate_batch_rejects_small_batch(self):
        N, A = 1, 601
        batch = {
            "obs": np.zeros((N, 1456), dtype=np.float32),
            "action_features": np.zeros((N, A, 171), dtype=np.float32),
            "mask": np.ones((N, A), dtype=np.float32),
            "action_ids": np.array([0], dtype=np.int32),
            "log_probs": np.zeros(N, dtype=np.float32),
            "advantages": np.zeros(N, dtype=np.float32),
            "returns": np.zeros(N, dtype=np.float32),
        }

        with pytest.raises(ValueError, match="min_batch_transitions"):
            _validate_batch(batch, PPOConfig(min_batch_transitions=3))

    def test_validate_batch_rejects_non_finite(self):
        N, A = 3, 601
        batch = {
            "obs": np.zeros((N, 1456), dtype=np.float32),
            "action_features": np.zeros((N, A, 171), dtype=np.float32),
            "mask": np.ones((N, A), dtype=np.float32),
            "action_ids": np.array([0, 1, 2], dtype=np.int32),
            "log_probs": np.zeros(N, dtype=np.float32),
            "advantages": np.array([0.0, np.nan, 0.0], dtype=np.float32),
            "returns": np.zeros(N, dtype=np.float32),
        }

        with pytest.raises(ValueError, match="contains non-finite values"):
            _validate_batch(batch, PPOConfig(min_batch_transitions=2))


class TestAdvantageNormalization:
    def test_prepare_batch_constant_advantages_no_nan(self):
        transitions = [
            {
                "obs": np.zeros(1456, dtype=np.float32),
                "action_features": np.zeros((601, 171), dtype=np.float32),
                "mask": np.ones(601, dtype=np.float32),
                "action_id": 0,
                "reward": 0.0,
                "done": False,
                "truncated": False,
                "value": 0.5,
                "log_prob": 0.0,
                "player_id": 1,
                "next_obs": np.zeros(1456, dtype=np.float32),
            },
            {
                "obs": np.zeros(1456, dtype=np.float32),
                "action_features": np.zeros((601, 171), dtype=np.float32),
                "mask": np.ones(601, dtype=np.float32),
                "action_id": 1,
                "reward": 0.0,
                "done": False,
                "truncated": False,
                "value": 0.5,
                "log_prob": 0.0,
                "player_id": 1,
                "next_obs": np.zeros(1456, dtype=np.float32),
            },
            {
                "obs": np.zeros(1456, dtype=np.float32),
                "action_features": np.zeros((601, 171), dtype=np.float32),
                "mask": np.ones(601, dtype=np.float32),
                "action_id": 2,
                "reward": 0.0,
                "done": True,
                "truncated": False,
                "value": 0.5,
                "log_prob": 0.0,
                "player_id": 1,
                "next_obs": np.zeros(1456, dtype=np.float32),
            },
        ]
        config = PPOConfig(
            total_updates=1,
            episodes_per_update=1,
            max_steps_per_episode=10,
            hidden_dim=32,
            action_hidden_dim=16,
            minibatch_size=8,
            epochs=1,
            seed=42,
        )
        batch = _prepare_batch(transitions, config)
        assert np.all(np.isfinite(batch["advantages"]))
        assert np.all(np.isfinite(batch["returns"]))


class TestTrainStability:
    def test_train_tiny_stability_smoke(self):
        import shutil
        ckpt_dir = "/tmp/_t15_stability_ckpts"
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        try:
            config = PPOConfig(
                total_updates=1,
                episodes_per_update=1,
                max_steps_per_episode=10,
                hidden_dim=32,
                action_hidden_dim=16,
                minibatch_size=8,
                epochs=1,
                seed=42,
                checkpoint_dir=ckpt_dir,
                fail_on_non_finite=True,
                min_batch_transitions=2,
            )
            result = train(config)
            assert result["updates"] == 1
            assert result["episodes"] >= 1
            assert np.isfinite(result["last_loss"])
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

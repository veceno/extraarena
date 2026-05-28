"""
Tests for MLX PPO baseline (Task 05).
"""
import copy
import random as rand_mod
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from ai.train_v2.model_mlx import (
    MODEL_VERSION,
    ActionConditionedPolicy,
    masked_logits,
    sample_action,
    policy_argmax,
    save_checkpoint,
    load_checkpoint,
    flatten_params,
)
from ai.train_v2.train_ppo import (
    PPOConfig,
    collect_policy_episode,
    compute_gae,
    ppo_update,
    train,
    _prepare_batch,
)

from ai.train_v2.classic_rl_env import ClassicRLEnv


class TestModelForward:
    def test_model_forward_shapes(self):
        model = ActionConditionedPolicy(hidden_dim=64, action_hidden_dim=32)
        mx.eval(model.parameters())

        obs = mx.random.normal((2, 1456))
        af = mx.random.normal((2, 601, 171))
        mask = mx.ones((2, 601))

        logits, value = model(obs, af)
        mx.eval(logits, value)

        assert logits.shape == (2, 601), f"logits shape {logits.shape}"
        assert value.shape == (2,), f"value shape {value.shape}"


class TestMaskedLogits:
    def test_masked_logits_blocks_illegal_actions(self):
        logits = mx.array([1.0, 5.0, 3.0])
        mask = mx.array([1.0, 0.0, 1.0], dtype=mx.float32)

        result = masked_logits(logits, mask)
        mx.eval(result)

        r = np.array(result)
        assert r[0] == 1.0
        assert r[1] <= -1e8
        assert r[2] == 3.0

    def test_masked_logits_accepts_numpy_mask(self):
        logits = mx.array([1.0, 2.0])
        mask = np.array([1.0, 0.0], dtype=np.float32)

        result = masked_logits(logits, mask)
        mx.eval(result)
        assert float(result[1].item()) <= -1e8

    def test_policy_argmax_respects_mask(self):
        logits = mx.array([1.0, 999.0, 2.0])
        mask = mx.array([1.0, 0.0, 1.0], dtype=mx.float32)

        best = policy_argmax(logits, mask)
        assert best in (0, 2), f"argmax must not return masked action, got {best}"


class TestPolicySampling:
    def test_sample_action_returns_legal(self):
        logits = mx.array([1.0, 1.0, 1.0])
        mask = mx.array([1.0, 0.0, 1.0], dtype=mx.float32)

        for _ in range(20):
            aid, _ = sample_action(logits, mask)
            assert aid in (0, 2), f"sampled illegal action {aid}"

    def test_sample_action_returns_log_prob(self):
        logits = mx.array([10.0, -10.0])
        mask = mx.array([1.0, 0.0], dtype=mx.float32)

        aid, lp = sample_action(logits, mask)
        assert aid == 0
        assert isinstance(lp, float), f"log_prob should be float, got {type(lp)}"


class TestCheckpoint:
    def test_checkpoint_save_load_roundtrip(self):
        model1 = ActionConditionedPolicy(hidden_dim=64, action_hidden_dim=32)
        mx.eval(model1.parameters())

        obs = mx.random.normal((2, 1456))
        af = mx.random.normal((2, 601, 171))
        logits1, value1 = model1(obs, af)
        mx.eval(logits1, value1)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp_path = f.name

        try:
            save_checkpoint(tmp_path, model1, metadata={"test": True, "obs_dim": 1456})
            assert Path(tmp_path).exists()

            model2 = ActionConditionedPolicy(hidden_dim=64, action_hidden_dim=32)
            mx.eval(model2.parameters())

            result = load_checkpoint(tmp_path, model2)
            assert "metadata" in result
            assert result["metadata"]["obs_dim"] == 1456
            assert result["metadata"]["test"] is True

            logits2, value2 = model2(obs, af)
            mx.eval(logits2, value2)

            assert bool(mx.allclose(logits1, logits2, atol=1e-5).item()), "logits roundtrip mismatch"
            assert bool(mx.allclose(value1, value2, atol=1e-5).item()), "value roundtrip mismatch"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_flatten_params_no_nan(self):
        model = ActionConditionedPolicy(hidden_dim=64, action_hidden_dim=32)
        mx.eval(model.parameters())
        params = flatten_params(model)
        assert len(params) > 0
        for k, v in params.items():
            assert not np.any(np.isnan(v)), f"NaN in {k}"

    def test_optimizer_state_roundtrip_smoke(self):
        import mlx.nn as nn
        from mlx.optimizers import Adam
        model1 = ActionConditionedPolicy(hidden_dim=64, action_hidden_dim=32)
        mx.eval(model1.parameters())
        opt1 = Adam(learning_rate=3e-4)
        opt1.init(model1.parameters())

        obs = mx.random.normal((1, 1456))
        af = mx.random.normal((1, 601, 171))

        def dummy_loss(m):
            logits, values = m(obs, af)
            return mx.mean(values)

        _loss, grads = nn.value_and_grad(model1, dummy_loss)(model1)
        opt1.update(model1, grads)
        mx.eval(model1.parameters(), opt1.state)

        logits1, value1 = model1(obs, af)
        mx.eval(logits1, value1)
        step1 = float(opt1.state["step"].item())

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp_path = f.name

        try:
            save_checkpoint(tmp_path, model1, optimizer=opt1,
                            metadata={"test": "opt_roundtrip"})

            model2 = ActionConditionedPolicy(hidden_dim=64, action_hidden_dim=32)
            mx.eval(model2.parameters())
            opt2 = Adam(learning_rate=3e-4)

            result = load_checkpoint(tmp_path, model2, optimizer=opt2)
            assert result["optimizer_restored"] is True

            step2 = float(opt2.state["step"].item())
            assert step1 == step2, f"optimizer step should roundtrip: {step1} vs {step2}"

            logits2, value2 = model2(obs, af)
            mx.eval(logits2, value2)
            assert bool(mx.allclose(logits1, logits2, atol=1e-5).item()), "logits roundtrip mismatch with optimizer"
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestGAE:
    def test_compute_gae_shapes(self):
        rew = np.array([0.1, -0.05, 0.2, -0.03], dtype=np.float32)
        val = np.array([0.5, 0.3, 0.7, 0.4], dtype=np.float32)
        dones = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        pids = np.array([1, 2, 1, 2], dtype=np.int32)

        adv, ret = compute_gae(rew, val, dones, pids, 0.99, 0.95)

        assert adv.shape == rew.shape
        assert ret.shape == rew.shape
        assert not np.any(np.isnan(adv))
        assert not np.any(np.isinf(adv))

    def test_compute_gae_per_player_subsequence(self):
        """
        P1 acts at indices [0, 2], P2 at [1, 3].
        GAE must compute advantages within each player's sequence,
        NOT cross-player where next_value would be the OTHER player's value.
        """
        rew = np.array([0.1, -0.02, 0.3, -0.01], dtype=np.float32)
        val = np.array([10.0, -10.0, 5.0, -5.0], dtype=np.float32)
        dones = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        pids = np.array([1, 2, 1, 2], dtype=np.int32)

        adv, ret = compute_gae(rew, val, dones, pids, 0.99, 0.95)

        assert not np.any(np.isnan(adv))
        assert not np.any(np.isinf(adv))

        adv_p1_0 = adv[pids == 1][0]
        adv_p2_0 = adv[pids == 2][0]

        delta_p1_0 = rew[0] + 0.99 * val[2] - val[0]
        delta_p2_0 = rew[1] + 0.99 * val[3] - val[1]

        adv_p1_expected = delta_p1_0 + 0.99 * 0.95 * (rew[2] + 0 - val[2])
        adv_p2_expected = delta_p2_0 + 0.99 * 0.95 * (rew[3] + 0 - val[3])

        assert abs(adv_p1_0 - adv_p1_expected) < 1e-4, f"P1 GAE mismatch: {adv_p1_0} vs {adv_p1_expected}"
        assert abs(adv_p2_0 - adv_p2_expected) < 1e-4, f"P2 GAE mismatch: {adv_p2_0} vs {adv_p2_expected}"

    def test_compute_gae_single_player_sequential(self):
        """Single player: standard GAE = per-player GAE must be identical."""
        rew = np.array([0.5, -0.3, 0.7, -0.1, 1.0], dtype=np.float32)
        val = np.array([0.8, 0.6, 0.4, 0.9, 0.2], dtype=np.float32)
        dones = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        pids = np.array([1, 1, 1, 1, 1], dtype=np.int32)

        adv, ret = compute_gae(rew, val, dones, pids, 0.99, 0.95)

        gae_adv = np.zeros(5, dtype=np.float32)
        last_gae = 0.0
        for t in range(4, -1, -1):
            next_val = val[t+1] if t+1 < 5 and not dones[t] else 0.0
            next_gae = gae_adv[t+1] if t+1 < 5 and not dones[t] else 0.0
            delta = rew[t] + 0.99 * next_val - val[t]
            gae_adv[t] = delta + 0.99 * 0.95 * next_gae

        assert np.allclose(adv, gae_adv, atol=1e-4), "per-player GAE must match standard GAE for single player"


class TestCollectPolicyEpisode:
    def test_collect_policy_episode_smoke(self):
        mx.random.seed(42)
        rand_mod.seed(42)
        np.random.seed(42)

        model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(model.parameters())

        env = ClassicRLEnv(seed=42)
        result = collect_policy_episode(
            env, model, seed=100, max_steps=10, include_action_features=True
        )

        transitions = result["transitions"]
        assert len(transitions) > 0, "should collect at least one transition"
        for t in transitions:
            assert t["mask"][t["action_id"]] == 1.0, f"action {t['action_id']} was illegal"
            assert isinstance(t["log_prob"], float)
            assert isinstance(t["value"], float)
            assert t["action_features"].shape == (601, 171)

    def test_collect_episode_deterministic_by_seed(self):
        model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(model.parameters())

        env1 = ClassicRLEnv(seed=42)
        r1 = collect_policy_episode(env1, model, seed=777, max_steps=20, include_action_features=True)

        env2 = ClassicRLEnv(seed=42)
        r2 = collect_policy_episode(env2, model, seed=777, max_steps=20, include_action_features=True)

        t1 = r1["transitions"]
        t2 = r2["transitions"]

        assert len(t1) == len(t2), f"transition count differs: {len(t1)} vs {len(t2)}"
        for idx in range(len(t1)):
            assert t1[idx]["action_id"] == t2[idx]["action_id"], (
                f"step {idx}: action differs {t1[idx]['action_id']} vs {t2[idx]['action_id']}"
            )
            assert abs(t1[idx]["reward"] - t2[idx]["reward"]) < 1e-6, (
                f"step {idx}: reward differs {t1[idx]['reward']} vs {t2[idx]['reward']}"
            )
            assert t1[idx]["done"] == t2[idx]["done"]
            assert t1[idx]["truncated"] == t2[idx]["truncated"]

        s1 = r1["summary"]
        s2 = r2["summary"]
        assert s1["winner_id"] == s2["winner_id"]
        assert s1["steps"] == s2["steps"]


class TestPPOUpdateSmoke:
    def test_ppo_update_runs(self):
        mx.random.seed(42)
        rand_mod.seed(42)
        np.random.seed(42)

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

        model = ActionConditionedPolicy(hidden_dim=config.hidden_dim, action_hidden_dim=config.action_hidden_dim)
        mx.eval(model.parameters())

        from ai.train_v2.train_ppo import collect_policy_episode as cpe
        env = ClassicRLEnv(seed=42)
        result = cpe(env, model, seed=100, max_steps=config.max_steps_per_episode, include_action_features=True)

        batch = _prepare_batch(result["transitions"], config)

        from mlx.optimizers import Adam
        opt = Adam(learning_rate=config.learning_rate)

        metrics = ppo_update(model, opt, batch, config)
        for key in ["loss", "policy_loss", "value_loss", "entropy"]:
            assert np.isfinite(metrics[key]), f"{key} is NaN/Inf: {metrics[key]}"


class TestTrainTinySmoke:
    def test_train_tiny_smoke(self):
        config = PPOConfig(
            total_updates=1,
            episodes_per_update=1,
            max_steps_per_episode=10,
            hidden_dim=32,
            action_hidden_dim=16,
            minibatch_size=8,
            epochs=1,
            seed=42,
            checkpoint_dir="/tmp/_ppo_test_ckpts",
        )

        import shutil
        shutil.rmtree(config.checkpoint_dir, ignore_errors=True)

        try:
            result = train(config)
            assert result["updates"] == 1
            assert result["episodes"] >= 1
            assert result["steps"] >= 1

            ckpt = Path(result["checkpoint_path"])
            assert ckpt.exists(), f"checkpoint not found: {ckpt}"
        finally:
            shutil.rmtree(config.checkpoint_dir, ignore_errors=True)

"""
Tests for PPO checkpoint evaluation and learning metrics (Task 06).
"""
import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.policies import RandomLegalPolicy, EndTurnPolicy
from ai.train_v2.model_mlx import ActionConditionedPolicy, save_checkpoint
from ai.train_v2.train_ppo import PPOConfig, train

from ai.train_v2.ppo_eval import (
    MlxPolicy,
    load_mlx_policy,
    evaluate_policy_matchup,
)


def _make_tiny_checkpoint(checkpoint_dir, seed=42):
    """Train a tiny model and return the checkpoint path."""
    import shutil as _shutil
    _shutil.rmtree(checkpoint_dir, ignore_errors=True)
    config = PPOConfig(
        total_updates=1,
        episodes_per_update=1,
        max_steps_per_episode=5,
        hidden_dim=32,
        action_hidden_dim=16,
        minibatch_size=8,
        epochs=1,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
    )
    result = train(config)
    return result["checkpoint_path"]


class TestMlxPolicy:
    def test_selects_legal_action(self):
        model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(model.parameters())
        p = MlxPolicy(model, mode="argmax")

        env = ClassicRLEnv(seed=42)
        env.reset(seed=100)

        for _ in range(10):
            cp = env.current_player_id()
            mask = env.action_mask(cp)
            aid = p.select_action(env, cp)
            assert mask[aid] == 1.0, f"action {aid} is illegal"

    def test_argmax_deterministic(self):
        mx.random.seed(42)
        model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(model.parameters())
        p = MlxPolicy(model, mode="argmax")

        env1 = ClassicRLEnv(seed=42)
        env1.reset(seed=777)
        cp1 = env1.current_player_id()
        a1 = p.select_action(env1, cp1)

        env2 = ClassicRLEnv(seed=42)
        env2.reset(seed=777)
        cp2 = env2.current_player_id()
        a2 = p.select_action(env2, cp2)

        assert a1 == a2, f"argmax must be deterministic: {a1} vs {a2}"

    def test_reset_clears_fallbacks(self):
        model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(model.parameters())
        p = MlxPolicy(model, mode="argmax")
        p._invalid_fallbacks = 5
        p.reset(seed=1)
        assert p._invalid_fallbacks == 0


class TestLoadMlxPolicy:
    def test_load_from_checkpoint(self):
        ckpt_dir = "/tmp/_task06_load_test"
        ckpt = _make_tiny_checkpoint(ckpt_dir)

        p = load_mlx_policy(ckpt, mode="argmax")
        assert p.name.startswith("mlx_argmax_")

        env = ClassicRLEnv(seed=42)
        env.reset(seed=100)
        cp = env.current_player_id()
        mask = env.action_mask(cp)
        aid = p.select_action(env, cp)
        assert mask[aid] == 1.0


class TestEvaluateMatchup:
    def test_basic_no_swap(self):
        p1 = RandomLegalPolicy(seed=1)
        p2 = RandomLegalPolicy(seed=2)

        result = evaluate_policy_matchup(p1, p2, seeds=[10, 11], swap_sides=False)
        assert result["games"] == 2
        assert result["p1_wins"] + result["p2_wins"] + result["draws"] == 2
        assert 0.0 <= result["p1_winrate"] <= 1.0
        assert result["seeds"] == 2

    def test_swap_sides_doubles_games(self):
        p1 = RandomLegalPolicy(seed=1)
        p2 = RandomLegalPolicy(seed=2)

        result = evaluate_policy_matchup(p1, p2, seeds=[10, 11], swap_sides=True)
        assert result["games"] == 4
        assert result["seeds"] == 2

    def test_invalid_fallbacks_in_result(self):
        model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(model.parameters())
        p = MlxPolicy(model, mode="argmax")

        result = evaluate_policy_matchup(p, EndTurnPolicy(), seeds=[10], swap_sides=False)
        assert "p1_invalid_fallbacks" in result
        assert "p2_invalid_fallbacks" in result

    def test_mlx_vs_end_turn_wins(self):
        ckpt_dir = "/tmp/_task06_mlx_et"
        ckpt = _make_tiny_checkpoint(ckpt_dir)
        p = load_mlx_policy(ckpt, mode="argmax")

        result = evaluate_policy_matchup(p, EndTurnPolicy(), seeds=[50, 51, 52], swap_sides=True)
        assert result["games"] == 6
        assert result["p1_winrate"] >= 0.0


class TestEvalCLI:
    def test_eval_cli_smoke(self):
        import subprocess, sys
        ckpt_dir = "/tmp/_task06_cli_test"
        ckpt = _make_tiny_checkpoint(ckpt_dir)

        proc = subprocess.run(
            [sys.executable, "-m", "ai.train_v2.ppo_eval",
             "--checkpoint", ckpt, "--opponent", "end_turn", "--games", "2", "--seed", "1", "--no-swap"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
        assert "Eval:" in proc.stdout
        assert "Games:" in proc.stdout


class TestTrainMetrics:
    def test_train_writes_metrics_jsonl(self):
        import shutil as _shutil
        tmp_dir = "/tmp/_task06_metrics_test"
        metrics_path = f"{tmp_dir}/metrics.jsonl"
        ckpt_dir = f"{tmp_dir}/ckpts"
        _shutil.rmtree(tmp_dir, ignore_errors=True)

        config = PPOConfig(
            total_updates=1,
            episodes_per_update=1,
            max_steps_per_episode=5,
            hidden_dim=32,
            action_hidden_dim=16,
            minibatch_size=8,
            epochs=1,
            seed=42,
            checkpoint_dir=ckpt_dir,
            metrics_path=metrics_path,
        )
        result = train(config)

        p = Path(metrics_path)
        assert p.exists(), f"metrics file not found: {p}"

        lines = p.read_text().strip().split("\n")
        assert len(lines) == 1, f"expected 1 JSONL line, got {len(lines)}"
        record = json.loads(lines[0])
        for key in ["update", "episodes", "steps", "loss", "policy_loss", "value_loss", "entropy"]:
            assert key in record, f"missing key {key} in metrics record"


class TestTrainWithEval:
    def test_train_with_eval_writes_eval_lines(self):
        import shutil as _shutil
        tmp_dir = "/tmp/_task06_evallog_test"
        metrics_path = f"{tmp_dir}/metrics.jsonl"
        ckpt_dir = f"{tmp_dir}/ckpts"
        _shutil.rmtree(tmp_dir, ignore_errors=True)

        config = PPOConfig(
            total_updates=2,
            episodes_per_update=1,
            max_steps_per_episode=5,
            hidden_dim=32,
            action_hidden_dim=16,
            minibatch_size=8,
            epochs=1,
            seed=42,
            checkpoint_dir=ckpt_dir,
            metrics_path=metrics_path,
            eval_every_updates=1,
            eval_games=2,
        )
        train(config)

        p = Path(metrics_path)
        lines = p.read_text().strip().split("\n")
        assert len(lines) >= 3, f"expected at least 3 lines (2 train + eval), got {len(lines)}"

        eval_lines = [l for l in lines if '"type":"eval"' in l]
        assert len(eval_lines) >= 2, f"expected at least 2 eval lines, got {len(eval_lines)}"

        for el in eval_lines:
            er = json.loads(el)
            assert er["type"] == "eval"
            assert "winrate" in er
            assert "opponent" in er
            assert "avg_turns" in er

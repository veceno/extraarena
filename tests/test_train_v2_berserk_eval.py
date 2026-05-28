"""
Tests for berserk_eval shadow harness (Task 09).
"""
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ai.train_v2.berserk_eval import (
    BerserkBrainPolicy,
    make_train_v2_berserk_brain,
    evaluate_berserk_matchup,
    compare_berserk_to_onnx_policy,
)
from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.policies import RandomLegalPolicy, EndTurnPolicy
from ai.bot_brain import BerserkInference


def _make_tiny_onnx(tmp_path):
    from ai.train_v2.train_ppo import PPOConfig, train
    from ai.train_v2.export_onnx import export_checkpoint_to_onnx

    ckpt_dir = str(tmp_path / "ckpts")
    config = PPOConfig(
        total_updates=1, episodes_per_update=1, max_steps_per_episode=5,
        hidden_dim=32, action_hidden_dim=16, minibatch_size=8, epochs=1,
        seed=42, checkpoint_dir=ckpt_dir,
    )
    result = train(config)
    onnx_path = str(tmp_path / "model.onnx")
    export_checkpoint_to_onnx(result["checkpoint_path"], onnx_path, opset=17)
    return onnx_path


@pytest.fixture
def tiny_onnx(tmp_path):
    return _make_tiny_onnx(tmp_path)


class TestMakeBerserkBrain:
    def test_make_train_v2_berserk_brain_loads(self, tiny_onnx):
        brain = make_train_v2_berserk_brain(tiny_onnx, selection="argmax")
        assert "test" in brain.sessions
        s = brain.sessions["test"]
        assert s["format"] == "train_v2_classic_v1"
        assert s["obs_dim"] == 1456
        assert s["max_candidate_actions"] == 601

        from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState
        from core.engine import ArenaEnvironment
        from uuid import uuid4

        h1 = CardInstance(instance_id=uuid4(), card_id=0, card_type=CardType.HERO, hp=30, max_hp=30)
        h2 = CardInstance(instance_id=uuid4(), card_id=0, card_type=CardType.HERO, hp=30, max_hp=30)
        p1 = PlayerState(user_id=1, hero=h1, mana=1, max_mana=1, hand=[], board=[], deck=[])
        p2 = PlayerState(user_id=2, hero=h2, mana=0, max_mana=0, hand=[], board=[], deck=[])
        gs = GameState(p1=p1, p2=p2, current_turn_owner_id=1)
        env = ArenaEnvironment(gs)
        legal = env.get_legal_actions(1)

        idx = brain.get_action(gs, 1, legal, difficulty="test")
        assert 0 <= idx < len(legal), f"brain returned illegal index {idx}"


class TestBerserkPolicy:
    def test_selects_stable_legal_action(self, tiny_onnx):
        brain = make_train_v2_berserk_brain(tiny_onnx, selection="argmax")
        pol = BerserkBrainPolicy(brain, difficulty="test")

        env = ClassicRLEnv(seed=42)
        env.reset(seed=100)

        for _ in range(10):
            cp = env.current_player_id()
            mask = env.action_mask(cp)
            aid = pol.select_action(env, cp)
            assert 0 <= aid <= 600, f"aid {aid} out of range"
            assert mask[aid] == 1.0, f"action {aid} illegal"

    def test_invalid_legal_index_fallback(self, tiny_onnx):
        brain = make_train_v2_berserk_brain(tiny_onnx, selection="argmax")
        pol = BerserkBrainPolicy(brain, difficulty="test")

        env = ClassicRLEnv(seed=42)
        env.reset(seed=100)
        cp = env.current_player_id()

        # Return garbage legal_idx
        brain.sessions["test"]["_orig_get"] = brain.get_action
        brain.get_action = lambda *a, **kw: 999

        try:
            aid = pol.select_action(env, cp)
            assert aid == 0, f"expected fallback to 0, got {aid}"
            assert pol.invalid_actions == 1, f"tracker should be 1, got {pol.invalid_actions}"
        finally:
            brain.get_action = brain.sessions["test"].pop("_orig_get")


class TestBerserkEval:
    def test_basic(self, tiny_onnx):
        brain = make_train_v2_berserk_brain(tiny_onnx, selection="argmax")
        pol = BerserkBrainPolicy(brain, difficulty="test")
        opp = RandomLegalPolicy(seed=1)

        result = evaluate_berserk_matchup(pol, opp, seeds=[10, 11], swap_sides=False)
        assert result["games"] == 2
        assert result["p1_wins"] + result["p2_wins"] + result["draws"] == 2

    def test_latency_fields_present(self, tiny_onnx):
        brain = make_train_v2_berserk_brain(tiny_onnx, selection="argmax")
        pol = BerserkBrainPolicy(brain, difficulty="test")
        opp = EndTurnPolicy()

        result = evaluate_berserk_matchup(pol, opp, seeds=[10], swap_sides=False)
        assert result["p1_latency_ms_p50"] >= 0
        assert result["p1_latency_ms_p95"] >= 0
        assert result["p2_latency_ms_p50"] == 0.0  # EndTurn has no brain latencies
        assert result["p1_brain_invalid_actions"] >= 0
        assert result["p2_brain_invalid_actions"] >= 0
        assert not np.isnan(result["p1_latency_ms_p50"])

    def test_versus_direct_onnx_parity(self, tiny_onnx):
        result = compare_berserk_to_onnx_policy(tiny_onnx, seed=42, steps=10, selection="argmax")
        assert result["checked"] > 0
        assert result["mismatches"] == 0, f"argmax must have zero mismatches, got {result['mismatches']}"
        assert result["matches"] == result["checked"]

    def test_eval_cli_smoke(self, tiny_onnx):
        import subprocess, sys

        proc = subprocess.run(
            [sys.executable, "-m", "ai.train_v2.berserk_eval",
             "--onnx", tiny_onnx, "--opponent", "end_turn",
             "--games", "2", "--seed", "1", "--no-swap", "--selection", "argmax"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, f"CLI failed:\n{proc.stderr}"
        assert "Eval:" in proc.stdout
        assert "Games:" in proc.stdout
        assert "latency:" in proc.stdout

    def test_eval_with_parity_flag(self, tiny_onnx):
        import subprocess, sys

        proc = subprocess.run(
            [sys.executable, "-m", "ai.train_v2.berserk_eval",
             "--onnx", tiny_onnx, "--opponent", "end_turn",
             "--games", "1", "--seed", "1", "--no-swap", "--parity", "--selection", "argmax"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, f"CLI with --parity failed:\n{proc.stderr}"
        assert "Parity:" in proc.stdout

    def test_swap_sides_with_berserk_brain(self, tiny_onnx):
        brain = make_train_v2_berserk_brain(tiny_onnx, selection="argmax")
        pol = BerserkBrainPolicy(brain, difficulty="test")
        opp = EndTurnPolicy()

        result = evaluate_berserk_matchup(pol, opp, seeds=[10], swap_sides=True)
        assert result["games"] == 2
        assert result["seeds"] == 1
        assert result["p1_latency_ms_p50"] >= 0
        assert result["p2_latency_ms_p50"] >= 0  # EndTurn has no latencies but field must be present

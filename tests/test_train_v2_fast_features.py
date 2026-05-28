"""
Tests for fast action features (Task 10) — shape stability, prefix match, zero preview.
"""
import numpy as np
import pytest

from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.classic_actions_v1 import (
    encode_action_features,
    ACTION_FEATURE_DIM,
    MAX_CANDIDATE_ACTIONS,
)


class TestFastActionFeatures:
    def test_fast_action_features_shape(self):
        env = ClassicRLEnv(seed=42)
        env.reset(seed=100)
        cp = env.current_player_id()
        st = env.clone_state()

        fast = encode_action_features(st, cp, include_preview=False)
        assert fast.shape == (MAX_CANDIDATE_ACTIONS, ACTION_FEATURE_DIM)
        assert fast.dtype == np.float32

    def test_fast_action_features_prefix_matches_full(self):
        env = ClassicRLEnv(seed=42)
        env.reset(seed=100)
        cp = env.current_player_id()
        st = env.clone_state()

        full = encode_action_features(st, cp, include_preview=True)
        fast = encode_action_features(st, cp, include_preview=False)

        assert fast.shape == full.shape
        assert np.allclose(fast[:, :142], full[:, :142], atol=1e-7), (
            "prefix [0:142] must be identical between fast and full"
        )

    def test_fast_action_features_preview_zero(self):
        env = ClassicRLEnv(seed=42)
        env.reset(seed=100)
        cp = env.current_player_id()
        st = env.clone_state()

        fast = encode_action_features(st, cp, include_preview=False)
        assert np.all(fast[:, 142:171] == 0.0), (
            "preview slice [142:171] must be all zeros in fast mode"
        )

    def test_full_action_features_preview_nonzero_for_some(self):
        env = ClassicRLEnv(seed=42)
        env.reset(seed=100)

        # Create a state with a warrior in hand so there are multiple masked actions
        from core.state import CardInstance, CardType, PlayerState, GameState, GameStatus
        from uuid import uuid4
        h1 = CardInstance(instance_id=uuid4(), card_id=0, card_type=CardType.HERO, hp=30, max_hp=30)
        h2 = CardInstance(instance_id=uuid4(), card_id=0, card_type=CardType.HERO, hp=30, max_hp=30)
        w = CardInstance(instance_id=uuid4(), card_id=100, card_type=CardType.WARRIOR,
                         mana_cost=3, attack=4, hp=5, max_hp=5, mechanics=[], is_ready=False)
        p1 = PlayerState(user_id=1, hero=h1, mana=10, max_mana=10, hand=[w], board=[], deck=[])
        p2 = PlayerState(user_id=2, hero=h2, mana=0, max_mana=0, hand=[], board=[], deck=[])
        gs = GameState(p1=p1, p2=p2, current_turn_owner_id=1)

        full = encode_action_features(gs, 1, include_preview=True)
        from ai.train_v2.classic_actions_v1 import build_action_mask
        mask = build_action_mask(gs, 1)

        any_nonzero = False
        for aid in range(MAX_CANDIDATE_ACTIONS):
            if mask[aid] == 1.0:
                if np.any(np.abs(full[aid, 142:171]) > 1e-8):
                    any_nonzero = True
                    break

        assert any_nonzero, "expected at least one masked action to have nonzero preview"


class TestEnvActionFeatures:
    def test_action_features_include_preview_flag(self):
        env = ClassicRLEnv(seed=42)
        env.reset(seed=100)

        full = env.action_features(include_preview=True)
        fast = env.action_features(include_preview=False)

        assert full.shape == fast.shape
        assert np.allclose(full[:, :142], fast[:, :142], atol=1e-7)
        assert np.all(fast[:, 142:171] == 0.0)

    def test_default_is_full_preview(self):
        from core.state import CardInstance, CardType, PlayerState, GameState, GameStatus
        from uuid import uuid4

        h1 = CardInstance(instance_id=uuid4(), card_id=0, card_type=CardType.HERO, hp=30, max_hp=30)
        h2 = CardInstance(instance_id=uuid4(), card_id=0, card_type=CardType.HERO, hp=30, max_hp=30)
        w = CardInstance(instance_id=uuid4(), card_id=100, card_type=CardType.WARRIOR,
                         mana_cost=3, attack=4, hp=5, max_hp=5, mechanics=[], is_ready=False)
        p1 = PlayerState(user_id=1, hero=h1, mana=10, max_mana=10, hand=[w], board=[], deck=[])
        p2 = PlayerState(user_id=2, hero=h2, mana=0, max_mana=0, hand=[], board=[], deck=[])
        gs = GameState(p1=p1, p2=p2, current_turn_owner_id=1)
        env = ClassicRLEnv(seed=42)
        # Inject state directly
        from core.engine import ArenaEnvironment
        env._env = ArenaEnvironment(gs)

        fast = env.action_features(include_preview=False)
        default_result = env.action_features()
        assert np.any(default_result[:, 142:171] != 0.0), (
            "default (include_preview=True) should have non-zero preview for some actions"
        )


class TestPPOCollectWithoutPreview:
    def test_ppo_collect_without_preview(self):
        from ai.train_v2.train_ppo import PPOConfig, train
        import shutil, tempfile

        tmp = tempfile.mkdtemp()
        try:
            config = PPOConfig(
                total_updates=1,
                episodes_per_update=1,
                max_steps_per_episode=5,
                hidden_dim=32,
                action_hidden_dim=16,
                minibatch_size=8,
                epochs=1,
                seed=42,
                checkpoint_dir=f"{tmp}/ckpts",
                include_preview_features=False,
            )
            result = train(config)
            assert result["updates"] == 1
            assert result["episodes"] >= 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_ppo_collect_with_preview(self):
        from ai.train_v2.train_ppo import PPOConfig, train
        import shutil, tempfile

        tmp = tempfile.mkdtemp()
        try:
            config = PPOConfig(
                total_updates=1,
                episodes_per_update=1,
                max_steps_per_episode=5,
                hidden_dim=32,
                action_hidden_dim=16,
                minibatch_size=8,
                epochs=1,
                seed=42,
                checkpoint_dir=f"{tmp}/ckpts",
                include_preview_features=True,
            )
            result = train(config)
            assert result["updates"] == 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestBenchmarkFeatureModes:
    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("onnx"),
        reason="onnx not installed",
    )
    def test_benchmark_feature_modes_smoke(self):
        from ai.train_v2.berserk_eval import benchmark_feature_modes
        from ai.train_v2.export_onnx import export_checkpoint_to_onnx
        from ai.train_v2.train_ppo import PPOConfig, train
        import shutil, tempfile

        tmp = tempfile.mkdtemp()
        try:
            config = PPOConfig(total_updates=1, episodes_per_update=1, max_steps_per_episode=5,
                               hidden_dim=32, action_hidden_dim=16, minibatch_size=8, epochs=1,
                               seed=42, checkpoint_dir=f"{tmp}/ckpts")
            ckpt = train(config)["checkpoint_path"]
            onnx_path = f"{tmp}/model.onnx"
            export_checkpoint_to_onnx(ckpt, onnx_path, opset=17)

            result = benchmark_feature_modes(onnx_path, steps=5)
            assert result["steps"] == 5
            assert result["fast_features_ms_p50"] >= 0
            assert result["full_features_ms_p50"] >= 0
            assert result["brain_get_action_ms_p50"] >= 0
            assert result["fast_vs_full_speedup"] >= 0
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

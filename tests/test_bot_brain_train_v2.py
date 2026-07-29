"""
Tests for TrainV2 ONNX runtime in BerserkInference (Task 08).
"""
import json
import random as rand_mod
import shutil
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from core.actions import BaseAction, EndTurnAction, PlayCardAction, AttackAction
from core.engine import ArenaEnvironment
from core.state import CardInstance, CardType, GameState, GameStatus, PlayerState
from uuid import uuid4


def _make_hero(hp=30):
    return CardInstance(
        instance_id=uuid4(), card_id=0, name="Hero", card_type=CardType.HERO,
        mana_cost=0, attack=0, hp=hp, max_hp=hp, mechanics=[], is_ready=True,
    )


def _make_warrior(mana_cost=3, attack=4, hp=5, name="W", mechanics=None):
    return CardInstance(
        instance_id=uuid4(), card_id=100, name=name, card_type=CardType.WARRIOR,
        mana_cost=mana_cost, attack=attack, hp=hp, max_hp=hp,
        mechanics=mechanics or [], is_ready=True,
    )


def _make_potion(mana_cost=2, name="P", mechanics=None):
    return CardInstance(
        instance_id=uuid4(), card_id=200, name=name, card_type=CardType.POTION,
        mana_cost=mana_cost, attack=0, hp=0, max_hp=0,
        mechanics=mechanics or [], is_ready=False,
    )


def _classic_state_with_actions():
    hero1 = _make_hero(hp=30)
    hero2 = _make_hero(hp=30)
    w = _make_warrior(mana_cost=3, attack=4, hp=5, name="W1")
    p1 = PlayerState(user_id=1, hero=hero1, mana=5, max_mana=5,
                     hand=[w], board=[], deck=[])
    p2 = PlayerState(user_id=2, hero=hero2, mana=0, max_mana=0,
                     hand=[], board=[], deck=[])
    gs = GameState(p1=p1, p2=p2, current_turn_owner_id=1)
    env = ArenaEnvironment(gs)
    legal = env.get_legal_actions(1)
    return gs, legal


@pytest.fixture
def train_v2_onnx_model(tmp_path):
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


class TestTrainV2Profile:
    def test_production_v4_profiles_disable_preview_features(self):
        from infrastructure.config import BOT_MODEL_PROFILES

        assert BOT_MODEL_PROFILES["extra-lr-v4-micro"]["include_preview_features"] is False

    def test_production_model_registry_excludes_legacy_profiles(self):
        from infrastructure.config import BOT_MODEL_PROFILES

        assert set(BOT_MODEL_PROFILES) == {
            "extra-lr-v4-micro",
            "extra-lr-v5-lite",
            "extra-lr-v5",
            "extra-lr-v5-ultra",
        }

    def test_train_v2_profile_loads(self, train_v2_onnx_model):
        from ai.bot_brain import BerserkInference

        profile = {
            "model_path": train_v2_onnx_model,
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": (1.0, 1.0),
            "selection": "argmax",
        }

        brain = BerserkInference(profiles={"test": profile})
        sess = brain.sessions["test"]
        assert sess["format"] == "train_v2_classic_v1"
        assert sess["obs_dim"] == 1456
        assert sess["action_feature_dim"] == 171
        assert sess["max_candidate_actions"] == 601
        assert sess["selection"] == "argmax"
        assert "observation" in sess["input_names"] or "obs" in str(sess["input_names"])

    def test_train_v2_get_action_returns_legal_index(self, train_v2_onnx_model):
        from ai.bot_brain import BerserkInference

        profile = {
            "model_path": train_v2_onnx_model,
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": (1.0, 1.0),
            "selection": "argmax",
        }

        brain = BerserkInference(profiles={"test": profile})

        gs, legal = _classic_state_with_actions()
        idx = brain.get_action(gs, 1, legal, difficulty="test")
        assert 0 <= idx < len(legal), f"illegal index {idx}, legal count={len(legal)}"

    def test_train_v2_argmax_matches_direct_onnx_policy(self, train_v2_onnx_model):
        from ai.bot_brain import BerserkInference
        from ai.train_v2.onnx_policy import OnnxActionPolicy
        from ai.train_v2.classic_rl_env import ClassicRLEnv
        from ai.train_v2.classic_actions_v1 import decode_action

        profile = {
            "model_path": train_v2_onnx_model,
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": (1.0, 1.0),
            "selection": "argmax",
        }

        brain = BerserkInference(profiles={"test": profile})
        onnx_pol = OnnxActionPolicy(train_v2_onnx_model, mode="argmax")

        env = ClassicRLEnv(seed=42)
        env.reset(seed=777)

        for _ in range(5):
            cp = env.current_player_id()
            st = env.clone_state()
            engine = ArenaEnvironment(st)
            legal = engine.get_legal_actions(cp)

            onnx_aid = onnx_pol.select_action(env, cp)
            onnx_decoded = decode_action(st, cp, onnx_aid)

            _, _, _, _, _ = env.step(onnx_aid)

            if onnx_decoded is None:
                continue

            brain_idx = brain.get_action(st, cp, legal, difficulty="test")
            if brain_idx >= len(legal):
                continue

            legal_action = legal[brain_idx]
            assert onnx_decoded.to_dict().get("type") == legal_action.to_dict().get("type"), (
                f"direct ONNX={onnx_decoded.to_dict()}, brain legal={legal_action.to_dict()}"
            )

    def test_train_v2_fallback_on_unmatched_action(self, train_v2_onnx_model):
        from ai.bot_brain import BerserkInference

        profile = {
            "model_path": train_v2_onnx_model,
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": (1.0, 1.0),
            "selection": "argmax",
        }

        brain = BerserkInference(profiles={"test": profile})
        gs, legal = _classic_state_with_actions()

        with patch.object(
            brain, "_find_matching_legal_action_index", return_value=None
        ):
            idx = brain.get_action(gs, 1, legal, difficulty="test")
            assert 0 <= idx < len(legal), f"fallback returned illegal index {idx}"

    def test_legacy_profile_is_ignored(self):
        # Create a tiny dummy ONNX model with one input/output (633 format)
        import torch
        import warnings
        m = torch.nn.Linear(633, 1)
        dummy = torch.randn(1, 633)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            torch.onnx.export(m, dummy, "/tmp/_t08_dummy_legacy.onnx",
                              opset_version=17, input_names=["obs"], output_names=["logits"],
                              dynamo=False)

        from ai.bot_brain import BerserkInference
        profile = {
            "model_path": "/tmp/_t08_dummy_legacy.onnx",
            "obs_dim": 633,
            "temperature_range": (0.5, 0.5),
            "selection": "softmax",
        }

        brain = BerserkInference(profiles={"legacy": profile})
        assert brain.sessions == {}

    def test_action_matching_helper(self):
        from ai.bot_brain import BerserkInference

        h1 = _make_hero(30)
        h2 = _make_hero(30)
        w1 = _make_warrior(name="A")
        p1 = PlayerState(user_id=1, hero=h1, mana=10, max_mana=10,
                         hand=[w1], board=[], deck=[])
        p2 = PlayerState(user_id=2, hero=h2, mana=10, max_mana=10,
                         hand=[], board=[], deck=[])
        gs = GameState(p1=p1, p2=p2, current_turn_owner_id=1)
        env = ArenaEnvironment(gs)
        legal = env.get_legal_actions(1)

        # EndTurn
        et = EndTurnAction()
        idx = BerserkInference._find_matching_legal_action_index(et, legal)
        assert idx is not None
        assert legal[idx].to_dict()["type"] == "end_turn"

        # PlayCard by hand_index+target_id (ignore position)
        for la in legal:
            if la.to_dict().get("type") == "play_card":
                idx = BerserkInference._find_matching_legal_action_index(la, [la])
                assert idx == 0
                # Different position should still match
                clone_dict = la.to_dict()
                clone_dict["position"] = 999
                fake = PlayCardAction(
                    hand_index=clone_dict.get("hand_index", 0),
                    target_id=clone_dict.get("target_id"),
                    position=999,
                )
                idx2 = BerserkInference._find_matching_legal_action_index(fake, [la])
                assert idx2 == 0, "different position should still match"
                break

        # Attack by attacker+target_is_hero+target_id
        w_a = _make_warrior(name="Att", attack=3)
        p1.board = [w_a]
        env = ArenaEnvironment(gs)
        legal = env.get_legal_actions(1)
        for la in legal:
            if la.to_dict().get("type") == "attack":
                idx = BerserkInference._find_matching_legal_action_index(la, [la])
                assert idx == 0
                break

    def test_train_v2_decode_action_none_fallback(self, train_v2_onnx_model):
        from ai.bot_brain import BerserkInference
        from ai.train_v2.classic_actions_v1 import decode_action as real_decode

        profile = {
            "model_path": train_v2_onnx_model,
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": (1.0, 1.0),
            "selection": "argmax",
        }

        brain = BerserkInference(profiles={"test": profile})
        gs, legal = _classic_state_with_actions()

        with patch("ai.train_v2.classic_actions_v1.decode_action", return_value=None):
            idx = brain.get_action(gs, 1, legal, difficulty="test")
            assert 0 <= idx < len(legal), f"decode=None fallback: illegal index {idx}"

    def test_legal_fallback_prefers_hero_attack_then_attack_play_card_end_turn(self):
        from ai.bot_brain import _legal_fallback

        end_turn = EndTurnAction()
        play_card = PlayCardAction(hand_index=0)
        board_attack = AttackAction(attacker_id="a1", target_id="unit1", target_is_hero=False)
        hero_attack = AttackAction(attacker_id="a2", target_id=None, target_is_hero=True)

        assert _legal_fallback([end_turn, play_card, board_attack, hero_attack]) == 3
        assert _legal_fallback([end_turn, play_card, board_attack]) == 2
        assert _legal_fallback([end_turn, play_card]) == 1
        assert _legal_fallback([end_turn]) == 0
        assert _legal_fallback([]) == 0

    def test_decode_none_fallback_logs_reason_and_uses_simple_legal_policy(
        self, train_v2_onnx_model, caplog, monkeypatch
    ):
        from ai.bot_brain import BerserkInference
        import ai.train_v2.classic_actions_v1 as action_codec

        profile = {
            "model_path": train_v2_onnx_model,
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": (1.0, 1.0),
            "selection": "argmax",
        }

        brain = BerserkInference(profiles={"test": profile})
        h1 = _make_hero(30)
        h2 = _make_hero(30)
        attacker = _make_warrior(name="Ready", attack=3)
        p1 = PlayerState(user_id=1, hero=h1, mana=0, max_mana=0, hand=[], board=[attacker], deck=[])
        p2 = PlayerState(user_id=2, hero=h2, mana=0, max_mana=0, hand=[], board=[], deck=[])
        gs = GameState(p1=p1, p2=p2, current_turn_owner_id=1)
        legal = ArenaEnvironment(gs).get_legal_actions(1)

        mask = np.zeros(601, dtype=np.float32)
        mask[0] = 1.0
        monkeypatch.setattr(
            action_codec,
            "build_action_mask",
            lambda *_args, **_kwargs: mask,
        )

        with patch("ai.train_v2.classic_actions_v1.decode_action", return_value=None):
            idx = brain.get_action(gs, 1, legal, difficulty="test")

        assert legal[idx].to_dict()["type"] == "attack"
        assert legal[idx].to_dict()["target_is_hero"] is True
        assert "decode_action returned None, fallback" in caplog.text


class TestTrainV2IONames:
    def test_train_v2_logits_only_outputs_are_runnable(self):
        from ai.bot_brain import BerserkInference

        obs_name, af_name, output_names = BerserkInference._resolve_train_v2_io_names(
            {
                "input_names": ["observation", "action_features"],
                "output_names": ["logits"],
            }
        )

        assert obs_name == "observation"
        assert af_name == "action_features"
        assert output_names == ["logits"]

    def test_train_v2_uses_session_input_names(self, train_v2_onnx_model):
        from ai.bot_brain import BerserkInference

        profile = {
            "model_path": train_v2_onnx_model,
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": (1.0, 1.0),
            "selection": "argmax",
        }

        brain = BerserkInference(profiles={"test": profile})

        # Override input_names to non-standard names
        brain.sessions["test"]["input_names"] = ["obs_custom", "af_custom"]

        gs, legal = _classic_state_with_actions()

        # Replace session with a fake that asserts the correct feed dict keys
        real_session = brain.sessions["test"]["session"]

        class FakeSession:
            def __init__(self, real):
                self._real = real

            def run(self, output_names, input_feed):
                assert "obs_custom" in input_feed, f"expected obs_custom in feed, got {list(input_feed.keys())}"
                assert "af_custom" in input_feed, f"expected af_custom in feed, got {list(input_feed.keys())}"
                # Use real data but with custom feed keys
                return self._real.run(output_names, {
                    "observation": input_feed["obs_custom"],
                    "action_features": input_feed["af_custom"],
                })

        brain.sessions["test"]["session"] = FakeSession(real_session)

        idx = brain.get_action(gs, 1, legal, difficulty="test")
        assert 0 <= idx < len(legal), f"custom input names: illegal index {idx}"

    def test_train_v2_bad_input_names_fallback(self, train_v2_onnx_model):
        from ai.bot_brain import BerserkInference

        profile = {
            "model_path": train_v2_onnx_model,
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": (1.0, 1.0),
            "selection": "argmax",
        }

        brain = BerserkInference(profiles={"test": profile})
        brain.sessions["test"]["input_names"] = ["only_one"]

        gs, legal = _classic_state_with_actions()
        idx = brain.get_action(gs, 1, legal, difficulty="test")
        assert 0 <= idx < len(legal), f"bad input names fallback: illegal index {idx}"

    def test_train_v2_profile_respects_include_preview_false(self, train_v2_onnx_model, monkeypatch):
        from ai.bot_brain import BerserkInference
        import ai.train_v2.classic_actions_v1 as action_codec

        profile = {
            "model_path": train_v2_onnx_model,
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": (1.0, 1.0),
            "selection": "argmax",
            "include_preview_features": False,
        }

        seen_include_preview = []
        real_encode = action_codec.encode_action_features

        def wrapped_encode_action_features(*args, **kwargs):
            seen_include_preview.append(kwargs.get("include_preview", True))
            return real_encode(*args, **kwargs)

        monkeypatch.setattr(action_codec, "encode_action_features", wrapped_encode_action_features)

        brain = BerserkInference(profiles={"test": profile})
        gs, legal = _classic_state_with_actions()
        idx = brain.get_action(gs, 1, legal, difficulty="test")

        assert 0 <= idx < len(legal)
        assert seen_include_preview == [False]

    def test_train_v2_profile_defaults_preview_features_to_false(self, train_v2_onnx_model, monkeypatch):
        from ai.bot_brain import BerserkInference
        import ai.train_v2.classic_actions_v1 as action_codec

        profile = {
            "model_path": train_v2_onnx_model,
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": (1.0, 1.0),
            "selection": "argmax",
        }

        seen_include_preview = []
        real_encode = action_codec.encode_action_features

        def wrapped_encode_action_features(*args, **kwargs):
            seen_include_preview.append(kwargs.get("include_preview", True))
            return real_encode(*args, **kwargs)

        monkeypatch.setattr(action_codec, "encode_action_features", wrapped_encode_action_features)

        brain = BerserkInference(profiles={"test": profile})
        gs, legal = _classic_state_with_actions()
        idx = brain.get_action(gs, 1, legal, difficulty="test")

        assert 0 <= idx < len(legal)
        assert seen_include_preview == [False]

    @pytest.mark.parametrize(
        "profile_update",
        [
            {"obs_dim": 999},
            {"action_feature_dim": 170},
            {"max_candidate_actions": 600},
            {"temperature_range": (0.0, 0.0)},
            {"selection": "roulette"},
            {"placement_mode": "sideways"},
            {"action_codec": "classic_actions_v0"},
            {"observation_codec": "classic_obs_v0"},
        ],
    )
    def test_train_v2_profile_contract_errors_skip_profile(self, tmp_path, monkeypatch, profile_update):
        from ai.bot_brain import BerserkInference

        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake onnx")

        class FakeSession:
            def __init__(self, *_args, **_kwargs):
                pass

            def get_inputs(self):
                return [
                    SimpleNamespace(name="observation", shape=[1, 1456]),
                    SimpleNamespace(name="action_features", shape=[1, 601, 171]),
                ]

            def get_outputs(self):
                return [
                    SimpleNamespace(name="logits", shape=[1, 601]),
                    SimpleNamespace(name="value", shape=[1]),
                ]

        monkeypatch.setattr("ai.bot_brain.ort.InferenceSession", FakeSession)

        profile = {
            "model_path": str(model_path),
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": (1.0, 1.0),
            "selection": "argmax",
        }
        profile.update(profile_update)

        brain = BerserkInference(profiles={"test": profile})
        assert brain.sessions == {}

    def test_malformed_profile_without_model_path_does_not_crash_startup(self):
        from ai.bot_brain import BerserkInference

        profile = {
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": (1.0, 1.0),
            "selection": "argmax",
        }

        brain = BerserkInference(profiles={"test": profile})
        assert brain.sessions == {}

    def test_missing_model_file_does_not_crash_startup(self, tmp_path):
        from ai.bot_brain import BerserkInference

        profile = {
            "model_path": str(tmp_path / "missing.onnx"),
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": (1.0, 1.0),
            "selection": "argmax",
        }

        brain = BerserkInference(profiles={"test": profile})
        assert brain.sessions == {}

    def test_unknown_difficulty_raises_controlled_error(self, train_v2_onnx_model):
        from ai.bot_brain import BerserkInference

        profile = {
            "model_path": train_v2_onnx_model,
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": (1.0, 1.0),
            "selection": "argmax",
        }

        brain = BerserkInference(profiles={"test": profile})
        assert brain.has_profile("test") is True
        assert brain.has_profile("missing") is False
        gs, legal = _classic_state_with_actions()

        with pytest.raises(ValueError, match="Unknown Berserk difficulty"):
            brain.get_action(gs, 1, legal, difficulty="missing")

    def test_constructor_rejects_removed_action_dim_argument(self):
        from ai.bot_brain import BerserkInference

        with pytest.raises(TypeError):
            BerserkInference(profiles={}, action_dim=200)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_get_action_async_offloads_sync_inference(self, train_v2_onnx_model):
        from ai.bot_brain import BerserkInference

        profile = {
            "model_path": train_v2_onnx_model,
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": (1.0, 1.0),
            "selection": "argmax",
        }

        brain = BerserkInference(profiles={"test": profile})
        gs, legal = _classic_state_with_actions()

        idx = await brain.get_action_async(gs, 1, legal, difficulty="test")
        assert 0 <= idx < len(legal)

    def test_legacy_unsupported_obs_dim_profile_is_ignored(self, tmp_path):
        import torch
        import warnings
        from ai.bot_brain import BerserkInference

        model_path = tmp_path / "legacy_789.onnx"
        m = torch.nn.Linear(789, 3)
        dummy = torch.randn(1, 789)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            torch.onnx.export(
                m,
                dummy,
                str(model_path),
                opset_version=17,
                input_names=["obs"],
                output_names=["logits"],
                dynamo=False,
            )

        brain = BerserkInference(
            profiles={
                "legacy": {
                    "model_path": str(model_path),
                    "obs_dim": 789,
                    "temperature_range": (1.0, 1.0),
                    "selection": "softmax",
                }
            }
        )

        assert brain.sessions == {}

    def test_legacy_session_no_longer_runs(self):
        from ai.bot_brain import BerserkInference

        brain = BerserkInference.__new__(BerserkInference)
        brain.sessions = {}

        gs, legal = _classic_state_with_actions()
        with pytest.raises(ValueError, match="Unknown Berserk difficulty"):
            brain.get_action(gs, 1, legal, difficulty="legacy")

    def test_argmax_empty_mask_falls_back_without_decoding(self, train_v2_onnx_model, monkeypatch):
        from ai.bot_brain import BerserkInference, _legal_fallback
        import ai.train_v2.classic_actions_v1 as action_codec

        profile = {
            "model_path": train_v2_onnx_model,
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": (1.0, 1.0),
            "selection": "argmax",
        }

        brain = BerserkInference(profiles={"test": profile})
        gs, legal = _classic_state_with_actions()
        fallback_idx = _legal_fallback(legal)

        monkeypatch.setattr(
            action_codec,
            "build_action_mask",
            lambda *_args, **_kwargs: np.zeros(601, dtype=np.float32),
        )

        def fail_decode(*_args, **_kwargs):
            raise AssertionError("decode_action should not run when the mask has no legal actions")

        monkeypatch.setattr(action_codec, "decode_action", fail_decode)

        assert brain.get_action(gs, 1, legal, difficulty="test") == fallback_idx

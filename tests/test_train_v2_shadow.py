import json
import subprocess
import sys
from pathlib import Path

import pytest

from ai.train_v2.shadow import (
    describe_stable_action,
    actions_semantically_equal,
    OverlayBerserkPolicy,
    LegacyBerserkPolicy,
    FakeLegacyBrain,
    run_shadow_episode,
    run_shadow_matchup,
)
from core.state import CardInstance, CardType, PlayerState, GameState
from core.actions import EndTurnAction, PlayCardAction, AttackAction
from uuid import uuid4


def _make_hero(hp=30):
    return CardInstance(
        instance_id=uuid4(), card_id=0, name="Hero", card_type=CardType.HERO,
        mana_cost=0, attack=0, hp=hp, max_hp=hp, mechanics=[], is_ready=True,
    )


def _make_warrior(mana_cost=3, attack=4, hp=5, name="W"):
    return CardInstance(
        instance_id=uuid4(), card_id=100, name=name, card_type=CardType.WARRIOR,
        mana_cost=mana_cost, attack=attack, hp=hp, max_hp=hp,
        mechanics=[], is_ready=True,
    )


def _make_state_with_actions():
    h1 = _make_hero(30)
    h2 = _make_hero(30)
    w = _make_warrior(3, 4, 5, "W1")
    p1 = PlayerState(user_id=1, hero=h1, mana=5, max_mana=5, hand=[w], board=[], deck=[])
    p2 = PlayerState(user_id=2, hero=h2, mana=0, max_mana=0, hand=[], board=[], deck=[])
    return GameState(p1=p1, p2=p2, current_turn_owner_id=1)


class TestDescribeAndCompare:
    def test_describe_stable_action_end_turn(self):
        gs = _make_state_with_actions()
        d = describe_stable_action(gs, 1, 0)
        assert d["type"] == "end_turn"
        assert d["action_id"] == 0

    def test_describe_stable_action_play_card(self):
        gs = _make_state_with_actions()
        d = describe_stable_action(gs, 1, 1)  # hand[0], pos 0, tcode 0
        assert d["type"] == "play_card"
        assert d["hand_index"] == 0
        assert d["target_id"] is None

    def test_describe_stable_action_attack(self):
        gs = _make_state_with_actions()
        w = _make_warrior(3, 4, 5, "W1")
        gs.p1.board = [w]
        d = describe_stable_action(gs, 1, 545)  # board[0], tcode 0 (enemy board[0])
        assert d["type"] == "attack"
        assert d["attacker_id"] == str(w.instance_id)

    def test_describe_stable_action_invalid(self):
        gs = _make_state_with_actions()
        d = describe_stable_action(gs, 1, 999)
        assert d["type"] == "invalid"

    def test_actions_semantically_equal_end_turn(self):
        gs = _make_state_with_actions()
        assert actions_semantically_equal(gs, 1, 0, 0) is True

    def test_actions_semantically_equal_play_position_ignored(self):
        gs = _make_state_with_actions()
        # Same hand[0], same target (none), different position
        a = 1  # pos 0, tcode 0
        b = 1 + 17  # pos 1, tcode 0
        assert actions_semantically_equal(gs, 1, a, b) is True

    def test_actions_semantically_equal_attack(self):
        gs = _make_state_with_actions()
        w = _make_warrior(3, 4, 5, "W1")
        gs.p1.board = [w]
        # Same attacker, same target
        a = 545  # board[0], tcode 0
        b = 545
        assert actions_semantically_equal(gs, 1, a, b) is True

    def test_actions_semantically_equal_different_types(self):
        gs = _make_state_with_actions()
        assert actions_semantically_equal(gs, 1, 0, 1) is False

    def test_actions_semantically_equal_invalid_same_id(self):
        gs = _make_state_with_actions()
        assert actions_semantically_equal(gs, 1, 999, 999) is True
        assert actions_semantically_equal(gs, 1, 999, 998) is False


class TestPolicies:
    def test_legacy_policy_uses_live_env_legal_actions(self):
        from ai.train_v2.classic_rl_env import ClassicRLEnv

        class LastLegalBrain:
            seen_legal_len = None

            def get_action(self, game_state, player_id, legal_actions, difficulty):
                self.seen_legal_len = len(legal_actions)
                return len(legal_actions) - 1

        env = ClassicRLEnv(seed=5, verify_mask=False, placement_mode="append_only")
        env.reset(seed=5)
        cp = env.current_player_id()
        live_legal = env._env.get_legal_actions(cp)

        brain = LastLegalBrain()
        policy = LegacyBerserkPolicy(brain, difficulty="easy")
        idx = policy.select_legal_action_index(env, cp)

        assert brain.seen_legal_len == len(live_legal)
        assert idx == len(live_legal) - 1

    def test_legacy_berserk_policy_fake_brain(self):
        from ai.train_v2.classic_rl_env import ClassicRLEnv

        env = ClassicRLEnv(seed=42)
        env.reset(seed=42)
        cp = env.current_player_id()

        brain = FakeLegacyBrain()
        policy = LegacyBerserkPolicy(brain, difficulty="easy")
        policy.reset(42)
        aid = policy.select_action(env, cp)

        mask = env.action_mask(cp)
        assert mask[aid] == 1.0, f"legacy policy returned illegal aid={aid}"
        assert len(policy.latencies_ms) == 1
        assert policy.invalid_actions == 0

    def test_overlay_berserk_policy_selects_legal(self):
        from ai.train_v2.train_ppo import PPOConfig, train
        from ai.train_v2.export_onnx import export_checkpoint_to_onnx
        from ai.train_v2.candidate_profile import build_train_v2_profile
        from ai.train_v2.profile_registry import write_profile_overlay

        tmp = Path("/tmp/_t26_overlay_policy")
        tmp.mkdir(exist_ok=True)
        try:
            ckpt_dir = str(tmp / "ckpts")
            config = PPOConfig(
                total_updates=1, episodes_per_update=1, max_steps_per_episode=5,
                hidden_dim=32, action_hidden_dim=16, minibatch_size=8, epochs=1,
                seed=42, checkpoint_dir=ckpt_dir,
            )
            result = train(config)
            onnx_path = str(tmp / "model.onnx")
            export_checkpoint_to_onnx(result["checkpoint_path"], onnx_path, opset=17)

            candidate = {
                "candidate_onnx": onnx_path,
                "source_onnx": onnx_path,
                "model_name": "model",
                "score": 1.0,
                "source_run_dir": str(tmp),
            }
            pack = build_train_v2_profile(candidate)
            pack["_profile_path"] = str(tmp / "pack.json")
            overlay = write_profile_overlay(pack, str(tmp / "overlay.json"))

            policy = OverlayBerserkPolicy(str(tmp / "overlay.json"))
            policy.reset(42)

            from ai.train_v2.classic_rl_env import ClassicRLEnv
            env = ClassicRLEnv(seed=42)
            env.reset(seed=42)
            cp = env.current_player_id()
            aid = policy.select_action(env, cp)

            mask = env.action_mask(cp)
            assert mask[aid] == 1.0, f"overlay policy returned illegal aid={aid}"
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


class TestShadowEpisode:
    def test_run_shadow_episode_smoke(self):
        legacy_brain = FakeLegacyBrain()
        legacy_policy = LegacyBerserkPolicy(legacy_brain, difficulty="easy")

        # Create a fake overlay that always returns action 0 (end turn)
        from ai.train_v2.classic_rl_env import ClassicRLEnv
        env = ClassicRLEnv(seed=42)
        env.reset(seed=42)
        cp = env.current_player_id()
        first_legal = 0
        for i in range(601):
            if env.action_mask(cp)[i] == 1.0:
                first_legal = i
                break

        class FakeOverlayPolicy:
            name = "fake_overlay"

            def reset(self, seed):
                pass

            def select_action(self, env, player_id):
                return first_legal

        result = run_shadow_episode(
            overlay_policy=FakeOverlayPolicy(),
            legacy_policy=legacy_policy,
            seed=42,
            max_steps=10,
        )
        summary = result["summary"]
        assert summary["steps"] > 0
        assert summary["steps"] <= 10
        assert "matches" in summary
        assert "mismatches" in summary
        assert len(result["decisions"]) == summary["steps"]

    def test_run_shadow_episode_play_policy_random(self):
        legacy_brain = FakeLegacyBrain()
        legacy_policy = LegacyBerserkPolicy(legacy_brain, difficulty="easy")

        class FakeOverlayPolicy:
            name = "fake_overlay"

            def reset(self, seed):
                pass

            def select_action(self, env, player_id):
                return 0

        result = run_shadow_episode(
            overlay_policy=FakeOverlayPolicy(),
            legacy_policy=legacy_policy,
            seed=42,
            max_steps=5,
            play_policy="random",
        )
        assert result["summary"]["steps"] > 0

    def test_run_shadow_matchup_aggregate(self):
        legacy_brain = FakeLegacyBrain()
        legacy_policy = LegacyBerserkPolicy(legacy_brain, difficulty="easy")

        class FakeOverlayPolicy:
            name = "fake_overlay"
            _policy = None

            def reset(self, seed):
                pass

            def select_action(self, env, player_id):
                return 0

        seeds = [42, 43]
        results = []
        total_steps = 0
        for seed in seeds:
            r = run_shadow_episode(
                overlay_policy=FakeOverlayPolicy(),
                legacy_policy=LegacyBerserkPolicy(FakeLegacyBrain(), difficulty="easy"),
                seed=seed,
                max_steps=5,
            )
            results.append(r)
            total_steps += r["summary"]["steps"]

        assert len(results) == 2
        assert total_steps > 0


class TestCLI:
    def test_shadow_cli_smoke(self, tmp_path):
        from ai.train_v2.train_ppo import PPOConfig, train
        from ai.train_v2.export_onnx import export_checkpoint_to_onnx
        from ai.train_v2.candidate_profile import build_train_v2_profile
        from ai.train_v2.profile_registry import write_profile_overlay

        tmp = tmp_path / "cli_shadow"
        tmp.mkdir()

        ckpt_dir = str(tmp / "ckpts")
        config = PPOConfig(
            total_updates=1, episodes_per_update=1, max_steps_per_episode=5,
            hidden_dim=32, action_hidden_dim=16, minibatch_size=8, epochs=1,
            seed=42, checkpoint_dir=ckpt_dir,
        )
        result = train(config)
        onnx_path = str(tmp / "model.onnx")
        export_checkpoint_to_onnx(result["checkpoint_path"], onnx_path, opset=17)

        candidate = {
            "candidate_onnx": onnx_path,
            "source_onnx": onnx_path,
            "model_name": "model",
            "score": 1.0,
            "source_run_dir": str(tmp),
        }
        pack = build_train_v2_profile(candidate)
        pack["_profile_path"] = str(tmp / "pack.json")
        write_profile_overlay(pack, str(tmp / "overlay.json"))

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "ai.train_v2.shadow",
                "--overlay",
                str(tmp / "overlay.json"),
                "--seeds",
                "42",
                "--max-steps",
                "5",
                "--play-policy",
                "legacy",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "Shadow episodes" in proc.stdout
        assert "No legacy profile provided" in proc.stderr or "No legacy profile provided" in proc.stdout

    def test_shadow_output_json(self, tmp_path):
        from ai.train_v2.train_ppo import PPOConfig, train
        from ai.train_v2.export_onnx import export_checkpoint_to_onnx
        from ai.train_v2.candidate_profile import build_train_v2_profile
        from ai.train_v2.profile_registry import write_profile_overlay

        tmp = tmp_path / "cli_shadow_out"
        tmp.mkdir()

        ckpt_dir = str(tmp / "ckpts")
        config = PPOConfig(
            total_updates=1, episodes_per_update=1, max_steps_per_episode=5,
            hidden_dim=32, action_hidden_dim=16, minibatch_size=8, epochs=1,
            seed=42, checkpoint_dir=ckpt_dir,
        )
        result = train(config)
        onnx_path = str(tmp / "model.onnx")
        export_checkpoint_to_onnx(result["checkpoint_path"], onnx_path, opset=17)

        candidate = {
            "candidate_onnx": onnx_path,
            "source_onnx": onnx_path,
            "model_name": "model",
            "score": 1.0,
            "source_run_dir": str(tmp),
        }
        pack = build_train_v2_profile(candidate)
        pack["_profile_path"] = str(tmp / "pack.json")
        write_profile_overlay(pack, str(tmp / "overlay.json"))

        out = tmp / "shadow.json"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "ai.train_v2.shadow",
                "--overlay",
                str(tmp / "overlay.json"),
                "--seeds",
                "42",
                "--max-steps",
                "5",
                "--output",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["episodes"] == 1
        assert "steps" in loaded

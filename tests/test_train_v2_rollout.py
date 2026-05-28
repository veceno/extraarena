"""
Tests for rollout transition schema and profiler buckets (Task 04).
"""
import numpy as np
import pytest

from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.policies import RandomLegalPolicy, GreedyFacePolicy
from ai.train_v2.profile_env import benchmark_env
from ai.train_v2.rollout import collect_episode, ROLLOUT_VERSION


class TestProfileBuckets:
    def test_profile_buckets_have_policy_seconds(self):
        result = benchmark_env(episodes=2, seed=42, policy="random")
        assert result["policy_seconds"] > 0, "policy_seconds must be positive"
        assert "overhead_seconds" in result
        assert result["overhead_seconds"] >= 0

    def test_profile_buckets_non_overlapping(self):
        result = benchmark_env(episodes=2, seed=42, policy="random")
        accounted = (result["reset_seconds"] + result["mask_seconds"] +
                     result["features_seconds"] + result["policy_seconds"] +
                     result["step_seconds"] + result["overhead_seconds"])
        assert abs(accounted - result["seconds"]) < 0.1, (
            f"accounted {accounted:.4f} should approx match total {result['seconds']:.4f}"
        )

    def test_profile_with_greedy_face(self):
        result = benchmark_env(episodes=2, seed=42, policy="greedy_face")
        assert result["policy_seconds"] >= 0
        assert result["episodes"] == 2
        assert result["steps"] > 0


class TestCollectEpisode:
    def test_collect_episode_shapes(self):
        env = ClassicRLEnv(seed=42)
        p1 = RandomLegalPolicy(seed=1)
        p2 = RandomLegalPolicy(seed=2)
        result = collect_episode(env, p1, p2, seed=123)
        assert result["version"] == ROLLOUT_VERSION
        assert len(result["transitions"]) > 0

        t = result["transitions"][0]
        assert t.obs.shape == (1456,), f"obs shape {t.obs.shape}"
        assert t.mask.shape == (601,), f"mask shape {t.mask.shape}"
        assert isinstance(t.action_id, int)
        assert t.next_obs.shape == (1456,), f"next_obs shape {t.next_obs.shape}"
        assert t.action_features.shape == (0,), f"action_features shape when disabled: {t.action_features.shape}"
        assert t.value_player_id == 1

    def test_collect_episode_with_features(self):
        env = ClassicRLEnv(seed=42)
        p1 = RandomLegalPolicy(seed=1)
        p2 = RandomLegalPolicy(seed=2)
        result = collect_episode(env, p1, p2, seed=123, include_action_features=True)
        assert len(result["transitions"]) > 0
        t = result["transitions"][0]
        assert t.action_features.shape == (601, 171)

    def test_collect_episode_without_features(self):
        env = ClassicRLEnv(seed=42)
        p1 = RandomLegalPolicy(seed=1)
        p2 = RandomLegalPolicy(seed=2)
        result = collect_episode(env, p1, p2, seed=123, include_action_features=False)
        t = result["transitions"][0]
        assert t.action_features.shape == (0,)

    def test_collect_episode_deterministic(self):
        env1 = ClassicRLEnv(seed=42)
        p1a = RandomLegalPolicy(seed=10)
        p2a = RandomLegalPolicy(seed=20)
        r1 = collect_episode(env1, p1a, p2a, seed=456)

        env2 = ClassicRLEnv(seed=42)
        p1b = RandomLegalPolicy(seed=10)
        p2b = RandomLegalPolicy(seed=20)
        r2 = collect_episode(env2, p1b, p2b, seed=456)

        assert len(r1["transitions"]) == len(r2["transitions"])
        for idx, (t1, t2) in enumerate(zip(r1["transitions"], r2["transitions"])):
            assert t1.action_id == t2.action_id, f"transition {idx}: action {t1.action_id} vs {t2.action_id}"
            assert t1.reward == t2.reward, f"transition {idx}: reward {t1.reward} vs {t2.reward}"

        s1 = r1["summary"]
        s2 = r2["summary"]
        assert s1["winner_id"] == s2["winner_id"]
        assert s1["steps"] == s2["steps"]
        assert s1["p1_reward"] == s2["p1_reward"]

    def test_transition_action_was_legal(self):
        env = ClassicRLEnv(seed=42)
        p1 = RandomLegalPolicy(seed=1)
        p2 = RandomLegalPolicy(seed=2)
        result = collect_episode(env, p1, p2, seed=789)

        for t in result["transitions"]:
            assert t.mask[t.action_id] == 1.0, (
                f"action_id={t.action_id} was selected but mask says illegal"
            )

    def test_collect_episode_has_summary(self):
        env = ClassicRLEnv(seed=42)
        p1 = RandomLegalPolicy(seed=1)
        p2 = RandomLegalPolicy(seed=2)
        result = collect_episode(env, p1, p2, seed=999)

        s = result["summary"]
        assert "winner_id" in s
        assert "status" in s
        assert "turns" in s
        assert "steps" in s
        assert "p1_hp" in s
        assert "p2_hp" in s
        assert "seed" in s
        assert s["seed"] == 999

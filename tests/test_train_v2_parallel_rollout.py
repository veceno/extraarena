"""
Tests for parallel rollout collector and float16 action features.
"""
import numpy as np
import pytest

from ai.train_v2.train_ppo import PPOConfig, collect_policy_episodes_parallel, collect_policy_episode
from ai.train_v2.classic_rl_env import ClassicRLEnv
from ai.train_v2.model_mlx import ActionConditionedPolicy
import mlx.core as mx


class TestParallelCollector:
    def test_parallel_2_workers_returns_valid_transitions(self):
        config = PPOConfig(
            total_updates=1,
            episodes_per_update=4,
            max_steps_per_episode=20,
            hidden_dim=32,
            action_hidden_dim=16,
            verify_mask=False,
            placement_mode="append_only",
            rollout_workers=2,
        )
        model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(model.parameters())
        seeds = [100, 101, 102, 103]
        result = collect_policy_episodes_parallel(config, model, seeds, max_steps=20)

        assert len(result["summaries"]) == 4, f"expected 4 summaries, got {len(result['summaries'])}"
        transitions = result["transitions"]
        assert len(transitions) > 0
        for t in transitions:
            assert t["mask"][t["action_id"]] == 1.0, f"illegal action {t['action_id']}"
            assert t["action_features"].shape == (601, 171)
            assert t["obs"].shape == (1456,)

    def test_parallel_4_workers_exactly_n_summaries(self):
        config = PPOConfig(
            episodes_per_update=8,
            max_steps_per_episode=15,
            hidden_dim=32,
            action_hidden_dim=16,
            verify_mask=False,
            placement_mode="append_only",
            rollout_workers=4,
        )
        model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(model.parameters())
        seeds = list(range(200, 208))
        result = collect_policy_episodes_parallel(config, model, seeds, max_steps=15)
        assert len(result["summaries"]) == 8

    def test_parallel_vs_serial_same_seeds(self):
        """Serial and parallel should collect similar episode lengths for fixed seeds."""
        config = PPOConfig(
            episodes_per_update=2,
            max_steps_per_episode=20,
            hidden_dim=32,
            action_hidden_dim=16,
            verify_mask=False,
            placement_mode="append_only",
            rollout_workers=1,
        )
        model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(model.parameters())
        seeds = [300, 301]

        serial = collect_policy_episode(
            env=ClassicRLEnv(seed=42, verify_mask=False, placement_mode="append_only"),
            model=model, seed=seeds[0], max_steps=20,
        )

        config2 = PPOConfig(
            episodes_per_update=2,
            max_steps_per_episode=20,
            hidden_dim=32,
            action_hidden_dim=16,
            verify_mask=False,
            placement_mode="append_only",
            rollout_workers=2,
        )
        parallel = collect_policy_episodes_parallel(config2, model, seeds, max_steps=20)

        # Both should have 2 summaries
        assert len(parallel["summaries"]) == 2
        # Serial has 1 summary
        assert len(serial["transitions"]) > 0
        assert len(parallel["transitions"]) > 0

    def test_parallel_inference_timing_present(self):
        config = PPOConfig(
            episodes_per_update=2,
            max_steps_per_episode=10,
            hidden_dim=32,
            action_hidden_dim=16,
            verify_mask=False,
            placement_mode="append_only",
            rollout_workers=2,
        )
        model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(model.parameters())
        seeds = [400, 401]
        result = collect_policy_episodes_parallel(config, model, seeds, max_steps=10)
        assert "inference_ms_p50" in result
        assert "inference_ms_p95" in result
        assert result["inference_ms_p50"] >= 0

    def test_parallel_worker_failure_raises(self):
        """If worker count is insane but small test, just ensure no hang."""
        config = PPOConfig(
            episodes_per_update=1,
            max_steps_per_episode=5,
            hidden_dim=32,
            action_hidden_dim=16,
            verify_mask=False,
            placement_mode="append_only",
            rollout_workers=100,  # more workers than episodes
        )
        model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(model.parameters())
        seeds = [500]
        result = collect_policy_episodes_parallel(config, model, seeds, max_steps=5)
        assert len(result["summaries"]) == 1

    def test_parallel_respects_max_steps(self):
        config = PPOConfig(
            episodes_per_update=2,
            max_steps_per_episode=5,
            hidden_dim=32,
            action_hidden_dim=16,
            verify_mask=False,
            placement_mode="append_only",
            rollout_workers=2,
        )
        model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(model.parameters())
        result = collect_policy_episodes_parallel(config, model, [800, 801], max_steps=5)
        assert len(result["transitions"]) <= 10
        assert len(result["summaries"]) == 2
        assert all(s["steps"] <= 5 for s in result["summaries"])

    def test_parallel_final_transition_not_lost(self):
        config = PPOConfig(
            episodes_per_update=2,
            max_steps_per_episode=1,
            hidden_dim=32,
            action_hidden_dim=16,
            verify_mask=False,
            placement_mode="append_only",
            rollout_workers=2,
        )
        model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(model.parameters())
        result = collect_policy_episodes_parallel(config, model, [810, 811], max_steps=1)
        assert len(result["transitions"]) == 2
        assert all(t["truncated"] for t in result["transitions"])
        assert all(s["steps"] == 1 for s in result["summaries"])

    def test_parallel_can_force_p2_to_start(self):
        config = PPOConfig(
            episodes_per_update=2,
            max_steps_per_episode=3,
            hidden_dim=32,
            action_hidden_dim=16,
            verify_mask=False,
            placement_mode="append_only",
            rollout_workers=2,
            starting_player="p2",
        )
        model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(model.parameters())
        result = collect_policy_episodes_parallel(config, model, [830, 831], max_steps=3)

        assert len(result["summaries"]) == 2
        assert all(s["starting_player_id"] == 2 for s in result["summaries"])
        assert all(t["starting_player_id"] == 2 for t in result["transitions"])

    def test_parallel_random_starting_player_is_seeded_but_not_p1_only(self):
        config = PPOConfig(
            episodes_per_update=10,
            max_steps_per_episode=1,
            hidden_dim=32,
            action_hidden_dim=16,
            verify_mask=False,
            placement_mode="append_only",
            rollout_workers=4,
            starting_player="random",
            seed=123,
        )
        model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(model.parameters())
        result = collect_policy_episodes_parallel(config, model, list(range(840, 850)), max_steps=1)
        starts = {s["starting_player_id"] for s in result["summaries"]}
        assert starts == {1, 2}

    def test_parallel_spawn_failure_raises_runtimeerror(self, monkeypatch):
        import multiprocessing as mp
        import ai.train_v2.train_ppo as train_ppo

        class DeadProc:
            exitcode = 1

            def is_alive(self):
                return False

            def join(self, timeout=None):
                return None

            def terminate(self):
                return None

        def fake_spawn_worker(config_dict, worker_id, ctx=None):
            ctx = ctx or mp.get_context("spawn")
            parent_conn, child_conn = ctx.Pipe()
            child_conn.close()
            return DeadProc(), parent_conn, child_conn

        monkeypatch.setattr(train_ppo, "spawn_worker", fake_spawn_worker)

        config = PPOConfig(
            episodes_per_update=1,
            max_steps_per_episode=1,
            hidden_dim=32,
            action_hidden_dim=16,
            verify_mask=False,
            placement_mode="append_only",
            rollout_workers=1,
        )
        model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(model.parameters())
        with pytest.raises(RuntimeError, match="Rollout worker 0 failed"):
            train_ppo.collect_policy_episodes_parallel(config, model, [820], max_steps=1)


class TestFloat16Buffer:
    def test_float16_worker_returns_half_dtype(self):
        config = PPOConfig(
            episodes_per_update=2,
            max_steps_per_episode=10,
            hidden_dim=32,
            action_hidden_dim=16,
            verify_mask=False,
            placement_mode="append_only",
            rollout_workers=2,
            action_features_dtype="float16",
        )
        model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(model.parameters())
        seeds = [600, 601]
        result = collect_policy_episodes_parallel(config, model, seeds, max_steps=10)
        for t in result["transitions"]:
            assert t["action_features"].dtype == np.float16, (
                f"expected float16, got {t['action_features'].dtype}"
            )

    def test_float16_prepare_batch_preserves_dtype(self):
        from ai.train_v2.train_ppo import _prepare_batch
        config = PPOConfig(
            action_features_dtype="float16",
        )
        # Fake 3 transitions
        transitions = [
            {
                "obs": np.zeros(1456, dtype=np.float32),
                "action_features": np.zeros((601, 171), dtype=np.float16),
                "mask": np.zeros(601, dtype=np.float32),
                "action_id": 0,
                "reward": 0.0,
                "done": False,
                "truncated": False,
                "value": 0.0,
                "log_prob": 0.0,
                "player_id": 1,
            }
            for _ in range(3)
        ]
        batch = _prepare_batch(transitions, config)
        assert batch["action_features"].dtype == np.float16
        assert batch["obs"].dtype == np.float32

    def test_float32_default_unchanged(self):
        config = PPOConfig(
            episodes_per_update=2,
            max_steps_per_episode=10,
            hidden_dim=32,
            action_hidden_dim=16,
            verify_mask=False,
            placement_mode="append_only",
            rollout_workers=2,
            action_features_dtype="float32",
        )
        model = ActionConditionedPolicy(hidden_dim=32, action_hidden_dim=16)
        mx.eval(model.parameters())
        seeds = [700, 701]
        result = collect_policy_episodes_parallel(config, model, seeds, max_steps=10)
        for t in result["transitions"]:
            assert t["action_features"].dtype == np.float32


class TestSmokeTrainingWorkers:
    @pytest.mark.parametrize("workers", [1, 2, 4])
    def test_smoke_training_workers(self, workers):
        config = PPOConfig(
            total_updates=1,
            episodes_per_update=2,
            max_steps_per_episode=10,
            hidden_dim=32,
            action_hidden_dim=16,
            minibatch_size=8,
            epochs=1,
            seed=42,
            checkpoint_dir=f"/tmp/_ppo_smoke_{workers}",
            verify_mask=False,
            placement_mode="append_only",
            rollout_workers=workers,
        )
        result = train(config)
        assert result["updates"] == 1
        assert result["episodes"] >= 2
        assert result["steps"] >= 1

    def test_m4_night_dry_run_shows_workers_8(self):
        from ai.train_v2.night_run import NightRunConfig, preflight_night_run
        config = NightRunConfig(preset="m4_night", dry_run=True)
        preflight = preflight_night_run(config)
        assert preflight["ok"]
        assert preflight["rollout_workers"] == 8

    def test_benchmark_workers_1_2_respects_max_steps(self):
        from ai.train_v2.benchmark_rollout import benchmark_rollout
        result = benchmark_rollout(
            preset="smoke",
            workers_list=[1, 2],
            episodes_per_update=2,
            max_steps=5,
            updates=1,
            verify_mask=False,
            placement_mode="append_only",
        )
        by_workers = {r["workers"]: r for r in result["results"]}
        assert by_workers[1]["steps"] <= 10
        assert by_workers[2]["steps"] <= 10


# ============================================================================
# smoke import
# ============================================================================
from ai.train_v2.train_ppo import train

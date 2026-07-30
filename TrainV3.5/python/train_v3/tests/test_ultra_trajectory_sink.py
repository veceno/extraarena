from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from train_v3.ultra_trajectory_sink import UltraTrajectorySink


def _rollout() -> SimpleNamespace:
    steps, envs, obs_dim = 2, 3, 4
    transitions = SimpleNamespace(
        observations=np.arange(steps * envs * obs_dim, dtype=np.float32).reshape(
            steps, envs, obs_dim
        ),
        legal_action_counts=np.full((steps, envs), 2, dtype=np.int64),
        legal_action_offsets=np.arange(0, steps * envs * 2, 2, dtype=np.int64).reshape(
            steps, envs
        ),
        legal_action_ids=np.arange(steps * envs * 2, dtype=np.int64),
        legal_action_features=np.arange(
            steps * envs * 2 * 2, dtype=np.float32
        ).reshape(steps * envs * 2, 2),
        actions=np.arange(steps * envs, dtype=np.int64).reshape(steps, envs),
        rewards=np.zeros((steps, envs), dtype=np.float32),
        terminated=np.asarray(
            [[False, False, False], [True, False, True]], dtype=np.bool_
        ),
        truncated=np.zeros((steps, envs), dtype=np.bool_),
        values=np.zeros((steps, envs), dtype=np.float32),
        log_probs=np.zeros((steps, envs), dtype=np.float32),
        selected_local_indices=np.zeros((steps, envs), dtype=np.int32),
    )
    return SimpleNamespace(
        transitions=transitions,
        mana_draw_legal=np.ones((steps, envs), dtype=np.bool_),
        mana_draw_taken=np.zeros((steps, envs), dtype=np.bool_),
        learner_actor_ids=np.asarray([1, 2, 1], dtype=np.int32),
        opponent_identities=("self", "v4-orig-argmax", "stall"),
        learner_step_counts=np.asarray([2, 2, 2], dtype=np.int32),
        episode_counts=np.asarray([1, 1, 1], dtype=np.int32),
        final_observations=np.zeros((envs, obs_dim), dtype=np.float32),
    )


def test_sink_persists_compact_sample_and_checksums(tmp_path):
    sink = UltraTrajectorySink(tmp_path, sampled_envs=2)
    session = object()
    sink(1, _rollout(), {"opponent_mix_parsed": True}, session)
    manifest = sink.finalize()

    shard = next((tmp_path / "trajectory_shards").glob("*.npz"))
    with np.load(shard, allow_pickle=False) as data:
        assert data["observations"].shape == (2, 2, 4)
        assert data["env_indices"].tolist() == [0, 2]
        assert data["legal_action_counts"].tolist() == [[2, 2], [2, 2]]
        assert data["legal_action_ids"].tolist() == [0, 1, 4, 5, 6, 7, 10, 11]
    segments = [
        json.loads(line)
        for line in (tmp_path / "episode_segments.jsonl").read_text().splitlines()
    ]
    assert len(segments) == 2
    assert all(row["episode_ordinal_end"] == 1 for row in segments)
    assert manifest["updates_logged"] == 1
    assert manifest["learner_steps_logged"] == 4
    assert manifest["closed_episode_transitions"] == 2
    assert manifest["usage"]["assembler_authoritative_labels"] is False

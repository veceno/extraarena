"""Bounded, checksummed trajectory logging for Block-D/Ultra live self-play.

The Rust PPO collector intentionally keeps only learner transitions.  This sink
persists a deterministic lane sample before the driver discards the rollout,
so hard states and accepted learner decisions survive for later replay/SFT.
It is a supplemental corpus: authoritative Assembler matchup labels still need
paired seat/initiative cells, and CardOptimum labels still need counterfactual
draw branches.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class UltraTrajectorySink:
    """Persist a fixed set of lanes from every PPO update.

    Keeping the same sampled lane indices for the lifetime of a persistent
    ``LiveSelfPlaySession`` preserves cross-update episode lineage. Session
    rotation is detected by object identity and starts a new lineage.
    """

    schema = "extra_lr_v5_ultra_trajectory_sample_v1"

    def __init__(
        self,
        output_dir: str | Path,
        *,
        sampled_envs: int = 8,
    ) -> None:
        if int(sampled_envs) <= 0:
            raise ValueError("sampled_envs must be positive")
        self.output_dir = Path(output_dir).resolve()
        self.shard_dir = self.output_dir / "trajectory_shards"
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.sampled_envs = int(sampled_envs)
        self.index_path = self.output_dir / "episode_segments.jsonl"
        self._session_key: int | None = None
        self._session_id = -1
        self._selected_envs = np.zeros(0, dtype=np.intp)
        self._episode_ordinals: dict[int, int] = {}
        self._updates: list[dict[str, Any]] = []

    def _bind_session(self, session: Any, env_count: int) -> None:
        key = id(session)
        if key == self._session_key:
            return
        self._session_key = key
        self._session_id += 1
        count = min(int(env_count), self.sampled_envs)
        self._selected_envs = np.unique(
            np.linspace(0, int(env_count) - 1, num=count, dtype=np.intp)
        )
        self._episode_ordinals = {
            int(env_idx): 0 for env_idx in self._selected_envs.tolist()
        }

    @staticmethod
    def _compact_legal_tape(
        transitions: Any,
        selected: np.ndarray,
        step_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        counts_all = np.asarray(transitions.legal_action_counts)
        offsets_all = np.asarray(transitions.legal_action_offsets)
        ids_all = np.asarray(transitions.legal_action_ids)
        features_all = np.asarray(transitions.legal_action_features)
        steps = int(counts_all.shape[0])
        counts = np.zeros((steps, selected.size), dtype=counts_all.dtype)
        offsets = np.zeros((steps, selected.size), dtype=offsets_all.dtype)
        ids_parts: list[np.ndarray] = []
        feature_parts: list[np.ndarray] = []
        cursor = 0
        for step in range(steps):
            for local_env, env_idx in enumerate(selected.tolist()):
                if step >= int(step_counts[local_env]):
                    offsets[step, local_env] = cursor
                    continue
                count = int(counts_all[step, env_idx])
                source_offset = int(offsets_all[step, env_idx])
                counts[step, local_env] = count
                offsets[step, local_env] = cursor
                if count:
                    ids_parts.append(ids_all[source_offset : source_offset + count])
                    feature_parts.append(
                        features_all[source_offset : source_offset + count]
                    )
                    cursor += count
        ids = (
            np.concatenate(ids_parts, axis=0)
            if ids_parts
            else np.empty((0,), dtype=ids_all.dtype)
        )
        features = (
            np.concatenate(feature_parts, axis=0)
            if feature_parts
            else np.empty((0, *features_all.shape[1:]), dtype=features_all.dtype)
        )
        return counts, offsets, ids, features

    @staticmethod
    def _sample_time_env(
        values: Any,
        selected: np.ndarray,
        step_counts: np.ndarray,
        *,
        fill: float | int | bool = 0,
    ) -> np.ndarray:
        array = np.asarray(values)
        sampled = np.asarray(array[:, selected, ...]).copy()
        for local_env, valid_steps in enumerate(step_counts.tolist()):
            sampled[int(valid_steps) :, local_env, ...] = fill
        return sampled

    def __call__(
        self,
        update_number: int,
        rollout: Any,
        metrics: dict[str, Any],
        session: Any,
    ) -> None:
        if rollout is None:
            raise ValueError("trajectory sink received no rollout")
        transitions = rollout.transitions
        env_count = int(np.asarray(rollout.learner_actor_ids).shape[0])
        self._bind_session(session, env_count)
        selected = self._selected_envs
        step_counts = np.asarray(rollout.learner_step_counts, dtype=np.intp)[
            selected
        ]
        legal_counts, legal_offsets, legal_ids, legal_features = (
            self._compact_legal_tape(transitions, selected, step_counts)
        )
        arrays = {
            "observations": self._sample_time_env(
                transitions.observations, selected, step_counts
            ),
            "legal_action_counts": legal_counts,
            "legal_action_offsets": legal_offsets,
            "legal_action_ids": legal_ids,
            "legal_action_features": legal_features,
            "actions": self._sample_time_env(
                transitions.actions, selected, step_counts
            ),
            "rewards": self._sample_time_env(
                transitions.rewards, selected, step_counts
            ),
            "terminated": self._sample_time_env(
                transitions.terminated, selected, step_counts
            ),
            "truncated": self._sample_time_env(
                transitions.truncated, selected, step_counts
            ),
            "values": self._sample_time_env(
                transitions.values, selected, step_counts
            ),
            "log_probs": self._sample_time_env(
                transitions.log_probs, selected, step_counts
            ),
            "selected_local_indices": self._sample_time_env(
                transitions.selected_local_indices,
                selected,
                step_counts,
                fill=-1,
            ),
            "mana_draw_legal": self._sample_time_env(
                rollout.mana_draw_legal, selected, step_counts
            ),
            "mana_draw_taken": self._sample_time_env(
                rollout.mana_draw_taken, selected, step_counts
            ),
            "final_observations": np.asarray(rollout.final_observations)[
                selected
            ].copy(),
            "env_indices": selected.astype(np.int32),
            "learner_step_counts": step_counts.astype(np.int32),
            "learner_actor_ids": np.asarray(rollout.learner_actor_ids)[
                selected
            ].astype(np.int32),
            "episode_counts": np.asarray(rollout.episode_counts)[selected].astype(
                np.int32
            ),
        }
        stem = (
            f"session-{self._session_id:03d}-"
            f"update-{int(update_number):05d}"
        )
        final_npz = self.shard_dir / f"{stem}.npz"
        temp_npz = self.shard_dir / f".{stem}.tmp.npz"
        np.savez_compressed(temp_npz, **arrays)
        os.replace(temp_npz, final_npz)

        opponents = np.asarray(rollout.opponent_identities, dtype=object)
        segments: list[dict[str, Any]] = []
        terminated = np.asarray(arrays["terminated"], dtype=np.bool_)
        truncated = np.asarray(arrays["truncated"], dtype=np.bool_)
        for local_env, env_idx in enumerate(selected.tolist()):
            ordinal_start = self._episode_ordinals[int(env_idx)]
            terminal_rows = np.flatnonzero(
                terminated[:, local_env] | truncated[:, local_env]
            ).astype(int)
            self._episode_ordinals[int(env_idx)] += int(terminal_rows.size)
            segments.append(
                {
                    "schema": self.schema,
                    "update_number": int(update_number),
                    "session_id": int(self._session_id),
                    "env_index": int(env_idx),
                    "learner_actor_id": int(
                        np.asarray(rollout.learner_actor_ids)[env_idx]
                    ),
                    "opponent_identity": str(opponents[env_idx]),
                    "episode_ordinal_start": int(ordinal_start),
                    "episode_ordinal_end": int(
                        self._episode_ordinals[int(env_idx)]
                    ),
                    "terminal_rows": terminal_rows.tolist(),
                    "truncated_rows": np.flatnonzero(
                        truncated[:, local_env]
                    ).astype(int).tolist(),
                    "learner_steps": int(step_counts[local_env]),
                    "shard": final_npz.name,
                    "accepted_actions_only": True,
                    "degraded": False,
                }
            )
        with self.index_path.open("a", encoding="utf-8") as stream:
            for row in segments:
                stream.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())

        shard_record = {
            "update_number": int(update_number),
            "session_id": int(self._session_id),
            "path": str(final_npz.relative_to(self.output_dir)),
            "sha256": _sha256(final_npz),
            "bytes": final_npz.stat().st_size,
            "sampled_env_indices": selected.tolist(),
            "learner_steps": int(step_counts.sum()),
            "closed_episode_transitions": int(
                np.count_nonzero(terminated | truncated)
            ),
            "opponent_identities": [
                str(opponents[env_idx]) for env_idx in selected.tolist()
            ],
            "mix_used": _jsonable(metrics.get("mix_used")),
        }
        self._updates.append(shard_record)
        meta_path = self.shard_dir / f"{stem}.json"
        meta_path.write_text(
            json.dumps(
                {
                    "schema": self.schema,
                    **shard_record,
                    "limitations": {
                        "learner_transitions_only": True,
                        "opponent_actions_not_materialized": True,
                        "assembler_requires_paired_matchup_labels": True,
                        "cardoptimum_requires_counterfactual_branches": True,
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def finalize(self) -> dict[str, Any]:
        files = sorted(
            [
                path
                for path in self.output_dir.rglob("*")
                if path.is_file() and path.name != "files_manifest.json"
            ]
        )
        manifest = {
            "schema": "extra_lr_v5_ultra_trajectory_files_manifest_v1",
            "trajectory_schema": self.schema,
            "sampled_envs_target": self.sampled_envs,
            "updates_logged": len(self._updates),
            "learner_steps_logged": sum(
                int(row["learner_steps"]) for row in self._updates
            ),
            "closed_episode_transitions": sum(
                int(row["closed_episode_transitions"]) for row in self._updates
            ),
            "shards": list(self._updates),
            "files": [
                {
                    "path": str(path.relative_to(self.output_dir)),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in files
            ],
            "usage": {
                "policy_hard_state_supplement": True,
                "assembler_authoritative_labels": False,
                "cardoptimum_authoritative_labels": False,
            },
        }
        path = self.output_dir / "files_manifest.json"
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return manifest


__all__ = ["UltraTrajectorySink"]

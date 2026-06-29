"""Small VecEnv-style adapter over the training-only Rust TrainV3 worker."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .rust_ffi import RustBatchWorker


@dataclass(frozen=True)
class RustVecEnvReset:
    observations: np.ndarray
    infos: list[dict[str, Any]] | None
    action_mask: np.ndarray | None
    action_features: np.ndarray | None
    legal_action_counts: np.ndarray
    legal_action_offsets: np.ndarray
    legal_action_ids: np.ndarray
    legal_action_features: np.ndarray
    raw: dict[str, np.ndarray]


@dataclass(frozen=True)
class RustVecEnvStep:
    observations: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray | None
    infos: list[dict[str, Any]] | None
    action_mask: np.ndarray | None
    action_features: np.ndarray | None
    legal_action_counts: np.ndarray
    legal_action_offsets: np.ndarray
    legal_action_ids: np.ndarray
    legal_action_features: np.ndarray
    reset_flags: np.ndarray | None
    raw: dict[str, np.ndarray]


class RustVecEnv:
    """Dependency-light vectorized rollout adapter for `RustBatchWorker`.

    The adapter keeps the coarse FFI tensors available in `raw`, while exposing
    the usual collector-facing pieces as attributes.
    """

    def __init__(
        self,
        worker: RustBatchWorker,
        *,
        observation_key: str = "observation_v5",
        auto_reset: bool = True,
    ):
        if observation_key not in {"observation_v1", "observation_v5"}:
            raise ValueError("observation_key must be observation_v1 or observation_v5")
        if observation_key == "observation_v1" and worker.observation_mode == "v5_only":
            raise ValueError("observation_key='observation_v1' requires observation_mode='v1_and_v5'")
        self.worker = worker
        self.env_count = worker.env_count
        self.observation_key = observation_key
        self.auto_reset = bool(auto_reset)
        self._zero_truncated = np.zeros(self.env_count, dtype=np.bool_)
        self._zero_truncated.setflags(write=False)

    @classmethod
    def from_trace_file(
        cls,
        path: str | Path,
        *,
        env_count: int,
        library_path: str | Path | None = None,
        observation_key: str = "observation_v5",
        auto_reset: bool = True,
        action_features_dtype: str = "float32",
        action_features_mode: str = "dense_and_legal",
        observation_mode: str = "v1_and_v5",
        action_mask_mode: str = "dense",
        terminal_observation_mode: str = "full",
        diagnostic_mode: str = "full",
    ) -> "RustVecEnv":
        worker = RustBatchWorker.from_trace_file(
            path,
            env_count=env_count,
            library_path=library_path,
            action_features_dtype=action_features_dtype,
            action_features_mode=action_features_mode,
            observation_mode=observation_mode,
            action_mask_mode=action_mask_mode,
            terminal_observation_mode=terminal_observation_mode,
            diagnostic_mode=diagnostic_mode,
        )
        return cls(worker, observation_key=observation_key, auto_reset=auto_reset)

    @classmethod
    def from_trace_files(
        cls,
        paths: list[str | Path] | tuple[str | Path, ...],
        *,
        env_count: int,
        library_path: str | Path | None = None,
        observation_key: str = "observation_v5",
        auto_reset: bool = True,
        action_features_dtype: str = "float32",
        action_features_mode: str = "dense_and_legal",
        reset_pool_mode: str = "fixed",
        observation_mode: str = "v1_and_v5",
        action_mask_mode: str = "dense",
        terminal_observation_mode: str = "full",
        diagnostic_mode: str = "full",
    ) -> "RustVecEnv":
        worker = RustBatchWorker.from_trace_files(
            paths,
            env_count=env_count,
            library_path=library_path,
            action_features_dtype=action_features_dtype,
            action_features_mode=action_features_mode,
            reset_pool_mode=reset_pool_mode,
            observation_mode=observation_mode,
            action_mask_mode=action_mask_mode,
            terminal_observation_mode=terminal_observation_mode,
            diagnostic_mode=diagnostic_mode,
        )
        return cls(worker, observation_key=observation_key, auto_reset=auto_reset)

    def close(self) -> None:
        self.worker.close()

    def __enter__(self) -> "RustVecEnv":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def reset(self, *, copy: bool = True, include_infos: bool = True) -> RustVecEnvReset:
        raw = self.worker.reset(copy=copy)
        return RustVecEnvReset(
            observations=raw[self.observation_key],
            infos=[{} for _ in range(self.env_count)] if include_infos else None,
            action_mask=raw["action_mask"],
            action_features=raw["action_features"],
            legal_action_counts=raw["legal_action_counts"],
            legal_action_offsets=raw["legal_action_offsets"],
            legal_action_ids=raw["legal_action_ids"],
            legal_action_features=raw["legal_action_features"],
            raw=raw,
        )

    def step(
        self,
        action_ids,
        *,
        copy: bool = True,
        include_infos: bool = True,
        include_truncated: bool = True,
    ) -> RustVecEnvStep:
        raw = (
            self.worker.step_auto_reset(action_ids, copy=copy)
            if self.auto_reset
            else self.worker.step(action_ids, copy=copy)
        )
        return RustVecEnvStep(
            observations=raw[self.observation_key],
            rewards=raw["rewards"],
            terminated=raw["terminated"],
            truncated=self._zero_truncated if include_truncated else None,
            infos=self._infos(raw) if include_infos else None,
            action_mask=raw["action_mask"],
            action_features=raw["action_features"],
            legal_action_counts=raw["legal_action_counts"],
            legal_action_offsets=raw["legal_action_offsets"],
            legal_action_ids=raw["legal_action_ids"],
            legal_action_features=raw["legal_action_features"],
            reset_flags=raw["reset_flags"],
            raw=raw,
        )

    def current_actor_ids(self) -> np.ndarray:
        return self.worker.current_actor_ids()

    def select_rule_actions(self, agent_codes, *, salt: int = 0) -> np.ndarray:
        return self.worker.select_rule_actions(agent_codes, salt=salt)

    def advance_rule_until_actor(
        self,
        learner_actor_ids,
        agent_codes,
        *,
        max_actions_per_env: int = 64,
        salt: int = 0,
        auto_reset: bool | None = None,
        copy: bool = True,
    ) -> dict[str, np.ndarray]:
        return self.worker.advance_rule_until_actor(
            learner_actor_ids,
            agent_codes,
            max_actions_per_env=max_actions_per_env,
            salt=salt,
            auto_reset=self.auto_reset if auto_reset is None else bool(auto_reset),
            copy=copy,
        )

    def _infos(self, raw: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        if (
            raw["reset_flags"] is None
            or raw["terminal_observation_valid"] is None
            or raw["episode_returns"] is None
            or raw["episode_lengths"] is None
        ):
            raise ValueError("include_infos=True requires RustVecEnv diagnostic_mode='full'")
        infos: list[dict[str, Any]] = []
        terminal_key = "terminal_" + self.observation_key
        for idx in range(self.env_count):
            info: dict[str, Any] = {
                "reset": bool(raw["reset_flags"][idx]),
                "terminal_observation_valid": bool(raw["terminal_observation_valid"][idx]),
            }
            if raw["terminated"][idx]:
                info["episode"] = {
                    "r": float(raw["episode_returns"][idx]),
                    "l": int(raw["episode_lengths"][idx]),
                }
            if raw["terminal_observation_valid"][idx]:
                if raw[terminal_key] is not None:
                    info["terminal_observation"] = raw[terminal_key][idx]
                if raw["terminal_observation_v1"] is not None:
                    info["terminal_observation_v1"] = raw["terminal_observation_v1"][idx]
                if raw["terminal_observation_v5"] is not None:
                    info["terminal_observation_v5"] = raw["terminal_observation_v5"][idx]
            infos.append(info)
        return infos


__all__ = ["RustVecEnv", "RustVecEnvReset", "RustVecEnvStep"]

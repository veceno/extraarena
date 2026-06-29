"""Python rollout helpers over the training-only Rust TrainV3 FFI worker."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .rust_ffi import RustBatchWorker


@dataclass(frozen=True)
class RustRolloutStats:
    """Small benchmark summary for mirrored Rust trace rollouts."""

    env_count: int
    action_count: int
    iterations: int
    elapsed_seconds: float
    worker_reuse: bool
    action_features_mode: str = "legal_only"
    observation_mode: str = "v5_only"
    action_mask_mode: str = "legal_only"
    terminal_observation_mode: str = "none"
    coarse_rollout: bool = False

    @property
    def env_transitions(self) -> int:
        return self.env_count * self.action_count * self.iterations

    @property
    def env_transitions_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return float("inf")
        return self.env_transitions / self.elapsed_seconds

    def as_dict(self) -> dict[str, float | int | bool | str]:
        return {
            "env_count": self.env_count,
            "action_count": self.action_count,
            "iterations": self.iterations,
            "worker_reuse": self.worker_reuse,
            "coarse_rollout": self.coarse_rollout,
            "action_features_mode": self.action_features_mode,
            "observation_mode": self.observation_mode,
            "action_mask_mode": self.action_mask_mode,
            "terminal_observation_mode": self.terminal_observation_mode,
            "elapsed_seconds": self.elapsed_seconds,
            "env_transitions": self.env_transitions,
            "env_transitions_per_second": self.env_transitions_per_second,
        }


class RustTraceRolloutRunner:
    """Replay a golden-trace action script through the Rust batched worker.

    This is intentionally a narrow training helper. It mirrors one Python oracle
    trace into many Rust env states, then applies the same scripted action id to
    every mirrored env at each step.
    """

    def __init__(self, worker: RustBatchWorker, action_ids: list[int]):
        if not action_ids:
            raise ValueError("RustTraceRolloutRunner requires at least one action id")
        self.worker = worker
        self.action_ids = [int(action_id) for action_id in action_ids]
        self.env_count = worker.env_count
        self._coarse_action_ids = np.asarray(self.action_ids, dtype=np.uintp)
        self._action_vector_cache: dict[int, np.ndarray] = {}
        for action_id in self.action_ids:
            if action_id not in self._action_vector_cache:
                actions = np.empty(self.env_count, dtype=np.uintp)
                actions.fill(action_id)
                self._action_vector_cache[action_id] = actions

    @classmethod
    def from_trace_file(
        cls,
        path: str | Path,
        *,
        env_count: int,
        library_path: str | Path | None = None,
        action_features_dtype: str = "float32",
        action_features_mode: str = "dense_and_legal",
        observation_mode: str = "v1_and_v5",
        action_mask_mode: str = "dense",
        terminal_observation_mode: str = "full",
    ) -> "RustTraceRolloutRunner":
        trace_path = Path(path)
        action_ids = _action_ids_from_trace(trace_path)
        worker = RustBatchWorker.from_trace_file(
            trace_path,
            env_count=env_count,
            library_path=library_path,
            action_features_dtype=action_features_dtype,
            action_features_mode=action_features_mode,
            observation_mode=observation_mode,
            action_mask_mode=action_mask_mode,
            terminal_observation_mode=terminal_observation_mode,
        )
        return cls(worker, action_ids)

    def close(self) -> None:
        self.worker.close()

    def __enter__(self) -> "RustTraceRolloutRunner":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def initial(self, *, copy: bool = False) -> dict[str, np.ndarray]:
        return self.worker.encode(copy=copy)

    def reset(self, *, copy: bool = False) -> dict[str, np.ndarray]:
        return self.worker.reset(copy=copy)

    def reset_indices(self, indices, *, copy: bool = False) -> dict[str, np.ndarray]:
        return self.worker.reset_indices(indices, copy=copy)

    def replay(self, *, copy: bool = False) -> list[dict[str, np.ndarray]]:
        return [self.step(action_id, copy=copy) for action_id in self.action_ids]

    def replay_coarse(self, *, auto_reset: bool = False, copy: bool = False) -> dict[str, np.ndarray]:
        return self.worker.rollout_action_tape(
            self._coarse_action_ids,
            auto_reset=auto_reset,
            copy=copy,
        )

    def step(self, action_id: int, *, copy: bool = False) -> dict[str, np.ndarray]:
        return self.worker.step(self._action_vector_for(action_id), copy=copy)

    def step_auto_reset(self, action_id: int, *, copy: bool = False) -> dict[str, np.ndarray]:
        return self.worker.step_auto_reset(self._action_vector_for(action_id), copy=copy)

    def _action_vector_for(self, action_id: int) -> np.ndarray:
        action_id = int(action_id)
        cached = self._action_vector_cache.get(action_id)
        if cached is None:
            cached = np.full(self.env_count, action_id, dtype=np.uintp)
            self._action_vector_cache[action_id] = cached
        return cached


def benchmark_trace_file(
    path: str | Path,
    *,
    env_count: int,
    iterations: int,
    library_path: str | Path | None = None,
    reuse_worker: bool = True,
    action_features_dtype: str = "float32",
    action_features_mode: str = "legal_only",
    observation_mode: str = "v5_only",
    action_mask_mode: str = "legal_only",
    terminal_observation_mode: str = "none",
    coarse_rollout: bool = True,
) -> dict[str, float | int | bool | str]:
    """Benchmark Python->Rust FFI rollout over a scripted golden trace."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if env_count <= 0:
        raise ValueError("env_count must be positive")

    trace_path = Path(path)
    action_count = len(_action_ids_from_trace(trace_path))

    if reuse_worker:
        with RustTraceRolloutRunner.from_trace_file(
            trace_path,
            env_count=env_count,
            library_path=library_path,
            action_features_dtype=action_features_dtype,
            action_features_mode=action_features_mode,
            observation_mode=observation_mode,
            action_mask_mode=action_mask_mode,
            terminal_observation_mode=terminal_observation_mode,
        ) as runner:
            start = time.perf_counter()
            for _ in range(iterations):
                runner.reset(copy=False)
                if coarse_rollout:
                    runner.replay_coarse(copy=False)
                else:
                    for action_id in runner.action_ids:
                        runner.step(action_id, copy=False)
            elapsed = time.perf_counter() - start
    else:
        start = time.perf_counter()
        for _ in range(iterations):
            with RustTraceRolloutRunner.from_trace_file(
                trace_path,
                env_count=env_count,
                library_path=library_path,
                action_features_dtype=action_features_dtype,
                action_features_mode=action_features_mode,
                observation_mode=observation_mode,
                action_mask_mode=action_mask_mode,
                terminal_observation_mode=terminal_observation_mode,
            ) as runner:
                runner.initial(copy=False)
                if coarse_rollout:
                    runner.replay_coarse(copy=False)
                else:
                    for action_id in runner.action_ids:
                        runner.step(action_id, copy=False)
        elapsed = time.perf_counter() - start

    return RustRolloutStats(
        env_count=env_count,
        action_count=action_count,
        iterations=iterations,
        elapsed_seconds=elapsed,
        worker_reuse=reuse_worker,
        action_features_mode=action_features_mode,
        observation_mode=observation_mode,
        action_mask_mode=action_mask_mode,
        terminal_observation_mode=terminal_observation_mode,
        coarse_rollout=coarse_rollout,
    ).as_dict()


def _action_ids_from_trace(path: Path) -> list[int]:
    data = json.loads(path.read_text())
    steps = data.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"trace {path} does not contain a steps list")

    action_ids: list[int] = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict) or "action_id" not in step:
            raise ValueError(f"trace {path} step {idx} does not contain action_id")
        action_ids.append(int(step["action_id"]))
    return action_ids


__all__ = [
    "RustRolloutStats",
    "RustTraceRolloutRunner",
    "benchmark_trace_file",
]

"""Phase 9 V5 broad opponent runtime contracts.

This module deliberately separates real training integrations from legacy
`opponent_mix` metadata. A lane is valid only when it names an executable
source of behavior: Rust exploit logic, V5 policy control, offline teacher
labels, or sanity trace generation.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from .gauntlet_v5 import EXPLOIT_AGENT_KINDS

BROAD_OPPONENT_ENVIRONMENT_SCHEMA = "trainv3-v5-broad-opponent-environment-v1"

PHASE9_SPARRING_STRENGTHS = (0.25, 0.5, 0.75, 1.0)

PHASE9_BROAD_OPPONENT_LANES = (
    "v4max",
    "self",
    "v5_snapshot",
    *(f"sparring_strength_{strength}" for strength in PHASE9_SPARRING_STRENGTHS),
    "llm_teacher",
    "random",
    "greedy_face",
    "end_turn",
    *EXPLOIT_AGENT_KINDS,
)

REAL_EXECUTION_KINDS = frozenset(
    {
        "offline_v4max_teacher",
        "v5_policy_control",
        "v5_snapshot_policy",
        "offline_llm_teacher",
        "sanity_trace_policy",
        "rust_exploit",
    }
)


@dataclass(frozen=True)
class V5OpponentLane:
    kind: str
    execution_kind: str
    runtime: str = "python"
    weight: float = 1.0
    adaptive_strength: float | None = None
    probe_status: str = "not_run"
    probe_error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "V5OpponentLane":
        if not self.kind:
            raise ValueError("opponent lane kind must be non-empty")
        if self.execution_kind == "metadata_only":
            raise ValueError(f"{self.kind}: metadata_only opponent lanes are forbidden")
        if self.execution_kind not in REAL_EXECUTION_KINDS:
            raise ValueError(f"{self.kind}: unknown execution_kind {self.execution_kind!r}")
        if float(self.weight) <= 0.0:
            raise ValueError(f"{self.kind}: weight must be positive")
        if self.adaptive_strength is not None and not 0.0 <= float(self.adaptive_strength) <= 1.0:
            raise ValueError(f"{self.kind}: adaptive_strength must be in [0, 1]")
        if self.execution_kind == "rust_exploit" and self.runtime != "rust":
            raise ValueError(f"{self.kind}: rust_exploit lanes must use rust runtime")
        if self.kind == "v4max" and bool(self.metadata.get("v4_1_included", True)):
            raise ValueError("v4max lane must explicitly exclude V4.1")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data = asdict(self)
        data["metadata"] = _jsonable_dict(self.metadata)
        return data


def build_phase9_broad_opponent_lanes() -> list[V5OpponentLane]:
    """Return the complete real-or-offline-teacher lane set for Phase 9 prep."""
    lanes: list[V5OpponentLane] = [
        V5OpponentLane(
            kind="v4max",
            execution_kind="offline_v4max_teacher",
            runtime="python_onnx",
            weight=2.0,
            adaptive_strength=1.0,
            metadata={
                "model_name": "extra-lr-v4-max",
                "model_path": "ai/models/extra-lr-v4-max.onnx",
                "benchmark_script": "TrainV3/scripts/run_v5_vs_v4max_benchmark.py",
                "label_use": "teacher_action_ranking_and_hard_state_distillation",
                "v4_1_included": False,
            },
        ),
        V5OpponentLane(
            kind="self",
            execution_kind="v5_policy_control",
            runtime="mlx_plus_rust_collector",
            weight=1.0,
            metadata={
                "role": "current_policy_self_play_trace_control",
                "online_rollout_policy": "current_v5_policy_controls_actors",
            },
        ),
        V5OpponentLane(
            kind="v5_snapshot",
            execution_kind="v5_snapshot_policy",
            runtime="mlx_plus_rust_collector",
            weight=0.75,
            metadata={
                "role": "historical_v5_checkpoint_teacher_or_h2h_control",
                "requires_checkpoint_pool": True,
            },
        ),
        V5OpponentLane(
            kind="llm_teacher",
            execution_kind="offline_llm_teacher",
            runtime="openai_compatible_http",
            weight=0.35,
            metadata={
                "role": "offline_preference_labels",
                "online_rollout_dependency": False,
                "client": "train_v3.llm_teacher.OpenAICompatibleTeacherClient",
            },
        ),
        V5OpponentLane(
            kind="random",
            execution_kind="sanity_trace_policy",
            runtime="python_debug_only",
            weight=0.08,
            metadata={"role": "early_curriculum_and_sanity_trace_source"},
        ),
        V5OpponentLane(
            kind="greedy_face",
            execution_kind="sanity_trace_policy",
            runtime="python_debug_only",
            weight=0.08,
            metadata={"role": "sanity_trace_source_prefers_face_damage"},
        ),
        V5OpponentLane(
            kind="end_turn",
            execution_kind="sanity_trace_policy",
            runtime="python_debug_only",
            weight=0.02,
            metadata={"role": "degenerate_baseline_guardrail"},
        ),
    ]
    lanes.extend(
        V5OpponentLane(
            kind=f"sparring_strength_{strength}",
            execution_kind="v5_snapshot_policy",
            runtime="mlx_plus_rust_collector",
            weight=0.35,
            adaptive_strength=float(strength),
            metadata={
                "role": "training_only_adaptive_strength_sparring_snapshot",
                "requires_checkpoint_pool": True,
            },
        )
        for strength in PHASE9_SPARRING_STRENGTHS
    )
    lanes.extend(
        V5OpponentLane(
            kind=kind,
            execution_kind="rust_exploit",
            runtime="rust",
            weight=0.25,
            metadata={
                "role": "training_only_exploit_lane",
                "rust_module": "trainv3_core::exploit",
                "rust_cli": "trainv3_kernel gauntlet",
            },
        )
        for kind in EXPLOIT_AGENT_KINDS
    )
    return validate_broad_opponent_lanes(lanes)


def validate_broad_opponent_lanes(lanes: Iterable[V5OpponentLane]) -> list[V5OpponentLane]:
    validated = [lane.validate() for lane in lanes]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for lane in validated:
        if lane.kind in seen:
            duplicates.add(lane.kind)
        seen.add(lane.kind)
    if duplicates:
        raise ValueError(f"duplicate opponent lanes: {sorted(duplicates)}")
    required = set(PHASE9_BROAD_OPPONENT_LANES)
    present = {lane.kind for lane in validated}
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"missing opponent lanes: {missing}")
    extra = sorted(present - required)
    if extra:
        raise ValueError(f"unexpected opponent lanes: {extra}")
    return validated


def phase9_broad_opponent_mix(lanes: Iterable[V5OpponentLane]) -> str:
    validated = validate_broad_opponent_lanes(lanes)
    return ",".join(f"{lane.kind}:{float(lane.weight):g}" for lane in validated)


def prepare_phase9_broad_opponent_environment(
    *,
    output_path: str | Path,
    run_probes: bool = True,
    root: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    root_path = Path(root).resolve() if root is not None else Path.cwd().resolve()
    lanes = build_phase9_broad_opponent_lanes()
    if run_probes:
        lanes = [probe_phase9_opponent_lane(lane, root=root_path) for lane in lanes]
    manifest = {
        "schema": BROAD_OPPONENT_ENVIRONMENT_SCHEMA,
        "phase": "phase9_broad_opponent_runtime_prep",
        "created_at": created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root": str(root_path),
        "v4_1_included": False,
        "all_lanes_real_or_offline_teacher": all(
            lane.execution_kind in REAL_EXECUTION_KINDS and lane.execution_kind != "metadata_only"
            for lane in lanes
        ),
        "runtime_contract": {
            "legacy_v5_league_opponent_mix_is_not_authoritative_for_runtime": True,
            "rust_hot_path_required_for_mass_rollout": True,
            "llm_teacher_online_rollout_dependency": False,
            "v4max_online_rollout_dependency": False,
            "v4max_labels_are_offline_distillation_inputs": True,
        },
        "opponent_mix": phase9_broad_opponent_mix(lanes),
        "lanes": [lane.to_dict() for lane in lanes],
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def assert_phase9_broad_environment_ready(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema") != BROAD_OPPONENT_ENVIRONMENT_SCHEMA:
        raise ValueError("phase9 opponent environment schema is invalid")
    if bool(manifest.get("v4_1_included", True)):
        raise ValueError("Phase9 V4-max environment must not include V4.1")
    if not bool(manifest.get("all_lanes_real_or_offline_teacher", False)):
        raise ValueError("phase9 opponent environment contains non-real lanes")
    raw_lanes = manifest.get("lanes")
    if not isinstance(raw_lanes, list):
        raise ValueError("phase9 opponent environment lanes must be a list")
    lanes = [_lane_from_dict(item).validate() for item in raw_lanes]
    validate_broad_opponent_lanes(lanes)
    not_ready = [
        f"{lane.kind}:{lane.probe_status}"
        for lane in lanes
        if lane.probe_status != "ok"
    ]
    if not_ready:
        raise ValueError(f"phase9 opponent lane probe_status is not ok: {not_ready}")
    return manifest


def probe_phase9_opponent_lane(lane: V5OpponentLane, *, root: str | Path) -> V5OpponentLane:
    lane.validate()
    root_path = Path(root)
    required_paths = _required_paths_for_lane(lane)
    missing = [path for path in required_paths if not (root_path / path).exists()]
    if missing:
        return replace(lane, probe_status="missing", probe_error=f"missing required paths: {missing}")
    return replace(lane, probe_status="ok", probe_error="")


def _required_paths_for_lane(lane: V5OpponentLane) -> tuple[str, ...]:
    if lane.kind == "v4max":
        return (
            "ai/models/extra-lr-v4-max.onnx",
            "TrainV3/scripts/run_v5_vs_v4max_benchmark.py",
        )
    if lane.execution_kind == "rust_exploit":
        return (
            "TrainV3/rust/trainv3_core/src/exploit.rs",
            "TrainV3/rust/trainv3_core/src/bin/trainv3_kernel.rs",
        )
    if lane.execution_kind in {"v5_policy_control", "v5_snapshot_policy"}:
        return (
            "TrainV3/python/train_v3/rust_collector.py",
            "TrainV3/python/train_v3/rust_trainer.py",
            "TrainV3/python/train_v3/v5_policy.py",
        )
    if lane.execution_kind == "offline_llm_teacher":
        return ("TrainV3/python/train_v3/llm_teacher.py",)
    if lane.execution_kind == "sanity_trace_policy":
        return ("TrainV3/python/train_v3/golden_trace.py",)
    return ()


def _lane_from_dict(value: Any) -> V5OpponentLane:
    if not isinstance(value, dict):
        raise ValueError("opponent lane manifest entry must be an object")
    return V5OpponentLane(
        kind=str(value.get("kind", "")),
        execution_kind=str(value.get("execution_kind", "")),
        runtime=str(value.get("runtime", "python")),
        weight=float(value.get("weight", 1.0)),
        adaptive_strength=(
            None
            if value.get("adaptive_strength") is None
            else float(value.get("adaptive_strength"))
        ),
        probe_status=str(value.get("probe_status", "not_run")),
        probe_error=str(value.get("probe_error", "")),
        metadata=dict(value.get("metadata") or {}),
    )


def _jsonable_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(raw) for key, raw in value.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return _jsonable_dict(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "BROAD_OPPONENT_ENVIRONMENT_SCHEMA",
    "PHASE9_BROAD_OPPONENT_LANES",
    "PHASE9_SPARRING_STRENGTHS",
    "REAL_EXECUTION_KINDS",
    "V5OpponentLane",
    "assert_phase9_broad_environment_ready",
    "build_phase9_broad_opponent_lanes",
    "phase9_broad_opponent_mix",
    "prepare_phase9_broad_opponent_environment",
    "probe_phase9_opponent_lane",
    "validate_broad_opponent_lanes",
]

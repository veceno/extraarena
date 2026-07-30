"""V5 trace-pool generation for TrainV3 curriculum experiments."""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import AssistModeV5, InfoModeV5, OBS_V5_DIM
from .golden_trace import build_golden_trace
from .v5_artifacts import TRACE_POOL_SCHEMA, TracePoolEntry, TracePoolManifest, manifest_to_dict


_V5_ENV_CONFIG_FIELDS = (
    "adaptive_strength",
    "own_hand_identity_known",
    "own_deck_known",
    "enemy_hand_known",
    "enemy_deck_known",
    "enemy_deck_order_known",
    "draw_assist_enabled",
    "draw_assist_strength",
    "assembler_enabled",
    "assembler_strength",
    "desirerer_enabled",
    "desirerer_strength",
    "teacher_hint_available",
    "assist_profile_id",
)


class _LoadedV5TracePoolManifest(dict):
    def __init__(self, data: dict[str, Any], source_path: Path) -> None:
        super().__init__(data)
        self.source_path = source_path


@dataclass(frozen=True)
class V5TraceScenario:
    scenario_key: str
    seeds: tuple[int, ...] = (42,)
    steps: int = 16
    p1_deck_ids: tuple[int, ...] | None = None
    p2_deck_ids: tuple[int, ...] | None = None
    action_ids: tuple[int, ...] | None = None
    adaptive_strengths: tuple[float, ...] = (1.0,)
    visibility_modes: tuple[dict[str, Any], ...] = (
        {
            "own_hand_identity_known": True,
            "own_deck_known": True,
            "enemy_hand_known": True,
            "enemy_deck_known": True,
            "enemy_deck_order_known": True,
        },
    )
    draw_assist_modes: tuple[dict[str, Any], ...] = (
        {"draw_assist_enabled": False, "draw_assist_strength": 0.0},
    )
    assist_modes: tuple[dict[str, Any], ...] = ({"assist_profile_id": 0},)
    level_modes: tuple[dict[str, Any], ...] = ({},)
    placement_mode: str = "append_only"
    verify_mask: bool = False
    choose: str = "first"


def generate_v5_trace_pool(
    scenarios: list[V5TraceScenario] | tuple[V5TraceScenario, ...],
    out_dir: str | Path,
    *,
    created_at: str = "1970-01-01T00:00:00Z",
) -> TracePoolManifest:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    entries: list[TracePoolEntry] = []
    for scenario in scenarios:
        for seed in scenario.seeds:
            for strength in scenario.adaptive_strengths:
                for visibility in scenario.visibility_modes:
                    for draw_assist in scenario.draw_assist_modes:
                        for assist in scenario.assist_modes:
                            for level_mode in scenario.level_modes:
                                info_mode = _info_mode(strength, visibility, draw_assist)
                                assist_mode = _assist_mode(assist)
                                level_handicap = _level_handicap(level_mode)
                                trace = build_golden_trace(
                                    seed=int(seed),
                                    steps=int(scenario.steps),
                                    placement_mode=scenario.placement_mode,
                                    verify_mask=scenario.verify_mask,
                                    info_mode=info_mode,
                                    assist_mode=assist_mode,
                                    choose=scenario.choose,
                                    p1_deck_ids=_list_or_none(scenario.p1_deck_ids),
                                    p2_deck_ids=_list_or_none(scenario.p2_deck_ids),
                                    p1_level=level_handicap.get("p1_level"),
                                    p2_level=level_handicap.get("p2_level"),
                                    action_ids=_list_or_none(scenario.action_ids),
                                )
                                if level_handicap:
                                    trace["env_config"]["level_handicap"] = level_handicap
                                trace_path = out / _trace_filename(
                                    scenario,
                                    seed,
                                    strength,
                                    visibility,
                                    draw_assist,
                                    assist,
                                    level_handicap,
                                )
                                trace_path.write_text(
                                    json.dumps(trace, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8",
                                )
                                entries.append(
                                    TracePoolEntry(
                                        path=str(trace_path),
                                        scenario_key=scenario.scenario_key,
                                        strength=float(info_mode.clipped_strength()),
                                        visibility=_visibility_payload(info_mode),
                                        draw_assist=_draw_assist_payload(info_mode),
                                        assist_mode=assist_mode.to_dict(),
                                        deck_ids={
                                            "p1": list(scenario.p1_deck_ids or []),
                                            "p2": list(scenario.p2_deck_ids or []),
                                        },
                                        action_script=list(scenario.action_ids or []),
                                        oracle={
                                            "state_sha256": trace["initial"]["state_sha256"],
                                            "obs_v5_sha256_f32_le": trace["initial"]["obs_v5_sha256_f32_le"],
                                            "obs_v5_dim": int(trace["initial"]["obs_v5_dim"]),
                                            "steps": len(trace["steps"]),
                                        },
                                        level_handicap=level_handicap or None,
                                    )
                                )
    manifest = TracePoolManifest(traces=entries, created_at=created_at)
    manifest_dict = manifest_to_dict(manifest)
    return TracePoolManifest(
        traces=entries,
        created_at=created_at,
        manifest_id=str(manifest_dict["manifest_id"]),
    )


def load_v5_trace_pool_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _LoadedV5TracePoolManifest(data, manifest_path)
    _validate_v5_trace_pool_manifest(manifest)
    trace_paths = _resolve_v5_trace_paths(manifest, manifest_path.parent)
    _validate_v5_trace_files(trace_paths, manifest=manifest, base_dir=manifest_path.parent)
    return manifest


def resolve_v5_trace_paths(manifest_or_path: dict[str, Any] | str | Path) -> list[Path]:
    if isinstance(manifest_or_path, str | Path):
        return _resolve_v5_trace_paths(load_v5_trace_pool_manifest(manifest_or_path), Path(manifest_or_path).parent)
    if isinstance(manifest_or_path, TracePoolManifest):
        manifest_or_path = manifest_to_dict(manifest_or_path)
    manifest = manifest_or_path
    _validate_v5_trace_pool_manifest(manifest)
    source_path = getattr(manifest, "source_path", None)
    base_dir = Path(source_path).parent if source_path is not None else Path.cwd()
    trace_paths = _resolve_v5_trace_paths(manifest, base_dir)
    _validate_v5_trace_files(trace_paths, manifest=manifest, base_dir=base_dir)
    return trace_paths


def group_v5_trace_pool_by_mode(manifest_or_path: dict[str, Any] | str | Path) -> dict[tuple[Any, ...], list[Path]]:
    manifest, base_dir = _load_manifest_with_base(manifest_or_path)
    grouped: dict[tuple[Any, ...], list[Path]] = {}
    for entry in _trace_entries(manifest):
        key = _mode_key_from_entry(entry)
        path = Path(str(entry["path"]))
        resolved = path if path.is_absolute() else base_dir / path
        grouped.setdefault(key, []).append(resolved)
    return grouped


def select_v5_trace_paths_for_mode(
    manifest_or_path: dict[str, Any] | str | Path,
    mode: InfoModeV5 | dict[str, Any],
    *,
    assist_mode: AssistModeV5 | dict[str, Any] | None = None,
    runtime_mode_source: str = "manifest_cycle",
) -> list[Path]:
    if runtime_mode_source == "manifest_cycle":
        return resolve_v5_trace_paths(manifest_or_path)
    if runtime_mode_source != "league_schedule":
        raise ValueError("runtime_mode_source must be manifest_cycle or league_schedule")
    grouped = group_v5_trace_pool_by_mode(manifest_or_path)
    key = _mode_key_from_info_mode(mode) + _assist_key_from_assist_mode(assist_mode)
    selected = grouped.get(key, [])
    if not selected:
        raise ValueError(f"no V5 trace subset matches league-scheduled mode {key!r}")
    return selected


def _info_mode(strength: float, visibility: dict[str, Any], draw_assist: dict[str, Any]) -> InfoModeV5:
    return InfoModeV5(
        adaptive_strength=float(strength),
        own_hand_identity_known=bool(visibility.get("own_hand_identity_known", True)),
        own_deck_known=bool(visibility.get("own_deck_known", True)),
        enemy_hand_known=bool(visibility.get("enemy_hand_known", True)),
        enemy_deck_known=bool(visibility.get("enemy_deck_known", True)),
        enemy_deck_order_known=bool(visibility.get("enemy_deck_order_known", True)),
        draw_assist_enabled=bool(draw_assist.get("draw_assist_enabled", False)),
        draw_assist_strength=float(draw_assist.get("draw_assist_strength", 0.0) or 0.0),
    )


def _assist_mode(assist: dict[str, Any] | None) -> AssistModeV5:
    assist = assist or {}
    return AssistModeV5(
        assembler_enabled=bool(assist.get("assembler_enabled", False)),
        assembler_strength=float(assist.get("assembler_strength", 0.0) or 0.0),
        desirerer_enabled=bool(assist.get("desirerer_enabled", False)),
        desirerer_strength=float(assist.get("desirerer_strength", 0.0) or 0.0),
        teacher_hint_available=bool(assist.get("teacher_hint_available", False)),
        assist_profile_id=int(assist.get("assist_profile_id", 0) or 0),
    )


def _visibility_payload(info_mode: InfoModeV5) -> dict[str, Any]:
    return {
        "own_hand_identity_known": bool(info_mode.own_hand_identity_known),
        "own_deck_known": bool(info_mode.own_deck_known),
        "enemy_hand_known": bool(info_mode.enemy_hand_known),
        "enemy_deck_known": bool(info_mode.enemy_deck_known),
        "enemy_deck_order_known": bool(info_mode.enemy_deck_order_known),
    }


def _draw_assist_payload(info_mode: InfoModeV5) -> dict[str, Any]:
    return {
        "draw_assist_enabled": bool(info_mode.draw_assist_enabled),
        "draw_assist_strength": float(info_mode.clipped_draw_assist_strength()),
    }


def _trace_filename(
    scenario: V5TraceScenario,
    seed: int,
    strength: float,
    visibility: dict[str, Any],
    draw_assist: dict[str, Any],
    assist: dict[str, Any] | None,
    level_handicap: dict[str, Any] | None = None,
) -> str:
    enemy = "enemy_known" if visibility.get("enemy_hand_known") or visibility.get("enemy_deck_known") else "enemy_hidden"
    draw = "draw_on" if draw_assist.get("draw_assist_enabled") else "draw_off"
    strength_key = f"{float(strength):.2f}".replace(".", "_")
    payload = {
        "scenario_key": scenario.scenario_key,
        "seed": int(seed),
        "steps": int(scenario.steps),
        "p1_deck_ids": list(scenario.p1_deck_ids or []),
        "p2_deck_ids": list(scenario.p2_deck_ids or []),
        "action_ids": list(scenario.action_ids or []),
        "adaptive_strength": float(strength),
        "visibility": visibility,
        "draw_assist": draw_assist,
        "assist_mode": _assist_mode(assist).to_dict(),
        "level_handicap": level_handicap or {},
        "placement_mode": scenario.placement_mode,
        "verify_mask": bool(scenario.verify_mask),
        "choose": scenario.choose,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    level = _level_filename_suffix(level_handicap)
    return f"{scenario.scenario_key}_seed{int(seed)}_s{strength_key}_{enemy}_{draw}_{level}_{digest}.json"


def _list_or_none(value: tuple[int, ...] | None) -> list[int] | None:
    return None if value is None else [int(item) for item in value]


def _level_handicap(level_mode: dict[str, Any] | None) -> dict[str, Any]:
    level_mode = level_mode or {}
    payload: dict[str, Any] = {}
    if level_mode.get("p1_level") is not None:
        payload["p1_level"] = max(1, min(10, int(level_mode["p1_level"])))
    if level_mode.get("p2_level") is not None:
        payload["p2_level"] = max(1, min(10, int(level_mode["p2_level"])))
    if level_mode.get("label"):
        payload["label"] = str(level_mode["label"])
    elif payload:
        p1 = payload.get("p1_level", "default")
        p2 = payload.get("p2_level", "default")
        payload["label"] = f"p1_l{p1}_vs_p2_l{p2}"
    return payload


def _level_filename_suffix(level_handicap: dict[str, Any] | None) -> str:
    if not level_handicap:
        return "levels_default"
    p1 = level_handicap.get("p1_level", "default")
    p2 = level_handicap.get("p2_level", "default")
    return f"p1l{p1}_p2l{p2}"


def _validate_v5_trace_pool_manifest(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("trace pool manifest must be a JSON object")
    if manifest.get("schema") != TRACE_POOL_SCHEMA:
        raise ValueError(f"trace pool manifest schema must be {TRACE_POOL_SCHEMA!r}")
    if not str(manifest.get("manifest_id", "")).strip():
        raise ValueError("trace pool manifest manifest_id must be non-empty")
    _trace_path_strings(manifest)


def _resolve_v5_trace_paths(manifest: dict[str, Any], base_dir: Path) -> list[Path]:
    resolved: list[Path] = []
    for item in _trace_path_strings(manifest):
        path = Path(item)
        resolved.append(path if path.is_absolute() else base_dir / path)
    return resolved


def _trace_path_strings(manifest: dict[str, Any]) -> list[str]:
    raw_paths = manifest.get("trace_paths")
    if raw_paths is None:
        raw_paths = [entry.get("path") for entry in manifest.get("traces", []) if isinstance(entry, dict)]
    if not isinstance(raw_paths, list):
        raise ValueError("trace pool manifest trace paths must be a list")
    paths = [str(item) for item in raw_paths if str(item).strip()]
    if not paths:
        raise ValueError("trace pool manifest must resolve non-empty trace paths")
    return paths


def _load_manifest_with_base(manifest_or_path: dict[str, Any] | str | Path) -> tuple[dict[str, Any], Path]:
    if isinstance(manifest_or_path, str | Path):
        manifest = load_v5_trace_pool_manifest(manifest_or_path)
        return manifest, Path(manifest_or_path).parent
    if isinstance(manifest_or_path, TracePoolManifest):
        return manifest_to_dict(manifest_or_path), Path.cwd()
    _validate_v5_trace_pool_manifest(manifest_or_path)
    source_path = getattr(manifest_or_path, "source_path", None)
    base_dir = Path(source_path).parent if source_path is not None else Path.cwd()
    return manifest_or_path, base_dir


def _trace_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw_entries = manifest.get("traces")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("trace pool manifest traces must be a non-empty list for V5 mode selection")
    entries: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise ValueError(f"trace pool manifest trace {idx} must be an object")
        if not str(entry.get("path", "")).strip():
            raise ValueError(f"trace pool manifest trace {idx} path must be non-empty")
        entries.append(entry)
    return entries


def _mode_key_from_entry(entry: dict[str, Any]) -> tuple[Any, ...]:
    visibility = entry.get("visibility") if isinstance(entry.get("visibility"), dict) else {}
    draw_assist = entry.get("draw_assist") if isinstance(entry.get("draw_assist"), dict) else {}
    return (
        _mode_float(entry.get("strength", 1.0)),
        bool(visibility.get("own_hand_identity_known", True)),
        bool(visibility.get("own_deck_known", True)),
        bool(visibility.get("enemy_hand_known", False)),
        bool(visibility.get("enemy_deck_known", False)),
        bool(visibility.get("enemy_deck_order_known", False)),
        bool(draw_assist.get("draw_assist_enabled", False)),
        _mode_float(draw_assist.get("draw_assist_strength", 0.0)),
    ) + _assist_key_from_entry(entry)


def _mode_key_from_info_mode(mode: InfoModeV5 | dict[str, Any]) -> tuple[Any, ...]:
    if isinstance(mode, InfoModeV5):
        return (
            _mode_float(mode.clipped_strength()),
            bool(mode.own_hand_identity_known),
            bool(mode.own_deck_known),
            bool(mode.enemy_hand_known),
            bool(mode.enemy_deck_known),
            bool(mode.enemy_deck_order_known),
            bool(mode.draw_assist_enabled),
            _mode_float(mode.clipped_draw_assist_strength()),
        )
    return (
        _mode_float(mode.get("adaptive_strength", mode.get("strength", 1.0))),
        bool(mode.get("own_hand_identity_known", True)),
        bool(mode.get("own_deck_known", True)),
        bool(mode.get("enemy_hand_known", False)),
        bool(mode.get("enemy_deck_known", False)),
        bool(mode.get("enemy_deck_order_known", False)),
        bool(mode.get("draw_assist_enabled", False)),
        _mode_float(mode.get("draw_assist_strength", 0.0)),
    )


def _assist_key_from_entry(entry: dict[str, Any]) -> tuple[Any, ...]:
    assist = entry.get("assist_mode") if isinstance(entry.get("assist_mode"), dict) else {}
    return _assist_key_from_assist_mode(assist)


def _assist_key_from_assist_mode(mode: AssistModeV5 | dict[str, Any] | None) -> tuple[Any, ...]:
    if isinstance(mode, AssistModeV5):
        assist = mode
    else:
        assist = _assist_mode(mode)
    return (
        bool(assist.assembler_enabled),
        _mode_float(assist.clipped_assembler_strength()),
        bool(assist.desirerer_enabled),
        _mode_float(assist.clipped_desirerer_strength()),
        bool(assist.teacher_hint_available),
        int(assist.clipped_profile_id()),
    )


def _mode_float(value: Any) -> float:
    return round(max(0.0, min(1.0, float(value))), 6)


def _validate_v5_trace_files(
    trace_paths: list[Path],
    *,
    manifest: dict[str, Any] | None = None,
    base_dir: Path | None = None,
) -> None:
    entries_by_path = _trace_entries_by_resolved_path(manifest, base_dir) if manifest is not None else {}
    for trace_path in trace_paths:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        env_config = trace.get("env_config")
        if not isinstance(env_config, dict):
            raise ValueError(f"{trace_path}: env_config must be present")
        for field in _V5_ENV_CONFIG_FIELDS:
            if field not in env_config:
                raise ValueError(f"{trace_path}: env_config missing {field}")
        initial = trace.get("initial")
        if not isinstance(initial, dict):
            raise ValueError(f"{trace_path}: initial must be present")
        if int(initial.get("obs_v5_dim", -1)) != OBS_V5_DIM:
            raise ValueError(f"{trace_path}: initial obs_v5_dim must be exactly {OBS_V5_DIM}")
        entry = entries_by_path.get(trace_path.resolve())
        if entry is not None:
            _validate_trace_entry_matches_trace(trace_path, entry, trace)


def _trace_entries_by_resolved_path(
    manifest: dict[str, Any] | None,
    base_dir: Path | None,
) -> dict[Path, dict[str, Any]]:
    if manifest is None:
        return {}
    raw_entries = manifest.get("traces")
    if not isinstance(raw_entries, list) or not raw_entries:
        return {}
    root = base_dir or Path.cwd()
    entries: dict[Path, dict[str, Any]] = {}
    for idx, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise ValueError(f"trace pool manifest trace {idx} must be an object")
        path_text = str(entry.get("path", "")).strip()
        if not path_text:
            raise ValueError(f"trace pool manifest trace {idx} path must be non-empty")
        path = Path(path_text)
        resolved = path if path.is_absolute() else root / path
        entries[resolved.resolve()] = entry
    return entries


def _validate_trace_entry_matches_trace(trace_path: Path, entry: dict[str, Any], trace: dict[str, Any]) -> None:
    env_config = trace["env_config"]
    expected = {
        "adaptive_strength": _mode_float(entry.get("strength", 1.0)),
        "own_hand_identity_known": bool(_entry_visibility(entry).get("own_hand_identity_known", True)),
        "own_deck_known": bool(_entry_visibility(entry).get("own_deck_known", True)),
        "enemy_hand_known": bool(_entry_visibility(entry).get("enemy_hand_known", False)),
        "enemy_deck_known": bool(_entry_visibility(entry).get("enemy_deck_known", False)),
        "enemy_deck_order_known": bool(_entry_visibility(entry).get("enemy_deck_order_known", False)),
        "draw_assist_enabled": bool(_entry_draw_assist(entry).get("draw_assist_enabled", False)),
        "draw_assist_strength": _mode_float(_entry_draw_assist(entry).get("draw_assist_strength", 0.0)),
        "assembler_enabled": bool(_entry_assist_mode(entry).get("assembler_enabled", False)),
        "assembler_strength": _mode_float(_entry_assist_mode(entry).get("assembler_strength", 0.0)),
        "desirerer_enabled": bool(_entry_assist_mode(entry).get("desirerer_enabled", False)),
        "desirerer_strength": _mode_float(_entry_assist_mode(entry).get("desirerer_strength", 0.0)),
        "teacher_hint_available": bool(_entry_assist_mode(entry).get("teacher_hint_available", False)),
        "assist_profile_id": int(_entry_assist_mode(entry).get("assist_profile_id", 0) or 0),
    }
    actual = {
        "adaptive_strength": _mode_float(env_config.get("adaptive_strength", 1.0)),
        "own_hand_identity_known": bool(env_config.get("own_hand_identity_known", True)),
        "own_deck_known": bool(env_config.get("own_deck_known", True)),
        "enemy_hand_known": bool(env_config.get("enemy_hand_known", False)),
        "enemy_deck_known": bool(env_config.get("enemy_deck_known", False)),
        "enemy_deck_order_known": bool(env_config.get("enemy_deck_order_known", False)),
        "draw_assist_enabled": bool(env_config.get("draw_assist_enabled", False)),
        "draw_assist_strength": _mode_float(env_config.get("draw_assist_strength", 0.0)),
        "assembler_enabled": bool(env_config.get("assembler_enabled", False)),
        "assembler_strength": _mode_float(env_config.get("assembler_strength", 0.0)),
        "desirerer_enabled": bool(env_config.get("desirerer_enabled", False)),
        "desirerer_strength": _mode_float(env_config.get("desirerer_strength", 0.0)),
        "teacher_hint_available": bool(env_config.get("teacher_hint_available", False)),
        "assist_profile_id": int(env_config.get("assist_profile_id", 0) or 0),
    }
    if expected != actual:
        raise ValueError(f"{trace_path}: trace manifest entry V5 mode does not match env_config")
    oracle = entry.get("oracle")
    if isinstance(oracle, dict):
        initial = trace.get("initial") if isinstance(trace.get("initial"), dict) else {}
        for key in ("state_sha256", "obs_v5_sha256_f32_le"):
            if key in oracle and initial.get(key) != oracle[key]:
                raise ValueError(f"{trace_path}: trace manifest oracle {key} does not match trace initial")
        if "obs_v5_dim" in oracle and int(oracle["obs_v5_dim"]) != int(initial.get("obs_v5_dim", -1)):
            raise ValueError(f"{trace_path}: trace manifest oracle obs_v5_dim does not match trace initial")
    action_script = entry.get("action_script")
    if isinstance(action_script, list) and action_script:
        actual_actions = [step.get("action_id") for step in trace.get("steps", []) if isinstance(step, dict)]
        expected_actions = [int(item) for item in action_script]
        if actual_actions[: len(expected_actions)] != expected_actions:
            raise ValueError(f"{trace_path}: trace manifest action_script does not match trace steps")


def _entry_visibility(entry: dict[str, Any]) -> dict[str, Any]:
    visibility = entry.get("visibility")
    return visibility if isinstance(visibility, dict) else {}


def _entry_draw_assist(entry: dict[str, Any]) -> dict[str, Any]:
    draw_assist = entry.get("draw_assist")
    return draw_assist if isinstance(draw_assist, dict) else {}


def _entry_assist_mode(entry: dict[str, Any]) -> dict[str, Any]:
    assist_mode = entry.get("assist_mode")
    return assist_mode if isinstance(assist_mode, dict) else {}


__all__ = [
    "V5TraceScenario",
    "generate_v5_trace_pool",
    "group_v5_trace_pool_by_mode",
    "load_v5_trace_pool_manifest",
    "resolve_v5_trace_paths",
    "select_v5_trace_paths_for_mode",
]

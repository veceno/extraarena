"""JSON-safe artifact manifests for TrainV3 V5 training pipelines."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

TRACE_POOL_SCHEMA = "trainv3-v5-trace-pool-v1"
LEAGUE_RUN_SCHEMA = "trainv3-v5-league-run-v1"
AUX_DATASET_SCHEMA = "trainv3-v5-aux-dataset-v1"
DEFAULT_CREATED_AT = "1970-01-01T00:00:00Z"


@dataclass(frozen=True)
class TracePoolEntry:
    path: str | Path
    scenario_key: str
    strength: float
    visibility: dict[str, Any]
    draw_assist: dict[str, Any]
    assist_mode: dict[str, Any]
    deck_ids: dict[str, list[int]]
    action_script: list[int]
    oracle: dict[str, Any]
    level_handicap: dict[str, Any] | None = None


@dataclass(frozen=True)
class TracePoolManifest:
    traces: list[TracePoolEntry]
    created_at: str = DEFAULT_CREATED_AT
    schema: str = TRACE_POOL_SCHEMA
    manifest_id: str = ""

    @property
    def trace_paths(self) -> list[str]:
        return [str(Path(entry.path)) for entry in self.traces]


@dataclass(frozen=True)
class LeagueRunManifest:
    run_name: str
    model_name: str
    trace_manifest_id: str
    config: dict[str, Any]
    curriculum: dict[str, Any]
    metrics_path: str | Path | None = None
    checkpoint_path: str | Path | None = None
    created_at: str = DEFAULT_CREATED_AT
    schema: str = LEAGUE_RUN_SCHEMA
    manifest_id: str = ""


@dataclass(frozen=True)
class AuxDatasetManifest:
    dataset_kind: str
    dataset_path: str | Path
    rows: int
    source_manifest_ids: tuple[str, ...] = ()
    label: str = ""
    created_at: str = DEFAULT_CREATED_AT
    schema: str = AUX_DATASET_SCHEMA
    manifest_id: str = ""


def manifest_to_dict(manifest: Any) -> dict[str, Any]:
    data = _jsonable(manifest)
    if isinstance(manifest, TracePoolManifest):
        data["trace_paths"] = manifest.trace_paths
    if not data.get("manifest_id"):
        data["manifest_id"] = _stable_manifest_id(data)
    return data


def write_manifest_json(manifest: Any, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest_to_dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def read_manifest_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stable_manifest_id(data: dict[str, Any]) -> str:
    clean = dict(data)
    clean.pop("manifest_id", None)
    raw = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "v5_" + hashlib.sha256(raw).hexdigest()[:16]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


__all__ = [
    "AUX_DATASET_SCHEMA",
    "LEAGUE_RUN_SCHEMA",
    "TRACE_POOL_SCHEMA",
    "AuxDatasetManifest",
    "LeagueRunManifest",
    "TracePoolEntry",
    "TracePoolManifest",
    "manifest_to_dict",
    "read_manifest_json",
    "write_manifest_json",
]

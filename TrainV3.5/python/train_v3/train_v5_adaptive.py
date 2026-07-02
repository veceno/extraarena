"""Adaptive V5 league launcher for TrainV3."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from .aux_models import (
    AssemblerDatasetRow,
    build_assembler_rows_from_matchup_summaries,
    build_desirerer_rows_from_v5_trace,
    save_assembler_dataset_with_manifest,
    save_desirerer_dataset_with_manifest,
)
from .league_v5 import V5LeagueConfig
from .rust_trainer import RustPPOTrainingConfig, train_rust_ppo_trace_files
from .trace_factory_v5 import (
    V5TraceScenario,
    generate_v5_trace_pool,
    load_v5_trace_pool_manifest,
    resolve_v5_trace_paths,
)
from .v5_artifacts import read_manifest_json, write_manifest_json


def run_v5_adaptive_training_pipeline(
    config: RustPPOTrainingConfig,
    *,
    scenarios: Iterable[V5TraceScenario] | None = None,
    trace_pool_dir: str | Path | None = None,
    trace_manifest_path: str | Path | None = None,
    aux_output_dir: str | Path | None = None,
    model: Any = None,
    optimizer: Any = None,
    library_path: str | Path | None = None,
    allow_empty_aux: bool = False,
) -> dict[str, Any]:
    out_dir = _default_output_dir(config)
    trace_dir = Path(trace_pool_dir) if trace_pool_dir is not None else out_dir / "trace_pool"
    manifest_path = Path(trace_manifest_path or config.trace_manifest_path or out_dir / "trace_manifest.json")
    aux_dir = Path(aux_output_dir) if aux_output_dir is not None else out_dir / "aux"

    trace_manifest = _load_or_generate_trace_manifest(
        scenarios=scenarios,
        trace_dir=trace_dir,
        manifest_path=manifest_path,
    )

    train_config = replace(config, trace_manifest_path=manifest_path)
    training_result = train_rust_ppo_trace_files(
        [],
        model,
        optimizer,
        train_config,
        library_path=library_path,
    )
    _validate_training_result_artifacts(training_result, trace_manifest.manifest_id)

    assembler_rows = _derive_assembler_rows(training_result)
    trained_trace_paths = _selected_trace_paths_for_aux(training_result, manifest_path)
    desirerer_rows = _derive_desirerer_rows_from_paths(trained_trace_paths)
    if not allow_empty_aux:
        _require_non_empty_aux("assembler", assembler_rows)
        _require_non_empty_aux("desirerer", desirerer_rows)
    assembler_dataset_path, assembler_manifest_path = save_assembler_dataset_with_manifest(
        assembler_rows,
        aux_dir / "assembler.jsonl",
        aux_dir / "assembler_manifest.json",
        source_manifest_ids=(str(trace_manifest.manifest_id),),
    )
    desirerer_dataset_path, desirerer_manifest_path = save_desirerer_dataset_with_manifest(
        desirerer_rows,
        aux_dir / "desirerer.jsonl",
        aux_dir / "desirerer_manifest.json",
        source_manifest_ids=(str(trace_manifest.manifest_id),),
    )

    league_manifest_path = str(training_result.get("league_manifest_path") or train_config.league_manifest_path or "")
    return {
        "run_name": config.run_name,
        "model_name": config.model_name,
        "trace_manifest_path": str(manifest_path),
        "trace_manifest_id": trace_manifest.manifest_id,
        "league_manifest_path": league_manifest_path,
        "assembler_dataset_path": str(assembler_dataset_path),
        "assembler_manifest_path": str(assembler_manifest_path),
        "desirerer_dataset_path": str(desirerer_dataset_path),
        "desirerer_manifest_path": str(desirerer_manifest_path),
        "training_result": training_result,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run TrainV3 V5 adaptive league training.")
    parser.add_argument("--config", type=Path, help="JSON file containing RustPPOTrainingConfig fields.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--trace-pool-dir", type=Path, default=None)
    parser.add_argument("--trace-manifest-path", type=Path, default=None)
    parser.add_argument("--aux-output-dir", type=Path, default=None)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--action-hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--policy-kind", default="v5_split_encoder", choices=["v5_split_encoder", "baseline_mlp"])
    parser.add_argument("--allow-empty-aux", action="store_true")
    args = parser.parse_args(argv)

    config = _config_from_json(args.config) if args.config is not None else RustPPOTrainingConfig()
    output_dir = args.output_dir or _default_output_dir(config)
    model, optimizer = create_v5_default_model_optimizer(
        hidden_dim=args.hidden_dim,
        action_hidden_dim=args.action_hidden_dim,
        learning_rate=args.learning_rate,
        policy_kind=args.policy_kind,
    )
    result = run_v5_adaptive_training_pipeline(
        config,
        trace_pool_dir=args.trace_pool_dir or output_dir / "trace_pool",
        trace_manifest_path=args.trace_manifest_path or output_dir / "trace_manifest.json",
        aux_output_dir=args.aux_output_dir or output_dir / "aux",
        model=model,
        optimizer=optimizer,
        allow_empty_aux=args.allow_empty_aux,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _default_output_dir(config: RustPPOTrainingConfig) -> Path:
    if config.metrics_path is not None:
        return Path(config.metrics_path).parent
    if config.league_manifest_path is not None:
        return Path(config.league_manifest_path).parent
    if config.checkpoint_dir is not None:
        return Path(config.checkpoint_dir).parent
    return Path.cwd() / "train_v5_adaptive"


def _default_scenarios() -> tuple[V5TraceScenario, ...]:
    return (
        V5TraceScenario(
            scenario_key="v5_adaptive_default",
            seeds=(42,),
            steps=16,
            p1_deck_ids=(1, 37, 38, 40, 41, 42, 27, 28, 29),
            p2_deck_ids=(1, 37, 38, 40, 41, 42, 27, 28, 29),
        ),
    )


def create_v5_default_model_optimizer(
    *,
    hidden_dim: int = 256,
    action_hidden_dim: int = 128,
    learning_rate: float = 3e-4,
    policy_kind: str = "v5_split_encoder",
) -> tuple[Any, Any]:
    import mlx.optimizers as optim

    from .v5_policy import create_v5_policy

    model = create_v5_policy(
        policy_kind=policy_kind,
        hidden_dim=int(hidden_dim),
        action_hidden_dim=int(action_hidden_dim),
    )
    optimizer = optim.Adam(learning_rate=float(learning_rate))
    return model, optimizer


def _load_or_generate_trace_manifest(
    *,
    scenarios: Iterable[V5TraceScenario] | None,
    trace_dir: Path,
    manifest_path: Path,
) -> Any:
    if scenarios is None and manifest_path.exists():
        return _manifest_object_from_loaded(load_v5_trace_pool_manifest(manifest_path))
    trace_manifest = generate_v5_trace_pool(list(scenarios or _default_scenarios()), trace_dir)
    write_manifest_json(trace_manifest, manifest_path)
    return trace_manifest


def _manifest_object_from_loaded(manifest: dict[str, Any]) -> Any:
    return SimpleNamespace(manifest_id=str(manifest["manifest_id"]))


def _validate_training_result_artifacts(training_result: dict[str, Any], trace_manifest_id: str) -> None:
    result_manifest_id = str(training_result.get("trace_manifest_id", ""))
    if result_manifest_id != str(trace_manifest_id):
        raise ValueError(
            f"trainer trace_manifest_id {result_manifest_id!r} does not match launcher manifest {trace_manifest_id!r}"
        )
    league_manifest_path = training_result.get("league_manifest_path")
    if league_manifest_path:
        league_manifest = read_manifest_json(league_manifest_path)
        if str(league_manifest.get("trace_manifest_id", "")) != str(trace_manifest_id):
            raise ValueError("league manifest trace_manifest_id does not match launcher manifest")


def _derive_assembler_rows(_training_result: dict[str, Any]) -> list[AssemblerDatasetRow]:
    summaries = _training_result.get("assembler_matchup_summaries")
    if summaries is None:
        summaries = _training_result.get("matchup_summaries", [])
    return build_assembler_rows_from_matchup_summaries(summaries)


def _derive_desirerer_rows(trace_manifest_path: str | Path):
    return _derive_desirerer_rows_from_paths(resolve_v5_trace_paths(load_v5_trace_pool_manifest(trace_manifest_path)))


def _derive_desirerer_rows_from_paths(trace_paths: Iterable[str | Path]):
    rows = []
    for trace_path in trace_paths:
        trace = read_manifest_json(trace_path)
        rows.extend(build_desirerer_rows_from_v5_trace(trace, source_run=str(trace_path)))
    return rows


def _selected_trace_paths_for_aux(training_result: dict[str, Any], trace_manifest_path: str | Path) -> list[Path]:
    selected = training_result.get("selected_trace_paths")
    if isinstance(selected, list) and selected:
        return [Path(path) for path in selected]
    subsets = training_result.get("selected_trace_subsets")
    if isinstance(subsets, list):
        unique: dict[str, Path] = {}
        for subset in subsets:
            if not isinstance(subset, dict):
                continue
            for item in subset.get("selected_trace_paths", []):
                unique.setdefault(str(item), Path(item))
        if unique:
            return list(unique.values())
    return resolve_v5_trace_paths(load_v5_trace_pool_manifest(trace_manifest_path))


def _require_non_empty_aux(kind: str, rows: list[Any]) -> None:
    if not rows:
        raise ValueError(f"{kind} aux dataset would be empty; pass allow_empty_aux=True for smoke-only runs")


def _config_from_json(path: str | Path) -> RustPPOTrainingConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config JSON must be an object")
    league_config = data.get("v5_league_config")
    if isinstance(league_config, dict):
        data["v5_league_config"] = V5LeagueConfig(**league_config)
    return RustPPOTrainingConfig(**data)


__all__ = [
    "create_v5_default_model_optimizer",
    "main",
    "run_v5_adaptive_training_pipeline",
]


if __name__ == "__main__":
    main()

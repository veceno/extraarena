#!/usr/bin/env python3
"""Train an additive second-start recovery reranker for V5.

The base V5 checkpoint stays frozen. This script trains a separate V5-shaped
action scorer on second-start rollout-search labels. At inference the scorer is
used only as a centered bias on top of the base logits, so first-start behavior
does not forget.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAINV3_PYTHON = ROOT / "TrainV3" / "python"
TRAINV3_SCRIPTS = ROOT / "TrainV3" / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAINV3_PYTHON) not in sys.path:
    sys.path.insert(0, str(TRAINV3_PYTHON))
if str(TRAINV3_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TRAINV3_SCRIPTS))

from ai.train_v2.model_mlx import save_checkpoint  # noqa: E402
from run_phase10_v4max_distill import (  # noqa: E402
    DEFAULT_ASSEMBLER_DATASET,
    DEFAULT_V4_MAX,
    DistillConfig,
    _jsonable,
    collect_v5_rollout_search_dataset,
    train_teacher_cross_entropy,
)
from train_v3.v5_policy import create_v5_policy  # noqa: E402


DEFAULT_BASE_CHECKPOINT = (
    ROOT
    / "TrainV3"
    / "runs"
    / "phase10_v4max_distill_round2_from_15020_20260609_1324"
    / "extra_lr_v5_phase10_v4max_distill_61571_states.npz"
)


@dataclass(frozen=True)
class RecoveryRerankerConfig:
    base_checkpoint: Path
    v4_model: Path
    output_dir: Path
    games: int
    max_steps: int
    seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    assembler_dataset: Path | None = DEFAULT_ASSEMBLER_DATASET
    search_candidates: int = 8
    search_depth_plies: int = 6
    search_min_margin: float = 0.25
    hidden_dim: int = 256
    action_hidden_dim: int = 128
    save_dataset: bool = True


def run_recovery_reranker_training(config: RecoveryRerankerConfig) -> dict[str, Any]:
    _validate_recovery_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    distill_config = DistillConfig(
        source_checkpoint=config.base_checkpoint,
        v4_model=config.v4_model,
        output_dir=config.output_dir,
        games=config.games,
        max_steps=config.max_steps,
        seed=config.seed,
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        profile="strong",
        collection_mode="v5_rollout_search",
        focus_start_mode="v5_second",
        assembler_dataset=config.assembler_dataset,
        search_candidates=config.search_candidates,
        search_depth_plies=config.search_depth_plies,
        search_min_margin=config.search_min_margin,
        save_dataset=False,
        restore_optimizer=False,
    )
    dataset = collect_v5_rollout_search_dataset(distill_config)
    if config.save_dataset:
        np.savez_compressed(
            config.output_dir / "second_start_recovery_dataset.npz",
            observations=dataset["observations"],
            action_features=dataset["action_features"],
            masks=dataset["masks"],
            actions=dataset["actions"],
            seeds=dataset["seeds"],
            v5_started=dataset["v5_started"],
        )

    import mlx.optimizers as optim

    model = create_v5_policy(
        policy_kind="v5_split_encoder",
        hidden_dim=config.hidden_dim,
        action_hidden_dim=config.action_hidden_dim,
    )
    optimizer = optim.Adam(learning_rate=config.learning_rate)
    train_summary = train_teacher_cross_entropy(
        model,
        optimizer,
        observations=dataset["observations"],
        action_features=dataset["action_features"],
        masks=dataset["masks"],
        actions=dataset["actions"],
        epochs=config.epochs,
        batch_size=config.batch_size,
        seed=config.seed + 131,
    )

    checkpoint_path = config.output_dir / (
        f"extra_lr_v5_phase13_second_start_recovery_{int(dataset['actions'].shape[0])}_states.npz"
    )
    metadata = {
        "run_name": "phase13_second_start_recovery_reranker",
        "model_name": "extra-lr-v5-adaptive-recovery-reranker",
        "phase": "phase13_second_start_recovery_reranker",
        "base_checkpoint": str(config.base_checkpoint),
        "v4_model": str(config.v4_model),
        "obs_dim": 6480,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "activation": "only_when_v5_started_false",
        "inference_contract": "centered_additive_logits_bias",
        "not_standalone_policy": True,
        "config": _jsonable(asdict(config)),
        "dataset": dataset["summary"],
        "train_summary": train_summary,
        "v4_1_included": False,
    }
    save_checkpoint(str(checkpoint_path), model, optimizer=optimizer, metadata=metadata)
    result = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_summary": dataset["summary"],
        "train_summary": train_summary,
        "summary": {
            "status": "ok",
            "checkpoint_path": str(checkpoint_path),
            "states": int(dataset["actions"].shape[0]),
            "improved_labels": int(dataset["summary"].get("improved_labels", 0)),
            "avg_score_margin": float(dataset["summary"].get("avg_score_margin", 0.0)),
            "final_loss": float(train_summary["final_loss"]),
            "final_accuracy": float(train_summary["final_accuracy"]),
        },
    }
    (config.output_dir / "phase13_recovery_reranker_summary.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _validate_recovery_config(config: RecoveryRerankerConfig) -> None:
    if not config.base_checkpoint.exists():
        raise FileNotFoundError(f"base checkpoint not found: {config.base_checkpoint}")
    if not config.v4_model.exists():
        raise FileNotFoundError(f"V4 model not found: {config.v4_model}")
    if config.assembler_dataset is not None and not config.assembler_dataset.exists():
        raise FileNotFoundError(f"assembler dataset not found: {config.assembler_dataset}")
    for name in ("games", "max_steps", "batch_size", "epochs", "hidden_dim", "action_hidden_dim"):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if float(config.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if int(config.search_candidates) <= 0:
        raise ValueError("search_candidates must be positive")
    if int(config.search_depth_plies) < 0:
        raise ValueError("search_depth_plies must be non-negative")
    if float(config.search_min_margin) < 0.0:
        raise ValueError("search_min_margin must be non-negative")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Train a second-start V5 recovery reranker")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--v4-model", type=Path, default=DEFAULT_V4_MAX)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "TrainV3" / "runs" / f"phase13_second_start_recovery_reranker_{stamp}",
    )
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=21300000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--assembler-dataset", type=Path, default=DEFAULT_ASSEMBLER_DATASET)
    parser.add_argument("--search-candidates", type=int, default=8)
    parser.add_argument("--search-depth-plies", type=int, default=6)
    parser.add_argument("--search-min-margin", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--action-hidden-dim", type=int, default=128)
    parser.add_argument("--no-save-dataset", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = RecoveryRerankerConfig(
        base_checkpoint=args.base_checkpoint.resolve(),
        v4_model=args.v4_model.resolve(),
        output_dir=args.output_dir.resolve(),
        games=int(args.games),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        assembler_dataset=args.assembler_dataset.resolve() if args.assembler_dataset is not None else None,
        search_candidates=int(args.search_candidates),
        search_depth_plies=int(args.search_depth_plies),
        search_min_margin=float(args.search_min_margin),
        hidden_dim=int(args.hidden_dim),
        action_hidden_dim=int(args.action_hidden_dim),
        save_dataset=not bool(args.no_save_dataset),
    )
    result = run_recovery_reranker_training(config)
    print("PHASE13_RECOVERY_RERANKER_RESULT", json.dumps(result["summary"], sort_keys=True), flush=True)
    print(f"Saved: {result['checkpoint_path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

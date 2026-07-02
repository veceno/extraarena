#!/usr/bin/env python3
"""Phase 2 / 29: clean no-assist V4-league foundation distillation for V5.

V4 opponents are used as offline teachers, not as an online Rust rollout
dependency. This keeps the training hot path Rust-first for PPO phases while
letting us inject V4 league knowledge through deterministic MLX distillation.
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
TRAINV3_PYTHON = ROOT / "TrainV3.5" / "python"
TRAINV3_SCRIPTS = ROOT / "TrainV3.5" / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAINV3_PYTHON) not in sys.path:
    sys.path.insert(0, str(TRAINV3_PYTHON))
if str(TRAINV3_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TRAINV3_SCRIPTS))

from ai.train_v2.model_mlx import load_checkpoint, save_checkpoint  # noqa: E402
from run_phase10_v4max_distill import (  # noqa: E402
    DEFAULT_ASSEMBLER_DATASET,
    NOASSIST_BASELINE_DECK_IDS,
    DistillConfig,
    collect_teacher_dataset,
    compute_reference_log_probs,
    train_teacher_cross_entropy,
)
from train_v3.v5_policy import create_v5_policy  # noqa: E402


DEFAULT_SOURCE_CHECKPOINT = (
    ROOT
    / "TrainV3.5"
    / "runs"
    / "phase26_u0020_balanced_antidraw_repair_20260611_133450"
    / "checkpoints"
    / "trainv3_rust_legal_update_0010.npz"
)

DEFAULT_V4_LEAGUE = (
    ("v4-max", ROOT / "ai" / "models" / "extra-lr-v4-max.onnx", 0.45),
    ("v4-opti", ROOT / "ai" / "models" / "extra-lr-v4-opti.onnx", 0.20),
    ("v4-lite", ROOT / "ai" / "models" / "extra-lr-v4-lite.onnx", 0.10),
    ("v4-micro", ROOT / "ai" / "models" / "extra-lr-v4-micro.onnx", 0.05),
)

DEFAULT_NOASSIST_DECK_POOL = (
    (1, 37, 38, 40, 41, 42, 27, 28, 29),
    (1, 40, 43, 29, 31, 25, 27, 37, 28),
    (1, 32, 26, 16, 40, 33, 46, 28, 29),
    (1, 17, 23, 19, 8, 42, 28, 44, 43),
    (1, 32, 18, 45, 28, 16, 46, 38, 36),
    (1, 20, 42, 31, 21, 13, 27, 28, 38),
    (1, 41, 45, 31, 21, 40, 39, 43, 15),
)


@dataclass(frozen=True)
class V4LeagueTeacherSpec:
    name: str
    model_path: Path
    weight: float
    games: int


@dataclass(frozen=True)
class Phase29Config:
    source_checkpoint: Path
    output_dir: Path
    teacher_specs: tuple[V4LeagueTeacherSpec, ...]
    collection_mode: str
    focus_start_mode: str
    max_steps: int
    seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    source_kl_coef: float
    max_states: int
    noassist_deck_ids: tuple[int, ...]
    noassist_deck_pool: tuple[tuple[int, ...], ...]
    assembler_dataset: Path | None = DEFAULT_ASSEMBLER_DATASET
    save_dataset: bool = True


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = Phase29Config(
        source_checkpoint=args.source_checkpoint.resolve(),
        output_dir=args.output_dir.resolve(),
        teacher_specs=_resolve_teacher_specs(
            total_games=int(args.total_games),
            league_spec=str(args.v4_league),
        ),
        collection_mode=str(args.collection_mode),
        focus_start_mode=str(args.focus_start_mode),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        source_kl_coef=float(args.source_kl_coef),
        max_states=int(args.max_states),
        noassist_deck_ids=_parse_deck_ids(args.noassist_deck_ids),
        noassist_deck_pool=_parse_deck_pool(args.noassist_deck_pool),
        assembler_dataset=args.assembler_dataset.resolve() if args.assembler_dataset is not None else None,
        save_dataset=bool(args.save_dataset),
    )
    result = run_phase29(config)
    print("PHASE29_RESULT", json.dumps(result["summary"], sort_keys=True), flush=True)
    print(f"Saved: {result['checkpoint_path']}", flush=True)
    return 0


def run_phase29(config: Phase29Config) -> dict[str, Any]:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = collect_v4_league_dataset(config)
    if config.save_dataset:
        np.savez_compressed(
            config.output_dir / "phase29_v4_league_dataset.npz",
            observations=dataset["observations"],
            action_features=dataset["action_features"],
            masks=dataset["masks"],
            actions=dataset["actions"],
            seeds=dataset["seeds"],
            v5_started=dataset["v5_started"],
            teacher_names=dataset["teacher_names"],
        )

    import mlx.optimizers as optim

    model = create_v5_policy(policy_kind="v5_split_encoder", hidden_dim=256, action_hidden_dim=128)
    optimizer = optim.Adam(learning_rate=float(config.learning_rate))
    loaded = load_checkpoint(str(config.source_checkpoint), model, optimizer=None)
    reference_log_probs = (
        compute_reference_log_probs(
            model,
            observations=dataset["observations"],
            action_features=dataset["action_features"],
            masks=dataset["masks"],
            batch_size=int(config.batch_size),
        )
        if float(config.source_kl_coef) > 0.0
        else None
    )
    train_summary = train_teacher_cross_entropy(
        model,
        optimizer,
        observations=dataset["observations"],
        action_features=dataset["action_features"],
        masks=dataset["masks"],
        actions=dataset["actions"],
        reference_log_probs=reference_log_probs,
        source_kl_coef=float(config.source_kl_coef),
        epochs=int(config.epochs),
        batch_size=int(config.batch_size),
        seed=int(config.seed) + 2901,
    )
    state_count = int(dataset["actions"].shape[0])
    checkpoint_path = config.output_dir / f"extra_lr_v5_phase29_v4_league_{state_count}_states.npz"
    metadata = {
        "run_name": "phase29_v4_league_foundation",
        "model_name": "extra-lr-v5-adaptive",
        "phase": "phase29_v4_league_foundation",
        "source_checkpoint": str(config.source_checkpoint),
        "source_metadata": loaded.get("metadata", {}),
        "obs_dim": 6480,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "offline_v4_league_teacher_lane": True,
        "online_v4_rollout_dependency": False,
        "assist_policy": "off",
        "private_info_policy": "enemy_hidden_only",
        "draw_assist_policy": "off",
        "v4_1_included": False,
        "config": _jsonable(asdict(config)),
        "dataset": dataset["summary"],
        "train_summary": train_summary,
    }
    save_checkpoint(str(checkpoint_path), model, optimizer=optimizer, metadata=metadata)
    result = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_summary": dataset["summary"],
        "train_summary": train_summary,
        "summary": {
            "status": "ok",
            "checkpoint_path": str(checkpoint_path),
            "states": state_count,
            "teacher_counts": dataset["summary"]["teacher_counts"],
            "final_accuracy": float(train_summary["final_accuracy"]),
            "final_source_kl": float(train_summary["final_source_kl"]),
        },
    }
    (config.output_dir / "phase29_v4_league_summary.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def collect_v4_league_dataset(config: Phase29Config) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    teacher_names: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []
    for idx, spec in enumerate(config.teacher_specs):
        if int(spec.games) <= 0:
            continue
        print(
            f"phase29_collect teacher={spec.name} games={spec.games} weight={spec.weight:.3f}",
            flush=True,
        )
        sub_config = DistillConfig(
            source_checkpoint=config.source_checkpoint,
            v4_model=spec.model_path,
            output_dir=config.output_dir / f"dataset_{spec.name}",
            games=int(spec.games),
            max_steps=int(config.max_steps),
            seed=int(config.seed) + idx * 100_000,
            batch_size=int(config.batch_size),
            epochs=1,
            learning_rate=float(config.learning_rate),
            profile="noassist",
            collection_mode=config.collection_mode,
            focus_start_mode=config.focus_start_mode,
            assembler_dataset=config.assembler_dataset,
            noassist_deck_ids=config.noassist_deck_ids,
            noassist_deck_pool=config.noassist_deck_pool,
            source_kl_coef=0.0,
            save_dataset=False,
            restore_optimizer=False,
        )
        chunk = collect_teacher_dataset(sub_config)
        n = int(chunk["actions"].shape[0])
        chunks.append(chunk)
        teacher_names.append(np.asarray([spec.name] * n))
        summaries.append({"teacher": spec.name, "model_path": str(spec.model_path), **chunk["summary"]})
        print(f"phase29_collect_done teacher={spec.name} states={n}", flush=True)

    if not chunks:
        raise RuntimeError("V4 league dataset is empty")
    observations = np.concatenate([chunk["observations"] for chunk in chunks], axis=0).astype(np.float32, copy=False)
    action_features = np.concatenate([chunk["action_features"] for chunk in chunks], axis=0).astype(np.float32, copy=False)
    masks = np.concatenate([chunk["masks"] for chunk in chunks], axis=0).astype(np.float32, copy=False)
    actions = np.concatenate([chunk["actions"] for chunk in chunks], axis=0).astype(np.int32, copy=False)
    seeds = np.concatenate([chunk["seeds"] for chunk in chunks], axis=0).astype(np.int64, copy=False)
    v5_started = np.concatenate([chunk["v5_started"] for chunk in chunks], axis=0).astype(np.bool_, copy=False)
    names = np.concatenate(teacher_names, axis=0)

    if int(config.max_states) > 0 and actions.shape[0] > int(config.max_states):
        rng = np.random.default_rng(int(config.seed) + 29)
        keep = np.arange(actions.shape[0], dtype=np.int64)
        rng.shuffle(keep)
        keep = np.sort(keep[: int(config.max_states)])
        observations = observations[keep]
        action_features = action_features[keep]
        masks = masks[keep]
        actions = actions[keep]
        seeds = seeds[keep]
        v5_started = v5_started[keep]
        names = names[keep]

    teacher_counts = {str(name): int(np.sum(names == name)) for name in sorted(set(names.tolist()))}
    summary = {
        "schema": "extra_lr_v5_phase29_v4_league_dataset_v1",
        "collection_mode": config.collection_mode,
        "focus_start_mode": config.focus_start_mode,
        "states": int(actions.shape[0]),
        "max_states": int(config.max_states),
        "teacher_counts": teacher_counts,
        "teacher_summaries": summaries,
        "v5_started_states": int(np.sum(v5_started)),
        "v5_second_states": int(actions.shape[0] - int(np.sum(v5_started))),
        "profile": "noassist",
        "assist_policy": "off",
        "private_info_policy": "enemy_hidden_only",
        "draw_assist_policy": "off",
        "v4_1_included": False,
        "noassist_deck_ids": list(config.noassist_deck_ids),
        "noassist_deck_pool": [list(deck) for deck in config.noassist_deck_pool],
    }
    return {
        "observations": observations,
        "action_features": action_features,
        "masks": masks,
        "actions": actions,
        "seeds": seeds,
        "v5_started": v5_started,
        "teacher_names": names,
        "summary": summary,
    }


def _resolve_teacher_specs(*, total_games: int, league_spec: str) -> tuple[V4LeagueTeacherSpec, ...]:
    if int(total_games) <= 0:
        raise ValueError("total_games must be positive")
    raw_specs = DEFAULT_V4_LEAGUE if not league_spec.strip() else _parse_league_spec(league_spec)
    positive_specs = [(name, path, weight) for name, path, weight in raw_specs if float(weight) > 0.0]
    total_weight = sum(float(weight) for _name, _path, weight in positive_specs)
    if total_weight <= 0.0:
        raise ValueError("V4 league weights must sum to a positive value")
    base_min = 1 if int(total_games) >= len(positive_specs) else 0
    remaining = int(total_games) - base_min * len(positive_specs)
    fractional: list[tuple[float, int, tuple[str, Path, float]]] = []
    assigned_extra = 0
    for idx, spec in enumerate(positive_specs):
        exact = max(0.0, remaining) * float(spec[2]) / total_weight
        whole = int(np.floor(exact))
        assigned_extra += whole
        fractional.append((exact - whole, idx, spec))
    extras = {idx: int(np.floor(max(0.0, remaining) * float(spec[2]) / total_weight)) for _frac, idx, spec in fractional}
    for _frac, idx, _spec in sorted(fractional, key=lambda item: (-item[0], item[1]))[: max(0, remaining - assigned_extra)]:
        extras[idx] += 1
    specs: list[V4LeagueTeacherSpec] = []
    for idx, (name, model_path, weight) in enumerate(positive_specs):
        games = base_min + extras.get(idx, 0)
        specs.append(
            V4LeagueTeacherSpec(
                name=str(name),
                model_path=Path(model_path).resolve(),
                weight=float(weight),
                games=int(games),
            )
        )
    return tuple(specs)


def _parse_league_spec(value: str) -> tuple[tuple[str, Path, float], ...]:
    specs: list[tuple[str, Path, float]] = []
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [part.strip() for part in chunk.split(":")]
        if len(parts) != 3:
            raise ValueError("v4 league entries must be name:path:weight")
        specs.append((parts[0], Path(parts[1]), float(parts[2])))
    return tuple(specs)


def _parse_deck_ids(raw: str) -> tuple[int, ...]:
    deck = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if len(deck) < 2:
        raise ValueError("deck must include a hero and at least one card")
    return deck


def _parse_deck_pool(raw: str) -> tuple[tuple[int, ...], ...]:
    if not raw.strip():
        return DEFAULT_NOASSIST_DECK_POOL
    decks = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if chunk:
            decks.append(_parse_deck_ids(chunk))
    if not decks:
        raise ValueError("deck pool must not be empty")
    return tuple(decks)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_config(config: Phase29Config) -> None:
    if not config.source_checkpoint.exists():
        raise FileNotFoundError(f"source checkpoint not found: {config.source_checkpoint}")
    if str(config.source_checkpoint).lower().find("v4.1") >= 0:
        raise ValueError("V4.1 checkpoints must not be used for Phase 2")
    for spec in config.teacher_specs:
        if str(spec.model_path).lower().find("v4.1") >= 0 or str(spec.name).lower().find("v4.1") >= 0:
            raise ValueError("V4.1 teacher models must not be used for Phase 2")
        if not spec.model_path.exists():
            raise FileNotFoundError(f"V4 teacher model not found: {spec.model_path}")
    if config.collection_mode not in {"v5_on_policy", "v4_selfplay", "v5_rollout_search"}:
        raise ValueError("collection_mode must be v5_on_policy, v4_selfplay, or v5_rollout_search")
    if config.focus_start_mode not in {"both", "v5_first", "v5_second"}:
        raise ValueError("focus_start_mode must be both, v5_first, or v5_second")
    for name in ("max_steps", "batch_size", "epochs"):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if float(config.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if float(config.source_kl_coef) < 0.0:
        raise ValueError("source_kl_coef must be non-negative")
    if config.assembler_dataset is not None and not config.assembler_dataset.exists():
        raise FileNotFoundError(f"assembler dataset not found: {config.assembler_dataset}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "TrainV3.5" / "runs" / f"phase29_v4_league_foundation_{stamp}")
    parser.add_argument("--total-games", type=int, default=128)
    parser.add_argument("--v4-league", default="")
    parser.add_argument("--collection-mode", choices=["v5_on_policy", "v4_selfplay", "v5_rollout_search"], default="v5_on_policy")
    parser.add_argument("--focus-start-mode", choices=["both", "v5_first", "v5_second"], default="both")
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=29001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=4.0e-5)
    parser.add_argument("--source-kl-coef", type=float, default=0.2)
    parser.add_argument("--max-states", type=int, default=0)
    parser.add_argument("--assembler-dataset", type=Path, default=DEFAULT_ASSEMBLER_DATASET)
    parser.add_argument("--noassist-deck-ids", default=",".join(str(card_id) for card_id in NOASSIST_BASELINE_DECK_IDS))
    parser.add_argument("--noassist-deck-pool", default="")
    parser.add_argument("--save-dataset", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train a second-start recovery reranker with positive/negative trajectories."""
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
from ai.train_v2.onnx_policy import OnnxActionPolicy  # noqa: E402
from run_phase10_v4max_distill import (  # noqa: E402
    DEFAULT_ASSEMBLER_DATASET,
    DEFAULT_V4_MAX,
    _jsonable,
    _profile_modes,
    _v5_on_policy_game_specs,
)
from run_phase14_second_start_success_recovery import DEFAULT_BASE_CHECKPOINT  # noqa: E402
from run_v5_vs_v4max_benchmark import (  # noqa: E402
    V5AdaptivePolicy,
    _action_is_end_turn,
    _apply_draw_assist_to_player,
    _load_assembler_candidates,
    _player_card_pool_ids,
    _select_v5_deck,
)
from train_v3.aux_models import DeckMatchupEvaluator, DrawAssistController  # noqa: E402
from train_v3.env_v5 import TrainV3ClassicEnv, TrainV3EnvConfig  # noqa: E402
from train_v3.v5_policy import create_v5_policy  # noqa: E402


@dataclass(frozen=True)
class ContrastiveRecoveryConfig:
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
    min_hp_margin: int = 6
    min_positive_games: int = 4
    min_negative_games: int = 4
    negative_weight: float = 0.35
    hidden_dim: int = 256
    action_hidden_dim: int = 128
    initialize_from_base: bool = True
    save_dataset: bool = True


def run_contrastive_recovery_training(config: ContrastiveRecoveryConfig) -> dict[str, Any]:
    _validate_contrastive_recovery_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = collect_contrastive_recovery_dataset(config)
    summary = dataset["summary"]
    if int(summary["positive_games"]) < int(config.min_positive_games):
        raise RuntimeError(f"positive_games={summary['positive_games']} below min_positive_games={config.min_positive_games}")
    if int(summary["negative_games"]) < int(config.min_negative_games):
        raise RuntimeError(f"negative_games={summary['negative_games']} below min_negative_games={config.min_negative_games}")
    if config.save_dataset:
        np.savez_compressed(
            config.output_dir / "second_start_contrastive_recovery_dataset.npz",
            observations=dataset["observations"],
            action_features=dataset["action_features"],
            masks=dataset["masks"],
            actions=dataset["actions"],
            labels=dataset["labels"],
            seeds=dataset["seeds"],
        )

    import mlx.optimizers as optim

    model = create_v5_policy(
        policy_kind="v5_split_encoder",
        hidden_dim=config.hidden_dim,
        action_hidden_dim=config.action_hidden_dim,
    )
    optimizer = optim.Adam(learning_rate=config.learning_rate)
    loaded: dict[str, Any] | None = None
    if config.initialize_from_base:
        loaded = load_checkpoint(str(config.base_checkpoint), model, optimizer=None)
    train_summary = train_contrastive_action_loss(
        model,
        optimizer,
        observations=dataset["observations"],
        action_features=dataset["action_features"],
        masks=dataset["masks"],
        actions=dataset["actions"],
        labels=dataset["labels"],
        negative_weight=config.negative_weight,
        epochs=config.epochs,
        batch_size=config.batch_size,
        seed=config.seed + 151,
    )

    checkpoint_path = config.output_dir / (
        f"extra_lr_v5_phase15_second_start_contrastive_recovery_{int(dataset['actions'].shape[0])}_states.npz"
    )
    metadata = {
        "run_name": "phase15_second_start_contrastive_recovery",
        "model_name": "extra-lr-v5-adaptive-contrastive-recovery-reranker",
        "phase": "phase15_second_start_contrastive_recovery",
        "base_checkpoint": str(config.base_checkpoint),
        "base_metadata": loaded.get("metadata", {}) if loaded else {},
        "v4_model": str(config.v4_model),
        "obs_dim": 6480,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "activation": "only_when_v5_started_false",
        "inference_contract": "centered_additive_logits_bias",
        "label_source": "positive_and_negative_second_start_v5_trajectories",
        "not_standalone_policy": True,
        "config": _jsonable(asdict(config)),
        "dataset": summary,
        "train_summary": train_summary,
        "v4_1_included": False,
    }
    save_checkpoint(str(checkpoint_path), model, optimizer=optimizer, metadata=metadata)
    result = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_summary": summary,
        "train_summary": train_summary,
        "summary": {
            "status": "ok",
            "checkpoint_path": str(checkpoint_path),
            "states": int(dataset["actions"].shape[0]),
            "positive_states": int(summary["positive_states"]),
            "negative_states": int(summary["negative_states"]),
            "final_loss": float(train_summary["final_loss"]),
            "final_positive_accuracy": float(train_summary["final_positive_accuracy"]),
            "final_negative_avoidance": float(train_summary["final_negative_avoidance"]),
        },
    }
    (config.output_dir / "phase15_contrastive_recovery_summary.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def collect_contrastive_recovery_dataset(config: ContrastiveRecoveryConfig) -> dict[str, Any]:
    opponent = OnnxActionPolicy(str(config.v4_model), mode="argmax", seed=config.seed, verify_mask=False)
    v5_policy = V5AdaptivePolicy(config.base_checkpoint, adaptive_strength=1.0)
    assembler = DeckMatchupEvaluator()
    draw_controller = DrawAssistController()
    assembler_candidates = _load_assembler_candidates(config.assembler_dataset)
    info_mode, assist_mode = _profile_modes("strong")

    observations: list[np.ndarray] = []
    action_features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    labels: list[int] = []
    seeds: list[int] = []
    positive_games = 0
    negative_games = 0
    terminal_games = 0
    draw_assist_uses = 0
    specs = _v5_on_policy_game_specs(games=int(config.games), seed=int(config.seed), focus_start_mode="v5_second")

    for game_idx, spec in enumerate(specs, start=1):
        seed = int(spec["seed"])
        v5_player_id = int(spec["v5_player_id"])
        starting_player_id = int(spec["starting_player_id"])
        v4_player_id = 2 if v5_player_id == 1 else 1
        env = TrainV3ClassicEnv(
            TrainV3EnvConfig(
                seed=seed,
                verify_mask=False,
                placement_mode="append_only",
                include_legal_actions_in_info=False,
                info_mode=info_mode,
                assist_mode=assist_mode,
            )
        )
        base_env = env.env
        base_env.reset(seed=seed, starting_player_id=starting_player_id)
        v4_deck_ids = _player_card_pool_ids(base_env._env.state, v4_player_id)
        v5_deck_ids, _assembler_score = _select_v5_deck(
            opponent_deck_ids=v4_deck_ids,
            assembler=assembler,
            candidates=assembler_candidates,
        )
        env.reset(
            p1_deck_ids=v5_deck_ids if v5_player_id == 1 else v4_deck_ids,
            p2_deck_ids=v5_deck_ids if v5_player_id == 2 else v4_deck_ids,
            p1_is_bot=True,
            p2_is_bot=True,
            starting_player_id=starting_player_id,
            seed=seed,
        )
        opponent.reset(seed * 13 + v4_player_id)
        v5_policy.reset(seed * 11 + v5_player_id)
        game_obs: list[np.ndarray] = []
        game_features: list[np.ndarray] = []
        game_masks: list[np.ndarray] = []
        game_actions: list[int] = []

        for _step in range(int(config.max_steps)):
            current = env.current_player_id()
            if current == v5_player_id:
                obs = env.observe(current).astype(np.float32, copy=False)
                mask = env.action_mask(current).astype(np.float32, copy=False)
                features = env.action_features(current, include_preview=False).astype(np.float32, copy=False)
                action_id = int(v5_policy.select_action(env, current))
                if 0 <= action_id < mask.shape[0] and mask[action_id] == 1.0:
                    game_obs.append(obs.copy())
                    game_features.append(features.copy())
                    game_masks.append(mask.copy())
                    game_actions.append(action_id)
            else:
                action_id = int(opponent.select_action(env.env, current))

            if _action_is_end_turn(env.env._env.state, current, action_id):
                next_player_id = 2 if current == 1 else 1
                if next_player_id == v5_player_id:
                    assist_info = _apply_draw_assist_to_player(
                        env=env,
                        player_id=v5_player_id,
                        controller=draw_controller,
                        strength=info_mode.draw_assist_strength,
                    )
                    draw_assist_uses += int(assist_info.get("selected_card_id") is not None)

            _obs, _reward, terminated, truncated, _info = env.step(action_id)
            if terminated or truncated:
                terminal_games += 1
                break

        if not game_actions:
            continue
        state = env.env._env.state
        winner = env.env.winner_id()
        v5_hp = state.p1.hero.hp if v5_player_id == 1 else state.p2.hero.hp
        v4_hp = state.p1.hero.hp if v4_player_id == 1 else state.p2.hero.hp
        label = _phase15_outcome_label(
            winner=winner,
            v5_player_id=v5_player_id,
            v5_hp=int(v5_hp),
            v4_hp=int(v4_hp),
            min_hp_margin=int(config.min_hp_margin),
        )
        positive_games += int(label == 1)
        negative_games += int(label == 0)
        observations.extend(game_obs)
        action_features.extend(game_features)
        masks.extend(game_masks)
        actions.extend(game_actions)
        labels.extend([label] * len(game_actions))
        seeds.extend([seed] * len(game_actions))
        if game_idx % 64 == 0:
            print(
                f"phase15_collect games={game_idx}/{len(specs)} pos_games={positive_games} "
                f"neg_games={negative_games} states={len(actions)}",
                flush=True,
            )

    if not actions:
        raise RuntimeError("contrastive recovery dataset is empty")
    labels_np = np.asarray(labels, dtype=np.int32)
    return {
        "observations": np.stack(observations).astype(np.float32, copy=False),
        "action_features": np.stack(action_features).astype(np.float32, copy=False),
        "masks": np.stack(masks).astype(np.float32, copy=False),
        "actions": np.asarray(actions, dtype=np.int32),
        "labels": labels_np,
        "seeds": np.asarray(seeds, dtype=np.int64),
        "summary": {
            "schema": "extra_lr_v5_phase15_contrastive_recovery_dataset_v1",
            "collection_mode": "v5_second_contrastive_recovery",
            "games": int(config.games),
            "actual_games": int(len(specs)),
            "terminal_games": int(terminal_games),
            "positive_games": int(positive_games),
            "negative_games": int(negative_games),
            "states": int(len(actions)),
            "positive_states": int(np.sum(labels_np == 1)),
            "negative_states": int(np.sum(labels_np == 0)),
            "min_hp_margin": int(config.min_hp_margin),
            "negative_weight": float(config.negative_weight),
            "draw_assist_uses": int(draw_assist_uses),
            "info_mode": asdict(info_mode),
            "assist_mode": assist_mode.to_dict(),
        },
    }


def train_contrastive_action_loss(
    model: Any,
    optimizer: Any,
    *,
    observations: np.ndarray,
    action_features: np.ndarray,
    masks: np.ndarray,
    actions: np.ndarray,
    labels: np.ndarray,
    negative_weight: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn

    n = int(actions.shape[0])
    if n <= 0:
        raise ValueError("actions must contain at least one row")
    rng = np.random.default_rng(int(seed))
    metrics: list[dict[str, float]] = []
    eps = mx.array(1.0e-6, dtype=mx.float32)
    neg_w = float(negative_weight)

    for epoch in range(int(epochs)):
        order = np.arange(n, dtype=np.int64)
        rng.shuffle(order)
        losses: list[float] = []
        pos_accs: list[float] = []
        neg_avoids: list[float] = []
        for start in range(0, n, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            obs_b = mx.array(observations[idx])
            features_b = mx.array(action_features[idx])
            mask_b = mx.array(masks[idx])
            actions_b = mx.array(actions[idx], dtype=mx.int32)
            labels_b = mx.array(labels[idx].astype(np.float32, copy=False))

            def loss_fn(m):
                logits, _values = m(obs_b, features_b)
                masked = mx.where(mask_b.astype(mx.bool_), logits, mx.array(-1.0e9, dtype=mx.float32))
                log_probs = masked - mx.logsumexp(masked, axis=-1, keepdims=True)
                picked_logp = log_probs[mx.arange(actions_b.shape[0]), actions_b]
                picked_p = mx.exp(picked_logp)
                positive_loss = -picked_logp
                negative_loss = -mx.log(mx.maximum(1.0 - picked_p, eps))
                row_weights = labels_b + (1.0 - labels_b) * neg_w
                row_loss = labels_b * positive_loss + (1.0 - labels_b) * neg_w * negative_loss
                loss = mx.sum(row_loss) / mx.maximum(mx.sum(row_weights), eps)
                pred = mx.argmax(masked, axis=-1)
                pos_mask = labels_b > 0.5
                neg_mask = labels_b <= 0.5
                pos_denom = mx.maximum(mx.sum(pos_mask.astype(mx.float32)), eps)
                neg_denom = mx.maximum(mx.sum(neg_mask.astype(mx.float32)), eps)
                pos_acc = mx.sum(((pred == actions_b) & pos_mask).astype(mx.float32)) / pos_denom
                neg_avoid = mx.sum(((pred != actions_b) & neg_mask).astype(mx.float32)) / neg_denom
                return loss, {"positive_accuracy": pos_acc, "negative_avoidance": neg_avoid}

            value_and_grad = nn.value_and_grad(model, loss_fn)
            (loss_value, aux), grads = value_and_grad(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss_value, aux["positive_accuracy"], aux["negative_avoidance"])
            losses.append(float(loss_value.item()))
            pos_accs.append(float(aux["positive_accuracy"].item()))
            neg_avoids.append(float(aux["negative_avoidance"].item()))
        metrics.append(
            {
                "epoch": float(epoch + 1),
                "loss": float(np.mean(losses)),
                "positive_accuracy": float(np.mean(pos_accs)),
                "negative_avoidance": float(np.mean(neg_avoids)),
            }
        )
    return {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "states": int(n),
        "positive_states": int(np.sum(labels == 1)),
        "negative_states": int(np.sum(labels == 0)),
        "negative_weight": float(negative_weight),
        "epoch_metrics": metrics,
        "final_loss": float(metrics[-1]["loss"]),
        "final_positive_accuracy": float(metrics[-1]["positive_accuracy"]),
        "final_negative_avoidance": float(metrics[-1]["negative_avoidance"]),
    }


def _phase15_outcome_label(
    *,
    winner: int | None,
    v5_player_id: int,
    v5_hp: int,
    v4_hp: int,
    min_hp_margin: int,
) -> int:
    if winner == int(v5_player_id):
        return 1
    margin = int(v5_hp) - int(v4_hp)
    if int(v5_hp) > 0 and margin >= int(min_hp_margin):
        return 1
    return 0


def _validate_contrastive_recovery_config(config: ContrastiveRecoveryConfig) -> None:
    if not config.base_checkpoint.exists():
        raise FileNotFoundError(f"base checkpoint not found: {config.base_checkpoint}")
    if not config.v4_model.exists():
        raise FileNotFoundError(f"V4 model not found: {config.v4_model}")
    if config.assembler_dataset is not None and not config.assembler_dataset.exists():
        raise FileNotFoundError(f"assembler dataset not found: {config.assembler_dataset}")
    for name in ("games", "max_steps", "batch_size", "epochs", "hidden_dim", "action_hidden_dim"):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(config.min_positive_games) <= 0:
        raise ValueError("min_positive_games must be positive")
    if int(config.min_negative_games) <= 0:
        raise ValueError("min_negative_games must be positive")
    if float(config.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if float(config.negative_weight) <= 0.0:
        raise ValueError("negative_weight must be positive")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Train second-start contrastive V5 recovery reranker")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--v4-model", type=Path, default=DEFAULT_V4_MAX)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "TrainV3.5" / "runs" / f"phase15_second_start_contrastive_recovery_{stamp}",
    )
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=21500000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--assembler-dataset", type=Path, default=DEFAULT_ASSEMBLER_DATASET)
    parser.add_argument("--min-hp-margin", type=int, default=6)
    parser.add_argument("--min-positive-games", type=int, default=4)
    parser.add_argument("--min-negative-games", type=int, default=4)
    parser.add_argument("--negative-weight", type=float, default=0.35)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--action-hidden-dim", type=int, default=128)
    parser.add_argument("--no-initialize-from-base", action="store_true")
    parser.add_argument("--no-save-dataset", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = ContrastiveRecoveryConfig(
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
        min_hp_margin=int(args.min_hp_margin),
        min_positive_games=int(args.min_positive_games),
        min_negative_games=int(args.min_negative_games),
        negative_weight=float(args.negative_weight),
        hidden_dim=int(args.hidden_dim),
        action_hidden_dim=int(args.action_hidden_dim),
        initialize_from_base=not bool(args.no_initialize_from_base),
        save_dataset=not bool(args.no_save_dataset),
    )
    result = run_contrastive_recovery_training(config)
    print("PHASE15_CONTRASTIVE_RECOVERY_RESULT", json.dumps(result["summary"], sort_keys=True), flush=True)
    print(f"Saved: {result['checkpoint_path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

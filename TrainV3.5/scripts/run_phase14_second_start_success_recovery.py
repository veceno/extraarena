#!/usr/bin/env python3
"""Train a second-start recovery reranker from successful V5 trajectories.

Phase13 showed that shallow rollout-search labels can be noisy: small recovery
weights did nothing and large weights harmed second-start play. Phase14 keeps
the base V5 checkpoint frozen and trains a separate additive recovery scorer
only from second-start games that V5 actually wins or finishes with a strong HP
margin. The recovery scorer is still a separate inference layer, not a base
policy update.
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
from ai.train_v2.onnx_policy import OnnxActionPolicy  # noqa: E402
from run_phase10_v4max_distill import (  # noqa: E402
    DEFAULT_ASSEMBLER_DATASET,
    DEFAULT_V4_MAX,
    _jsonable,
    _profile_modes,
    _v5_on_policy_game_specs,
    train_teacher_cross_entropy,
)
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


DEFAULT_BASE_CHECKPOINT = (
    ROOT
    / "TrainV3.5"
    / "runs"
    / "phase10_v4max_distill_round2_from_15020_20260609_1324"
    / "extra_lr_v5_phase10_v4max_distill_61571_states.npz"
)


@dataclass(frozen=True)
class SuccessRecoveryConfig:
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
    min_kept_games: int = 4
    hidden_dim: int = 256
    action_hidden_dim: int = 128
    initialize_from_base: bool = True
    save_dataset: bool = True


def run_success_recovery_training(config: SuccessRecoveryConfig) -> dict[str, Any]:
    _validate_success_recovery_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = collect_success_recovery_dataset(config)
    if int(dataset["summary"]["kept_games"]) < int(config.min_kept_games):
        raise RuntimeError(
            f"kept_games={dataset['summary']['kept_games']} below min_kept_games={config.min_kept_games}"
        )
    if config.save_dataset:
        np.savez_compressed(
            config.output_dir / "second_start_success_recovery_dataset.npz",
            observations=dataset["observations"],
            action_features=dataset["action_features"],
            masks=dataset["masks"],
            actions=dataset["actions"],
            seeds=dataset["seeds"],
            v5_started=dataset["v5_started"],
            game_outcomes=dataset["game_outcomes"],
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
    train_summary = train_teacher_cross_entropy(
        model,
        optimizer,
        observations=dataset["observations"],
        action_features=dataset["action_features"],
        masks=dataset["masks"],
        actions=dataset["actions"],
        epochs=config.epochs,
        batch_size=config.batch_size,
        seed=config.seed + 141,
    )

    checkpoint_path = config.output_dir / (
        f"extra_lr_v5_phase14_second_start_success_recovery_{int(dataset['actions'].shape[0])}_states.npz"
    )
    metadata = {
        "run_name": "phase14_second_start_success_recovery",
        "model_name": "extra-lr-v5-adaptive-success-recovery-reranker",
        "phase": "phase14_second_start_success_recovery",
        "base_checkpoint": str(config.base_checkpoint),
        "base_metadata": loaded.get("metadata", {}) if loaded else {},
        "v4_model": str(config.v4_model),
        "obs_dim": 6480,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "activation": "only_when_v5_started_false",
        "inference_contract": "centered_additive_logits_bias",
        "label_source": "successful_or_strong_hp_margin_second_start_v5_trajectories",
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
            "kept_games": int(dataset["summary"]["kept_games"]),
            "win_kept_games": int(dataset["summary"]["win_kept_games"]),
            "hp_margin_kept_games": int(dataset["summary"]["hp_margin_kept_games"]),
            "final_loss": float(train_summary["final_loss"]),
            "final_accuracy": float(train_summary["final_accuracy"]),
        },
    }
    (config.output_dir / "phase14_success_recovery_summary.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def collect_success_recovery_dataset(config: SuccessRecoveryConfig) -> dict[str, Any]:
    opponent = OnnxActionPolicy(str(config.v4_model), mode="argmax", seed=config.seed, verify_mask=False)
    v5_policy = V5AdaptivePolicy(config.base_checkpoint, adaptive_strength=1.0)
    assembler = DeckMatchupEvaluator()
    draw_controller = DrawAssistController()
    assembler_candidates = _load_assembler_candidates(config.assembler_dataset)
    info_mode, assist_mode = _profile_modes("strong")

    kept_obs: list[np.ndarray] = []
    kept_features: list[np.ndarray] = []
    kept_masks: list[np.ndarray] = []
    kept_actions: list[int] = []
    kept_seeds: list[int] = []
    kept_started: list[bool] = []
    kept_outcomes: list[int] = []
    actual_games = 0
    terminal_games = 0
    kept_games = 0
    win_kept_games = 0
    hp_margin_kept_games = 0
    total_v5_states = 0
    draw_assist_uses = 0

    specs = _v5_on_policy_game_specs(
        games=int(config.games),
        seed=int(config.seed),
        focus_start_mode="v5_second",
    )
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
        actual_games += 1
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
                    total_v5_states += 1
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

        state = env.env._env.state
        winner = env.env.winner_id()
        v5_hp = state.p1.hero.hp if v5_player_id == 1 else state.p2.hero.hp
        v4_hp = state.p1.hero.hp if v4_player_id == 1 else state.p2.hero.hp
        keep = _keep_phase14_recovery_game(
            winner=winner,
            v5_player_id=v5_player_id,
            v5_hp=int(v5_hp),
            v4_hp=int(v4_hp),
            min_hp_margin=int(config.min_hp_margin),
        )
        if keep and game_actions:
            kept_games += 1
            win_kept_games += int(winner == v5_player_id)
            hp_margin_kept_games += int(winner != v5_player_id)
            kept_obs.extend(game_obs)
            kept_features.extend(game_features)
            kept_masks.extend(game_masks)
            kept_actions.extend(game_actions)
            kept_seeds.extend([seed] * len(game_actions))
            kept_started.extend([False] * len(game_actions))
            kept_outcomes.extend([1 if winner == v5_player_id else 0] * len(game_actions))
        if game_idx % 64 == 0:
            print(
                f"phase14_collect games={game_idx}/{len(specs)} kept_games={kept_games} states={len(kept_actions)}",
                flush=True,
            )

    if not kept_actions:
        raise RuntimeError("success recovery dataset is empty")
    return {
        "observations": np.stack(kept_obs).astype(np.float32, copy=False),
        "action_features": np.stack(kept_features).astype(np.float32, copy=False),
        "masks": np.stack(kept_masks).astype(np.float32, copy=False),
        "actions": np.asarray(kept_actions, dtype=np.int32),
        "seeds": np.asarray(kept_seeds, dtype=np.int64),
        "v5_started": np.asarray(kept_started, dtype=np.bool_),
        "game_outcomes": np.asarray(kept_outcomes, dtype=np.int32),
        "summary": {
            "schema": "extra_lr_v5_phase14_success_recovery_dataset_v1",
            "collection_mode": "v5_second_success_recovery",
            "games": int(config.games),
            "actual_games": int(actual_games),
            "terminal_games": int(terminal_games),
            "kept_games": int(kept_games),
            "win_kept_games": int(win_kept_games),
            "hp_margin_kept_games": int(hp_margin_kept_games),
            "states": int(len(kept_actions)),
            "total_v5_states": int(total_v5_states),
            "v5_second_states": int(len(kept_actions)),
            "min_hp_margin": int(config.min_hp_margin),
            "draw_assist_uses": int(draw_assist_uses),
            "info_mode": asdict(info_mode),
            "assist_mode": assist_mode.to_dict(),
        },
    }


def _keep_phase14_recovery_game(
    *,
    winner: int | None,
    v5_player_id: int,
    v5_hp: int,
    v4_hp: int,
    min_hp_margin: int,
) -> bool:
    if winner == int(v5_player_id):
        return True
    margin = int(v5_hp) - int(v4_hp)
    return int(v5_hp) > 0 and margin >= int(min_hp_margin)


def _validate_success_recovery_config(config: SuccessRecoveryConfig) -> None:
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
    if int(config.min_kept_games) <= 0:
        raise ValueError("min_kept_games must be positive")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Train second-start recovery from successful V5 trajectories")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--v4-model", type=Path, default=DEFAULT_V4_MAX)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "TrainV3.5" / "runs" / f"phase14_second_start_success_recovery_{stamp}",
    )
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=21400000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8.0e-5)
    parser.add_argument("--assembler-dataset", type=Path, default=DEFAULT_ASSEMBLER_DATASET)
    parser.add_argument("--min-hp-margin", type=int, default=6)
    parser.add_argument("--min-kept-games", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--action-hidden-dim", type=int, default=128)
    parser.add_argument("--no-initialize-from-base", action="store_true")
    parser.add_argument("--no-save-dataset", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = SuccessRecoveryConfig(
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
        min_kept_games=int(args.min_kept_games),
        hidden_dim=int(args.hidden_dim),
        action_hidden_dim=int(args.action_hidden_dim),
        initialize_from_base=not bool(args.no_initialize_from_base),
        save_dataset=not bool(args.no_save_dataset),
    )
    result = run_success_recovery_training(config)
    print("PHASE14_SUCCESS_RECOVERY_RESULT", json.dumps(result["summary"], sort_keys=True), flush=True)
    print(f"Saved: {result['checkpoint_path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

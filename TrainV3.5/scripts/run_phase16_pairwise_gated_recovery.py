#!/usr/bin/env python3
"""Train a pairwise gated second-start recovery reranker for V5.

The base V5 checkpoint is frozen and used only to generate on-policy games,
top-K candidate actions, and baseline actions. A separate V5-shaped scorer is
trained on hard second-start recovery states where a rollout-evaluated candidate
beats the base action by a configured margin.
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

from ai.train_v2.model_mlx import load_checkpoint, save_checkpoint  # noqa: E402
from ai.train_v2.onnx_policy import OnnxActionPolicy  # noqa: E402
from run_phase10_v4max_distill import (  # noqa: E402
    DEFAULT_ASSEMBLER_DATASET,
    DEFAULT_V4_MAX,
    _evaluate_rollout_candidate,
    _jsonable,
    _profile_modes,
    _rank_v5_policy_actions,
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
class PairwiseGatedRecoveryConfig:
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
    min_pairwise_margin: float = 0.25
    ranking_margin: float = 1.0
    gate_max_hp_margin: int = -4
    gate_max_own_hp: int = 18
    gate_max_board_margin: int = -6
    min_pairs: int = 16
    hidden_dim: int = 256
    action_hidden_dim: int = 128
    initialize_from_base: bool = True
    save_dataset: bool = True


PairwiseRecoveryConfig = PairwiseGatedRecoveryConfig


def run_pairwise_gated_recovery_training(config: PairwiseGatedRecoveryConfig) -> dict[str, Any]:
    _validate_pairwise_gated_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = collect_pairwise_gated_recovery_dataset(config)
    summary = dataset["summary"]
    if int(summary["pairs"]) < int(config.min_pairs):
        raise RuntimeError(f"pairs={summary['pairs']} below min_pairs={config.min_pairs}")
    if config.save_dataset:
        np.savez_compressed(
            config.output_dir / "second_start_pairwise_gated_recovery_dataset.npz",
            observations=dataset["observations"],
            action_features=dataset["action_features"],
            masks=dataset["masks"],
            positive_actions=dataset["positive_actions"],
            negative_actions=dataset["negative_actions"],
            candidate_scores=dataset["candidate_scores"],
            base_scores=dataset["base_scores"],
            score_margins=dataset["score_margins"],
            seeds=dataset["seeds"],
            v5_started=dataset["v5_started"],
            gate_reasons=dataset["gate_reasons"],
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
    train_summary = train_pairwise_margin_loss(
        model,
        optimizer,
        observations=dataset["observations"],
        action_features=dataset["action_features"],
        masks=dataset["masks"],
        positive_actions=dataset["positive_actions"],
        negative_actions=dataset["negative_actions"],
        ranking_margin=config.ranking_margin,
        epochs=config.epochs,
        batch_size=config.batch_size,
        seed=config.seed + 161,
    )

    checkpoint_path = config.output_dir / (
        f"extra_lr_v5_phase16_pairwise_gated_recovery_{int(summary['pairs'])}_pairs.npz"
    )
    metadata = {
        "run_name": "phase16_pairwise_gated_recovery",
        "model_name": "extra-lr-v5-adaptive-pairwise-gated-recovery-reranker",
        "phase": "phase16_pairwise_gated_recovery",
        "base_checkpoint": str(config.base_checkpoint),
        "base_metadata": loaded.get("metadata", {}) if loaded else {},
        "v4_model": str(config.v4_model),
        "obs_dim": 6480,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "activation": "only_when_v5_started_false_and_phase16_recovery_gate_true",
        "inference_contract": "centered_additive_logits_bias",
        "label_source": "top_k_base_policy_candidates_rollout_vs_base_pairwise_margin",
        "loss": "pairwise_margin_ranking",
        "not_standalone_policy": True,
        "base_policy_frozen": True,
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
            "pairs": int(summary["pairs"]),
            "gated_states": int(summary["gated_states"]),
            "accepted_pairs": int(summary["accepted_pairs"]),
            "avg_score_margin": float(summary["avg_score_margin"]),
            "final_loss": float(train_summary["final_loss"]),
            "final_pairwise_accuracy": float(train_summary["final_pairwise_accuracy"]),
        },
    }
    (config.output_dir / "phase16_pairwise_gated_recovery_summary.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def collect_pairwise_gated_recovery_dataset(config: PairwiseGatedRecoveryConfig) -> dict[str, Any]:
    opponent = OnnxActionPolicy(str(config.v4_model), mode="argmax", seed=config.seed, verify_mask=False)
    v5_policy = V5AdaptivePolicy(config.base_checkpoint, adaptive_strength=1.0)
    assembler = DeckMatchupEvaluator()
    draw_controller = DrawAssistController()
    assembler_candidates = _load_assembler_candidates(config.assembler_dataset)
    info_mode, assist_mode = _profile_modes("strong")

    observations: list[np.ndarray] = []
    action_features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    positive_actions: list[int] = []
    negative_actions: list[int] = []
    candidate_scores: list[float] = []
    base_scores: list[float] = []
    score_margins: list[float] = []
    seeds: list[int] = []
    v5_started_flags: list[bool] = []
    gate_reasons: list[int] = []

    actual_games = 0
    terminal_games = 0
    total_steps = 0
    v5_states = 0
    gated_states = 0
    searched_states = 0
    accepted_pairs = 0
    draw_assist_uses = 0
    candidate_evals = 0

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
        actual_games += 1

        for _step in range(int(config.max_steps)):
            current = env.current_player_id()
            if current == v5_player_id:
                obs = env.observe(current).astype(np.float32, copy=False)
                mask = env.action_mask(current).astype(np.float32, copy=False)
                features = env.action_features(current, include_preview=False).astype(np.float32, copy=False)
                base_action = int(v5_policy.select_action(env, current))
                v5_states += 1
                gate_reason = _phase16_recovery_gate_from_env(
                    env=env,
                    v5_player_id=v5_player_id,
                    v5_started=bool(spec["v5_started"]),
                    max_hp_margin=config.gate_max_hp_margin,
                    max_own_hp=config.gate_max_own_hp,
                    max_board_margin=config.gate_max_board_margin,
                )
                if gate_reason:
                    gated_states += 1
                    candidates = _rank_v5_policy_actions(
                        v5_policy=v5_policy,
                        obs=obs,
                        action_features=features,
                        mask=mask,
                        max_candidates=config.search_candidates,
                    )
                    if base_action not in candidates:
                        candidates.append(base_action)
                    scored: list[tuple[int, float]] = []
                    for action_id in candidates:
                        if 0 <= int(action_id) < mask.shape[0] and mask[int(action_id)] == 1.0:
                            scored.append(
                                (
                                    int(action_id),
                                    _evaluate_rollout_candidate(
                                        env=env,
                                        candidate_action=int(action_id),
                                        v5_player_id=v5_player_id,
                                        v5_policy=v5_policy,
                                        opponent_policy=opponent,
                                        draw_controller=draw_controller,
                                        draw_assist_strength=info_mode.draw_assist_strength,
                                        depth_plies=config.search_depth_plies,
                                    ),
                                )
                            )
                    candidate_evals += len(scored)
                    if scored:
                        searched_states += 1
                        base_score = next(score for action_id, score in scored if action_id == base_action)
                        for candidate_action, candidate_score in scored:
                            if int(candidate_action) == int(base_action):
                                continue
                            margin = float(candidate_score - base_score)
                            if _pairwise_margin_label(
                                candidate_score=candidate_score,
                                baseline_score=base_score,
                                min_pairwise_margin=config.min_pairwise_margin,
                            ):
                                observations.append(obs.copy())
                                action_features.append(features.copy())
                                masks.append(mask.copy())
                                positive_actions.append(int(candidate_action))
                                negative_actions.append(int(base_action))
                                candidate_scores.append(float(candidate_score))
                                base_scores.append(float(base_score))
                                score_margins.append(margin)
                                seeds.append(seed)
                                v5_started_flags.append(False)
                                gate_reasons.append(int(gate_reason))
                                accepted_pairs += 1
                action_id = base_action
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
            total_steps += 1
            if terminated or truncated:
                terminal_games += 1
                break

        if game_idx % 64 == 0:
            print(
                f"phase16_collect games={game_idx}/{len(specs)} gated={gated_states} "
                f"pairs={accepted_pairs}",
                flush=True,
            )

    if not positive_actions:
        raise RuntimeError("pairwise gated recovery dataset is empty")
    margins_np = np.asarray(score_margins, dtype=np.float32)
    return {
        "observations": np.stack(observations).astype(np.float32, copy=False),
        "action_features": np.stack(action_features).astype(np.float32, copy=False),
        "masks": np.stack(masks).astype(np.float32, copy=False),
        "positive_actions": np.asarray(positive_actions, dtype=np.int32),
        "negative_actions": np.asarray(negative_actions, dtype=np.int32),
        "candidate_scores": np.asarray(candidate_scores, dtype=np.float32),
        "base_scores": np.asarray(base_scores, dtype=np.float32),
        "score_margins": margins_np,
        "seeds": np.asarray(seeds, dtype=np.int64),
        "v5_started": np.asarray(v5_started_flags, dtype=np.bool_),
        "gate_reasons": np.asarray(gate_reasons, dtype=np.int32),
        "summary": {
            "schema": "extra_lr_v5_phase16_pairwise_gated_recovery_dataset_v1",
            "collection_mode": "v5_second_pairwise_gated_recovery",
            "games": int(config.games),
            "actual_games": int(actual_games),
            "terminal_games": int(terminal_games),
            "pairs": int(len(positive_actions)),
            "accepted_pairs": int(accepted_pairs),
            "total_steps": int(total_steps),
            "v5_states": int(v5_states),
            "gated_states": int(gated_states),
            "searched_states": int(searched_states),
            "candidate_evals": int(candidate_evals),
            "avg_score_margin": float(np.mean(margins_np)),
            "min_score_margin": float(np.min(margins_np)),
            "max_score_margin": float(np.max(margins_np)),
            "draw_assist_uses": int(draw_assist_uses),
            "search_candidates": int(config.search_candidates),
            "search_depth_plies": int(config.search_depth_plies),
            "min_pairwise_margin": float(config.min_pairwise_margin),
            "gate_max_hp_margin": int(config.gate_max_hp_margin),
            "gate_max_own_hp": int(config.gate_max_own_hp),
            "gate_max_board_margin": int(config.gate_max_board_margin),
            "v5_started_states": 0,
            "v5_second_states": int(len(positive_actions)),
            "gate_reason_counts": _count_gate_reasons(gate_reasons),
            "profile": "strong",
            "info_mode": asdict(info_mode),
            "assist_mode": assist_mode.to_dict(),
        },
    }


def train_pairwise_margin_loss(
    model: Any,
    optimizer: Any,
    *,
    observations: np.ndarray,
    action_features: np.ndarray,
    masks: np.ndarray,
    positive_actions: np.ndarray,
    negative_actions: np.ndarray,
    ranking_margin: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn

    n = int(positive_actions.shape[0])
    if n <= 0:
        raise ValueError("positive_actions must contain at least one row")
    if negative_actions.shape[0] != n:
        raise ValueError("positive_actions and negative_actions must have the same length")
    rng = np.random.default_rng(int(seed))
    metrics: list[dict[str, float]] = []
    margin_value = float(ranking_margin)

    for epoch in range(int(epochs)):
        order = np.arange(n, dtype=np.int64)
        rng.shuffle(order)
        losses: list[float] = []
        accuracies: list[float] = []
        gaps: list[float] = []
        for start in range(0, n, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            obs_b = mx.array(observations[idx])
            features_b = mx.array(action_features[idx])
            mask_b = mx.array(masks[idx])
            pos_b = mx.array(positive_actions[idx], dtype=mx.int32)
            neg_b = mx.array(negative_actions[idx], dtype=mx.int32)

            def loss_fn(m):
                logits, _values = m(obs_b, features_b)
                masked = mx.where(mask_b.astype(mx.bool_), logits, mx.array(-1.0e9, dtype=mx.float32))
                row = mx.arange(pos_b.shape[0])
                pos_logits = masked[row, pos_b]
                neg_logits = masked[row, neg_b]
                diff = pos_logits - neg_logits
                loss = mx.mean(mx.maximum(mx.array(0.0, dtype=mx.float32), mx.array(margin_value) - diff))
                accuracy = mx.mean((diff > 0.0).astype(mx.float32))
                avg_gap = mx.mean(diff)
                return loss, {"pairwise_accuracy": accuracy, "avg_logit_gap": avg_gap}

            value_and_grad = nn.value_and_grad(model, loss_fn)
            (loss_value, aux), grads = value_and_grad(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss_value, aux["pairwise_accuracy"], aux["avg_logit_gap"])
            losses.append(float(loss_value.item()))
            accuracies.append(float(aux["pairwise_accuracy"].item()))
            gaps.append(float(aux["avg_logit_gap"].item()))
        metrics.append(
            {
                "epoch": float(epoch + 1),
                "loss": float(np.mean(losses)),
                "pairwise_accuracy": float(np.mean(accuracies)),
                "avg_logit_gap": float(np.mean(gaps)),
            }
        )
    return {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "pairs": int(n),
        "ranking_margin": float(ranking_margin),
        "epoch_metrics": metrics,
        "final_loss": float(metrics[-1]["loss"]),
        "final_pairwise_accuracy": float(metrics[-1]["pairwise_accuracy"]),
        "final_avg_logit_gap": float(metrics[-1]["avg_logit_gap"]),
    }


def _phase16_recovery_gate(
    *,
    v5_hp: int,
    opponent_hp: int,
    v5_board_power: int,
    opponent_board_power: int,
    v5_board_count: int,
    opponent_board_count: int,
    max_hp_margin: int = -4,
    max_own_hp: int = 12,
    max_board_margin: int = -4,
) -> bool:
    hp_margin = int(v5_hp) - int(opponent_hp)
    board_margin = int(v5_board_power) - int(opponent_board_power)
    return (
        hp_margin <= int(max_hp_margin)
        or int(v5_hp) <= int(max_own_hp)
        or board_margin <= int(max_board_margin)
        or (int(v5_board_count) == 0 and int(opponent_board_count) > 0)
    )


def _phase16_recovery_gate_from_env(
    *,
    env: TrainV3ClassicEnv,
    v5_player_id: int,
    v5_started: bool,
    max_hp_margin: int,
    max_own_hp: int,
    max_board_margin: int,
) -> int:
    if bool(v5_started):
        return 0
    state = env.env._env.state
    v4_player_id = 2 if int(v5_player_id) == 1 else 1
    v5 = state.p1 if state.p1.user_id == int(v5_player_id) else state.p2
    v4 = state.p1 if state.p1.user_id == v4_player_id else state.p2
    hp_margin = int(v5.hero.hp) - int(v4.hero.hp)
    own_board = sum(max(0, int(card.attack)) * max(0, int(card.hp)) for card in v5.board)
    opp_board = sum(max(0, int(card.attack)) * max(0, int(card.hp)) for card in v4.board)
    board_margin = int(own_board - opp_board)
    reason = 0
    if hp_margin <= int(max_hp_margin):
        reason |= 1
    if int(v5.hero.hp) <= int(max_own_hp):
        reason |= 2
    if board_margin <= int(max_board_margin):
        reason |= 4
    if len(v5.board) == 0 and len(v4.board) > 0:
        reason |= 8
    return reason


def _pairwise_margin_label(*, candidate_score: float, baseline_score: float, min_pairwise_margin: float) -> bool:
    return float(candidate_score) >= float(baseline_score) + float(min_pairwise_margin)


def _count_gate_reasons(reasons: list[int]) -> dict[str, int]:
    return {
        "hp_margin": int(sum(1 for reason in reasons if int(reason) & 1)),
        "own_hp": int(sum(1 for reason in reasons if int(reason) & 2)),
        "board_margin": int(sum(1 for reason in reasons if int(reason) & 4)),
        "empty_board_pressure": int(sum(1 for reason in reasons if int(reason) & 8)),
    }


def _validate_pairwise_gated_config(config: PairwiseGatedRecoveryConfig) -> None:
    if not config.base_checkpoint.exists():
        raise FileNotFoundError(f"base checkpoint not found: {config.base_checkpoint}")
    if not config.v4_model.exists():
        raise FileNotFoundError(f"V4 model not found: {config.v4_model}")
    if config.assembler_dataset is not None and not config.assembler_dataset.exists():
        raise FileNotFoundError(f"assembler dataset not found: {config.assembler_dataset}")
    for name in (
        "games",
        "max_steps",
        "batch_size",
        "epochs",
        "search_candidates",
        "hidden_dim",
        "action_hidden_dim",
        "min_pairs",
    ):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(config.search_depth_plies) < 0:
        raise ValueError("search_depth_plies must be non-negative")
    if float(config.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if float(config.min_pairwise_margin) <= 0.0:
        raise ValueError("min_pairwise_margin must be positive")
    if float(config.ranking_margin) <= 0.0:
        raise ValueError("ranking_margin must be positive")


def _validate_pairwise_recovery_config(config: PairwiseRecoveryConfig) -> None:
    _validate_pairwise_gated_config(config)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Train second-start pairwise gated V5 recovery reranker")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--v4-model", type=Path, default=DEFAULT_V4_MAX)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "TrainV3" / "runs" / f"phase16_pairwise_gated_recovery_{stamp}",
    )
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=21600000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--assembler-dataset", type=Path, default=DEFAULT_ASSEMBLER_DATASET)
    parser.add_argument("--search-candidates", type=int, default=8)
    parser.add_argument("--search-depth-plies", type=int, default=6)
    parser.add_argument("--min-pairwise-margin", type=float, default=0.25)
    parser.add_argument("--ranking-margin", type=float, default=1.0)
    parser.add_argument("--gate-max-hp-margin", type=int, default=-4)
    parser.add_argument("--gate-max-own-hp", type=int, default=18)
    parser.add_argument("--gate-max-board-margin", type=int, default=-6)
    parser.add_argument("--min-pairs", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--action-hidden-dim", type=int, default=128)
    parser.add_argument("--no-initialize-from-base", action="store_true")
    parser.add_argument("--no-save-dataset", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = PairwiseGatedRecoveryConfig(
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
        min_pairwise_margin=float(args.min_pairwise_margin),
        ranking_margin=float(args.ranking_margin),
        gate_max_hp_margin=int(args.gate_max_hp_margin),
        gate_max_own_hp=int(args.gate_max_own_hp),
        gate_max_board_margin=int(args.gate_max_board_margin),
        min_pairs=int(args.min_pairs),
        hidden_dim=int(args.hidden_dim),
        action_hidden_dim=int(args.action_hidden_dim),
        initialize_from_base=not bool(args.no_initialize_from_base),
        save_dataset=not bool(args.no_save_dataset),
    )
    result = run_pairwise_gated_recovery_training(config)
    print("PHASE16_PAIRWISE_GATED_RECOVERY_RESULT", json.dumps(result["summary"], sort_keys=True), flush=True)
    print(f"Saved: {result['checkpoint_path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

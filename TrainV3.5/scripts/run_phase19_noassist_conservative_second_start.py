#!/usr/bin/env python3
"""No-assist conservative second-start hardening for V5.

Phase18 showed that direct CE imitation of rollout-search labels can destroy
real play. Phase19 keeps this update conservative: it only applies pairwise
improvement pressure on V5-second no-assist states, while KL anchoring keeps the
policy close to the source checkpoint on both recovery states and V5-first
anchor states.
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
for path in (ROOT, TRAINV3_PYTHON, TRAINV3_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ai.train_v2.model_mlx import load_checkpoint, save_checkpoint  # noqa: E402
from ai.train_v2.onnx_policy import OnnxActionPolicy  # noqa: E402
from run_phase10_v4max_distill import (  # noqa: E402
    DEFAULT_V4_MAX,
    _evaluate_rollout_candidate,
    _jsonable,
    _profile_modes,
    _rank_v5_policy_actions,
    _v5_on_policy_game_specs,
)
from run_phase16_pairwise_gated_recovery import (  # noqa: E402
    _count_gate_reasons,
    _pairwise_margin_label,
    _phase16_recovery_gate_from_env,
)
from run_v5_vs_v4max_benchmark import (  # noqa: E402
    NOASSIST_BASELINE_DECK_IDS,
    V5AdaptivePolicy,
    _player_card_pool_ids,
    _parse_deck_ids,
)
from train_v3.aux_models import DrawAssistController  # noqa: E402
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
class Phase19Config:
    base_checkpoint: Path
    v4_model: Path
    output_dir: Path
    games: int
    anchor_games: int
    max_steps: int
    seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    noassist_deck_ids: tuple[int, ...] = NOASSIST_BASELINE_DECK_IDS
    search_candidates: int = 8
    search_depth_plies: int = 8
    min_pairwise_margin: float = 0.25
    ranking_margin: float = 0.5
    pairwise_coef: float = 1.0
    kl_coef: float = 1.5
    anchor_kl_coef: float = 2.0
    gate_max_hp_margin: int = -2
    gate_max_own_hp: int = 20
    gate_max_board_margin: int = -4
    min_pairs: int = 16
    hidden_dim: int = 256
    action_hidden_dim: int = 128
    save_dataset: bool = True


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = Phase19Config(
        base_checkpoint=args.base_checkpoint.resolve(),
        v4_model=args.v4_model.resolve(),
        output_dir=args.output_dir.resolve(),
        games=int(args.games),
        anchor_games=int(args.anchor_games),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        noassist_deck_ids=_parse_deck_ids(args.noassist_deck_ids),
        search_candidates=int(args.search_candidates),
        search_depth_plies=int(args.search_depth_plies),
        min_pairwise_margin=float(args.min_pairwise_margin),
        ranking_margin=float(args.ranking_margin),
        pairwise_coef=float(args.pairwise_coef),
        kl_coef=float(args.kl_coef),
        anchor_kl_coef=float(args.anchor_kl_coef),
        gate_max_hp_margin=int(args.gate_max_hp_margin),
        gate_max_own_hp=int(args.gate_max_own_hp),
        gate_max_board_margin=int(args.gate_max_board_margin),
        min_pairs=int(args.min_pairs),
        hidden_dim=int(args.hidden_dim),
        action_hidden_dim=int(args.action_hidden_dim),
        save_dataset=not bool(args.no_save_dataset),
    )
    result = run_phase19(config)
    print("PHASE19_RESULT", json.dumps(result["summary"], sort_keys=True), flush=True)
    print(f"Saved: {result['checkpoint_path']}", flush=True)
    return 0


def run_phase19(config: Phase19Config) -> dict[str, Any]:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = collect_phase19_dataset(config)
    if int(dataset["summary"]["pairs"]) < int(config.min_pairs):
        raise RuntimeError(f"pairs={dataset['summary']['pairs']} below min_pairs={config.min_pairs}")
    if config.save_dataset:
        np.savez_compressed(
            config.output_dir / "phase19_noassist_second_start_dataset.npz",
            observations=dataset["observations"],
            action_features=dataset["action_features"],
            masks=dataset["masks"],
            positive_actions=dataset["positive_actions"],
            negative_actions=dataset["negative_actions"],
            score_margins=dataset["score_margins"],
            anchor_observations=dataset["anchor_observations"],
            anchor_action_features=dataset["anchor_action_features"],
            anchor_masks=dataset["anchor_masks"],
            seeds=dataset["seeds"],
            gate_reasons=dataset["gate_reasons"],
        )

    import mlx.optimizers as optim

    model = create_v5_policy(
        policy_kind="v5_split_encoder",
        hidden_dim=config.hidden_dim,
        action_hidden_dim=config.action_hidden_dim,
    )
    reference_model = create_v5_policy(
        policy_kind="v5_split_encoder",
        hidden_dim=config.hidden_dim,
        action_hidden_dim=config.action_hidden_dim,
    )
    optimizer = optim.Adam(learning_rate=config.learning_rate)
    loaded = load_checkpoint(str(config.base_checkpoint), model, optimizer=None)
    load_checkpoint(str(config.base_checkpoint), reference_model, optimizer=None)
    train_summary = train_conservative_pairwise_kl(
        model,
        reference_model,
        optimizer,
        observations=dataset["observations"],
        action_features=dataset["action_features"],
        masks=dataset["masks"],
        positive_actions=dataset["positive_actions"],
        negative_actions=dataset["negative_actions"],
        anchor_observations=dataset["anchor_observations"],
        anchor_action_features=dataset["anchor_action_features"],
        anchor_masks=dataset["anchor_masks"],
        ranking_margin=config.ranking_margin,
        pairwise_coef=config.pairwise_coef,
        kl_coef=config.kl_coef,
        anchor_kl_coef=config.anchor_kl_coef,
        epochs=config.epochs,
        batch_size=config.batch_size,
        seed=config.seed + 191,
    )

    checkpoint_path = config.output_dir / f"extra_lr_v5_phase19_noassist_conservative_{int(dataset['summary']['pairs'])}_pairs.npz"
    metadata = {
        "run_name": "phase19_noassist_conservative_second_start",
        "model_name": "extra-lr-v5-adaptive",
        "phase": "phase19_noassist_conservative_second_start",
        "source_checkpoint": str(config.base_checkpoint),
        "source_metadata": loaded.get("metadata", {}),
        "v4_model": str(config.v4_model),
        "obs_dim": 6480,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "config": _jsonable(asdict(config)),
        "dataset": dataset["summary"],
        "train_summary": train_summary,
        "profile": "noassist",
        "base_policy_updated": True,
        "assist_or_submodel_used": False,
        "v4_1_included": False,
        "online_v4max_rollout": True,
        "loss": "pairwise_margin_plus_reference_kl",
    }
    save_checkpoint(str(checkpoint_path), model, optimizer=optimizer, metadata=metadata)
    result = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_summary": dataset["summary"],
        "train_summary": train_summary,
        "summary": {
            "status": "ok",
            "checkpoint_path": str(checkpoint_path),
            "pairs": int(dataset["summary"]["pairs"]),
            "anchor_states": int(dataset["summary"]["anchor_states"]),
            "avg_score_margin": float(dataset["summary"]["avg_score_margin"]),
            "final_loss": float(train_summary["final_loss"]),
            "final_pairwise_accuracy": float(train_summary["final_pairwise_accuracy"]),
            "final_policy_kl": float(train_summary["final_policy_kl"]),
            "final_anchor_kl": float(train_summary["final_anchor_kl"]),
        },
    }
    (config.output_dir / "phase19_summary.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def collect_phase19_dataset(config: Phase19Config) -> dict[str, Any]:
    pairwise = collect_noassist_pairwise_second_start_dataset(config)
    anchors = collect_noassist_anchor_dataset(config)
    return {**pairwise, **anchors, "summary": {**pairwise["summary"], **anchors["summary"]}}


def collect_noassist_pairwise_second_start_dataset(config: Phase19Config) -> dict[str, Any]:
    opponent = OnnxActionPolicy(str(config.v4_model), mode="argmax", seed=config.seed, verify_mask=False)
    v5_policy = V5AdaptivePolicy(config.base_checkpoint, adaptive_strength=1.0)
    draw_controller = DrawAssistController()
    info_mode, assist_mode = _profile_modes("noassist")

    observations: list[np.ndarray] = []
    action_features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    positive_actions: list[int] = []
    negative_actions: list[int] = []
    score_margins: list[float] = []
    seeds: list[int] = []
    gate_reasons: list[int] = []

    actual_games = 0
    terminal_games = 0
    total_steps = 0
    v5_states = 0
    gated_states = 0
    searched_states = 0
    accepted_pairs = 0
    candidate_evals = 0

    specs = _v5_on_policy_game_specs(games=int(config.games), seed=int(config.seed), focus_start_mode="v5_second")
    for game_idx, spec in enumerate(specs, start=1):
        seed = int(spec["seed"])
        v5_player_id = int(spec["v5_player_id"])
        starting_player_id = int(spec["starting_player_id"])
        v4_player_id = 2 if v5_player_id == 1 else 1
        env = _new_noassist_env(seed=seed, starting_player_id=starting_player_id, info_mode=info_mode, assist_mode=assist_mode)
        base_env = env.env
        v4_deck_ids = _player_card_pool_ids(base_env._env.state, v4_player_id)
        v5_deck_ids = list(config.noassist_deck_ids)
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
                    v5_started=False,
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
                    scored = []
                    for action_id in candidates:
                        if 0 <= int(action_id) < mask.shape[0] and mask[int(action_id)] == 1.0:
                            score = _evaluate_rollout_candidate(
                                env=env,
                                candidate_action=int(action_id),
                                v5_player_id=v5_player_id,
                                v5_policy=v5_policy,
                                opponent_policy=opponent,
                                draw_controller=draw_controller,
                                draw_assist_strength=0.0,
                                depth_plies=config.search_depth_plies,
                            )
                            scored.append((int(action_id), float(score)))
                    candidate_evals += len(scored)
                    if scored:
                        searched_states += 1
                        base_score = next((score for action_id, score in scored if action_id == base_action), None)
                        if base_score is not None:
                            best_action, best_score = max(scored, key=lambda item: (item[1], -item[0]))
                            margin = float(best_score - base_score)
                            if best_action != base_action and _pairwise_margin_label(
                                candidate_score=best_score,
                                baseline_score=base_score,
                                min_pairwise_margin=config.min_pairwise_margin,
                            ):
                                observations.append(obs.copy())
                                action_features.append(features.copy())
                                masks.append(mask.copy())
                                positive_actions.append(int(best_action))
                                negative_actions.append(int(base_action))
                                score_margins.append(margin)
                                seeds.append(seed)
                                gate_reasons.append(int(gate_reason))
                                accepted_pairs += 1
                action_id = base_action
            else:
                action_id = int(opponent.select_action(env.env, current))

            _obs, _reward, terminated, truncated, _info = env.step(action_id)
            total_steps += 1
            if terminated or truncated:
                terminal_games += 1
                break

        if game_idx % 64 == 0:
            print(f"phase19_collect_second games={game_idx}/{len(specs)} pairs={accepted_pairs}", flush=True)

    if not positive_actions:
        raise RuntimeError("phase19 pairwise dataset is empty")
    margins_np = np.asarray(score_margins, dtype=np.float32)
    return {
        "observations": np.stack(observations).astype(np.float32, copy=False),
        "action_features": np.stack(action_features).astype(np.float32, copy=False),
        "masks": np.stack(masks).astype(np.float32, copy=False),
        "positive_actions": np.asarray(positive_actions, dtype=np.int32),
        "negative_actions": np.asarray(negative_actions, dtype=np.int32),
        "score_margins": margins_np,
        "seeds": np.asarray(seeds, dtype=np.int64),
        "gate_reasons": np.asarray(gate_reasons, dtype=np.int32),
        "summary": {
            "schema": "extra_lr_v5_phase19_noassist_second_pairwise_dataset_v1",
            "collection_mode": "v5_second_noassist_pairwise_rollout_search",
            "profile": "noassist",
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
            "search_candidates": int(config.search_candidates),
            "search_depth_plies": int(config.search_depth_plies),
            "min_pairwise_margin": float(config.min_pairwise_margin),
            "gate_reason_counts": _count_gate_reasons(gate_reasons),
            "info_mode": asdict(info_mode),
            "assist_mode": assist_mode.to_dict(),
            "draw_assist_uses": 0,
            "assembler_dataset": None,
        },
    }


def collect_noassist_anchor_dataset(config: Phase19Config) -> dict[str, Any]:
    opponent = OnnxActionPolicy(str(config.v4_model), mode="argmax", seed=config.seed + 101, verify_mask=False)
    v5_policy = V5AdaptivePolicy(config.base_checkpoint, adaptive_strength=1.0)
    info_mode, assist_mode = _profile_modes("noassist")

    observations: list[np.ndarray] = []
    action_features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actual_games = 0
    terminal_games = 0
    total_steps = 0
    specs = _v5_on_policy_game_specs(
        games=int(config.anchor_games),
        seed=int(config.seed) + 50_000,
        focus_start_mode="v5_first",
    )
    for spec in specs:
        seed = int(spec["seed"])
        v5_player_id = int(spec["v5_player_id"])
        starting_player_id = int(spec["starting_player_id"])
        v4_player_id = 2 if v5_player_id == 1 else 1
        env = _new_noassist_env(seed=seed, starting_player_id=starting_player_id, info_mode=info_mode, assist_mode=assist_mode)
        v4_deck_ids = _player_card_pool_ids(env.env._env.state, v4_player_id)
        v5_deck_ids = list(config.noassist_deck_ids)
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
                observations.append(env.observe(current).astype(np.float32, copy=True))
                masks.append(env.action_mask(current).astype(np.float32, copy=True))
                action_features.append(env.action_features(current, include_preview=False).astype(np.float32, copy=True))
                action_id = int(v5_policy.select_action(env, current))
            else:
                action_id = int(opponent.select_action(env.env, current))
            _obs, _reward, terminated, truncated, _info = env.step(action_id)
            total_steps += 1
            if terminated or truncated:
                terminal_games += 1
                break

    if not observations:
        raise RuntimeError("phase19 anchor dataset is empty")
    return {
        "anchor_observations": np.stack(observations).astype(np.float32, copy=False),
        "anchor_action_features": np.stack(action_features).astype(np.float32, copy=False),
        "anchor_masks": np.stack(masks).astype(np.float32, copy=False),
        "summary": {
            "anchor_games": int(config.anchor_games),
            "anchor_actual_games": int(actual_games),
            "anchor_terminal_games": int(terminal_games),
            "anchor_states": int(len(observations)),
            "anchor_total_steps": int(total_steps),
        },
    }


def train_conservative_pairwise_kl(
    model: Any,
    reference_model: Any,
    optimizer: Any,
    *,
    observations: np.ndarray,
    action_features: np.ndarray,
    masks: np.ndarray,
    positive_actions: np.ndarray,
    negative_actions: np.ndarray,
    anchor_observations: np.ndarray,
    anchor_action_features: np.ndarray,
    anchor_masks: np.ndarray,
    ranking_margin: float,
    pairwise_coef: float,
    kl_coef: float,
    anchor_kl_coef: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn

    n = int(positive_actions.shape[0])
    if n <= 0:
        raise ValueError("positive_actions must contain at least one row")
    anchor_n = int(anchor_observations.shape[0])
    if anchor_n <= 0:
        raise ValueError("anchor_observations must contain at least one row")
    rng = np.random.default_rng(int(seed))
    metrics: list[dict[str, float]] = []

    for epoch in range(int(epochs)):
        order = np.arange(n, dtype=np.int64)
        rng.shuffle(order)
        losses: list[float] = []
        pairwise_losses: list[float] = []
        policy_kls: list[float] = []
        anchor_kls: list[float] = []
        accuracies: list[float] = []
        gaps: list[float] = []
        for start in range(0, n, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            anchor_idx = rng.integers(0, anchor_n, size=max(1, len(idx)), dtype=np.int64)
            obs_b = mx.array(observations[idx])
            features_b = mx.array(action_features[idx])
            mask_b = mx.array(masks[idx])
            pos_b = mx.array(positive_actions[idx], dtype=mx.int32)
            neg_b = mx.array(negative_actions[idx], dtype=mx.int32)
            anchor_obs_b = mx.array(anchor_observations[anchor_idx])
            anchor_features_b = mx.array(anchor_action_features[anchor_idx])
            anchor_mask_b = mx.array(anchor_masks[anchor_idx])

            ref_logits, _ref_values = reference_model(obs_b, features_b)
            ref_anchor_logits, _ref_anchor_values = reference_model(anchor_obs_b, anchor_features_b)
            mx.eval(ref_logits, ref_anchor_logits)

            def loss_fn(m):
                logits, _values = m(obs_b, features_b)
                masked = mx.where(mask_b.astype(mx.bool_), logits, mx.array(-1.0e9, dtype=mx.float32))
                row = mx.arange(pos_b.shape[0])
                pos_logits = masked[row, pos_b]
                neg_logits = masked[row, neg_b]
                diff = pos_logits - neg_logits
                pair_loss = mx.mean(mx.maximum(mx.array(0.0, dtype=mx.float32), mx.array(float(ranking_margin)) - diff))
                policy_kl = _masked_ref_kl(ref_logits, logits, mask_b)

                anchor_logits, _anchor_values = m(anchor_obs_b, anchor_features_b)
                anchor_kl = _masked_ref_kl(ref_anchor_logits, anchor_logits, anchor_mask_b)
                loss = (
                    mx.array(float(pairwise_coef), dtype=mx.float32) * pair_loss
                    + mx.array(float(kl_coef), dtype=mx.float32) * policy_kl
                    + mx.array(float(anchor_kl_coef), dtype=mx.float32) * anchor_kl
                )
                accuracy = mx.mean((diff > 0.0).astype(mx.float32))
                avg_gap = mx.mean(diff)
                return loss, {
                    "pairwise_loss": pair_loss,
                    "policy_kl": policy_kl,
                    "anchor_kl": anchor_kl,
                    "pairwise_accuracy": accuracy,
                    "avg_logit_gap": avg_gap,
                }

            value_and_grad = nn.value_and_grad(model, loss_fn)
            (loss_value, aux), grads = value_and_grad(model)
            optimizer.update(model, grads)
            mx.eval(
                model.parameters(),
                optimizer.state,
                loss_value,
                aux["pairwise_loss"],
                aux["policy_kl"],
                aux["anchor_kl"],
                aux["pairwise_accuracy"],
                aux["avg_logit_gap"],
            )
            losses.append(float(loss_value.item()))
            pairwise_losses.append(float(aux["pairwise_loss"].item()))
            policy_kls.append(float(aux["policy_kl"].item()))
            anchor_kls.append(float(aux["anchor_kl"].item()))
            accuracies.append(float(aux["pairwise_accuracy"].item()))
            gaps.append(float(aux["avg_logit_gap"].item()))
        metrics.append(
            {
                "epoch": float(epoch + 1),
                "loss": float(np.mean(losses)),
                "pairwise_loss": float(np.mean(pairwise_losses)),
                "policy_kl": float(np.mean(policy_kls)),
                "anchor_kl": float(np.mean(anchor_kls)),
                "pairwise_accuracy": float(np.mean(accuracies)),
                "avg_logit_gap": float(np.mean(gaps)),
            }
        )
    return {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "pairs": int(n),
        "anchor_states": int(anchor_n),
        "ranking_margin": float(ranking_margin),
        "pairwise_coef": float(pairwise_coef),
        "kl_coef": float(kl_coef),
        "anchor_kl_coef": float(anchor_kl_coef),
        "epoch_metrics": metrics,
        "final_loss": float(metrics[-1]["loss"]),
        "final_pairwise_loss": float(metrics[-1]["pairwise_loss"]),
        "final_policy_kl": float(metrics[-1]["policy_kl"]),
        "final_anchor_kl": float(metrics[-1]["anchor_kl"]),
        "final_pairwise_accuracy": float(metrics[-1]["pairwise_accuracy"]),
        "final_avg_logit_gap": float(metrics[-1]["avg_logit_gap"]),
    }


def _masked_ref_kl(ref_logits, logits, mask):
    import mlx.core as mx

    valid = mask.astype(mx.bool_)
    neg = mx.array(-1.0e9, dtype=mx.float32)
    ref_masked = mx.where(valid, ref_logits, neg)
    logits_masked = mx.where(valid, logits, neg)
    ref_log_probs = ref_masked - mx.logsumexp(ref_masked, axis=-1, keepdims=True)
    log_probs = logits_masked - mx.logsumexp(logits_masked, axis=-1, keepdims=True)
    ref_probs = mx.exp(ref_log_probs)
    terms = mx.where(valid, ref_probs * (ref_log_probs - log_probs), mx.array(0.0, dtype=mx.float32))
    return mx.mean(mx.sum(terms, axis=-1))


def _new_noassist_env(*, seed: int, starting_player_id: int, info_mode: Any, assist_mode: Any) -> TrainV3ClassicEnv:
    env = TrainV3ClassicEnv(
        TrainV3EnvConfig(
            seed=int(seed),
            verify_mask=False,
            placement_mode="append_only",
            include_legal_actions_in_info=False,
            info_mode=info_mode,
            assist_mode=assist_mode,
        )
    )
    env.env.reset(seed=int(seed), starting_player_id=int(starting_player_id))
    return env


def _validate_config(config: Phase19Config) -> None:
    if not config.base_checkpoint.exists():
        raise FileNotFoundError(f"base checkpoint not found: {config.base_checkpoint}")
    if not config.v4_model.exists():
        raise FileNotFoundError(f"V4 model not found: {config.v4_model}")
    for name in (
        "games",
        "anchor_games",
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
    if len(tuple(config.noassist_deck_ids)) < 2:
        raise ValueError("noassist_deck_ids must include a hero and at least one card")
    for name in ("learning_rate", "min_pairwise_margin", "ranking_margin", "pairwise_coef", "kl_coef", "anchor_kl_coef"):
        if float(getattr(config, name)) <= 0.0:
            raise ValueError(f"{name} must be positive")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run no-assist conservative second-start V5 hardening")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--v4-model", type=Path, default=DEFAULT_V4_MAX)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "TrainV3.5" / "runs" / f"phase19_noassist_conservative_{stamp}")
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--anchor-games", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=21900000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--noassist-deck-ids", default=",".join(str(card_id) for card_id in NOASSIST_BASELINE_DECK_IDS))
    parser.add_argument("--search-candidates", type=int, default=8)
    parser.add_argument("--search-depth-plies", type=int, default=8)
    parser.add_argument("--min-pairwise-margin", type=float, default=0.25)
    parser.add_argument("--ranking-margin", type=float, default=0.5)
    parser.add_argument("--pairwise-coef", type=float, default=1.0)
    parser.add_argument("--kl-coef", type=float, default=1.5)
    parser.add_argument("--anchor-kl-coef", type=float, default=2.0)
    parser.add_argument("--gate-max-hp-margin", type=int, default=-2)
    parser.add_argument("--gate-max-own-hp", type=int, default=20)
    parser.add_argument("--gate-max-board-margin", type=int, default=-4)
    parser.add_argument("--min-pairs", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--action-hidden-dim", type=int, default=128)
    parser.add_argument("--no-save-dataset", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

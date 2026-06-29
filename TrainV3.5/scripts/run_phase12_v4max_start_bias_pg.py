#!/usr/bin/env python3
"""Phase12 V4-max start-bias policy-gradient correction.

This phase tests a concrete hypothesis from the V5/V4-max probes: V5 is strong
when it starts first, but undertrained in second-start recovery states. Unlike
trace-only Rust PPO, this runner plays real Python-oracle games against V4-max
and skews the start distribution toward V5-second games.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
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
from run_v5_vs_v4max_benchmark import (  # noqa: E402
    DEFAULT_ASSEMBLER_DATASET,
    DEFAULT_V4_MAX,
    NOASSIST_BASELINE_DECK_IDS,
    V5AdaptivePolicy,
    _action_is_end_turn,
    _apply_draw_assist_to_player,
    _load_assembler_candidates,
    _parse_deck_ids,
    _player_card_pool_ids,
    _select_v5_deck_for_config,
    _strong_assist_mode,
    _strong_info_mode,
    BenchmarkConfig,
)
from train_v3.aux_models import DeckMatchupEvaluator, DrawAssistController  # noqa: E402
from train_v3.env_v5 import TrainV3ClassicEnv, TrainV3EnvConfig  # noqa: E402
from train_v3.v5_policy import create_v5_policy  # noqa: E402


DEFAULT_SOURCE_CHECKPOINT = (
    ROOT
    / "TrainV3"
    / "runs"
    / "phase10_v4max_distill_round2_from_15020_20260609_1324"
    / "extra_lr_v5_phase10_v4max_distill_61571_states.npz"
)


@dataclass(frozen=True)
class Phase12Config:
    source_checkpoint: Path
    v4_model: Path
    assembler_dataset: Path | None
    output_dir: Path
    games: int
    max_steps: int
    seed: int
    v5_first_rate: float
    batch_size: int
    epochs: int
    learning_rate: float
    entropy_coef: float
    return_clip: float
    win_reward: float
    loss_reward: float
    hp_reward_scale: float
    v5_action_mode: str
    v5_temperature: float
    v5_epsilon: float
    algorithm: str
    updates: int
    clip_epsilon: float
    value_coef: float
    checkpoint_interval: int = 0
    profile: str = "strong"
    noassist_deck_ids: tuple[int, ...] = NOASSIST_BASELINE_DECK_IDS
    noassist_deck_pool: tuple[tuple[int, ...], ...] = ()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = Phase12Config(
        source_checkpoint=args.source_checkpoint.resolve(),
        v4_model=args.v4_model.resolve(),
        assembler_dataset=args.assembler_dataset.resolve() if args.assembler_dataset is not None else None,
        output_dir=args.output_dir.resolve(),
        games=int(args.games),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        v5_first_rate=float(args.v5_first_rate),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        entropy_coef=float(args.entropy_coef),
        return_clip=float(args.return_clip),
        win_reward=float(args.win_reward),
        loss_reward=float(args.loss_reward),
        hp_reward_scale=float(args.hp_reward_scale),
        v5_action_mode=str(args.v5_action_mode),
        v5_temperature=float(args.v5_temperature),
        v5_epsilon=float(args.v5_epsilon),
        algorithm=str(args.algorithm),
        updates=int(args.updates),
        checkpoint_interval=int(args.checkpoint_interval),
        clip_epsilon=float(args.clip_epsilon),
        value_coef=float(args.value_coef),
        profile=str(args.profile),
        noassist_deck_ids=_parse_deck_ids(args.noassist_deck_ids),
        noassist_deck_pool=_parse_deck_pool(args.noassist_deck_pool),
    )
    result = run_phase12(config)
    print("PHASE12_RESULT", json.dumps(result["summary"], sort_keys=True), flush=True)
    print(f"Saved: {result['checkpoint_path']}", flush=True)
    return 0


def run_phase12(config: Phase12Config) -> dict[str, Any]:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    import mlx.optimizers as optim

    model = create_v5_policy(policy_kind="v5_split_encoder", hidden_dim=256, action_hidden_dim=128)
    optimizer = optim.Adam(learning_rate=config.learning_rate)
    loaded = load_checkpoint(str(config.source_checkpoint), model, optimizer=None)
    if config.algorithm == "ppo":
        return run_phase12_online_ppo(config, model, optimizer, loaded)

    dataset = collect_phase12_rollouts(config)
    train_summary = train_outcome_policy_gradient(
        model,
        optimizer,
        observations=dataset["observations"],
        action_features=dataset["action_features"],
        masks=dataset["masks"],
        actions=dataset["actions"],
        returns=dataset["returns"],
        epochs=config.epochs,
        batch_size=config.batch_size,
        entropy_coef=config.entropy_coef,
        seed=config.seed + 31,
    )
    checkpoint_path = config.output_dir / f"extra_lr_v5_phase12_v4max_start_bias_pg_{dataset['actions'].shape[0]}_states.npz"
    metadata = {
        "run_name": "phase12_v4max_start_bias_pg",
        "model_name": "extra-lr-v5-adaptive",
        "phase": "phase12_v4max_start_bias_pg",
        "source_checkpoint": str(config.source_checkpoint),
        "source_metadata": loaded.get("metadata", {}),
        "v4_model": str(config.v4_model),
        "obs_dim": 6480,
        "action_feature_dim": 171,
        "config": _jsonable(asdict(config)),
        "dataset": dataset["summary"],
        "train_summary": train_summary,
        "profile_contract": _profile_contract_for_metadata(config),
        "v4_1_included": False,
        "online_v4max_rollout": True,
        "start_bias_policy": "v5_first_10_v5_second_90",
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
            "games": int(dataset["summary"]["games"]),
            "v5_first_games": int(dataset["summary"]["v5_first_games"]),
            "v5_second_games": int(dataset["summary"]["v5_second_games"]),
            "v5_winrate": float(dataset["summary"]["v5_winrate"]),
            "v5_second_winrate": float(dataset["summary"]["v5_second_winrate"]),
            "final_loss": float(train_summary["final_loss"]),
            "final_action_accuracy": float(train_summary["final_action_accuracy"]),
        },
    }
    (config.output_dir / "phase12_summary.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


class _ModelBackedV5Policy:
    def __init__(self, model: Any):
        self.model = model

    def reset(self, _seed: int) -> None:
        return None


def run_phase12_online_ppo(
    config: Phase12Config,
    model: Any,
    optimizer: Any,
    loaded: dict[str, Any],
) -> dict[str, Any]:
    progress_path = config.output_dir / "phase12_ppo_progress.jsonl"
    checkpoints_dir = config.output_dir / "checkpoints"
    update_metrics: list[dict[str, Any]] = []
    last_dataset: dict[str, Any] | None = None
    policy = _ModelBackedV5Policy(model)
    for update in range(1, int(config.updates) + 1):
        dataset = collect_phase12_rollouts(
            config,
            v5_policy=policy,
            seed_offset=(update - 1) * int(config.games),
            collect_policy_stats=True,
        )
        train_summary = train_online_ppo_batch(
            model,
            optimizer,
            observations=dataset["observations"],
            action_features=dataset["action_features"],
            masks=dataset["masks"],
            actions=dataset["actions"],
            old_log_probs=dataset["old_log_probs"],
            old_values=dataset["old_values"],
            returns=dataset["returns"],
            epochs=config.epochs,
            batch_size=config.batch_size,
            clip_epsilon=config.clip_epsilon,
            value_coef=config.value_coef,
            entropy_coef=config.entropy_coef,
            seed=config.seed + 97 + update,
        )
        metric = {"update": update, "dataset_summary": dataset["summary"], "train_summary": train_summary}
        update_metrics.append(metric)
        last_dataset = dataset
        if int(config.checkpoint_interval) > 0 and update % int(config.checkpoint_interval) == 0:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            states_so_far = int(sum(item["dataset_summary"]["states"] for item in update_metrics))
            checkpoint_path = (
                checkpoints_dir
                / f"extra_lr_v5_phase12_v4max_start_bias_ppo_update_{update:04d}_{states_so_far}_states.npz"
            )
            checkpoint_metadata = _phase12_ppo_metadata(
                config=config,
                loaded=loaded,
                last_dataset=last_dataset,
                update_metrics=update_metrics,
                total_states=states_so_far,
                completed_updates=update,
                partial=True,
            )
            save_checkpoint(str(checkpoint_path), model, optimizer=optimizer, metadata=checkpoint_metadata)
            metric["checkpoint_path"] = str(checkpoint_path)
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(metric), sort_keys=True) + "\n")

    assert last_dataset is not None
    total_states = int(sum(item["dataset_summary"]["states"] for item in update_metrics))
    checkpoint_path = config.output_dir / f"extra_lr_v5_phase12_v4max_start_bias_ppo_{total_states}_states.npz"
    metadata = _phase12_ppo_metadata(
        config=config,
        loaded=loaded,
        last_dataset=last_dataset,
        update_metrics=update_metrics,
        total_states=total_states,
        completed_updates=int(config.updates),
        partial=False,
    )
    save_checkpoint(str(checkpoint_path), model, optimizer=optimizer, metadata=metadata)
    result = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_summary": last_dataset["summary"],
        "train_summary": update_metrics[-1]["train_summary"],
        "update_metrics": update_metrics,
        "summary": {
            "status": "ok",
            "checkpoint_path": str(checkpoint_path),
            "algorithm": "ppo",
            "updates": int(config.updates),
            "checkpoint_interval": int(config.checkpoint_interval),
            "states": total_states,
            "last_v5_winrate": float(last_dataset["summary"]["v5_winrate"]),
            "last_v5_second_winrate": float(last_dataset["summary"]["v5_second_winrate"]),
            "final_loss": float(update_metrics[-1]["train_summary"]["final_loss"]),
            "final_approx_kl": float(update_metrics[-1]["train_summary"]["final_approx_kl"]),
        },
    }
    (config.output_dir / "phase12_summary.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _phase12_ppo_metadata(
    *,
    config: Phase12Config,
    loaded: dict[str, Any],
    last_dataset: dict[str, Any],
    update_metrics: list[dict[str, Any]],
    total_states: int,
    completed_updates: int,
    partial: bool,
) -> dict[str, Any]:
    return {
        "run_name": "phase12_v4max_start_bias_ppo",
        "model_name": "extra-lr-v5-adaptive",
        "phase": "phase12_v4max_start_bias_ppo",
        "source_checkpoint": str(config.source_checkpoint),
        "source_metadata": loaded.get("metadata", {}),
        "v4_model": str(config.v4_model),
        "obs_dim": 6480,
        "action_feature_dim": 171,
        "config": _jsonable(asdict(config)),
        "last_dataset": last_dataset["summary"],
        "update_metrics": update_metrics,
        "total_states": int(total_states),
        "completed_updates": int(completed_updates),
        "partial_checkpoint": bool(partial),
        "profile_contract": _profile_contract_for_metadata(config),
        "v4_1_included": False,
        "online_v4max_rollout": True,
        "start_bias_policy": "v5_first_10_v5_second_90",
    }


def _profile_contract_for_metadata(config: Phase12Config) -> dict[str, Any]:
    profile = str(config.profile).strip().lower()
    noassist = profile == "noassist"
    return {
        "profile": profile,
        "private_info_enabled": not noassist,
        "draw_assist_enabled": not noassist,
        "assist_mode_enabled": not noassist,
        "deck_assist_enabled": not noassist,
        "second_start_search_enabled": False,
        "recovery_reranker_enabled": False,
        "fixed_noassist_deck_ids": list(config.noassist_deck_ids) if noassist else [],
        "noassist_deck_pool": [list(deck) for deck in config.noassist_deck_pool] if noassist else [],
        "v4_1_included": False,
    }


def collect_phase12_rollouts(
    config: Phase12Config,
    *,
    v5_policy: Any | None = None,
    seed_offset: int = 0,
    collect_policy_stats: bool = False,
) -> dict[str, Any]:
    if v5_policy is None:
        v5_policy = V5AdaptivePolicy(config.source_checkpoint, adaptive_strength=1.0)
    v4_policy = OnnxActionPolicy(str(config.v4_model), mode="argmax", seed=config.seed, verify_mask=False)
    assembler = DeckMatchupEvaluator()
    draw_controller = DrawAssistController()
    profile = str(config.profile).strip().lower()
    assembler_candidates = [] if profile == "noassist" else _load_assembler_candidates(config.assembler_dataset)
    bench_config = BenchmarkConfig(
        v4_model_path=config.v4_model,
        v5_checkpoint_path=config.source_checkpoint,
        assembler_dataset_path=config.assembler_dataset,
        output_dir=config.output_dir,
        games=config.games,
        seed=config.seed,
        max_steps=config.max_steps,
        private_info_enabled=profile != "noassist",
        draw_assist_enabled=profile != "noassist",
        assist_mode_enabled=profile != "noassist",
        deck_assist_enabled=profile != "noassist",
        noassist_deck_ids=tuple(config.noassist_deck_ids),
    )
    observations: list[np.ndarray] = []
    action_features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    returns: list[float] = []
    old_log_probs: list[float] = []
    old_values: list[float] = []
    v5_started_flags: list[bool] = []
    game_summaries: list[dict[str, Any]] = []
    draw_assist_uses = 0
    rng = np.random.default_rng(int(config.seed) + 12_012)

    for game_index, spec in enumerate(_phase12_game_specs(
        games=config.games,
        seed=int(config.seed) + int(seed_offset),
        v5_first_rate=config.v5_first_rate,
    )):
        seed = int(spec["seed"])
        v5_player_id = int(spec["v5_player_id"])
        starting_player_id = int(spec["starting_player_id"])
        v4_player_id = 2 if v5_player_id == 1 else 1
        game_bench_config = replace(
            bench_config,
            noassist_deck_ids=_select_noassist_training_deck(
                config,
                game_index=game_index,
                seed=seed,
            ),
        )
        env = TrainV3ClassicEnv(
            TrainV3EnvConfig(
                seed=seed,
                verify_mask=False,
                placement_mode="append_only",
                include_legal_actions_in_info=False,
                info_mode=_strong_info_mode(game_bench_config),
                assist_mode=_strong_assist_mode(game_bench_config),
            )
        )
        env.env.reset(seed=seed, starting_player_id=starting_player_id)
        v4_deck_ids = _player_card_pool_ids(env.env._env.state, v4_player_id)
        v5_deck_ids, assembler_score = _select_v5_deck_for_config(
            config=game_bench_config,
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
        v5_policy.reset(seed * 11 + v5_player_id)
        v4_policy.reset(seed * 13 + v4_player_id)
        game_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, bool, float, float]] = []
        invalid = 0
        steps = 0
        for steps in range(1, config.max_steps + 1):
            current = env.current_player_id()
            if current == v5_player_id:
                obs = env.observe(current).astype(np.float32, copy=False)
                mask = env.action_mask(current).astype(np.float32, copy=False)
                features = env.action_features(current, include_preview=False).astype(np.float32, copy=False)
                action_info = _select_phase12_v5_action_with_stats(
                    v5_policy=v5_policy,
                    obs=obs,
                    action_features=features,
                    mask=mask,
                    rng=rng,
                    mode=config.v5_action_mode,
                    temperature=config.v5_temperature,
                    epsilon=config.v5_epsilon,
                )
                action_id = int(action_info["action_id"])
                game_rows.append(
                    (
                        obs.copy(),
                        features.copy(),
                        mask.copy(),
                        action_id,
                        bool(spec["v5_started"]),
                        float(action_info["log_prob"]),
                        float(action_info["value"]),
                    )
                )
            else:
                action_id = int(v4_policy.select_action(env.env, current))
            if _action_is_end_turn(env.env._env.state, current, action_id):
                next_player_id = 2 if current == 1 else 1
                if next_player_id == v5_player_id and bool(game_bench_config.draw_assist_enabled):
                    assist_info = _apply_draw_assist_to_player(
                        env=env,
                        player_id=v5_player_id,
                        controller=draw_controller,
                        strength=game_bench_config.draw_assist_strength,
                    )
                    draw_assist_uses += int(assist_info.get("selected_card_id") is not None)
            _obs, _reward, terminated, truncated, info = env.step(action_id)
            invalid += int(bool(info.get("invalid_action")))
            if terminated or truncated:
                break

        state = env.env._env.state
        winner = env.env.winner_id()
        v5_hp = int(state.p1.hero.hp if v5_player_id == 1 else state.p2.hero.hp)
        v4_hp = int(state.p1.hero.hp if v4_player_id == 1 else state.p2.hero.hp)
        game_return = _phase12_game_return(
            winner=winner,
            v5_player_id=v5_player_id,
            v5_hp=v5_hp,
            v4_hp=v4_hp,
            win_reward=config.win_reward,
            loss_reward=config.loss_reward,
            hp_reward_scale=config.hp_reward_scale,
        )
        game_return = float(np.clip(game_return, -config.return_clip, config.return_clip))
        for obs, features, mask, action_id, v5_started, old_log_prob, old_value in game_rows:
            observations.append(obs)
            action_features.append(features)
            masks.append(mask)
            actions.append(int(action_id))
            returns.append(game_return)
            old_log_probs.append(float(old_log_prob))
            old_values.append(float(old_value))
            v5_started_flags.append(v5_started)
        game_summaries.append(
            {
                "seed": seed,
                "v5_player_id": v5_player_id,
                "starting_player_id": starting_player_id,
                "v5_started": bool(spec["v5_started"]),
                "winner": winner,
                "v5_win": winner == v5_player_id,
                "v5_hp": v5_hp,
                "v4_hp": v4_hp,
                "return": game_return,
                "steps": steps,
                "invalid_actions": invalid,
                "assembler_score": float(assembler_score),
                "v5_deck_ids": list(v5_deck_ids),
            }
        )

    if not actions:
        raise RuntimeError("phase12 rollout dataset is empty")
    return_values = np.asarray(returns, dtype=np.float32)
    norm_returns = _normalize_returns(return_values)
    first_games = [game for game in game_summaries if game["v5_started"]]
    second_games = [game for game in game_summaries if not game["v5_started"]]
    summary = {
        "schema": "extra_lr_v5_phase12_v4max_start_bias_pg_dataset_v1",
        "games": int(len(game_summaries)),
        "states": int(len(actions)),
        "v5_first_games": int(len(first_games)),
        "v5_second_games": int(len(second_games)),
        "v5_first_rate": float(len(first_games) / max(1, len(game_summaries))),
        "v5_winrate": _winrate(game_summaries),
        "v5_first_winrate": _winrate(first_games),
        "v5_second_winrate": _winrate(second_games),
        "avg_raw_return": float(np.mean(return_values)),
        "avg_normalized_return": float(np.mean(norm_returns)),
        "draw_assist_uses": int(draw_assist_uses),
        "invalid_actions": int(sum(game["invalid_actions"] for game in game_summaries)),
        "start_bias_policy": "v5_first_10_v5_second_90",
        "profile": profile,
        "noassist_deck_ids": list(config.noassist_deck_ids),
        "noassist_deck_pool": [list(deck) for deck in config.noassist_deck_pool],
    }
    result = {
        "observations": np.stack(observations).astype(np.float32, copy=False),
        "action_features": np.stack(action_features).astype(np.float32, copy=False),
        "masks": np.stack(masks).astype(np.float32, copy=False),
        "actions": np.asarray(actions, dtype=np.int32),
        "returns": norm_returns.astype(np.float32, copy=False),
        "v5_started": np.asarray(v5_started_flags, dtype=np.bool_),
        "games": game_summaries,
        "summary": summary,
    }
    if collect_policy_stats:
        result["old_log_probs"] = np.asarray(old_log_probs, dtype=np.float32)
        result["old_values"] = np.asarray(old_values, dtype=np.float32)
        advantages = result["returns"].astype(np.float32, copy=False) - result["old_values"]
        result["advantages"] = _normalize_returns(advantages).astype(np.float32, copy=False)
        result["summary"]["old_value_mean"] = float(np.mean(result["old_values"]))
        result["summary"]["old_log_prob_mean"] = float(np.mean(result["old_log_probs"]))
        result["summary"]["advantage_mean"] = float(np.mean(result["advantages"]))
        result["summary"]["advantage_std"] = float(np.std(result["advantages"]))
    return result


def train_outcome_policy_gradient(
    model: Any,
    optimizer: Any,
    *,
    observations: np.ndarray,
    action_features: np.ndarray,
    masks: np.ndarray,
    actions: np.ndarray,
    returns: np.ndarray,
    epochs: int,
    batch_size: int,
    entropy_coef: float,
    seed: int,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn

    n = int(actions.shape[0])
    if n <= 0:
        raise ValueError("actions must contain at least one row")
    rng = np.random.default_rng(int(seed))
    metrics: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        order = np.arange(n, dtype=np.int64)
        rng.shuffle(order)
        losses: list[float] = []
        accs: list[float] = []
        entropies: list[float] = []
        for start in range(0, n, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            obs_b = mx.array(observations[idx])
            features_b = mx.array(action_features[idx])
            mask_b = mx.array(masks[idx])
            actions_b = mx.array(actions[idx], dtype=mx.int32)
            returns_b = mx.array(returns[idx], dtype=mx.float32)

            def loss_fn(model):
                logits, _values = model(obs_b, features_b)
                masked = mx.where(mask_b.astype(mx.bool_), logits, mx.array(-1.0e9, dtype=mx.float32))
                log_probs = masked - mx.logsumexp(masked, axis=-1, keepdims=True)
                probs = mx.exp(log_probs) * mask_b
                picked = log_probs[mx.arange(actions_b.shape[0]), actions_b]
                entropy = -mx.sum(probs * log_probs * mask_b, axis=-1)
                pg_loss = -mx.mean(returns_b * picked)
                loss = pg_loss - float(entropy_coef) * mx.mean(entropy)
                pred = mx.argmax(masked, axis=-1)
                accuracy = mx.mean((pred == actions_b).astype(mx.float32))
                return loss, {"accuracy": accuracy, "entropy": mx.mean(entropy), "pg_loss": pg_loss}

            value_and_grad = nn.value_and_grad(model, loss_fn)
            (loss_value, aux), grads = value_and_grad(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss_value, aux["accuracy"], aux["entropy"])
            losses.append(float(loss_value.item()))
            accs.append(float(aux["accuracy"].item()))
            entropies.append(float(aux["entropy"].item()))
        metrics.append(
            {
                "epoch": float(epoch + 1),
                "loss": float(np.mean(losses)),
                "action_accuracy": float(np.mean(accs)),
                "entropy": float(np.mean(entropies)),
            }
        )
    return {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "states": int(n),
        "epoch_metrics": metrics,
        "final_loss": float(metrics[-1]["loss"]),
        "final_action_accuracy": float(metrics[-1]["action_accuracy"]),
        "final_entropy": float(metrics[-1]["entropy"]),
    }


def train_online_ppo_batch(
    model: Any,
    optimizer: Any,
    *,
    observations: np.ndarray,
    action_features: np.ndarray,
    masks: np.ndarray,
    actions: np.ndarray,
    old_log_probs: np.ndarray,
    old_values: np.ndarray,
    returns: np.ndarray,
    epochs: int,
    batch_size: int,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    seed: int,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn

    n = int(actions.shape[0])
    if n <= 0:
        raise ValueError("actions must contain at least one row")
    advantages_np = _normalize_returns(np.asarray(returns, dtype=np.float32) - np.asarray(old_values, dtype=np.float32))
    rng = np.random.default_rng(int(seed))
    metrics: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        order = np.arange(n, dtype=np.int64)
        rng.shuffle(order)
        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        clip_fracs: list[float] = []
        for start in range(0, n, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            obs_b = mx.array(observations[idx])
            features_b = mx.array(action_features[idx])
            mask_b = mx.array(masks[idx])
            actions_b = mx.array(actions[idx], dtype=mx.int32)
            old_log_probs_b = mx.array(old_log_probs[idx], dtype=mx.float32)
            returns_b = mx.array(returns[idx], dtype=mx.float32)
            advantages_b = mx.array(advantages_np[idx], dtype=mx.float32)

            def loss_fn(model):
                logits, values = model(obs_b, features_b)
                masked = mx.where(mask_b.astype(mx.bool_), logits, mx.array(-1.0e9, dtype=mx.float32))
                log_probs = masked - mx.logsumexp(masked, axis=-1, keepdims=True)
                probs = mx.exp(log_probs) * mask_b
                picked = log_probs[mx.arange(actions_b.shape[0]), actions_b]
                ratios = mx.exp(picked - old_log_probs_b)
                surr1 = ratios * advantages_b
                surr2 = mx.clip(ratios, 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon)) * advantages_b
                policy_loss = -mx.mean(mx.minimum(surr1, surr2))
                value_loss = float(value_coef) * mx.mean((returns_b - values) ** 2)
                entropy = mx.mean(-mx.sum(probs * log_probs * mask_b, axis=-1))
                loss = policy_loss + value_loss - float(entropy_coef) * entropy
                approx_kl = mx.mean(old_log_probs_b - picked)
                clip_fraction = mx.mean(
                    mx.where(
                        ratios < 1.0 - float(clip_epsilon),
                        mx.ones_like(ratios),
                        mx.where(ratios > 1.0 + float(clip_epsilon), mx.ones_like(ratios), mx.zeros_like(ratios)),
                    )
                )
                return loss, {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                }

            value_and_grad = nn.value_and_grad(model, loss_fn)
            (loss_value, aux), grads = value_and_grad(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss_value, *aux.values())
            losses.append(float(loss_value.item()))
            policy_losses.append(float(aux["policy_loss"].item()))
            value_losses.append(float(aux["value_loss"].item()))
            entropies.append(float(aux["entropy"].item()))
            approx_kls.append(float(aux["approx_kl"].item()))
            clip_fracs.append(float(aux["clip_fraction"].item()))
        metrics.append(
            {
                "epoch": float(epoch + 1),
                "loss": float(np.mean(losses)),
                "policy_loss": float(np.mean(policy_losses)),
                "value_loss": float(np.mean(value_losses)),
                "entropy": float(np.mean(entropies)),
                "approx_kl": float(np.mean(approx_kls)),
                "clip_fraction": float(np.mean(clip_fracs)),
            }
        )
    return {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "states": int(n),
        "epoch_metrics": metrics,
        "final_loss": float(metrics[-1]["loss"]),
        "final_policy_loss": float(metrics[-1]["policy_loss"]),
        "final_value_loss": float(metrics[-1]["value_loss"]),
        "final_entropy": float(metrics[-1]["entropy"]),
        "final_approx_kl": float(metrics[-1]["approx_kl"]),
        "final_clip_fraction": float(metrics[-1]["clip_fraction"]),
    }


def _phase12_game_specs(*, games: int, seed: int, v5_first_rate: float) -> list[dict[str, Any]]:
    if int(games) <= 0:
        raise ValueError("games must be positive")
    if not 0.0 <= float(v5_first_rate) <= 1.0:
        raise ValueError("v5_first_rate must be in [0, 1]")
    first_period = round(1.0 / max(float(v5_first_rate), 1.0e-9)) if v5_first_rate > 0.0 else 0
    specs: list[dict[str, Any]] = []
    for idx in range(int(games)):
        v5_player_id = 1 if idx % 2 == 0 else 2
        v5_started = first_period > 0 and idx % first_period == 0
        starting_player_id = v5_player_id if v5_started else 2 if v5_player_id == 1 else 1
        specs.append(
            {
                "seed": int(seed) + idx,
                "v5_player_id": int(v5_player_id),
                "starting_player_id": int(starting_player_id),
                "v5_started": bool(v5_started),
            }
        )
    return specs


def _select_noassist_training_deck(config: Phase12Config, *, game_index: int, seed: int) -> tuple[int, ...]:
    profile = str(config.profile).strip().lower()
    if profile != "noassist" or not config.noassist_deck_pool:
        return tuple(config.noassist_deck_ids)
    pool = tuple(tuple(deck) for deck in config.noassist_deck_pool)
    idx = (int(game_index) + int(seed)) % len(pool)
    return pool[idx]


def _select_phase12_v5_action(
    *,
    v5_policy: V5AdaptivePolicy,
    obs: np.ndarray,
    action_features: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    mode: str,
    temperature: float,
    epsilon: float,
) -> int:
    return int(
        _select_phase12_v5_action_with_stats(
            v5_policy=v5_policy,
            obs=obs,
            action_features=action_features,
            mask=mask,
            rng=rng,
            mode=mode,
            temperature=temperature,
            epsilon=epsilon,
        )["action_id"]
    )


def _select_phase12_v5_action_with_stats(
    *,
    v5_policy: Any,
    obs: np.ndarray,
    action_features: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    mode: str,
    temperature: float,
    epsilon: float,
) -> dict[str, float | int]:
    import mlx.core as mx

    logits, value = v5_policy.model(
        mx.array(obs[None, :].astype(np.float32, copy=False)),
        mx.array(action_features[None, :, :].astype(np.float32, copy=False)),
    )
    mx.eval(logits, value)
    logits_np = np.asarray(logits, dtype=np.float32)[0]
    action_id = _sample_masked_action(
        logits=np.asarray(logits, dtype=np.float32)[0],
        mask=mask,
        rng=rng,
        mode=mode,
        temperature=temperature,
        epsilon=epsilon,
    )
    log_probs = _masked_log_probs_np(logits_np, mask)
    return {
        "action_id": int(action_id),
        "log_prob": float(log_probs[int(action_id)]),
        "value": float(np.asarray(value, dtype=np.float32)[0]),
    }


def _masked_log_probs_np(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(np.asarray(mask, dtype=bool), np.asarray(logits, dtype=np.float32), -1.0e9)
    max_value = float(np.max(masked))
    shifted = masked - max_value
    exp_values = np.exp(shifted) * np.asarray(mask, dtype=np.float32)
    denom = max(float(np.sum(exp_values)), 1.0e-12)
    return shifted - math.log(denom)


def _sample_masked_action(
    *,
    logits: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    mode: str,
    temperature: float,
    epsilon: float,
) -> int:
    legal = np.flatnonzero(np.asarray(mask) == 1.0)
    if legal.size == 0:
        return 0
    mode = str(mode).strip().lower()
    if mode not in {"argmax", "sample"}:
        raise ValueError("mode must be argmax or sample")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= float(epsilon) <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    if float(epsilon) > 0.0 and float(rng.random()) < float(epsilon):
        return int(rng.choice(legal))
    legal_logits = np.asarray(logits, dtype=np.float32)[legal]
    if mode == "argmax":
        order = np.lexsort((legal, -legal_logits))
        return int(legal[int(order[0])])
    scaled = legal_logits / float(temperature)
    scaled = scaled - float(np.max(scaled))
    probs = np.exp(scaled).astype(np.float64)
    probs = probs / max(float(np.sum(probs)), 1.0e-12)
    return int(rng.choice(legal, p=probs))


def _phase12_game_return(
    *,
    winner: int | None,
    v5_player_id: int,
    v5_hp: int,
    v4_hp: int,
    win_reward: float,
    loss_reward: float,
    hp_reward_scale: float,
) -> float:
    terminal = float(win_reward) if winner == int(v5_player_id) else float(loss_reward) if winner is not None else 0.0
    return terminal + float(hp_reward_scale) * float(v5_hp - v4_hp)


def _normalize_returns(returns: np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=np.float32)
    std = float(values.std())
    if std <= 1.0e-6:
        return values - float(values.mean())
    return (values - float(values.mean())) / std


def _winrate(games: list[dict[str, Any]]) -> float:
    if not games:
        return 0.0
    return sum(1 for game in games if bool(game["v5_win"])) / len(games)


def _validate_config(config: Phase12Config) -> None:
    if not config.source_checkpoint.exists():
        raise FileNotFoundError(f"source checkpoint not found: {config.source_checkpoint}")
    if not config.v4_model.exists():
        raise FileNotFoundError(f"V4 model not found: {config.v4_model}")
    if config.assembler_dataset is not None and not config.assembler_dataset.exists():
        raise FileNotFoundError(f"assembler dataset not found: {config.assembler_dataset}")
    if str(config.profile).strip().lower() not in {"strong", "noassist"}:
        raise ValueError("profile must be strong or noassist")
    if len(tuple(config.noassist_deck_ids)) < 2:
        raise ValueError("noassist_deck_ids must include a hero and at least one card")
    for deck in tuple(config.noassist_deck_pool):
        if len(tuple(deck)) < 2:
            raise ValueError("every noassist_deck_pool deck must include a hero and at least one card")
    if int(config.games) <= 0:
        raise ValueError("games must be positive")
    if int(config.max_steps) <= 0:
        raise ValueError("max_steps must be positive")
    if not 0.0 <= float(config.v5_first_rate) <= 1.0:
        raise ValueError("v5_first_rate must be in [0, 1]")
    if int(config.batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(config.epochs) <= 0:
        raise ValueError("epochs must be positive")
    if float(config.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not math.isfinite(float(config.entropy_coef)) or float(config.entropy_coef) < 0.0:
        raise ValueError("entropy_coef must be non-negative")
    if float(config.return_clip) <= 0.0:
        raise ValueError("return_clip must be positive")
    if config.v5_action_mode not in {"argmax", "sample"}:
        raise ValueError("v5_action_mode must be argmax or sample")
    if float(config.v5_temperature) <= 0.0:
        raise ValueError("v5_temperature must be positive")
    if not 0.0 <= float(config.v5_epsilon) <= 1.0:
        raise ValueError("v5_epsilon must be in [0, 1]")
    if config.algorithm not in {"pg", "ppo"}:
        raise ValueError("algorithm must be pg or ppo")
    if int(config.updates) <= 0:
        raise ValueError("updates must be positive")
    if int(config.checkpoint_interval) < 0:
        raise ValueError("checkpoint_interval must be non-negative")
    if not 0.0 < float(config.clip_epsilon) <= 1.0:
        raise ValueError("clip_epsilon must be in (0, 1]")
    if float(config.value_coef) < 0.0:
        raise ValueError("value_coef must be non-negative")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _parse_deck_pool(raw: str | None) -> tuple[tuple[int, ...], ...]:
    if raw is None or str(raw).strip() == "":
        return ()
    decks: list[tuple[int, ...]] = []
    for chunk in str(raw).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        decks.append(_parse_deck_ids(chunk))
    return tuple(decks)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run Phase12 V4-max 10:90 start-bias policy-gradient correction")
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--v4-model", type=Path, default=DEFAULT_V4_MAX)
    parser.add_argument("--assembler-dataset", type=Path, default=DEFAULT_ASSEMBLER_DATASET)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "TrainV3" / "runs" / f"phase12_v4max_start_bias_pg_{stamp}")
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261250)
    parser.add_argument("--v5-first-rate", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--return-clip", type=float, default=2.5)
    parser.add_argument("--win-reward", type=float, default=1.0)
    parser.add_argument("--loss-reward", type=float, default=-1.0)
    parser.add_argument("--hp-reward-scale", type=float, default=0.03)
    parser.add_argument("--v5-action-mode", choices=["argmax", "sample"], default="argmax")
    parser.add_argument("--v5-temperature", type=float, default=1.0)
    parser.add_argument("--v5-epsilon", type=float, default=0.0)
    parser.add_argument("--algorithm", choices=["pg", "ppo"], default="pg")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--clip-epsilon", type=float, default=0.16)
    parser.add_argument("--value-coef", type=float, default=0.25)
    parser.add_argument("--profile", choices=["strong", "noassist"], default="strong")
    parser.add_argument("--noassist-deck-ids", default=",".join(str(card_id) for card_id in NOASSIST_BASELINE_DECK_IDS))
    parser.add_argument(
        "--noassist-deck-pool",
        default="",
        help="Semicolon-separated static no-assist decks, each as comma-separated card ids.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

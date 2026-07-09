#!/usr/bin/env python3
"""DEPRECATED: offline V4-max teacher distillation for Extra-LR V5 adaptive.

This is not the current Phase-A / Block-B training path. Phase A is now the
teacher-free random bootstrap and Block B is live Rust league PPO. This script is
kept only as a diagnostic / repair-lane opt-in because later experimental runners
still import its dataset helpers.

This phase is intentionally different from trace PPO: V4-max plays real games
inside the Python battle oracle, and V5 is trained to imitate the teacher's
legal action choices from V5 observations. It is an offline teacher lane, not an
online rollout dependency.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
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
from run_v5_vs_v4max_benchmark import (  # noqa: E402
    DEFAULT_ASSEMBLER_DATASET,
    NOASSIST_BASELINE_DECK_IDS,
    V5AdaptivePolicy,
    _action_is_end_turn,
    _apply_draw_assist_to_player,
    _load_assembler_candidates,
    _player_card_pool_ids,
    _parse_deck_ids,
    _select_v5_deck,
)
from train_v3.aux_models import DeckMatchupEvaluator, DrawAssistController  # noqa: E402
from train_v3.contracts import AssistModeV5, InfoModeV5  # noqa: E402
from train_v3.env_v5 import TrainV3ClassicEnv, TrainV3EnvConfig  # noqa: E402
from train_v3.v5_policy import create_v5_policy  # noqa: E402


DEFAULT_V4_MAX = ROOT / "ai" / "models" / "extra-lr-v4-max.onnx"
DEFAULT_SOURCE_CHECKPOINT = (
    ROOT
    / "TrainV3.5"
    / "runs"
    / "phase9_broad_opponent_blend_5k_bucketed_contiguous_20260608_185729"
    / "checkpoints"
    / "trainv3_rust_legal_update_1000.npz"
)


@dataclass(frozen=True)
class DistillConfig:
    source_checkpoint: Path
    v4_model: Path
    output_dir: Path
    games: int
    max_steps: int
    seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    profile: str = "strong"
    collection_mode: str = "v4_selfplay"
    focus_start_mode: str = "both"
    assembler_dataset: Path | None = DEFAULT_ASSEMBLER_DATASET
    noassist_deck_ids: tuple[int, ...] = NOASSIST_BASELINE_DECK_IDS
    noassist_deck_pool: tuple[tuple[int, ...], ...] = ()
    teacher_dataset_path: Path | None = None
    search_candidates: int = 8
    search_depth_plies: int = 6
    search_min_margin: float = 0.25
    source_kl_coef: float = 0.0
    save_dataset: bool = False
    restore_optimizer: bool = False


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    assembler_dataset = _assembler_dataset_for_profile(
        profile=str(args.profile),
        assembler_dataset=args.assembler_dataset,
    )
    config = DistillConfig(
        source_checkpoint=args.source_checkpoint.resolve(),
        v4_model=args.v4_model.resolve(),
        output_dir=args.output_dir.resolve(),
        games=int(args.games),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        profile=str(args.profile),
        collection_mode=str(args.collection_mode),
        focus_start_mode=str(args.focus_start_mode),
        assembler_dataset=assembler_dataset,
        noassist_deck_ids=_parse_deck_ids(args.noassist_deck_ids),
        noassist_deck_pool=_parse_deck_pool(args.noassist_deck_pool),
        teacher_dataset_path=args.teacher_dataset_path.resolve() if args.teacher_dataset_path is not None else None,
        search_candidates=int(args.search_candidates),
        search_depth_plies=int(args.search_depth_plies),
        search_min_margin=float(args.search_min_margin),
        source_kl_coef=float(args.source_kl_coef),
        save_dataset=bool(args.save_dataset),
        restore_optimizer=bool(args.restore_optimizer),
    )
    result = run_distillation(config)
    print("PHASE10_DISTILL_RESULT", json.dumps(result["summary"], sort_keys=True), flush=True)
    print(f"Saved: {result['checkpoint_path']}", flush=True)
    return 0


def run_distillation(config: DistillConfig) -> dict[str, Any]:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = (
        load_teacher_dataset_npz(config.teacher_dataset_path)
        if config.teacher_dataset_path is not None
        else collect_teacher_dataset(config)
    )
    if config.save_dataset:
        np.savez_compressed(
            config.output_dir / "teacher_dataset.npz",
            observations=dataset["observations"],
            action_features=dataset["action_features"],
            masks=dataset["masks"],
            actions=dataset["actions"],
            seeds=dataset["seeds"],
            v5_started=dataset.get("v5_started", np.zeros_like(dataset["seeds"], dtype=np.bool_)),
        )

    import mlx.optimizers as optim

    model = create_v5_policy(policy_kind="v5_split_encoder", hidden_dim=256, action_hidden_dim=128)
    optimizer = optim.Adam(learning_rate=config.learning_rate)
    loaded = load_checkpoint(
        str(config.source_checkpoint),
        model,
        optimizer=optimizer if config.restore_optimizer else None,
    )
    reference_log_probs = None
    if float(config.source_kl_coef) > 0.0:
        reference_log_probs = compute_reference_log_probs(
            model,
            observations=dataset["observations"],
            action_features=dataset["action_features"],
            masks=dataset["masks"],
            batch_size=config.batch_size,
        )
    train_summary = train_teacher_cross_entropy(
        model,
        optimizer,
        observations=dataset["observations"],
        action_features=dataset["action_features"],
        masks=dataset["masks"],
        actions=dataset["actions"],
        reference_log_probs=reference_log_probs,
        source_kl_coef=config.source_kl_coef,
        epochs=config.epochs,
        batch_size=config.batch_size,
        seed=config.seed + 17,
    )

    checkpoint_path = config.output_dir / f"extra_lr_v5_phase10_v4max_distill_{int(dataset['actions'].shape[0])}_states.npz"
    metadata = {
        "run_name": "phase10_v4max_teacher_distill",
        "model_name": "extra-lr-v5-adaptive",
        "phase": "phase10_v4max_distill",
        "source_checkpoint": str(config.source_checkpoint),
        "source_metadata": loaded.get("metadata", {}),
        "v4_model": str(config.v4_model),
        "obs_dim": 6480,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "config": _jsonable(asdict(config)),
        "dataset": dataset["summary"],
        "train_summary": train_summary,
        "offline_teacher_lane": True,
        "online_rollout_dependency": False,
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
            "teacher_invalid_fallbacks": int(dataset["summary"]["teacher_invalid_fallbacks"]),
            "final_loss": float(train_summary["final_loss"]),
            "final_accuracy": float(train_summary["final_accuracy"]),
        },
    }
    (config.output_dir / "phase10_distill_summary.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def collect_teacher_dataset(config: DistillConfig) -> dict[str, Any]:
    if config.collection_mode == "v4_selfplay":
        return collect_v4max_teacher_dataset(config)
    if config.collection_mode == "v5_on_policy":
        return collect_v5_on_policy_teacher_dataset(config)
    if config.collection_mode == "v5_success_imitation":
        return collect_v5_success_imitation_dataset(config)
    if config.collection_mode == "v5_rollout_search":
        return collect_v5_rollout_search_dataset(config)
    raise ValueError(
        "collection_mode must be v4_selfplay, v5_on_policy, v5_success_imitation, or v5_rollout_search"
    )


def load_teacher_dataset_npz(path: Path) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        raise FileNotFoundError(f"teacher dataset not found: {path}")
    loaded = np.load(Path(path), allow_pickle=True)
    required = ("observations", "action_features", "masks", "actions", "seeds")
    missing = [name for name in required if name not in loaded.files]
    if missing:
        raise ValueError(f"teacher dataset missing required arrays: {missing}")
    observations = np.asarray(loaded["observations"], dtype=np.float32)
    action_features = np.asarray(loaded["action_features"], dtype=np.float32)
    masks = np.asarray(loaded["masks"], dtype=np.float32)
    actions = np.asarray(loaded["actions"], dtype=np.int32)
    seeds = np.asarray(loaded["seeds"], dtype=np.int64)
    v5_started = (
        np.asarray(loaded["v5_started"], dtype=np.bool_)
        if "v5_started" in loaded.files
        else np.zeros_like(seeds, dtype=np.bool_)
    )
    n = int(actions.shape[0])
    if observations.shape[0] != n or action_features.shape[0] != n or masks.shape[0] != n or seeds.shape[0] != n:
        raise ValueError("teacher dataset arrays must have matching first dimension")
    if v5_started.shape[0] != n:
        raise ValueError("v5_started must have the same first dimension as actions")
    return {
        "observations": observations,
        "action_features": action_features,
        "masks": masks,
        "actions": actions,
        "seeds": seeds,
        "v5_started": v5_started,
        "summary": {
            "schema": "extra_lr_v5_v4max_teacher_dataset_v1",
            "collection_mode": "loaded_npz",
            "source_path": str(Path(path)),
            "games": 0,
            "actual_games": 0,
            "states": n,
            "teacher_invalid_fallbacks": 0,
            "v5_started_states": int(np.sum(v5_started)),
            "v5_second_states": int(n - int(np.sum(v5_started))),
        },
    }


def collect_v4max_teacher_dataset(config: DistillConfig) -> dict[str, Any]:
    teacher = OnnxActionPolicy(str(config.v4_model), mode="argmax", seed=config.seed, verify_mask=False)
    info_mode, assist_mode = _profile_modes(config.profile)
    observations: list[np.ndarray] = []
    action_features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    seeds: list[int] = []
    v5_started_flags: list[bool] = []
    invalid_fallbacks = 0
    terminal_games = 0
    total_steps = 0

    for game_idx in range(int(config.games)):
        seed = int(config.seed) + game_idx
        starting_player_id = 1 if game_idx % 2 == 0 else 2
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
        env.reset(
            p1_is_bot=True,
            p2_is_bot=True,
            starting_player_id=starting_player_id,
            seed=seed,
        )
        teacher.reset(seed)
        for _step in range(int(config.max_steps)):
            player_id = env.current_player_id()
            obs = env.observe(player_id).astype(np.float32, copy=False)
            mask = env.action_mask(player_id).astype(np.float32, copy=False)
            features = env.action_features(player_id, include_preview=False).astype(np.float32, copy=False)
            action_id = int(teacher.select_action(env.env, player_id))
            if action_id < 0 or action_id >= mask.shape[0] or mask[action_id] != 1.0:
                invalid_fallbacks += 1
                legal = np.flatnonzero(mask == 1.0)
                if legal.size == 0:
                    break
                action_id = int(legal[0])
            observations.append(obs.copy())
            action_features.append(features.copy())
            masks.append(mask.copy())
            actions.append(action_id)
            seeds.append(seed)
            v5_started_flags.append(starting_player_id == player_id)
            _obs, _reward, terminated, truncated, _info = env.step(action_id)
            total_steps += 1
            if terminated or truncated:
                terminal_games += 1
                break

    if not actions:
        raise RuntimeError("teacher dataset is empty")
    dataset = {
        "observations": np.stack(observations).astype(np.float32, copy=False),
        "action_features": np.stack(action_features).astype(np.float32, copy=False),
        "masks": np.stack(masks).astype(np.float32, copy=False),
        "actions": np.asarray(actions, dtype=np.int32),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "v5_started": np.asarray(v5_started_flags, dtype=np.bool_),
        "summary": {
            "schema": "extra_lr_v5_v4max_teacher_dataset_v1",
            "collection_mode": config.collection_mode,
            "focus_start_mode": config.focus_start_mode,
            "games": int(config.games),
            "actual_games": int(config.games),
            "terminal_games": int(terminal_games),
            "states": int(len(actions)),
            "total_steps": int(total_steps),
            "teacher_invalid_fallbacks": int(invalid_fallbacks),
            "v5_started_states": int(sum(v5_started_flags)),
            "v5_second_states": int(len(v5_started_flags) - sum(v5_started_flags)),
            "profile": config.profile,
            "noassist_deck_ids": list(config.noassist_deck_ids),
            "noassist_deck_pool": [list(deck) for deck in config.noassist_deck_pool],
            "info_mode": asdict(info_mode),
            "assist_mode": assist_mode.to_dict(),
        },
    }
    return dataset


def collect_v5_on_policy_teacher_dataset(config: DistillConfig) -> dict[str, Any]:
    teacher = OnnxActionPolicy(str(config.v4_model), mode="argmax", seed=config.seed, verify_mask=False)
    v5_policy = V5AdaptivePolicy(config.source_checkpoint, adaptive_strength=1.0)
    assembler = DeckMatchupEvaluator()
    draw_controller = DrawAssistController()
    assembler_candidates = _load_assembler_candidates(config.assembler_dataset)
    info_mode, assist_mode = _profile_modes(config.profile)
    observations: list[np.ndarray] = []
    action_features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    seeds: list[int] = []
    v5_started_flags: list[bool] = []
    invalid_fallbacks = 0
    terminal_games = 0
    total_steps = 0
    actual_games = 0
    draw_assist_uses = 0

    for spec in _v5_on_policy_game_specs(
        games=int(config.games),
        seed=int(config.seed),
        focus_start_mode=config.focus_start_mode,
    ):
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
        v5_deck_ids, _assembler_score = _select_distill_v5_deck(
            config=config,
            opponent_deck_ids=v4_deck_ids,
            assembler=assembler,
            candidates=assembler_candidates,
            seed=seed,
        )
        env.reset(
            p1_deck_ids=v5_deck_ids if v5_player_id == 1 else v4_deck_ids,
            p2_deck_ids=v5_deck_ids if v5_player_id == 2 else v4_deck_ids,
            p1_is_bot=True,
            p2_is_bot=True,
            starting_player_id=starting_player_id,
            seed=seed,
        )
        teacher.reset(seed * 13 + v4_player_id)
        v5_policy.reset(seed * 11 + v5_player_id)
        actual_games += 1

        for _step in range(int(config.max_steps)):
            current = env.current_player_id()
            if current == v5_player_id:
                obs = env.observe(current).astype(np.float32, copy=False)
                mask = env.action_mask(current).astype(np.float32, copy=False)
                features = env.action_features(current, include_preview=False).astype(np.float32, copy=False)
                teacher_action = int(teacher.select_action(env.env, current))
                if teacher_action < 0 or teacher_action >= mask.shape[0] or mask[teacher_action] != 1.0:
                    invalid_fallbacks += 1
                    legal = np.flatnonzero(mask == 1.0)
                    if legal.size == 0:
                        break
                    teacher_action = int(legal[0])
                observations.append(obs.copy())
                action_features.append(features.copy())
                masks.append(mask.copy())
                actions.append(teacher_action)
                seeds.append(seed)
                v5_started_flags.append(bool(spec["v5_started"]))
                action_id = v5_policy.select_action(env, current)
            else:
                action_id = int(teacher.select_action(env.env, current))

            if _action_is_end_turn(env.env._env.state, current, action_id):
                next_player_id = 2 if current == 1 else 1
                if next_player_id == v5_player_id and float(info_mode.draw_assist_strength) > 0.0:
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

    if not actions:
        raise RuntimeError("teacher dataset is empty")
    return {
        "observations": np.stack(observations).astype(np.float32, copy=False),
        "action_features": np.stack(action_features).astype(np.float32, copy=False),
        "masks": np.stack(masks).astype(np.float32, copy=False),
        "actions": np.asarray(actions, dtype=np.int32),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "v5_started": np.asarray(v5_started_flags, dtype=np.bool_),
        "summary": {
            "schema": "extra_lr_v5_v4max_teacher_dataset_v1",
            "collection_mode": config.collection_mode,
            "focus_start_mode": config.focus_start_mode,
            "games": int(config.games),
            "actual_games": int(actual_games),
            "terminal_games": int(terminal_games),
            "states": int(len(actions)),
            "total_steps": int(total_steps),
            "teacher_invalid_fallbacks": int(invalid_fallbacks),
            "v5_started_states": int(sum(v5_started_flags)),
            "v5_second_states": int(len(v5_started_flags) - sum(v5_started_flags)),
            "draw_assist_uses": int(draw_assist_uses),
            "profile": config.profile,
            "noassist_deck_ids": list(config.noassist_deck_ids),
            "noassist_deck_pool": [list(deck) for deck in config.noassist_deck_pool],
            "info_mode": asdict(info_mode),
            "assist_mode": assist_mode.to_dict(),
        },
    }


def collect_v5_success_imitation_dataset(config: DistillConfig) -> dict[str, Any]:
    opponent = OnnxActionPolicy(str(config.v4_model), mode="argmax", seed=config.seed, verify_mask=False)
    v5_policy = V5AdaptivePolicy(config.source_checkpoint, adaptive_strength=1.0)
    assembler = DeckMatchupEvaluator()
    draw_controller = DrawAssistController()
    assembler_candidates = _load_assembler_candidates(config.assembler_dataset)
    info_mode, assist_mode = _profile_modes(config.profile)
    observations: list[np.ndarray] = []
    action_features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    seeds: list[int] = []
    v5_started_flags: list[bool] = []
    terminal_games = 0
    total_steps = 0
    actual_games = 0
    kept_games = 0
    discarded_games = 0
    draw_assist_uses = 0

    for spec in _v5_on_policy_game_specs(
        games=int(config.games),
        seed=int(config.seed),
        focus_start_mode=config.focus_start_mode,
    ):
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
        v5_deck_ids, _assembler_score = _select_distill_v5_deck(
            config=config,
            opponent_deck_ids=v4_deck_ids,
            assembler=assembler,
            candidates=assembler_candidates,
            seed=seed,
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
        game_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int, bool]] = []

        for _step in range(int(config.max_steps)):
            current = env.current_player_id()
            if current == v5_player_id:
                obs = env.observe(current).astype(np.float32, copy=False)
                mask = env.action_mask(current).astype(np.float32, copy=False)
                features = env.action_features(current, include_preview=False).astype(np.float32, copy=False)
                action_id = int(v5_policy.select_action(env, current))
                game_rows.append(
                    (
                        obs.copy(),
                        features.copy(),
                        mask.copy(),
                        action_id,
                        seed,
                        bool(spec["v5_started"]),
                    )
                )
            else:
                action_id = int(opponent.select_action(env.env, current))

            if _action_is_end_turn(env.env._env.state, current, action_id):
                next_player_id = 2 if current == 1 else 1
                if next_player_id == v5_player_id and float(info_mode.draw_assist_strength) > 0.0:
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

        state = env.env._env.state
        v5_hp = int(state.p1.hero.hp if v5_player_id == 1 else state.p2.hero.hp)
        v4_hp = int(state.p1.hero.hp if v4_player_id == 1 else state.p2.hero.hp)
        if _keep_v5_self_imitation_game(
            winner=env.env.winner_id(),
            v5_player_id=v5_player_id,
            v5_hp=v5_hp,
            v4_hp=v4_hp,
        ):
            kept_games += 1
            for obs, features, mask, action_id, row_seed, v5_started in game_rows:
                observations.append(obs)
                action_features.append(features)
                masks.append(mask)
                actions.append(action_id)
                seeds.append(row_seed)
                v5_started_flags.append(v5_started)
        else:
            discarded_games += 1

    if not actions:
        raise RuntimeError("teacher dataset is empty")
    return {
        "observations": np.stack(observations).astype(np.float32, copy=False),
        "action_features": np.stack(action_features).astype(np.float32, copy=False),
        "masks": np.stack(masks).astype(np.float32, copy=False),
        "actions": np.asarray(actions, dtype=np.int32),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "v5_started": np.asarray(v5_started_flags, dtype=np.bool_),
        "summary": {
            "schema": "extra_lr_v5_v4max_teacher_dataset_v1",
            "collection_mode": config.collection_mode,
            "focus_start_mode": config.focus_start_mode,
            "games": int(config.games),
            "actual_games": int(actual_games),
            "kept_games": int(kept_games),
            "discarded_games": int(discarded_games),
            "terminal_games": int(terminal_games),
            "states": int(len(actions)),
            "total_steps": int(total_steps),
            "teacher_invalid_fallbacks": 0,
            "v5_started_states": int(sum(v5_started_flags)),
            "v5_second_states": int(len(v5_started_flags) - sum(v5_started_flags)),
            "draw_assist_uses": int(draw_assist_uses),
            "profile": config.profile,
            "noassist_deck_ids": list(config.noassist_deck_ids),
            "noassist_deck_pool": [list(deck) for deck in config.noassist_deck_pool],
            "info_mode": asdict(info_mode),
            "assist_mode": assist_mode.to_dict(),
        },
    }


def collect_v5_rollout_search_dataset(config: DistillConfig) -> dict[str, Any]:
    opponent = OnnxActionPolicy(str(config.v4_model), mode="argmax", seed=config.seed, verify_mask=False)
    v5_policy = V5AdaptivePolicy(config.source_checkpoint, adaptive_strength=1.0)
    assembler = DeckMatchupEvaluator()
    draw_controller = DrawAssistController()
    assembler_candidates = _load_assembler_candidates(config.assembler_dataset)
    info_mode, assist_mode = _profile_modes(config.profile)
    observations: list[np.ndarray] = []
    action_features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    seeds: list[int] = []
    v5_started_flags: list[bool] = []
    terminal_games = 0
    total_steps = 0
    actual_games = 0
    searched_states = 0
    accepted_labels = 0
    improved_labels = 0
    draw_assist_uses = 0
    score_margins: list[float] = []

    for spec in _v5_on_policy_game_specs(
        games=int(config.games),
        seed=int(config.seed),
        focus_start_mode=config.focus_start_mode,
    ):
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
        v5_deck_ids, _assembler_score = _select_distill_v5_deck(
            config=config,
            opponent_deck_ids=v4_deck_ids,
            assembler=assembler,
            candidates=assembler_candidates,
            seed=seed,
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
                baseline_action = int(v5_policy.select_action(env, current))
                legal = np.flatnonzero(mask == 1.0)
                if legal.size:
                    searched_states += 1
                    candidates = _rank_v5_policy_actions(
                        v5_policy=v5_policy,
                        obs=obs,
                        action_features=features,
                        mask=mask,
                        max_candidates=config.search_candidates,
                    )
                    if baseline_action not in candidates:
                        candidates.append(baseline_action)
                    scored = [
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
                        for action_id in candidates
                        if 0 <= int(action_id) < mask.shape[0] and mask[int(action_id)] == 1.0
                    ]
                    if scored:
                        best_action, best_score = max(scored, key=lambda item: (item[1], -item[0]))
                        baseline_score = next(
                            (score for action_id, score in scored if action_id == baseline_action),
                            _evaluate_rollout_candidate(
                                env=env,
                                candidate_action=baseline_action,
                                v5_player_id=v5_player_id,
                                v5_policy=v5_policy,
                                opponent_policy=opponent,
                                draw_controller=draw_controller,
                                draw_assist_strength=info_mode.draw_assist_strength,
                                depth_plies=config.search_depth_plies,
                            ),
                        )
                        margin = float(best_score - baseline_score)
                        if _accept_rollout_search_label(
                            best_score=best_score,
                            baseline_score=baseline_score,
                            min_margin=config.search_min_margin,
                        ):
                            observations.append(obs.copy())
                            action_features.append(features.copy())
                            masks.append(mask.copy())
                            actions.append(int(best_action))
                            seeds.append(seed)
                            v5_started_flags.append(bool(spec["v5_started"]))
                            accepted_labels += 1
                            score_margins.append(margin)
                            improved_labels += int(best_action != baseline_action)
                action_id = baseline_action
            else:
                action_id = int(opponent.select_action(env.env, current))

            if _action_is_end_turn(env.env._env.state, current, action_id):
                next_player_id = 2 if current == 1 else 1
                if next_player_id == v5_player_id and float(info_mode.draw_assist_strength) > 0.0:
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

    if not actions:
        raise RuntimeError("teacher dataset is empty")
    return {
        "observations": np.stack(observations).astype(np.float32, copy=False),
        "action_features": np.stack(action_features).astype(np.float32, copy=False),
        "masks": np.stack(masks).astype(np.float32, copy=False),
        "actions": np.asarray(actions, dtype=np.int32),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "v5_started": np.asarray(v5_started_flags, dtype=np.bool_),
        "summary": {
            "schema": "extra_lr_v5_v4max_teacher_dataset_v1",
            "collection_mode": config.collection_mode,
            "focus_start_mode": config.focus_start_mode,
            "games": int(config.games),
            "actual_games": int(actual_games),
            "terminal_games": int(terminal_games),
            "states": int(len(actions)),
            "total_steps": int(total_steps),
            "searched_states": int(searched_states),
            "accepted_labels": int(accepted_labels),
            "improved_labels": int(improved_labels),
            "avg_score_margin": float(np.mean(score_margins)) if score_margins else 0.0,
            "teacher_invalid_fallbacks": 0,
            "v5_started_states": int(sum(v5_started_flags)),
            "v5_second_states": int(len(v5_started_flags) - sum(v5_started_flags)),
            "draw_assist_uses": int(draw_assist_uses),
            "search_candidates": int(config.search_candidates),
            "search_depth_plies": int(config.search_depth_plies),
            "search_min_margin": float(config.search_min_margin),
            "profile": config.profile,
            "noassist_deck_ids": list(config.noassist_deck_ids),
            "noassist_deck_pool": [list(deck) for deck in config.noassist_deck_pool],
            "info_mode": asdict(info_mode),
            "assist_mode": assist_mode.to_dict(),
        },
    }


def _keep_v5_self_imitation_game(*, winner: int | None, v5_player_id: int, v5_hp: int, v4_hp: int) -> bool:
    if winner == int(v5_player_id):
        return True
    if winner is None:
        return int(v5_hp) > int(v4_hp)
    return int(v5_hp) > int(v4_hp) and int(v5_hp) > 0


def _accept_rollout_search_label(*, best_score: float, baseline_score: float, min_margin: float) -> bool:
    return float(best_score) >= float(baseline_score) + float(min_margin)


def _rank_v5_policy_actions(
    *,
    v5_policy: V5AdaptivePolicy,
    obs: np.ndarray,
    action_features: np.ndarray,
    mask: np.ndarray,
    max_candidates: int,
) -> list[int]:
    import mlx.core as mx

    legal = np.flatnonzero(mask == 1.0)
    if legal.size == 0:
        return []
    model_output = v5_policy.model(
        mx.array(obs[None, :].astype(np.float32, copy=False)),
        mx.array(action_features[None, :, :].astype(np.float32, copy=False)),
    )
    logits = model_output[0] if isinstance(model_output, tuple) else model_output
    mx.eval(logits)
    logits_np = np.asarray(logits, dtype=np.float32)[0]
    legal_scores = logits_np[legal]
    order = np.lexsort((legal, -legal_scores))
    limit = legal.size if int(max_candidates) <= 0 else min(int(max_candidates), legal.size)
    return [int(legal[idx]) for idx in order[:limit]]


def _evaluate_rollout_candidate(
    *,
    env: TrainV3ClassicEnv,
    candidate_action: int,
    v5_player_id: int,
    v5_policy: V5AdaptivePolicy,
    opponent_policy: OnnxActionPolicy,
    draw_controller: DrawAssistController,
    draw_assist_strength: float,
    depth_plies: int,
) -> float:
    sim = copy.deepcopy(env)
    terminated = truncated = False
    current = sim.current_player_id()
    if current != int(v5_player_id):
        return -math.inf
    mask = sim.action_mask(current)
    if int(candidate_action) < 0 or int(candidate_action) >= mask.shape[0] or mask[int(candidate_action)] != 1.0:
        return -math.inf
    terminated, truncated = _step_simulated_action(
        sim,
        int(candidate_action),
        v5_player_id=v5_player_id,
        draw_controller=draw_controller,
        draw_assist_strength=draw_assist_strength,
    )
    for _ in range(max(0, int(depth_plies))):
        if terminated or truncated:
            break
        current = sim.current_player_id()
        if current == int(v5_player_id):
            action_id = int(v5_policy.select_action(sim, current))
        else:
            action_id = int(opponent_policy.select_action(sim.env, current))
        terminated, truncated = _step_simulated_action(
            sim,
            action_id,
            v5_player_id=v5_player_id,
            draw_controller=draw_controller,
            draw_assist_strength=draw_assist_strength,
        )
    return _score_v5_rollout_state(sim, v5_player_id=int(v5_player_id))


def _step_simulated_action(
    env: TrainV3ClassicEnv,
    action_id: int,
    *,
    v5_player_id: int,
    draw_controller: DrawAssistController,
    draw_assist_strength: float,
) -> tuple[bool, bool]:
    current = env.current_player_id()
    if _action_is_end_turn(env.env._env.state, current, int(action_id)):
        next_player_id = 2 if current == 1 else 1
        if next_player_id == int(v5_player_id) and float(draw_assist_strength) > 0.0:
            _apply_draw_assist_to_player(
                env=env,
                player_id=int(v5_player_id),
                controller=draw_controller,
                strength=float(draw_assist_strength),
            )
    _obs, _reward, terminated, truncated, _info = env.step(int(action_id))
    return bool(terminated), bool(truncated)


def _score_v5_rollout_state(env: TrainV3ClassicEnv, *, v5_player_id: int) -> float:
    state = env.env._env.state
    winner = env.env.winner_id()
    v4_player_id = 2 if int(v5_player_id) == 1 else 1
    v5 = state.p1 if state.p1.user_id == int(v5_player_id) else state.p2
    v4 = state.p1 if state.p1.user_id == v4_player_id else state.p2
    hp_margin = float(v5.hero.hp - v4.hero.hp)
    own_board = sum(max(0, int(card.attack)) * max(0, int(card.hp)) for card in v5.board)
    opp_board = sum(max(0, int(card.attack)) * max(0, int(card.hp)) for card in v4.board)
    hand_margin = float(len(v5.hand) - len(v4.hand))
    if winner == int(v5_player_id):
        terminal = 1000.0
    elif winner == v4_player_id:
        terminal = -1000.0
    else:
        terminal = 0.0
    return terminal + 10.0 * hp_margin + 0.75 * float(own_board - opp_board) + 0.25 * hand_margin


def _v5_on_policy_game_specs(*, games: int, seed: int, focus_start_mode: str) -> list[dict[str, Any]]:
    if int(games) <= 0:
        raise ValueError("games must be positive")
    focus = str(focus_start_mode).strip().lower()
    if focus not in {"both", "v5_first", "v5_second"}:
        raise ValueError("focus_start_mode must be both, v5_first, or v5_second")
    specs: list[dict[str, Any]] = []
    for game_idx in range(int(games)):
        game_seed = int(seed) + game_idx
        for v5_player_id in (1, 2):
            if focus == "both":
                starting_players = (1, 2)
            elif focus == "v5_first":
                starting_players = (v5_player_id,)
            else:
                starting_players = (2 if v5_player_id == 1 else 1,)
            for starting_player_id in starting_players:
                specs.append(
                    {
                        "seed": game_seed,
                        "v5_player_id": v5_player_id,
                        "starting_player_id": int(starting_player_id),
                        "v5_started": int(starting_player_id) == int(v5_player_id),
                    }
                )
    return specs


def compute_reference_log_probs(
    model: Any,
    *,
    observations: np.ndarray,
    action_features: np.ndarray,
    masks: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    import mlx.core as mx

    n = int(observations.shape[0])
    if n <= 0:
        raise ValueError("observations must contain at least one row")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    rows: list[np.ndarray] = []
    for start in range(0, n, int(batch_size)):
        obs_b = mx.array(observations[start : start + int(batch_size)])
        features_b = mx.array(action_features[start : start + int(batch_size)])
        mask_b = mx.array(masks[start : start + int(batch_size)])
        model_output = model(obs_b, features_b)
        logits = model_output[0] if isinstance(model_output, tuple) else model_output
        masked = mx.where(mask_b.astype(mx.bool_), logits, mx.array(-1.0e9, dtype=mx.float32))
        log_probs = masked - mx.logsumexp(masked, axis=-1, keepdims=True)
        mx.eval(log_probs)
        rows.append(np.asarray(log_probs, dtype=np.float32))
    return np.concatenate(rows, axis=0).astype(np.float32, copy=False)


def train_teacher_cross_entropy(
    model: Any,
    optimizer: Any,
    *,
    observations: np.ndarray,
    action_features: np.ndarray,
    masks: np.ndarray,
    actions: np.ndarray,
    reference_log_probs: np.ndarray | None = None,
    source_kl_coef: float = 0.0,
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
    if reference_log_probs is not None and reference_log_probs.shape != masks.shape:
        raise ValueError("reference_log_probs must have the same shape as masks")
    kl_enabled = reference_log_probs is not None and float(source_kl_coef) > 0.0
    for epoch in range(int(epochs)):
        order = np.arange(n, dtype=np.int64)
        rng.shuffle(order)
        epoch_losses: list[float] = []
        epoch_accs: list[float] = []
        epoch_ce_losses: list[float] = []
        epoch_kls: list[float] = []
        for start in range(0, n, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            obs_b = mx.array(observations[idx])
            features_b = mx.array(action_features[idx])
            mask_b = mx.array(masks[idx])
            actions_b = mx.array(actions[idx], dtype=mx.int32)
            ref_log_probs_b = mx.array(reference_log_probs[idx]) if kl_enabled else None

            def loss_fn(model):
                model_output = model(obs_b, features_b)
                logits = model_output[0] if isinstance(model_output, tuple) else model_output
                masked = mx.where(mask_b.astype(mx.bool_), logits, mx.array(-1.0e9, dtype=mx.float32))
                log_probs = masked - mx.logsumexp(masked, axis=-1, keepdims=True)
                picked = log_probs[mx.arange(actions_b.shape[0]), actions_b]
                ce_loss = -mx.mean(picked)
                kl_loss = mx.array(0.0, dtype=mx.float32)
                if ref_log_probs_b is not None:
                    ref_probs = mx.exp(ref_log_probs_b) * mask_b
                    kl_per_row = mx.sum(ref_probs * (ref_log_probs_b - log_probs) * mask_b, axis=-1)
                    kl_loss = mx.mean(kl_per_row)
                loss = ce_loss + float(source_kl_coef) * kl_loss
                pred = mx.argmax(masked, axis=-1)
                accuracy = mx.mean((pred == actions_b).astype(mx.float32))
                return loss, {"accuracy": accuracy, "ce_loss": ce_loss, "source_kl": kl_loss}

            value_and_grad = nn.value_and_grad(model, loss_fn)
            (loss_value, aux), grads = value_and_grad(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss_value, *aux.values())
            epoch_losses.append(float(loss_value.item()))
            epoch_accs.append(float(aux["accuracy"].item()))
            epoch_ce_losses.append(float(aux["ce_loss"].item()))
            epoch_kls.append(float(aux["source_kl"].item()))
        metrics.append(
            {
                "epoch": float(epoch + 1),
                "loss": float(np.mean(epoch_losses)),
                "accuracy": float(np.mean(epoch_accs)),
                "ce_loss": float(np.mean(epoch_ce_losses)),
                "source_kl": float(np.mean(epoch_kls)),
            }
        )
    return {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "states": int(n),
        "source_kl_coef": float(source_kl_coef),
        "epoch_metrics": metrics,
        "final_loss": float(metrics[-1]["loss"]),
        "final_accuracy": float(metrics[-1]["accuracy"]),
        "final_ce_loss": float(metrics[-1]["ce_loss"]),
        "final_source_kl": float(metrics[-1]["source_kl"]),
    }


def _profile_modes(profile: str) -> tuple[InfoModeV5, AssistModeV5]:
    profile = profile.strip().lower()
    if profile == "strong":
        return (
            InfoModeV5(
                adaptive_strength=1.0,
                own_hand_identity_known=True,
                own_deck_known=True,
                enemy_hand_known=True,
                enemy_deck_known=True,
                enemy_deck_order_known=True,
                draw_assist_enabled=True,
                draw_assist_strength=1.0,
            ),
            AssistModeV5(
                assembler_enabled=True,
                assembler_strength=1.0,
                desirerer_enabled=True,
                desirerer_strength=1.0,
                teacher_hint_available=True,
                assist_profile_id=15,
            ),
        )
    if profile == "noassist":
        return (
            InfoModeV5(
                adaptive_strength=1.0,
                own_hand_identity_known=True,
                own_deck_known=True,
                enemy_hand_known=False,
                enemy_deck_known=False,
                enemy_deck_order_known=False,
                draw_assist_enabled=False,
                draw_assist_strength=0.0,
            ),
            AssistModeV5(),
        )
    raise ValueError("profile must be strong or noassist")


def _assembler_dataset_for_profile(*, profile: str, assembler_dataset: Path | None) -> Path | None:
    if str(profile).strip().lower() == "noassist":
        return None
    return assembler_dataset.resolve() if assembler_dataset is not None else None


def _select_distill_v5_deck(
    *,
    config: DistillConfig,
    opponent_deck_ids: list[int],
    assembler: DeckMatchupEvaluator,
    candidates: list[Any],
    seed: int | None = None,
) -> tuple[list[int], float]:
    if str(config.profile).strip().lower() == "noassist":
        deck = list(_select_noassist_distill_deck(config, seed=seed))
        return deck, float(assembler.score_candidate([], deck))
    return _select_v5_deck(opponent_deck_ids=opponent_deck_ids, assembler=assembler, candidates=candidates)


def _select_noassist_distill_deck(config: DistillConfig, *, seed: int | None = None) -> tuple[int, ...]:
    if str(config.profile).strip().lower() != "noassist" or not config.noassist_deck_pool:
        return tuple(config.noassist_deck_ids)
    pool = tuple(tuple(deck) for deck in config.noassist_deck_pool)
    idx = int(seed or 0) % len(pool)
    return pool[idx]


def _validate_config(config: DistillConfig) -> None:
    if not config.source_checkpoint.exists():
        raise FileNotFoundError(f"source checkpoint not found: {config.source_checkpoint}")
    if not config.v4_model.exists():
        raise FileNotFoundError(f"V4 model not found: {config.v4_model}")
    if int(config.games) <= 0:
        raise ValueError("games must be positive")
    if int(config.max_steps) <= 0:
        raise ValueError("max_steps must be positive")
    if int(config.batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(config.epochs) <= 0:
        raise ValueError("epochs must be positive")
    if float(config.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.collection_mode not in {"v4_selfplay", "v5_on_policy", "v5_success_imitation", "v5_rollout_search"}:
        raise ValueError(
            "collection_mode must be v4_selfplay, v5_on_policy, v5_success_imitation, or v5_rollout_search"
        )
    if config.focus_start_mode not in {"both", "v5_first", "v5_second"}:
        raise ValueError("focus_start_mode must be both, v5_first, or v5_second")
    if (
        config.collection_mode in {"v5_on_policy", "v5_success_imitation", "v5_rollout_search"}
        and config.assembler_dataset is not None
    ):
        if not config.assembler_dataset.exists():
            raise FileNotFoundError(f"assembler dataset not found: {config.assembler_dataset}")
    if config.teacher_dataset_path is not None and not config.teacher_dataset_path.exists():
        raise FileNotFoundError(f"teacher dataset not found: {config.teacher_dataset_path}")
    if int(config.search_candidates) < 0:
        raise ValueError("search_candidates must be non-negative")
    if int(config.search_depth_plies) < 0:
        raise ValueError("search_depth_plies must be non-negative")
    if float(config.search_min_margin) < 0.0:
        raise ValueError("search_min_margin must be non-negative")
    if not math.isfinite(float(config.source_kl_coef)) or float(config.source_kl_coef) < 0.0:
        raise ValueError("source_kl_coef must be non-negative")
    if len(tuple(config.noassist_deck_ids)) < 2:
        raise ValueError("noassist_deck_ids must include a hero and at least one card")
    for deck in tuple(config.noassist_deck_pool):
        if len(tuple(deck)) < 2:
            raise ValueError("every noassist_deck_pool deck must include a hero and at least one card")


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
    parser = argparse.ArgumentParser(description="Run offline V4-max teacher distillation for V5")
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--v4-model", type=Path, default=DEFAULT_V4_MAX)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "TrainV3.5" / "runs" / f"phase10_v4max_distill_{stamp}")
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20301000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=8.0e-5)
    parser.add_argument("--profile", choices=["strong", "noassist"], default="strong")
    parser.add_argument(
        "--collection-mode",
        choices=["v4_selfplay", "v5_on_policy", "v5_success_imitation", "v5_rollout_search"],
        default="v4_selfplay",
    )
    parser.add_argument("--focus-start-mode", choices=["both", "v5_first", "v5_second"], default="both")
    parser.add_argument("--assembler-dataset", type=Path, default=DEFAULT_ASSEMBLER_DATASET)
    parser.add_argument("--noassist-deck-ids", default=",".join(str(card_id) for card_id in NOASSIST_BASELINE_DECK_IDS))
    parser.add_argument(
        "--noassist-deck-pool",
        default="",
        help="Semicolon-separated static no-assist decks, each as comma-separated card ids.",
    )
    parser.add_argument("--search-candidates", type=int, default=8)
    parser.add_argument("--search-depth-plies", type=int, default=6)
    parser.add_argument("--search-min-margin", type=float, default=0.25)
    parser.add_argument("--source-kl-coef", type=float, default=0.0)
    parser.add_argument("--teacher-dataset-path", type=Path, default=None)
    parser.add_argument("--save-dataset", action="store_true")
    parser.add_argument("--restore-optimizer", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

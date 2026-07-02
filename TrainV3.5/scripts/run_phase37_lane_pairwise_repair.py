#!/usr/bin/env python3
"""Targeted no-assist pairwise repair against exploit lanes.

This is an offline label-mining phase. It does not replace Rust rollout/PPO hot
paths; it mines a compact set of hard states with the Python oracle, applies a
conservative pairwise/KL update, and leaves promotion to paired external guard
benches.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
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
from run_phase10_v4max_distill import (  # noqa: E402
    DEFAULT_V4_MAX,
    _jsonable,
    _profile_modes,
    _rank_v5_policy_actions,
    _score_v5_rollout_state,
    _step_simulated_action,
    _v5_on_policy_game_specs,
)
from run_phase19_noassist_conservative_second_start import (  # noqa: E402
    Phase19Config,
    collect_noassist_anchor_dataset,
    train_conservative_pairwise_kl,
)
from run_phase1_runtime_acceptance_bench import (  # noqa: E402
    LANES as PHASE1_LANES,
    select_lane_action,
)
from run_v5_vs_v4max_benchmark import NOASSIST_BASELINE_DECK_IDS, V5AdaptivePolicy, _parse_deck_ids  # noqa: E402
from train_v3.aux_models import DrawAssistController  # noqa: E402
from train_v3.env_v5 import TrainV3ClassicEnv, TrainV3EnvConfig  # noqa: E402
from train_v3.v5_policy import create_v5_policy  # noqa: E402


DEFAULT_BASE_CHECKPOINT = (
    ROOT
    / "TrainV3.5"
    / "runs"
    / "phase34c_pairwise_paired_accept_20260611_202032"
    / "best_checkpoint.npz"
)
DEFAULT_TARGET_LANES = ("greedy_trade", "anti_hand_leak_overfit", "anti_draw_greed", "board_control")


@dataclass(frozen=True)
class Phase37Config:
    base_checkpoint: Path
    output_dir: Path
    lanes: tuple[str, ...]
    games_per_lane: int
    anchor_games: int
    max_steps: int
    seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    noassist_deck_ids: tuple[int, ...] = NOASSIST_BASELINE_DECK_IDS
    focus_start_mode: str = "both"
    search_candidates: int = 12
    search_depth_plies: int = 8
    min_pairwise_margin: float = 0.25
    ranking_margin: float = 0.5
    pairwise_coef: float = 0.35
    kl_coef: float = 4.0
    anchor_kl_coef: float = 6.0
    min_pairs: int = 24
    hidden_dim: int = 256
    action_hidden_dim: int = 128
    dataset_path: Path | None = None
    save_dataset: bool = True


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = Phase37Config(
        base_checkpoint=args.base_checkpoint.resolve(),
        output_dir=args.output_dir.resolve(),
        lanes=_parse_lanes(args.lanes),
        games_per_lane=int(args.games_per_lane),
        anchor_games=int(args.anchor_games),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        noassist_deck_ids=_parse_deck_ids(args.noassist_deck_ids),
        focus_start_mode=str(args.focus_start_mode),
        search_candidates=int(args.search_candidates),
        search_depth_plies=int(args.search_depth_plies),
        min_pairwise_margin=float(args.min_pairwise_margin),
        ranking_margin=float(args.ranking_margin),
        pairwise_coef=float(args.pairwise_coef),
        kl_coef=float(args.kl_coef),
        anchor_kl_coef=float(args.anchor_kl_coef),
        min_pairs=int(args.min_pairs),
        hidden_dim=int(args.hidden_dim),
        action_hidden_dim=int(args.action_hidden_dim),
        dataset_path=args.dataset_path.resolve() if args.dataset_path is not None else None,
        save_dataset=not bool(args.no_save_dataset),
    )
    result = run_phase37(config)
    print("PHASE37_LANE_PAIRWISE_RESULT", json.dumps(result["summary"], sort_keys=True), flush=True)
    print(f"Saved: {result['checkpoint_path']}", flush=True)
    return 0


def run_phase37(config: Phase37Config) -> dict[str, Any]:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_phase37_dataset_npz(config.dataset_path) if config.dataset_path is not None else collect_phase37_dataset(config)
    if int(dataset["summary"]["pairs"]) < int(config.min_pairs):
        raise RuntimeError(f"pairs={dataset['summary']['pairs']} below min_pairs={config.min_pairs}")
    if config.save_dataset:
        np.savez_compressed(
            config.output_dir / "phase37_lane_pairwise_dataset.npz",
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
            lane_names=dataset["lane_names"],
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
        seed=config.seed + 3701,
    )

    checkpoint_path = config.output_dir / f"extra_lr_v5_phase37_lane_pairwise_{int(dataset['summary']['pairs'])}_pairs.npz"
    metadata = {
        "run_name": "phase37_lane_pairwise_repair",
        "model_name": "extra-lr-v5-adaptive",
        "phase": "phase37_lane_pairwise_repair",
        "source_checkpoint": str(config.base_checkpoint),
        "source_metadata": loaded.get("metadata", {}),
        "obs_dim": 6480,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "config": _jsonable(asdict(config)),
        "dataset": dataset["summary"],
        "train_summary": train_summary,
        "profile": "noassist",
        "assist_or_submodel_used": False,
        "v4_1_included": False,
        "offline_lane_label_mining": True,
        "loss": "lane_pairwise_margin_plus_reference_kl",
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
            "lane_counts": dataset["summary"]["lane_counts"],
            "final_loss": float(train_summary["final_loss"]),
            "final_policy_kl": float(train_summary["final_policy_kl"]),
            "final_anchor_kl": float(train_summary["final_anchor_kl"]),
        },
    }
    (config.output_dir / "phase37_summary.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def collect_phase37_dataset(config: Phase37Config) -> dict[str, Any]:
    pairwise = collect_lane_pairwise_dataset(config)
    anchors = collect_noassist_anchor_dataset(
        Phase19Config(
            base_checkpoint=config.base_checkpoint,
            v4_model=DEFAULT_V4_MAX,
            output_dir=config.output_dir / "anchor",
            games=1,
            anchor_games=config.anchor_games,
            max_steps=config.max_steps,
            seed=config.seed + 900_000,
            batch_size=config.batch_size,
            epochs=1,
            learning_rate=config.learning_rate,
            noassist_deck_ids=config.noassist_deck_ids,
            min_pairs=1,
            hidden_dim=config.hidden_dim,
            action_hidden_dim=config.action_hidden_dim,
            save_dataset=False,
        )
    )
    return {**pairwise, **anchors, "summary": {**pairwise["summary"], **anchors["summary"]}}


def load_phase37_dataset_npz(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        raise FileNotFoundError(f"phase37 dataset not found: {path}")
    loaded = np.load(Path(path), allow_pickle=True)
    required = (
        "observations",
        "action_features",
        "masks",
        "positive_actions",
        "negative_actions",
        "score_margins",
        "anchor_observations",
        "anchor_action_features",
        "anchor_masks",
        "seeds",
        "lane_names",
    )
    missing = [name for name in required if name not in loaded.files]
    if missing:
        raise ValueError(f"phase37 dataset missing required arrays: {missing}")
    observations = np.asarray(loaded["observations"], dtype=np.float32)
    action_features = np.asarray(loaded["action_features"], dtype=np.float32)
    masks = np.asarray(loaded["masks"], dtype=np.float32)
    positive_actions = np.asarray(loaded["positive_actions"], dtype=np.int32)
    negative_actions = np.asarray(loaded["negative_actions"], dtype=np.int32)
    score_margins = np.asarray(loaded["score_margins"], dtype=np.float32)
    anchor_observations = np.asarray(loaded["anchor_observations"], dtype=np.float32)
    anchor_action_features = np.asarray(loaded["anchor_action_features"], dtype=np.float32)
    anchor_masks = np.asarray(loaded["anchor_masks"], dtype=np.float32)
    seeds = np.asarray(loaded["seeds"], dtype=np.int64)
    lane_names = np.asarray(loaded["lane_names"]).astype(str)
    n = int(positive_actions.shape[0])
    if n <= 0:
        raise ValueError("phase37 dataset must contain at least one pair")
    for name, array in (
        ("observations", observations),
        ("action_features", action_features),
        ("masks", masks),
        ("negative_actions", negative_actions),
        ("score_margins", score_margins),
        ("seeds", seeds),
        ("lane_names", lane_names),
    ):
        if int(array.shape[0]) != n:
            raise ValueError(f"{name} must have first dimension {n}")
    lane_counts = {str(lane): int(np.sum(lane_names == lane)) for lane in sorted(set(lane_names.tolist()))}
    return {
        "observations": observations,
        "action_features": action_features,
        "masks": masks,
        "positive_actions": positive_actions,
        "negative_actions": negative_actions,
        "score_margins": score_margins,
        "anchor_observations": anchor_observations,
        "anchor_action_features": anchor_action_features,
        "anchor_masks": anchor_masks,
        "seeds": seeds,
        "lane_names": lane_names,
        "summary": {
            "schema": "extra_lr_v5_phase37_lane_pairwise_dataset_v1",
            "collection_mode": "loaded_phase37_lane_pairwise_dataset",
            "dataset_path": str(Path(path)),
            "pairs": n,
            "lane_counts": lane_counts,
            "lanes": sorted(lane_counts),
            "avg_score_margin": float(np.mean(score_margins)),
            "min_score_margin": float(np.min(score_margins)),
            "max_score_margin": float(np.max(score_margins)),
            "anchor_states": int(anchor_observations.shape[0]),
            "profile": "noassist",
        },
    }


def collect_lane_pairwise_dataset(config: Phase37Config) -> dict[str, Any]:
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
    lane_names: list[str] = []

    lane_counts: dict[str, int] = {lane: 0 for lane in config.lanes}
    actual_games = 0
    terminal_games = 0
    total_steps = 0
    v5_states = 0
    searched_states = 0
    candidate_evals = 0

    for lane_idx, lane in enumerate(config.lanes):
        specs = _v5_on_policy_game_specs(
            games=config.games_per_lane,
            seed=config.seed + lane_idx * 100_000,
            focus_start_mode=config.focus_start_mode,
        )
        for game_idx, spec in enumerate(specs, start=1):
            seed = int(spec["seed"])
            v5_player_id = int(spec["v5_player_id"])
            starting_player_id = int(spec["starting_player_id"])
            env = _new_noassist_env(seed=seed, starting_player_id=starting_player_id, info_mode=info_mode, assist_mode=assist_mode)
            env.reset(
                p1_deck_ids=list(config.noassist_deck_ids),
                p2_deck_ids=list(config.noassist_deck_ids),
                p1_is_bot=True,
                p2_is_bot=True,
                starting_player_id=starting_player_id,
                seed=seed,
            )
            v5_policy.reset(seed * 11 + v5_player_id)
            actual_games += 1

            for step in range(1, int(config.max_steps) + 1):
                current = env.current_player_id()
                if current == v5_player_id:
                    obs = env.observe(current).astype(np.float32, copy=False)
                    mask = env.action_mask(current).astype(np.float32, copy=False)
                    features = env.action_features(current, include_preview=False).astype(np.float32, copy=False)
                    baseline_action = int(v5_policy.select_action(env, current))
                    v5_states += 1
                    candidates = _rank_v5_policy_actions(
                        v5_policy=v5_policy,
                        obs=obs,
                        action_features=features,
                        mask=mask,
                        max_candidates=config.search_candidates,
                    )
                    if baseline_action not in candidates:
                        candidates.append(baseline_action)
                    scored = []
                    for action_id in candidates:
                        if 0 <= int(action_id) < mask.shape[0] and mask[int(action_id)] == 1.0:
                            score = _evaluate_lane_rollout_candidate(
                                env=env,
                                candidate_action=int(action_id),
                                v5_player_id=v5_player_id,
                                v5_policy=v5_policy,
                                lane=lane,
                                seed=seed,
                                step=step,
                                draw_controller=draw_controller,
                                depth_plies=config.search_depth_plies,
                            )
                            scored.append((int(action_id), float(score)))
                    candidate_evals += len(scored)
                    if scored:
                        searched_states += 1
                        base_score = next((score for action_id, score in scored if action_id == baseline_action), None)
                        if base_score is not None:
                            best_action, best_score = max(scored, key=lambda item: (item[1], -item[0]))
                            margin = float(best_score - base_score)
                            if best_action != baseline_action and margin >= float(config.min_pairwise_margin):
                                observations.append(obs.copy())
                                action_features.append(features.copy())
                                masks.append(mask.copy())
                                positive_actions.append(int(best_action))
                                negative_actions.append(int(baseline_action))
                                score_margins.append(margin)
                                seeds.append(seed)
                                lane_names.append(lane)
                                lane_counts[lane] += 1
                    action_id = baseline_action
                else:
                    action_id = select_lane_action(
                        lane,
                        env,
                        current,
                        seed=seed,
                        step=step,
                        exploit_policy="state_score",
                        legal_source="classic_legal_action_ids",
                        rng=random.Random(seed * 1000 + step),
                    )
                _obs, _reward, terminated, truncated, _info = env.step(int(action_id))
                total_steps += 1
                if terminated or truncated:
                    terminal_games += 1
                    break
            if game_idx % 64 == 0:
                print(f"phase37_collect lane={lane} games={game_idx}/{len(specs)} pairs={sum(lane_counts.values())}", flush=True)

    if not positive_actions:
        raise RuntimeError("phase37 pairwise dataset is empty")
    margins_np = np.asarray(score_margins, dtype=np.float32)
    return {
        "observations": np.stack(observations).astype(np.float32, copy=False),
        "action_features": np.stack(action_features).astype(np.float32, copy=False),
        "masks": np.stack(masks).astype(np.float32, copy=False),
        "positive_actions": np.asarray(positive_actions, dtype=np.int32),
        "negative_actions": np.asarray(negative_actions, dtype=np.int32),
        "score_margins": margins_np,
        "seeds": np.asarray(seeds, dtype=np.int64),
        "lane_names": np.asarray(lane_names),
        "summary": {
            "schema": "extra_lr_v5_phase37_lane_pairwise_dataset_v1",
            "collection_mode": "lane_pairwise_rollout_search",
            "profile": "noassist",
            "lanes": list(config.lanes),
            "lane_counts": lane_counts,
            "games_per_lane": int(config.games_per_lane),
            "actual_games": int(actual_games),
            "terminal_games": int(terminal_games),
            "pairs": int(len(positive_actions)),
            "total_steps": int(total_steps),
            "v5_states": int(v5_states),
            "searched_states": int(searched_states),
            "candidate_evals": int(candidate_evals),
            "avg_score_margin": float(np.mean(margins_np)),
            "min_score_margin": float(np.min(margins_np)),
            "max_score_margin": float(np.max(margins_np)),
            "search_candidates": int(config.search_candidates),
            "search_depth_plies": int(config.search_depth_plies),
            "min_pairwise_margin": float(config.min_pairwise_margin),
            "info_mode": asdict(info_mode),
            "assist_mode": assist_mode.to_dict(),
        },
    }


def _evaluate_lane_rollout_candidate(
    *,
    env: TrainV3ClassicEnv,
    candidate_action: int,
    v5_player_id: int,
    v5_policy: V5AdaptivePolicy,
    lane: str,
    seed: int,
    step: int,
    draw_controller: DrawAssistController,
    depth_plies: int,
) -> float:
    sim = copy.deepcopy(env)
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
        draw_assist_strength=0.0,
    )
    for ply in range(max(0, int(depth_plies))):
        if terminated or truncated:
            break
        current = sim.current_player_id()
        if current == int(v5_player_id):
            action_id = int(v5_policy.select_action(sim, current))
        else:
            action_id = select_lane_action(
                lane,
                sim,
                current,
                seed=int(seed),
                step=int(step) + ply + 1,
                exploit_policy="state_score",
                legal_source="classic_legal_action_ids",
                rng=random.Random(int(seed) * 1000 + int(step) + ply),
            )
        terminated, truncated = _step_simulated_action(
            sim,
            int(action_id),
            v5_player_id=v5_player_id,
            draw_controller=draw_controller,
            draw_assist_strength=0.0,
        )
    return _score_v5_rollout_state(sim, v5_player_id=int(v5_player_id))


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


def _parse_lanes(raw: str) -> tuple[str, ...]:
    lanes = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not lanes:
        raise ValueError("lanes must not be empty")
    allowed = set(PHASE1_LANES) - {"legal_random", "stall"}
    unknown = [lane for lane in lanes if lane not in allowed]
    if unknown:
        raise ValueError(f"unknown or unsupported phase37 lanes: {unknown}; allowed={sorted(allowed)}")
    return lanes


def _validate_config(config: Phase37Config) -> None:
    if not config.base_checkpoint.exists():
        raise FileNotFoundError(f"base checkpoint not found: {config.base_checkpoint}")
    if "v4.1" in str(config.base_checkpoint).lower():
        raise ValueError("V4.1 checkpoints must not be used for Phase37")
    if config.focus_start_mode not in {"both", "v5_first", "v5_second"}:
        raise ValueError("focus_start_mode must be both, v5_first, or v5_second")
    if config.dataset_path is not None and not config.dataset_path.exists():
        raise FileNotFoundError(f"phase37 dataset not found: {config.dataset_path}")
    for name in (
        "games_per_lane",
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
    for name in ("learning_rate", "min_pairwise_margin", "ranking_margin", "pairwise_coef", "kl_coef", "anchor_kl_coef"):
        if float(getattr(config, name)) <= 0.0:
            raise ValueError(f"{name} must be positive")
    if len(tuple(config.noassist_deck_ids)) < 2:
        raise ValueError("noassist_deck_ids must include a hero and at least one card")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "TrainV3.5" / "runs" / f"phase37_lane_pairwise_{stamp}")
    parser.add_argument("--lanes", default=",".join(DEFAULT_TARGET_LANES))
    parser.add_argument("--games-per-lane", type=int, default=24)
    parser.add_argument("--anchor-games", type=int, default=48)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=37001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=6.0e-6)
    parser.add_argument("--noassist-deck-ids", default=",".join(str(card_id) for card_id in NOASSIST_BASELINE_DECK_IDS))
    parser.add_argument("--focus-start-mode", choices=["both", "v5_first", "v5_second"], default="both")
    parser.add_argument("--search-candidates", type=int, default=12)
    parser.add_argument("--search-depth-plies", type=int, default=8)
    parser.add_argument("--min-pairwise-margin", type=float, default=0.25)
    parser.add_argument("--ranking-margin", type=float, default=0.5)
    parser.add_argument("--pairwise-coef", type=float, default=0.35)
    parser.add_argument("--kl-coef", type=float, default=4.0)
    parser.add_argument("--anchor-kl-coef", type=float, default=6.0)
    parser.add_argument("--min-pairs", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--action-hidden-dim", type=int, default=128)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--no-save-dataset", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

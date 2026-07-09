#!/usr/bin/env python3
"""First-pass V5 adaptive-strength 1.0 assist benchmark against V4 max.

This is an intentionally narrow benchmark: V4.1 is not touched, V4-max is the
only opponent, and V5 is evaluated in its strongest currently wired mode.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAINV3_PYTHON = ROOT / "TrainV3.5" / "python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAINV3_PYTHON) not in sys.path:
    sys.path.insert(0, str(TRAINV3_PYTHON))

from ai.train_v2.classic_actions_v1 import decode_action  # noqa: E402
from ai.train_v2.onnx_policy import OnnxActionPolicy  # noqa: E402
from ai.train_v2.model_mlx import load_checkpoint  # noqa: E402
from core.actions import ManaDrawAction  # noqa: E402
from train_v3.aux_models import (  # noqa: E402
    AssemblerCandidate,
    DeckMatchupEvaluator,
    DrawAssistController,
    load_assembler_dataset,
)
from train_v3.contracts import AssistModeV5, InfoModeV5  # noqa: E402
from train_v3.env_v5 import TrainV3ClassicEnv, TrainV3EnvConfig  # noqa: E402
from train_v3.mana_draw_head_v5 import mana_draw_legal_mask, select_includes_mana_draw  # noqa: E402
from train_v3.v5_policy import create_v5_policy  # noqa: E402


DEFAULT_V4_MAX = ROOT / "ai" / "models" / "extra-lr-v4-max.onnx"
DEFAULT_V5_CHECKPOINT = (
    ROOT
    / "TrainV3.5"
    / "runs"
    / "phase6_noassist_entropy_recovery_20260607_112603"
    / "checkpoints"
    / "trainv3_rust_legal_update_0300.npz"
)
DEFAULT_ASSEMBLER_DATASET = (
    ROOT
    / "TrainV3.5"
    / "runs"
    / "phase5_aux_models_from_phase4_20260606_231349"
    / "aux"
    / "assembler.jsonl"
)
HERO_CARD_IDS = frozenset({1, 3, 4, 5, 6, 7})
NOASSIST_BASELINE_DECK_IDS = (1, 37, 38, 40, 41, 42, 27, 28, 29)


@dataclass(frozen=True)
class BenchmarkConfig:
    v4_model_path: Path
    v5_checkpoint_path: Path
    assembler_dataset_path: Path | None
    output_dir: Path
    games: int
    seed: int
    max_steps: int
    start_mode: str = "both"
    adaptive_strength: float = 1.0
    draw_assist_strength: float = 1.0
    assembler_strength: float = 1.0
    desirerer_strength: float = 1.0
    private_info_enabled: bool = True
    draw_assist_enabled: bool = True
    assist_mode_enabled: bool = True
    deck_assist_enabled: bool = True
    noassist_deck_ids: tuple[int, ...] = NOASSIST_BASELINE_DECK_IDS
    second_start_search: bool = False
    search_candidates: int = 4
    search_depth_plies: int = 4
    adaptive_strength_runtime: bool = False
    search_hp_weight: float = 10.0
    search_board_power_weight: float = 0.75
    search_hand_weight: float = 0.25
    search_attack_weight: float = 0.0
    search_board_count_weight: float = 0.0
    search_empty_board_penalty: float = 0.0
    search_board_disadvantage_penalty: float = 0.0
    search_board_disadvantage_ratio: float = 0.7
    search_rollout_empty_board_penalty: float = 0.0
    search_rollout_board_disadvantage_penalty: float = 0.0
    search_rollout_board_disadvantage_ratio: float = 0.7
    recovery_reranker_path: Path | None = None
    recovery_reranker_weight: float = 0.0
    recovery_bias_clip: float = 4.0
    recovery_gated: bool = False
    recovery_hp_deficit_threshold: int = 4
    recovery_board_power_ratio_threshold: float = 0.5
    recovery_empty_board_gate: bool = False
    battle_log_path: Path | None = None
    log_lost_only: bool = True
    log_max_battles: int = 0


class V5AdaptivePolicy:
    name = "extra-lr-v5-adaptive-s1-assist"

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        adaptive_strength: float = 1.0,
        recovery_reranker_path: Path | None = None,
        recovery_reranker_weight: float = 0.0,
        recovery_bias_clip: float = 4.0,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.model = create_v5_policy(policy_kind="v5_split_encoder", hidden_dim=256, action_hidden_dim=128)
        self.loaded = load_checkpoint(str(self.checkpoint_path), self.model)
        self.adaptive_strength = float(adaptive_strength)
        self.invalid_fallbacks = 0
        self.recovery_reranker_path = Path(recovery_reranker_path) if recovery_reranker_path is not None else None
        self.recovery_reranker_weight = float(recovery_reranker_weight)
        self.recovery_bias_clip = float(recovery_bias_clip)
        self.recovery_model = None
        self.recovery_loaded: dict[str, Any] | None = None
        self.recovery_rerank_uses = 0
        if self.recovery_reranker_path is not None:
            self.recovery_model = create_v5_policy(policy_kind="v5_split_encoder", hidden_dim=256, action_hidden_dim=128)
            self.recovery_loaded = load_checkpoint(str(self.recovery_reranker_path), self.recovery_model)

    def reset(self, _seed: int) -> None:
        self.invalid_fallbacks = 0
        self.recovery_rerank_uses = 0

    def select_action(self, env: TrainV3ClassicEnv, player_id: int, *, recovery_active: bool = False) -> Any:
        import mlx.core as mx

        obs = env.observe(player_id).astype(np.float32)
        mask = env.action_mask(player_id).astype(np.float32)
        action_features = env.action_features(player_id, include_preview=False).astype(np.float32)
        md_legal = mana_draw_legal_mask(env.env._env.state, player_id)
        output = self.model(
            mx.array(obs[None, :]),
            mx.array(action_features[None, :, :]),
            mana_draw_legal=mx.array([md_legal]),
        )
        logits = output[0] if isinstance(output, tuple) else output
        mana_draw_logit = output[2] if isinstance(output, tuple) and len(output) >= 3 else None
        if mana_draw_logit is not None:
            mx.eval(logits, mana_draw_logit)
        else:
            mx.eval(logits)
        logits_np = np.array(logits, dtype=np.float32)[0]
        base_masked = np.where(mask.astype(bool), logits_np, -1e9)
        base_action = int(np.argmax(base_masked))
        if recovery_active and self.recovery_model is not None and self.recovery_reranker_weight > 0.0:
            recovery_output = self.recovery_model(
                mx.array(obs[None, :]),
                mx.array(action_features[None, :, :]),
            )
            recovery_logits = recovery_output[0] if isinstance(recovery_output, tuple) else recovery_output
            mx.eval(recovery_logits)
            logits_np = _combine_base_and_recovery_logits(
                base_logits=logits_np,
                recovery_logits=np.array(recovery_logits, dtype=np.float32)[0],
                mask=mask,
                weight=self.recovery_reranker_weight,
                bias_clip=self.recovery_bias_clip,
            )
        masked = np.where(mask.astype(bool), logits_np, -1e9)
        action_id = int(np.argmax(masked))
        self.recovery_rerank_uses += int(recovery_active and action_id != base_action)
        if mask[action_id] != 1.0:
            self.invalid_fallbacks += 1
            legal = np.flatnonzero(mask == 1.0)
            return int(legal[0]) if len(legal) else 0
        if mana_draw_logit is not None and select_includes_mana_draw(
            float(np.array(mana_draw_logit, dtype=np.float32).reshape(-1)[0]),
            float(masked[action_id]),
            bool(md_legal),
        ):
            return ManaDrawAction()
        return action_id


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    v5_policy = V5AdaptivePolicy(
        config.v5_checkpoint_path,
        adaptive_strength=config.adaptive_strength,
        recovery_reranker_path=config.recovery_reranker_path,
        recovery_reranker_weight=config.recovery_reranker_weight,
        recovery_bias_clip=config.recovery_bias_clip,
    )
    v4_policy = OnnxActionPolicy(str(config.v4_model_path), mode="argmax", seed=config.seed, verify_mask=False)
    assembler = DeckMatchupEvaluator()
    draw_controller = DrawAssistController()
    assembler_candidates = _assembler_candidates_for_config(
        config,
        _load_assembler_candidates(config.assembler_dataset_path),
    )

    games: list[dict[str, Any]] = []
    logged_battles = 0
    seeds = list(range(config.seed, config.seed + config.games))
    for idx, seed in enumerate(seeds, start=1):
        for v5_player_id in (1, 2):
            for starting_player_id in _starting_players(config.start_mode):
                print(
                    f"[{idx}/{len(seeds)}] seed={seed} v5_as=p{v5_player_id} start=p{starting_player_id}",
                    flush=True,
                )
                game = _run_game(
                    seed=seed,
                    max_steps=config.max_steps,
                    v5_player_id=v5_player_id,
                    starting_player_id=starting_player_id,
                    v5_policy=v5_policy,
                    v4_policy=v4_policy,
                    assembler=assembler,
                    assembler_candidates=assembler_candidates,
                    draw_controller=draw_controller,
                    config=config,
                )
                battle_log = game.pop("_battle_log", None)
                games.append(game)
                if _should_write_battle_log(config, game, logged_battles) and battle_log is not None:
                    _append_battle_log(config.battle_log_path, battle_log)
                    logged_battles += 1

    result = {
        "schema": "extra_lr_v5_vs_v4max_benchmark_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": _config_to_json(config),
        "v5_policy_metadata": v5_policy.loaded.get("metadata", {}),
        "modes": {
            "info_mode": asdict(_strong_info_mode(config)),
            "assist_mode": _strong_assist_mode(config).to_dict(),
            "v4_1_included": False,
            "second_start_search": bool(config.second_start_search),
            "search_candidates": int(config.search_candidates),
            "search_depth_plies": int(config.search_depth_plies),
            "adaptive_strength_runtime": bool(config.adaptive_strength_runtime),
            "second_start_recovery_reranker": {
                "enabled": config.recovery_reranker_path is not None and config.recovery_reranker_weight > 0.0,
                "path": str(config.recovery_reranker_path) if config.recovery_reranker_path is not None else None,
                "weight": float(config.recovery_reranker_weight),
                "bias_clip": float(config.recovery_bias_clip),
                "activation": (
                    "v5_started_false_and_gate_passes" if config.recovery_gated else "v5_started_false"
                ),
                "gating": _recovery_gating_config_to_json(config),
                "metadata": v5_policy.recovery_loaded.get("metadata", {}) if v5_policy.recovery_loaded else {},
            },
        },
        "games": games,
        "summary": _summarize(games),
    }
    _write_outputs(result, config.output_dir / "v5_s1_assist_vs_v4max.json")
    return result


def _run_game(
    *,
    seed: int,
    max_steps: int,
    v5_player_id: int,
    starting_player_id: int,
    v5_policy: V5AdaptivePolicy,
    v4_policy: OnnxActionPolicy,
    assembler: DeckMatchupEvaluator,
    assembler_candidates: list[AssemblerCandidate],
    draw_controller: DrawAssistController,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    v4_player_id = 2 if v5_player_id == 1 else 1
    env = TrainV3ClassicEnv(
        TrainV3EnvConfig(
            seed=seed,
            verify_mask=False,
            placement_mode="append_only",
            include_legal_actions_in_info=False,
            info_mode=_strong_info_mode(config),
            assist_mode=_strong_assist_mode(config),
        )
    )
    base_env = env.env
    base_env.reset(seed=seed, starting_player_id=starting_player_id)
    v4_deck_ids = _player_card_pool_ids(base_env._env.state, v4_player_id)
    v5_deck_ids, assembler_score = _select_v5_deck_for_config(
        config=config,
        opponent_deck_ids=v4_deck_ids,
        assembler=assembler,
        candidates=assembler_candidates,
    )
    p1_deck_ids = v5_deck_ids if v5_player_id == 1 else v4_deck_ids
    p2_deck_ids = v5_deck_ids if v5_player_id == 2 else v4_deck_ids

    env.reset(
        p1_deck_ids=p1_deck_ids,
        p2_deck_ids=p2_deck_ids,
        p1_is_bot=True,
        p2_is_bot=True,
        starting_player_id=starting_player_id,
        seed=seed,
    )
    v5_policy.reset(seed * 11 + v5_player_id)
    v4_policy.reset(seed * 13 + v4_player_id)

    invalid = 0
    draw_assist_uses = 0
    draw_assist_ranked = 0
    search_rerank_uses = 0
    steps = 0
    events: list[dict[str, Any]] = []
    for steps in range(1, max_steps + 1):
        current = env.current_player_id()
        if current == v5_player_id:
            action_id = v5_policy.select_action(
                env,
                current,
                recovery_active=_recovery_active_for_state(
                    config,
                    env=env,
                    v5_player_id=v5_player_id,
                    starting_player_id=starting_player_id,
                ),
            )
            if config.second_start_search and starting_player_id != v5_player_id:
                reranked = _select_second_start_search_action(
                    env=env,
                    v5_player_id=v5_player_id,
                    baseline_action=action_id,
                    v5_policy=v5_policy,
                    v4_policy=v4_policy,
                    draw_controller=draw_controller,
                    config=config,
                )
                search_rerank_uses += int(reranked != action_id)
                action_id = reranked
            policy_name = "v5"
        else:
            action_id = v4_policy.select_action(env.env, current)
            policy_name = "v4max"

        if config.battle_log_path is not None:
            events.append(
                {
                    "step": int(steps),
                    "current_player_id": int(current),
                    "policy": policy_name,
                    "action_id": _action_id_for_log(action_id),
                    "action": _action_to_json(env.env._env.state, current, action_id),
                    "state_before": _state_snapshot(env.env._env.state),
                }
            )

        if _action_is_end_turn(env.env._env.state, current, action_id):
            next_player_id = 2 if current == 1 else 1
            if next_player_id == v5_player_id and bool(config.draw_assist_enabled):
                assist_info = _apply_draw_assist_to_player(
                    env=env,
                    player_id=v5_player_id,
                    controller=draw_controller,
                    strength=config.draw_assist_strength,
                )
                draw_assist_ranked += int(bool(assist_info["ranked_options"]))
                draw_assist_uses += int(assist_info["selected_card_id"] is not None)

        _obs, _reward, terminated, truncated, info = _step_env_action(env, action_id)
        invalid += int(bool(info.get("invalid_action")))
        if terminated or truncated:
            break

    state = env.env._env.state
    winner = env.env.winner_id()
    winner_name = "v5" if winner == v5_player_id else "v4max" if winner == v4_player_id else None
    game = {
        "seed": seed,
        "v5_player_id": v5_player_id,
        "v4_player_id": v4_player_id,
        "starting_player_id": starting_player_id,
        "v5_started": starting_player_id == v5_player_id,
        "winner": winner,
        "winner_name": winner_name,
        "v5_win": winner == v5_player_id,
        "draw": winner is None,
        "steps": steps,
        "turns": state.turn_number,
        "p1_hp": state.p1.hero.hp,
        "p2_hp": state.p2.hero.hp,
        "v5_hp": state.p1.hero.hp if v5_player_id == 1 else state.p2.hero.hp,
        "v4_hp": state.p1.hero.hp if v4_player_id == 1 else state.p2.hero.hp,
        "invalid_actions": invalid,
        "last_policy": policy_name,
        "v5_deck_ids": list(v5_deck_ids),
        "v4_deck_ids": list(v4_deck_ids),
        "assembler_score": float(assembler_score),
        "draw_assist_uses": draw_assist_uses,
        "draw_assist_ranked": draw_assist_ranked,
        "search_rerank_uses": search_rerank_uses,
        "recovery_rerank_uses": int(v5_policy.recovery_rerank_uses),
    }
    if config.battle_log_path is not None:
        game["_battle_log"] = {
            "summary": dict(game),
            "events": events,
            "final_state": _state_snapshot(state),
        }
    return game


def _recovery_active_for_game(config: BenchmarkConfig, v5_player_id: int, starting_player_id: int) -> bool:
    return (
        config.recovery_reranker_path is not None
        and float(config.recovery_reranker_weight) > 0.0
        and int(starting_player_id) != int(v5_player_id)
    )


def _recovery_active_for_state(
    config: BenchmarkConfig,
    *,
    env: TrainV3ClassicEnv,
    v5_player_id: int,
    starting_player_id: int,
) -> bool:
    if not _recovery_active_for_game(config, v5_player_id, starting_player_id):
        return False
    if not bool(config.recovery_gated):
        return True
    return _recovery_gate_passes(config, env=env, v5_player_id=v5_player_id)


def _recovery_gate_passes(config: BenchmarkConfig, *, env: TrainV3ClassicEnv, v5_player_id: int) -> bool:
    state = env.env._env.state
    v4_player_id = 2 if int(v5_player_id) == 1 else 1
    v5 = state.p1 if state.p1.user_id == int(v5_player_id) else state.p2
    v4 = state.p1 if state.p1.user_id == v4_player_id else state.p2
    v5_power = _board_power(v5.board)
    v4_power = _board_power(v4.board)
    hp_deficit = int(v4.hero.hp) - int(v5.hero.hp)
    board_ratio = _raw_board_power_ratio(v5_power, v4_power)
    return (
        hp_deficit >= int(config.recovery_hp_deficit_threshold)
        or board_ratio <= float(config.recovery_board_power_ratio_threshold)
        or (bool(config.recovery_empty_board_gate) and len(v5.board) == 0 and len(v4.board) > 0)
    )


def _board_power(board: Iterable[Any]) -> int:
    return int(sum(max(0, int(card.attack)) * max(0, int(card.hp)) for card in board))


def _raw_board_power_ratio(v5_board_power: int, opponent_board_power: int) -> float:
    if int(opponent_board_power) <= 0:
        return 1.0 if int(v5_board_power) <= 0 else math.inf
    return float(v5_board_power) / float(opponent_board_power)


def _recovery_gating_config_to_json(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "enabled": bool(config.recovery_gated),
        "hp_deficit_threshold": int(config.recovery_hp_deficit_threshold),
        "board_power_ratio_threshold": float(config.recovery_board_power_ratio_threshold),
        "empty_board_gate": bool(config.recovery_empty_board_gate),
        "logic": "disabled" if not config.recovery_gated else "any_threshold",
    }


def _combine_base_and_recovery_logits(
    *,
    base_logits: np.ndarray,
    recovery_logits: np.ndarray,
    mask: np.ndarray,
    weight: float,
    bias_clip: float,
) -> np.ndarray:
    base = np.asarray(base_logits, dtype=np.float32)
    recovery = np.asarray(recovery_logits, dtype=np.float32)
    legal = np.asarray(mask, dtype=np.float32) == 1.0
    if base.shape != recovery.shape or base.shape != legal.shape:
        raise ValueError("base_logits, recovery_logits and mask must have the same shape")
    if not np.any(legal) or float(weight) <= 0.0:
        return base.copy()
    bias = recovery - float(np.mean(recovery[legal]))
    if math.isfinite(float(bias_clip)) and float(bias_clip) > 0.0:
        bias = np.clip(bias, -float(bias_clip), float(bias_clip))
    combined = base + float(weight) * bias.astype(np.float32, copy=False)
    return combined.astype(np.float32, copy=False)


def _select_second_start_search_action(
    *,
    env: TrainV3ClassicEnv,
    v5_player_id: int,
    baseline_action: int,
    v5_policy: V5AdaptivePolicy,
    v4_policy: OnnxActionPolicy,
    draw_controller: DrawAssistController,
    config: BenchmarkConfig,
) -> int:
    obs = env.observe(v5_player_id).astype(np.float32, copy=False)
    mask = env.action_mask(v5_player_id).astype(np.float32, copy=False)
    features = env.action_features(v5_player_id, include_preview=False).astype(np.float32, copy=False)
    effective_candidates, effective_depth = _effective_second_start_search_budget(config)
    candidate_scores = _rank_v5_policy_action_scores(
        v5_policy=v5_policy,
        obs=obs,
        action_features=features,
        mask=mask,
        max_candidates=effective_candidates,
    )
    candidates = [action_id for action_id, _score in candidate_scores]
    policy_score_by_action = {int(action_id): float(score) for action_id, score in candidate_scores}
    if baseline_action not in candidates:
        candidates.append(int(baseline_action))
        policy_score_by_action[int(baseline_action)] = (
            max(policy_score_by_action.values()) + 1.0e-6 if policy_score_by_action else 0.0
        )
    scored: list[tuple[int, float, float]] = []
    for action_id in candidates:
        if 0 <= int(action_id) < mask.shape[0] and mask[int(action_id)] == 1.0:
            scored.append(
                (
                    int(action_id),
                    float(policy_score_by_action.get(int(action_id), 0.0)),
                    _evaluate_second_start_candidate(
                        env=env,
                        candidate_action=int(action_id),
                        v5_player_id=v5_player_id,
                        v5_policy=v5_policy,
                        v4_policy=v4_policy,
                        draw_controller=draw_controller,
                        config=config,
                        search_depth_plies=effective_depth,
                    ),
                )
            )
    if not scored:
        return int(baseline_action)
    if bool(config.adaptive_strength_runtime):
        return _select_strength_blended_search_action(scored, adaptive_strength=config.adaptive_strength)
    return int(max(scored, key=lambda item: (item[2], -item[0]))[0])


def _effective_second_start_search_budget(config: BenchmarkConfig) -> tuple[int, int]:
    candidates = int(config.search_candidates)
    depth = int(config.search_depth_plies)
    if not bool(config.adaptive_strength_runtime):
        return candidates, depth
    strength = max(0.0, min(1.0, float(config.adaptive_strength)))
    if candidates > 0:
        candidates = max(1, int(math.ceil(candidates * strength)))
    depth = max(0, int(math.ceil(max(0, depth) * strength)))
    return candidates, depth


def _select_strength_blended_search_action(
    scored: list[tuple[int, float, float]],
    *,
    adaptive_strength: float,
) -> int:
    if not scored:
        raise ValueError("scored candidates must not be empty")
    strength = max(0.0, min(1.0, float(adaptive_strength)))
    policy_values = _normalize_candidate_scores([item[1] for item in scored])
    search_values = _normalize_candidate_scores([item[2] for item in scored])
    best: tuple[int, float] | None = None
    for idx, (action_id, _policy_score, _search_score) in enumerate(scored):
        blended = (1.0 - strength) * policy_values[idx] + strength * search_values[idx]
        candidate = (int(action_id), float(blended))
        if best is None or (candidate[1], -candidate[0]) > (best[1], -best[0]):
            best = candidate
    return int(best[0])


def _normalize_candidate_scores(values: list[float]) -> list[float]:
    finite = [float(value) if math.isfinite(float(value)) else -math.inf for value in values]
    real = [value for value in finite if math.isfinite(value)]
    if not real:
        return [0.0 for _value in values]
    lo = min(real)
    hi = max(real)
    if hi <= lo:
        return [0.5 if math.isfinite(value) else 0.0 for value in finite]
    return [0.0 if not math.isfinite(value) else (value - lo) / (hi - lo) for value in finite]


def _rank_v5_policy_actions(
    *,
    v5_policy: V5AdaptivePolicy,
    obs: np.ndarray,
    action_features: np.ndarray,
    mask: np.ndarray,
    max_candidates: int,
) -> list[int]:
    return [
        action_id
        for action_id, _score in _rank_v5_policy_action_scores(
            v5_policy=v5_policy,
            obs=obs,
            action_features=action_features,
            mask=mask,
            max_candidates=max_candidates,
        )
    ]


def _rank_v5_policy_action_scores(
    *,
    v5_policy: V5AdaptivePolicy,
    obs: np.ndarray,
    action_features: np.ndarray,
    mask: np.ndarray,
    max_candidates: int,
) -> list[tuple[int, float]]:
    import mlx.core as mx

    legal = np.flatnonzero(mask == 1.0)
    if legal.size == 0:
        return []
    output = v5_policy.model(
        mx.array(obs[None, :].astype(np.float32, copy=False)),
        mx.array(action_features[None, :, :].astype(np.float32, copy=False)),
    )
    logits = output[0] if isinstance(output, tuple) else output
    mx.eval(logits)
    logits_np = np.asarray(logits, dtype=np.float32)[0]
    scores = logits_np[legal]
    order = np.lexsort((legal, -scores))
    limit = legal.size if int(max_candidates) <= 0 else min(int(max_candidates), legal.size)
    return [(int(legal[idx]), float(scores[idx])) for idx in order[:limit]]


def _evaluate_second_start_candidate(
    *,
    env: TrainV3ClassicEnv,
    candidate_action: int,
    v5_player_id: int,
    v5_policy: V5AdaptivePolicy,
    v4_policy: OnnxActionPolicy,
    draw_controller: DrawAssistController,
    config: BenchmarkConfig,
    search_depth_plies: int | None = None,
) -> float:
    sim = copy.deepcopy(env)
    mask = sim.action_mask(v5_player_id)
    if int(candidate_action) < 0 or int(candidate_action) >= mask.shape[0] or mask[int(candidate_action)] != 1.0:
        return -math.inf
    trajectory_penalty = 0.0
    draw_strength = float(config.draw_assist_strength) if bool(config.draw_assist_enabled) else 0.0
    terminated, truncated = _step_search_sim(
        sim,
        int(candidate_action),
        v5_player_id=v5_player_id,
        draw_controller=draw_controller,
        draw_assist_strength=draw_strength,
    )
    depth = int(config.search_depth_plies) if search_depth_plies is None else int(search_depth_plies)
    for _ in range(max(0, depth)):
        if terminated or truncated:
            break
        current = sim.current_player_id()
        if current == v5_player_id:
            action_id = v5_policy.select_action(sim, current)
        else:
            action_id = v4_policy.select_action(sim.env, current)
        terminated, truncated = _step_search_sim(
            sim,
            action_id,
            v5_player_id=v5_player_id,
            draw_controller=draw_controller,
            draw_assist_strength=draw_strength,
        )
        if current != v5_player_id:
            trajectory_penalty += _search_rollout_penalty(sim, v5_player_id=v5_player_id, config=config)
    return _score_search_state(sim, v5_player_id=v5_player_id, config=config) - trajectory_penalty


def _step_search_sim(
    env: TrainV3ClassicEnv,
    action_id: Any,
    *,
    v5_player_id: int,
    draw_controller: DrawAssistController,
    draw_assist_strength: float,
) -> tuple[bool, bool]:
    current = env.current_player_id()
    if _action_is_end_turn(env.env._env.state, current, action_id):
        next_player_id = 2 if current == 1 else 1
        if next_player_id == v5_player_id and float(draw_assist_strength) > 0.0:
            _apply_draw_assist_to_player(
                env=env,
                player_id=v5_player_id,
                controller=draw_controller,
                strength=draw_assist_strength,
            )
    _obs, _reward, terminated, truncated, _info = _step_env_action(env, action_id)
    return bool(terminated), bool(truncated)


def _score_search_state(env: TrainV3ClassicEnv, *, v5_player_id: int, config: BenchmarkConfig) -> float:
    state = env.env._env.state
    winner = env.env.winner_id()
    v4_player_id = 2 if v5_player_id == 1 else 1
    v5 = state.p1 if state.p1.user_id == v5_player_id else state.p2
    v4 = state.p1 if state.p1.user_id == v4_player_id else state.p2
    hp_margin = float(v5.hero.hp - v4.hero.hp)
    own_board = sum(max(0, int(card.attack)) * max(0, int(card.hp)) for card in v5.board)
    opp_board = sum(max(0, int(card.attack)) * max(0, int(card.hp)) for card in v4.board)
    own_attack = sum(max(0, int(card.attack)) for card in v5.board)
    opp_attack = sum(max(0, int(card.attack)) for card in v4.board)
    hand_margin = float(len(v5.hand) - len(v4.hand))
    board_count_margin = float(len(v5.board) - len(v4.board))
    if winner == v5_player_id:
        terminal = 1000.0
    elif winner == v4_player_id:
        terminal = -1000.0
    else:
        terminal = 0.0
    score = (
        terminal
        + float(config.search_hp_weight) * hp_margin
        + float(config.search_board_power_weight) * float(own_board - opp_board)
        + float(config.search_hand_weight) * hand_margin
        + float(config.search_attack_weight) * float(own_attack - opp_attack)
        + float(config.search_board_count_weight) * board_count_margin
    )
    if float(config.search_empty_board_penalty) > 0.0 and len(v5.board) == 0 and len(v4.board) > 0:
        score -= float(config.search_empty_board_penalty)
    if float(config.search_board_disadvantage_penalty) > 0.0:
        ratio = _raw_board_power_ratio(own_board, opp_board)
        if math.isfinite(ratio) and ratio < float(config.search_board_disadvantage_ratio):
            score -= float(config.search_board_disadvantage_penalty) * (
                float(config.search_board_disadvantage_ratio) - ratio
            )
    return float(score)


def _search_rollout_penalty(
    env: TrainV3ClassicEnv,
    *,
    v5_player_id: int,
    config: BenchmarkConfig,
) -> float:
    penalty = 0.0
    state = env.env._env.state
    v4_player_id = 2 if v5_player_id == 1 else 1
    v5 = state.p1 if state.p1.user_id == v5_player_id else state.p2
    v4 = state.p1 if state.p1.user_id == v4_player_id else state.p2
    own_board = sum(max(0, int(card.attack)) * max(0, int(card.hp)) for card in v5.board)
    opp_board = sum(max(0, int(card.attack)) * max(0, int(card.hp)) for card in v4.board)
    if float(config.search_rollout_empty_board_penalty) > 0.0 and len(v5.board) == 0 and len(v4.board) > 0:
        penalty += float(config.search_rollout_empty_board_penalty)
    if float(config.search_rollout_board_disadvantage_penalty) > 0.0:
        ratio = _raw_board_power_ratio(own_board, opp_board)
        if math.isfinite(ratio) and ratio < float(config.search_rollout_board_disadvantage_ratio):
            penalty += float(config.search_rollout_board_disadvantage_penalty) * (
                float(config.search_rollout_board_disadvantage_ratio) - ratio
            )
    return float(penalty)


def _strong_info_mode(config: BenchmarkConfig) -> InfoModeV5:
    private_info = bool(config.private_info_enabled)
    draw_assist = bool(config.draw_assist_enabled)
    return InfoModeV5(
        adaptive_strength=config.adaptive_strength,
        own_hand_identity_known=True,
        own_deck_known=True,
        enemy_hand_known=private_info,
        enemy_deck_known=private_info,
        enemy_deck_order_known=private_info,
        draw_assist_enabled=draw_assist,
        draw_assist_strength=float(config.draw_assist_strength) if draw_assist else 0.0,
    )


def _strong_assist_mode(config: BenchmarkConfig) -> AssistModeV5:
    if not bool(config.assist_mode_enabled):
        return AssistModeV5()
    return AssistModeV5(
        assembler_enabled=True,
        assembler_strength=config.assembler_strength,
        desirerer_enabled=True,
        desirerer_strength=config.desirerer_strength,
        teacher_hint_available=True,
        assist_profile_id=15,
    )


def _assembler_candidates_for_config(
    config: BenchmarkConfig,
    candidates: list[AssemblerCandidate],
) -> list[AssemblerCandidate]:
    if not bool(config.deck_assist_enabled):
        return []
    return candidates


def _select_v5_deck(
    *,
    opponent_deck_ids: Iterable[int],
    assembler: DeckMatchupEvaluator,
    candidates: list[AssemblerCandidate],
) -> tuple[list[int], float]:
    if not candidates:
        deck = _normalize_candidate_deck(opponent_deck_ids, opponent_deck_ids=opponent_deck_ids)
        return deck, float(assembler.score_candidate(opponent_deck_ids, deck))
    normalized = [
        AssemblerCandidate(
            deck_ids=_normalize_candidate_deck(candidate.deck_ids, opponent_deck_ids=opponent_deck_ids),
            metadata=dict(candidate.metadata),
        )
        for candidate in candidates
    ]
    best = assembler.search_best(opponent_deck_ids, normalized)
    return list(best.deck_ids), float(best.score)


def _select_v5_deck_for_config(
    *,
    config: BenchmarkConfig,
    opponent_deck_ids: Iterable[int],
    assembler: DeckMatchupEvaluator,
    candidates: list[AssemblerCandidate],
) -> tuple[list[int], float]:
    if not bool(config.deck_assist_enabled):
        deck = _normalize_candidate_deck(config.noassist_deck_ids, opponent_deck_ids=NOASSIST_BASELINE_DECK_IDS)
        return deck, float(assembler.score_candidate([], deck))
    return _select_v5_deck(opponent_deck_ids=opponent_deck_ids, assembler=assembler, candidates=candidates)


def _load_assembler_candidates(path: Path | None) -> list[AssemblerCandidate]:
    if path is None or not path.exists():
        return []
    seen: set[tuple[int, ...]] = set()
    candidates: list[AssemblerCandidate] = []
    for row in load_assembler_dataset(path):
        key = tuple(int(card_id) for card_id in row.candidate_deck_ids)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            AssemblerCandidate(
                deck_ids=list(key),
                metadata={"source_run": row.source_run, "target_winrate": row.target_winrate},
            )
        )
    return candidates


def _normalize_candidate_deck(
    deck_ids: Iterable[int],
    *,
    opponent_deck_ids: Iterable[int],
) -> list[int]:
    raw = [int(card_id) for card_id in deck_ids if int(card_id) > 0]
    hero_ids = [card_id for card_id in raw if card_id in HERO_CARD_IDS]
    if hero_ids:
        hero_id = hero_ids[0]
        non_heroes = [card_id for card_id in raw if card_id not in HERO_CARD_IDS]
    else:
        opponent_heroes = [int(card_id) for card_id in opponent_deck_ids if int(card_id) in HERO_CARD_IDS]
        hero_id = opponent_heroes[0] if opponent_heroes else 1
        non_heroes = raw
    seen_non_heroes: list[int] = []
    for card_id in non_heroes:
        if card_id in HERO_CARD_IDS:
            continue
        seen_non_heroes.append(card_id)
    if not seen_non_heroes:
        seen_non_heroes = [37, 38, 40, 41, 42, 27, 28, 29]
    return [hero_id, *seen_non_heroes[:8]]


def _apply_draw_assist_to_player(
    *,
    env: TrainV3ClassicEnv,
    player_id: int,
    controller: DrawAssistController,
    strength: float,
) -> dict[str, Any]:
    state = env.env._env.state
    player = state.p1 if state.p1.user_id == player_id else state.p2
    if not player.deck:
        return {
            "selected_card_id": None,
            "ranked_options": [],
            "draw_assist_enabled": True,
            "draw_assist_strength": float(strength),
        }
    result = controller.choose_draw(
        deck_ids=[card.card_id for card in player.deck],
        hand_ids=[card.card_id for card in player.hand],
        board_power_ratio=_board_power_ratio(state, player_id),
        draw_assist_enabled=True,
        draw_assist_strength=strength,
    )
    selected = result.get("selected_card_id")
    if selected is not None:
        for idx, card in enumerate(player.deck):
            if int(card.card_id) == int(selected):
                player.deck.insert(0, player.deck.pop(idx))
                break
    return result


def _should_write_battle_log(config: BenchmarkConfig, game: dict[str, Any], logged_battles: int) -> bool:
    if config.battle_log_path is None:
        return False
    if int(config.log_max_battles) > 0 and int(logged_battles) >= int(config.log_max_battles):
        return False
    if bool(config.log_lost_only) and bool(game.get("v5_win")):
        return False
    return True


def _append_battle_log(path: Path | None, battle_log: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(battle_log, ensure_ascii=False, sort_keys=True) + "\n")


def _step_env_action(env: TrainV3ClassicEnv, action: Any):
    if hasattr(action, "to_dict"):
        return env.step_core_action(action)
    return env.step(int(action))


def _action_id_for_log(action: Any) -> int:
    return -1 if hasattr(action, "to_dict") else int(action)


def _action_to_json(state: Any, player_id: int, action_id: Any) -> dict[str, Any] | None:
    if hasattr(action_id, "to_dict"):
        data = action_id.to_dict()
        return data if isinstance(data, dict) else {"repr": repr(data)}
    action = decode_action(state, int(player_id), int(action_id))
    if action is None:
        return None
    if hasattr(action, "to_dict"):
        data = action.to_dict()
        return data if isinstance(data, dict) else {"repr": repr(data)}
    return {"repr": repr(action)}


def _state_snapshot(state: Any) -> dict[str, Any]:
    return {
        "turn_number": int(getattr(state, "turn_number", 0) or 0),
        "current_player_id": int(getattr(state, "current_player_id", 0) or 0),
        "p1": _player_snapshot(getattr(state, "p1", None)),
        "p2": _player_snapshot(getattr(state, "p2", None)),
    }


def _player_snapshot(player: Any) -> dict[str, Any]:
    if player is None:
        return {}
    hero = getattr(player, "hero", None)
    return {
        "user_id": int(getattr(player, "user_id", 0) or 0),
        "hero": {
            "card_id": int(getattr(hero, "card_id", 0) or 0),
            "hp": int(getattr(hero, "hp", 0) or 0),
        },
        "mana": int(getattr(player, "mana", 0) or 0),
        "max_mana": int(getattr(player, "max_mana", 0) or 0),
        "hand_ids": [int(getattr(card, "card_id", 0) or 0) for card in list(getattr(player, "hand", []) or [])],
        "deck_count": len(list(getattr(player, "deck", []) or [])),
        "board": [_card_snapshot(card) for card in list(getattr(player, "board", []) or [])],
        "graveyard_ids": [
            int(getattr(card, "card_id", 0) or 0)
            for card in list(getattr(player, "graveyard", []) or [])
        ],
    }


def _card_snapshot(card: Any) -> dict[str, Any]:
    return {
        "card_id": int(getattr(card, "card_id", 0) or 0),
        "attack": int(getattr(card, "attack", 0) or 0),
        "hp": int(getattr(card, "hp", 0) or 0),
        "cost": int(getattr(card, "cost", 0) or 0),
    }


def _board_power_ratio(state, player_id: int) -> float:
    player = state.p1 if state.p1.user_id == player_id else state.p2
    enemy = state.p2 if state.p1.user_id == player_id else state.p1
    own = sum(max(0, int(card.attack)) * max(0, int(card.hp)) for card in player.board)
    opp = sum(max(0, int(card.attack)) * max(0, int(card.hp)) for card in enemy.board)
    return float(own + 1.0) / float(opp + 1.0)


def _action_is_end_turn(state, player_id: int, action_id: Any) -> bool:
    if hasattr(action_id, "to_dict"):
        return str(action_id.to_dict().get("type")) == "end_turn"
    action = decode_action(state, player_id, int(action_id))
    if action is None:
        return False
    return str(action.to_dict().get("type")) == "end_turn"


def _player_card_pool_ids(state, player_id: int) -> list[int]:
    player = state.p1 if state.p1.user_id == player_id else state.p2
    return [
        int(card.card_id)
        for card in [player.hero, *player.hand, *player.deck, *player.board, *player.graveyard]
        if int(card.card_id) > 0
    ]


def _summarize(games: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(games)
    v5_wins = sum(1 for game in games if game["winner_name"] == "v5")
    v4_wins = sum(1 for game in games if game["winner_name"] == "v4max")
    draws = sum(1 for game in games if game["winner_name"] is None)
    v5_p1 = [game for game in games if game["v5_player_id"] == 1]
    v5_p2 = [game for game in games if game["v5_player_id"] == 2]
    v5_first = [game for game in games if game["v5_started"]]
    v5_second = [game for game in games if not game["v5_started"]]
    return {
        "games": total,
        "v5_wins": v5_wins,
        "v4max_wins": v4_wins,
        "draws": draws,
        "v5_score_rate": (v5_wins + 0.5 * draws) / total if total else 0.0,
        "v5_winrate": v5_wins / total if total else 0.0,
        "v4max_winrate": v4_wins / total if total else 0.0,
        "v5_p1_winrate": _winrate(v5_p1),
        "v5_p2_winrate": _winrate(v5_p2),
        "v5_first_winrate": _winrate(v5_first),
        "v5_second_winrate": _winrate(v5_second),
        "avg_v5_hp_margin": sum(game["v5_hp"] - game["v4_hp"] for game in games) / total if total else 0.0,
        "avg_steps": sum(game["steps"] for game in games) / total if total else 0.0,
        "invalid_actions": sum(int(game["invalid_actions"]) for game in games),
        "draw_assist_uses": sum(int(game["draw_assist_uses"]) for game in games),
        "draw_assist_ranked": sum(int(game["draw_assist_ranked"]) for game in games),
        "search_rerank_uses": sum(int(game.get("search_rerank_uses", 0)) for game in games),
        "recovery_rerank_uses": sum(int(game.get("recovery_rerank_uses", 0)) for game in games),
    }


def _starting_players(mode: str) -> tuple[int, ...]:
    if mode == "both":
        return (1, 2)
    if mode == "p1":
        return (1,)
    if mode == "p2":
        return (2,)
    raise ValueError(f"unknown start_mode: {mode}")


def _winrate(games: list[dict[str, Any]]) -> float:
    if not games:
        return 0.0
    return sum(1 for game in games if game["winner_name"] == "v5") / len(games)


def _write_outputs(result: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output_path.with_suffix(".csv")
    fields = [
        "seed",
        "v5_player_id",
        "starting_player_id",
        "v5_started",
        "winner_name",
        "v5_win",
        "draw",
        "steps",
        "turns",
        "v5_hp",
        "v4_hp",
        "invalid_actions",
        "assembler_score",
        "draw_assist_uses",
        "search_rerank_uses",
        "recovery_rerank_uses",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in result["games"]:
            writer.writerow({field: row.get(field) for field in fields})


def _validate_config(config: BenchmarkConfig) -> None:
    if not config.v4_model_path.exists():
        raise FileNotFoundError(f"V4 max model not found: {config.v4_model_path}")
    if not config.v5_checkpoint_path.exists():
        raise FileNotFoundError(f"V5 checkpoint not found: {config.v5_checkpoint_path}")
    if config.games <= 0:
        raise ValueError("games must be positive")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    for name in ("adaptive_strength", "draw_assist_strength", "assembler_strength", "desirerer_strength"):
        value = float(getattr(config, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if int(config.search_candidates) < 0:
        raise ValueError("search_candidates must be non-negative")
    if int(config.search_depth_plies) < 0:
        raise ValueError("search_depth_plies must be non-negative")
    for name in (
        "search_hp_weight",
        "search_board_power_weight",
        "search_hand_weight",
        "search_attack_weight",
        "search_board_count_weight",
        "search_empty_board_penalty",
        "search_board_disadvantage_penalty",
        "search_board_disadvantage_ratio",
        "search_rollout_empty_board_penalty",
        "search_rollout_board_disadvantage_penalty",
        "search_rollout_board_disadvantage_ratio",
    ):
        value = float(getattr(config, name))
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    for name in (
        "search_empty_board_penalty",
        "search_board_disadvantage_penalty",
        "search_board_disadvantage_ratio",
        "search_rollout_empty_board_penalty",
        "search_rollout_board_disadvantage_penalty",
        "search_rollout_board_disadvantage_ratio",
    ):
        if float(getattr(config, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if config.recovery_reranker_path is not None and not config.recovery_reranker_path.exists():
        raise FileNotFoundError(f"recovery reranker not found: {config.recovery_reranker_path}")
    if float(config.recovery_reranker_weight) < 0.0:
        raise ValueError("recovery_reranker_weight must be non-negative")
    if float(config.recovery_bias_clip) < 0.0:
        raise ValueError("recovery_bias_clip must be non-negative")
    if int(config.recovery_hp_deficit_threshold) < 0:
        raise ValueError("recovery_hp_deficit_threshold must be non-negative")
    if not math.isfinite(float(config.recovery_board_power_ratio_threshold)):
        raise ValueError("recovery_board_power_ratio_threshold must be finite")
    if float(config.recovery_board_power_ratio_threshold) < 0.0:
        raise ValueError("recovery_board_power_ratio_threshold must be non-negative")
    if len(tuple(config.noassist_deck_ids)) < 2:
        raise ValueError("noassist_deck_ids must include a hero and at least one card")


def _config_to_json(config: BenchmarkConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def _parse_deck_ids(raw: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in raw]
    deck = tuple(int(part) for part in parts if int(part) > 0)
    if len(deck) < 2:
        raise ValueError("noassist deck must include a hero and at least one card")
    return deck


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V5 adaptive S=1 assist benchmark against V4 max only")
    parser.add_argument("--v4-model", type=Path, default=DEFAULT_V4_MAX)
    parser.add_argument("--v5-checkpoint", type=Path, default=DEFAULT_V5_CHECKPOINT)
    parser.add_argument("--assembler-dataset", type=Path, default=DEFAULT_ASSEMBLER_DATASET)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--games", type=int, default=32, help="Seeds; two side-balanced games are run per seed")
    parser.add_argument("--seed", type=int, default=88000)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--start-mode", choices=["both", "p1", "p2"], default="both")
    parser.add_argument("--adaptive-strength", type=float, default=1.0)
    parser.add_argument("--draw-assist-strength", type=float, default=1.0)
    parser.add_argument("--assembler-strength", type=float, default=1.0)
    parser.add_argument("--desirerer-strength", type=float, default=1.0)
    parser.add_argument("--no-bonuses", action="store_true")
    parser.add_argument("--no-private-info", action="store_true")
    parser.add_argument("--disable-draw-assist", action="store_true")
    parser.add_argument("--disable-assist-mode", action="store_true")
    parser.add_argument("--disable-deck-assist", action="store_true")
    parser.add_argument("--noassist-deck-ids", default=",".join(str(card_id) for card_id in NOASSIST_BASELINE_DECK_IDS))
    parser.add_argument("--second-start-search", action="store_true")
    parser.add_argument("--search-candidates", type=int, default=4)
    parser.add_argument("--search-depth-plies", type=int, default=4)
    parser.add_argument("--adaptive-strength-runtime", action="store_true")
    parser.add_argument("--search-hp-weight", type=float, default=10.0)
    parser.add_argument("--search-board-power-weight", type=float, default=0.75)
    parser.add_argument("--search-hand-weight", type=float, default=0.25)
    parser.add_argument("--search-attack-weight", type=float, default=0.0)
    parser.add_argument("--search-board-count-weight", type=float, default=0.0)
    parser.add_argument("--search-empty-board-penalty", type=float, default=0.0)
    parser.add_argument("--search-board-disadvantage-penalty", type=float, default=0.0)
    parser.add_argument("--search-board-disadvantage-ratio", type=float, default=0.7)
    parser.add_argument("--search-rollout-empty-board-penalty", type=float, default=0.0)
    parser.add_argument("--search-rollout-board-disadvantage-penalty", type=float, default=0.0)
    parser.add_argument("--search-rollout-board-disadvantage-ratio", type=float, default=0.7)
    parser.add_argument("--recovery-reranker", type=Path, default=None)
    parser.add_argument("--recovery-reranker-weight", type=float, default=0.0)
    parser.add_argument("--recovery-bias-clip", type=float, default=4.0)
    parser.add_argument("--recovery-gated", action="store_true")
    parser.add_argument("--recovery-hp-deficit-threshold", type=int, default=4)
    parser.add_argument("--recovery-board-power-ratio-threshold", type=float, default=0.5)
    parser.add_argument("--recovery-empty-board-gate", action="store_true")
    parser.add_argument(
        "--log-battles",
        action="store_true",
        help="Write per-step battle logs to battle_logs.jsonl in the output dir.",
    )
    parser.add_argument("--battle-log-path", type=Path, default=None)
    parser.add_argument("--log-all-battles", action="store_true")
    parser.add_argument("--log-max-battles", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = args.output_dir
    if output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "TrainV3.5" / "runs" / f"v5_s1_assist_vs_v4max_{stamp}"
    no_bonuses = bool(args.no_bonuses)
    battle_log_path = args.battle_log_path
    if battle_log_path is None and bool(args.log_battles):
        battle_log_path = output_dir / "battle_logs.jsonl"
    result = run_benchmark(
        BenchmarkConfig(
            v4_model_path=args.v4_model,
            v5_checkpoint_path=args.v5_checkpoint,
            assembler_dataset_path=args.assembler_dataset,
            output_dir=output_dir,
            games=args.games,
            seed=args.seed,
            max_steps=args.max_steps,
            start_mode=args.start_mode,
            adaptive_strength=float(args.adaptive_strength),
            draw_assist_strength=float(args.draw_assist_strength),
            assembler_strength=float(args.assembler_strength),
            desirerer_strength=float(args.desirerer_strength),
            private_info_enabled=not (no_bonuses or bool(args.no_private_info)),
            draw_assist_enabled=not (no_bonuses or bool(args.disable_draw_assist)),
            assist_mode_enabled=not (no_bonuses or bool(args.disable_assist_mode)),
            deck_assist_enabled=not (no_bonuses or bool(args.disable_deck_assist)),
            noassist_deck_ids=_parse_deck_ids(args.noassist_deck_ids),
            second_start_search=bool(args.second_start_search) and not no_bonuses,
            search_candidates=int(args.search_candidates),
            search_depth_plies=int(args.search_depth_plies),
            adaptive_strength_runtime=bool(args.adaptive_strength_runtime),
            search_hp_weight=float(args.search_hp_weight),
            search_board_power_weight=float(args.search_board_power_weight),
            search_hand_weight=float(args.search_hand_weight),
            search_attack_weight=float(args.search_attack_weight),
            search_board_count_weight=float(args.search_board_count_weight),
            search_empty_board_penalty=float(args.search_empty_board_penalty),
            search_board_disadvantage_penalty=float(args.search_board_disadvantage_penalty),
            search_board_disadvantage_ratio=float(args.search_board_disadvantage_ratio),
            search_rollout_empty_board_penalty=float(args.search_rollout_empty_board_penalty),
            search_rollout_board_disadvantage_penalty=float(args.search_rollout_board_disadvantage_penalty),
            search_rollout_board_disadvantage_ratio=float(args.search_rollout_board_disadvantage_ratio),
            recovery_reranker_path=None if no_bonuses else args.recovery_reranker,
            recovery_reranker_weight=0.0 if no_bonuses else float(args.recovery_reranker_weight),
            recovery_bias_clip=float(args.recovery_bias_clip),
            recovery_gated=bool(args.recovery_gated),
            recovery_hp_deficit_threshold=int(args.recovery_hp_deficit_threshold),
            recovery_board_power_ratio_threshold=float(args.recovery_board_power_ratio_threshold),
            recovery_empty_board_gate=bool(args.recovery_empty_board_gate),
            battle_log_path=battle_log_path,
            log_lost_only=not bool(args.log_all_battles),
            log_max_battles=int(args.log_max_battles),
        )
    )
    summary = result["summary"]
    print(
        "\n"
        f"V5 score={summary['v5_score_rate'] * 100:.1f}% "
        f"wins={summary['v5_wins']} "
        f"losses={summary['v4max_wins']} "
        f"draws={summary['draws']} "
        f"p1={summary['v5_p1_winrate'] * 100:.1f}% "
        f"p2={summary['v5_p2_winrate'] * 100:.1f}% "
        f"first={summary['v5_first_winrate'] * 100:.1f}% "
        f"second={summary['v5_second_winrate'] * 100:.1f}% "
        f"hp_margin={summary['avg_v5_hp_margin']:.2f} "
        f"search_rerank_uses={summary.get('search_rerank_uses', 0)} "
        f"recovery_rerank_uses={summary.get('recovery_rerank_uses', 0)}"
    )
    print(f"Saved: {output_dir / 'v5_s1_assist_vs_v4max.json'}")


if __name__ == "__main__":
    main()

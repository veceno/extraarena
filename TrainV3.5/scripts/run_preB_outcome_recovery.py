#!/usr/bin/env python3
"""Outcome-labelled second-start recovery before Block B.

Unlike ``run_preB_counterfactual_recovery.py``, this collector never uses a
hand-written HP/board score.  Warm V5 first plays a complete mirror-deck game
as the second starter.  Only states from actual losses are searched, and every
candidate one-step deviation is played to a terminal result.  A pair is kept
only when the deviation turns the deterministic loss into a draw or win.

The V4-max action is included as a teacher candidate, but is not trusted
blindly: it receives a label only when the full continuation outcome improves.
Mana draw is treated exactly like every other candidate, so its recovery head
cannot learn generic draw greed from an intermediate shaping score.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRAINV3_PYTHON = ROOT / "TrainV3.5" / "python"
for path in (ROOT, TRAINV3_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ai.train_v2.onnx_policy import OnnxActionPolicy  # noqa: E402
from train_v3.mana_draw_head_v5 import mana_draw_legal_mask  # noqa: E402

from run_preB_counterfactual_recovery import (  # noqa: E402
    DEFAULT_V4_MAX,
    DRAW_ACTION,
    PreBRecoveryConfig,
    _V5Policy,
    _append_pair,
    _collect_anchor_states,
    _jsonable,
    _new_mirror_env,
    _save_dataset,
    _stack_dataset_values,
    _step_action,
    action_kind,
    recovery_gate_reason,
    train_counterfactual_recovery,
)


@dataclass(frozen=True)
class OutcomeRecoveryConfig:
    base_checkpoint: Path
    v4_model: Path
    output_dir: Path
    games: int = 64
    anchor_games: int = 24
    max_steps: int = 300
    continuation_max_steps: int = 300
    continuation_policy: str = "v4_teacher"
    seed: int = 71430001
    ranked_candidates: int = 6
    max_candidate_actions: int = 10
    max_states_per_loss: int = 6
    max_pairs: int = 4096
    min_pairs: int = 32
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 3.0e-4
    ranking_margin: float = 0.5
    action_pair_coef: float = 1.0
    draw_bce_coef: float = 0.15
    recovery_policy_kl_coef: float = 2.0
    recovery_draw_kl_coef: float = 4.0
    anchor_policy_kl_coef: float = 4.0
    anchor_draw_kl_coef: float = 8.0
    train_mana_draw_recovery: bool = False


@dataclass
class _RecordedState:
    env: Any
    obs: np.ndarray
    features: np.ndarray
    mask: np.ndarray
    draw_legal: bool
    base_action: int
    teacher_action: int
    ranked_actions: list[int]
    gate_reason: int
    seed: int
    step: int


def run_outcome_recovery(config: OutcomeRecoveryConfig) -> dict[str, Any]:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    training_config = _training_config(config)
    dataset = collect_outcome_dataset(config, training_config=training_config)
    if int(dataset["summary"]["pairs"]) < int(config.min_pairs):
        raise RuntimeError(
            f"outcome pairs={dataset['summary']['pairs']} below min_pairs={config.min_pairs}"
        )
    dataset_path = config.output_dir / "preB_outcome_dataset.npz"
    _save_dataset(dataset_path, dataset)
    checkpoint, train_summary = train_counterfactual_recovery(training_config, dataset)
    result = {
        "summary": {
            "status": "ok",
            "checkpoint_path": str(checkpoint),
            "pairs": int(dataset["summary"]["pairs"]),
            "action_pairs": int(dataset["summary"]["action_pairs"]),
            "draw_pairs": int(dataset["summary"]["draw_pairs"]),
            "loss_games": int(dataset["summary"]["loss_games"]),
            "flipped_to_win": int(dataset["summary"]["flipped_to_win"]),
            "flipped_to_draw": int(dataset["summary"]["flipped_to_draw"]),
            "final_loss": float(train_summary["final_loss"]),
        },
        "dataset_summary": dataset["summary"],
        "train_summary": train_summary,
        "config": _jsonable(asdict(config)),
    }
    (config.output_dir / "preB_outcome_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def collect_outcome_dataset(
    config: OutcomeRecoveryConfig, *, training_config: PreBRecoveryConfig
) -> dict[str, Any]:
    v5 = _V5Policy(config.base_checkpoint)
    v4 = OnnxActionPolicy(
        str(config.v4_model), mode="argmax", seed=config.seed, verify_mask=False
    )
    rows: dict[str, list[Any]] = {
        "observations": [],
        "action_features": [],
        "masks": [],
        "mana_draw_legal": [],
        "positive_actions": [],
        "negative_actions": [],
        "action_pair_mask": [],
        "draw_supervision_mask": [],
        "draw_targets": [],
        "score_margins": [],
        "base_action_codes": [],
        "preferred_action_codes": [],
        "preferred_action_kinds": [],
        "base_action_kinds": [],
        "gate_reasons": [],
        "seeds": [],
    }
    anchors: dict[str, list[Any]] = {
        "anchor_observations": [],
        "anchor_action_features": [],
        "anchor_masks": [],
        "anchor_mana_draw_legal": [],
    }
    stats = {
        "games": 0,
        "terminal_games": 0,
        "win_games": 0,
        "loss_games": 0,
        "draw_games": 0,
        "recorded_states": 0,
        "searched_states": 0,
        "candidate_evals": 0,
        "base_outcome_mismatches": 0,
        "no_improvement_states": 0,
        "flipped_to_win": 0,
        "flipped_to_draw": 0,
        "teacher_preferred": 0,
        "teacher_takeover_evals": 0,
    }

    for game_idx in range(int(config.games)):
        seed = int(config.seed) + game_idx
        v5_player_id = 1 if game_idx % 2 == 0 else 2
        starting_player_id = 2 if v5_player_id == 1 else 1
        env = _new_mirror_env(seed=seed, starting_player_id=starting_player_id)
        v4.reset(seed * 13 + (3 - v5_player_id))
        records: list[_RecordedState] = []
        stats["games"] += 1
        terminated = truncated = False
        for step in range(int(config.max_steps)):
            current = env.current_player_id()
            if current == v5_player_id:
                obs = env.observe(current).astype(np.float32, copy=True)
                features = env.action_features(current, include_preview=False).astype(
                    np.float32, copy=True
                )
                mask = env.action_mask(current).astype(np.float32, copy=True)
                draw_legal = bool(mana_draw_legal_mask(env.env._env.state, current))
                base_action = int(v5.select(env, current))
                teacher_action = int(v4.select_action(env.env, current))
                reason = recovery_gate_reason(env, current, base_action)
                base_kind = action_kind(env.env._env.state, current, base_action)
                if reason and (
                    teacher_action != base_action
                    or base_kind in {"attack_face", "mana_draw"}
                    or draw_legal
                ):
                    records.append(
                        _RecordedState(
                            env=copy.deepcopy(env),
                            obs=obs,
                            features=features,
                            mask=mask,
                            draw_legal=draw_legal,
                            base_action=base_action,
                            teacher_action=teacher_action,
                            ranked_actions=v5.ranked_actions(
                                obs, features, mask, config.ranked_candidates
                            ),
                            gate_reason=reason,
                            seed=seed,
                            step=step,
                        )
                    )
                action = base_action
            else:
                action = int(v4.select_action(env.env, current))
            terminated, truncated = _step_action(env, action)
            if terminated or truncated:
                stats["terminal_games"] += 1
                break

        base_outcome = _outcome(env, v5_player_id)
        if base_outcome > 0:
            stats["win_games"] += 1
        elif base_outcome < 0:
            stats["loss_games"] += 1
        else:
            stats["draw_games"] += 1
        if base_outcome >= 0:
            continue

        selected_records = _sample_records(records, int(config.max_states_per_loss))
        stats["recorded_states"] += len(selected_records)
        for record in selected_records:
            candidates = _outcome_candidates(
                record, max_actions=int(config.max_candidate_actions)
            )
            # Reproduce the observed loss under the original V5 continuation.
            replayed_base = evaluate_terminal_candidate(
                record.env,
                candidate=record.base_action,
                v5_player_id=v5_player_id,
                v5=v5,
                v4=v4,
                max_steps=int(config.continuation_max_steps),
                continuation_policy="v5",
            )
            outcomes: list[tuple[int, int]] = [(record.base_action, replayed_base)]
            for candidate in candidates:
                if candidate == record.base_action:
                    continue
                continuation_policy = (
                    "v4_teacher"
                    if config.continuation_policy == "v4_teacher"
                    and candidate == record.teacher_action
                    else "v5"
                )
                outcome = evaluate_terminal_candidate(
                    record.env,
                    candidate=candidate,
                    v5_player_id=v5_player_id,
                    v5=v5,
                    v4=v4,
                    max_steps=int(config.continuation_max_steps),
                    continuation_policy=continuation_policy,
                )
                outcomes.append((candidate, outcome))
                stats["teacher_takeover_evals"] += int(
                    continuation_policy == "v4_teacher"
                )
            stats["searched_states"] += 1
            stats["candidate_evals"] += len(outcomes)
            if replayed_base != -1:
                stats["base_outcome_mismatches"] += 1
                continue
            improved = [(action, outcome) for action, outcome in outcomes if outcome > replayed_base]
            if not improved:
                stats["no_improvement_states"] += 1
                continue
            preferred, preferred_outcome = _choose_preferred(record, improved)
            _append_pair(
                rows,
                obs=record.obs,
                features=record.features,
                mask=record.mask,
                draw_legal=record.draw_legal,
                preferred=preferred,
                base_action=record.base_action,
                margin=float(preferred_outcome - replayed_base),
                gate_reason=record.gate_reason,
                seed=record.seed,
                state=record.env.env._env.state,
                player_id=v5_player_id,
            )
            if preferred_outcome > 0:
                stats["flipped_to_win"] += 1
            else:
                stats["flipped_to_draw"] += 1
            if preferred == record.teacher_action:
                stats["teacher_preferred"] += 1
            if len(rows["positive_actions"]) >= int(config.max_pairs):
                break
        if len(rows["positive_actions"]) >= int(config.max_pairs):
            break
        if (game_idx + 1) % 8 == 0:
            print(
                f"OUTCOME_COLLECT games={game_idx + 1}/{config.games} "
                f"losses={stats['loss_games']} pairs={len(rows['positive_actions'])} "
                f"evals={stats['candidate_evals']}",
                flush=True,
            )

    _collect_anchor_states(training_config, v5=v5, v4=v4, anchors=anchors)
    if not rows["positive_actions"]:
        raise RuntimeError("outcome recovery dataset is empty")
    action_pair_mask = np.asarray(rows["action_pair_mask"], dtype=np.bool_)
    draw_mask = np.asarray(rows["draw_supervision_mask"], dtype=np.bool_)
    preferred_kinds = list(rows["preferred_action_kinds"])
    base_kinds = list(rows["base_action_kinds"])
    margins = np.asarray(rows["score_margins"], dtype=np.float32)
    summary = {
        "schema": "extra_lr_v5_preB_terminal_outcome_v1",
        "collection_mode": "model_benchmark_mirror_decks_v5_second_terminal_outcome",
        **{key: int(value) for key, value in stats.items()},
        "pairs": int(len(rows["positive_actions"])),
        "action_pairs": int(action_pair_mask.sum()),
        "draw_pairs": int(draw_mask.sum()),
        "preferred_draw_pairs": int(sum(kind == "mana_draw" for kind in preferred_kinds)),
        "preferred_trade_pairs": int(sum(kind == "attack_unit" for kind in preferred_kinds)),
        "base_draw_pairs": int(sum(kind == "mana_draw" for kind in base_kinds)),
        "base_face_pairs": int(sum(kind == "attack_face" for kind in base_kinds)),
        "anchor_states": int(len(anchors["anchor_observations"])),
        "avg_outcome_margin": float(margins.mean()),
        "mirror_decks": True,
        "v5_second_only": True,
        "v4_teacher_gated_by_terminal_outcome": True,
        "continuation_policy": config.continuation_policy,
        "intermediate_reward_or_score": False,
        "reward_shaping_changed": False,
    }
    return {
        **{key: _stack_dataset_values(key, value) for key, value in rows.items()},
        **{key: _stack_dataset_values(key, value) for key, value in anchors.items()},
        "summary": summary,
    }


def evaluate_terminal_candidate(
    env: Any,
    *,
    candidate: int,
    v5_player_id: int,
    v5: _V5Policy,
    v4: OnnxActionPolicy,
    max_steps: int,
    continuation_policy: str,
) -> int:
    sim = copy.deepcopy(env)
    if sim.current_player_id() != int(v5_player_id):
        raise ValueError("candidate state is not on the V5 turn")
    terminated, truncated = _step_action(sim, int(candidate))
    for _ in range(max(0, int(max_steps) - 1)):
        if terminated or truncated:
            break
        current = sim.current_player_id()
        if current == int(v5_player_id) and continuation_policy == "v5":
            action = v5.select(sim, current)
        else:
            action = int(v4.select_action(sim.env, current))
        terminated, truncated = _step_action(sim, int(action))
    return _outcome(sim, v5_player_id)


def _outcome(env: Any, v5_player_id: int) -> int:
    winner = env.env.winner_id()
    if winner == int(v5_player_id):
        return 1
    if winner in {1, 2}:
        return -1
    return 0


def _sample_records(records: list[_RecordedState], limit: int) -> list[_RecordedState]:
    if len(records) <= limit:
        return records
    indices = np.linspace(0, len(records) - 1, num=limit, dtype=np.int64)
    return [records[int(index)] for index in np.unique(indices)]


def _outcome_candidates(record: _RecordedState, *, max_actions: int) -> list[int]:
    state = record.env.env._env.state
    player_id = record.env.current_player_id()
    unit_trades = [
        int(action)
        for action in np.flatnonzero(record.mask == 1.0)
        if action_kind(state, player_id, int(action)) == "attack_unit"
    ]
    ordered = [record.base_action, record.teacher_action]
    if record.draw_legal:
        ordered.append(DRAW_ACTION)
    ordered.extend(unit_trades)
    ordered.extend(record.ranked_actions)
    selected: list[int] = []
    for action in ordered:
        if action not in selected:
            selected.append(int(action))
        if len(selected) >= max_actions:
            break
    return selected


def _choose_preferred(
    record: _RecordedState, improved: list[tuple[int, int]]
) -> tuple[int, int]:
    best_outcome = max(outcome for _action, outcome in improved)
    best = [action for action, outcome in improved if outcome == best_outcome]
    if record.teacher_action in best:
        return record.teacher_action, best_outcome
    non_draw = [action for action in best if action != DRAW_ACTION]
    return (non_draw[0] if non_draw else best[0]), best_outcome


def _training_config(config: OutcomeRecoveryConfig) -> PreBRecoveryConfig:
    return PreBRecoveryConfig(
        base_checkpoint=config.base_checkpoint,
        v4_model=config.v4_model,
        output_dir=config.output_dir,
        games=config.games,
        anchor_games=config.anchor_games,
        max_steps=config.max_steps,
        seed=config.seed,
        search_candidates=config.ranked_candidates,
        search_depth_plies=1,
        min_score_margin=0.5,
        max_pairs=config.max_pairs,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        ranking_margin=config.ranking_margin,
        action_pair_coef=config.action_pair_coef,
        draw_bce_coef=config.draw_bce_coef,
        recovery_policy_kl_coef=config.recovery_policy_kl_coef,
        recovery_draw_kl_coef=config.recovery_draw_kl_coef,
        anchor_policy_kl_coef=config.anchor_policy_kl_coef,
        anchor_draw_kl_coef=config.anchor_draw_kl_coef,
        min_pairs=config.min_pairs,
        greedy_face_fraction=0.0,
        hard_negative_only=False,
        train_mana_draw_recovery=config.train_mana_draw_recovery,
    )


def _validate_config(config: OutcomeRecoveryConfig) -> None:
    if not config.base_checkpoint.exists():
        raise FileNotFoundError(config.base_checkpoint)
    if not config.v4_model.exists():
        raise FileNotFoundError(config.v4_model)
    if config.continuation_policy not in {"v5", "v4_teacher"}:
        raise ValueError("continuation_policy must be v5 or v4_teacher")
    for name in (
        "games",
        "anchor_games",
        "max_steps",
        "continuation_max_steps",
        "ranked_candidates",
        "max_candidate_actions",
        "max_states_per_loss",
        "max_pairs",
        "min_pairs",
        "epochs",
        "batch_size",
    ):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in (
        "learning_rate",
        "ranking_margin",
        "action_pair_coef",
        "draw_bce_coef",
        "recovery_policy_kl_coef",
        "recovery_draw_kl_coef",
        "anchor_policy_kl_coef",
        "anchor_draw_kl_coef",
    ):
        if float(getattr(config, name)) <= 0.0:
            raise ValueError(f"{name} must be positive")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-model", type=Path, default=DEFAULT_V4_MAX)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--anchor-games", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--continuation-max-steps", type=int, default=300)
    parser.add_argument(
        "--continuation-policy", choices=["v5", "v4_teacher"], default="v4_teacher"
    )
    parser.add_argument("--seed", type=int, default=71430001)
    parser.add_argument("--ranked-candidates", type=int, default=6)
    parser.add_argument("--max-candidate-actions", type=int, default=10)
    parser.add_argument("--max-states-per-loss", type=int, default=6)
    parser.add_argument("--max-pairs", type=int, default=4096)
    parser.add_argument("--min-pairs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ranking-margin", type=float, default=0.5)
    parser.add_argument("--action-pair-coef", type=float, default=1.0)
    parser.add_argument("--draw-bce-coef", type=float, default=0.15)
    parser.add_argument("--recovery-policy-kl-coef", type=float, default=2.0)
    parser.add_argument("--recovery-draw-kl-coef", type=float, default=4.0)
    parser.add_argument("--anchor-policy-kl-coef", type=float, default=4.0)
    parser.add_argument("--anchor-draw-kl-coef", type=float, default=8.0)
    parser.add_argument(
        "--train-mana-draw-recovery",
        action="store_true",
        help="Also train exact-outcome mana-draw labels; default keeps warm draw behavior exact.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = OutcomeRecoveryConfig(
        **{
            **vars(args),
            "base_checkpoint": args.base_checkpoint.resolve(),
            "v4_model": args.v4_model.resolve(),
            "output_dir": args.output_dir.resolve(),
        }
    )
    result = run_outcome_recovery(config)
    print("OUTCOME_PREB_RESULT", json.dumps(result["summary"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

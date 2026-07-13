#!/usr/bin/env python3
"""Counterfactual pre-B recovery for the V5 policy.

The collector deliberately mirrors ``/ai/model_benchmark``: both players get
the same seed-generated deck and card levels.  Only second-start V5 states are
labelled.  Short deterministic rollouts choose between the base action, legal
unit trades, high-ranked policy actions, and the factorized mana-draw action.

Training is conservative: accepted action-vs-action labels use a pairwise
margin, draw-vs-nondraw labels supervise the separate mana-draw head, and both
heads are KL-anchored to the source checkpoint on recovery and ordinary states.
This is an opt-in repair lane before Block B; it does not change environment
rewards or the normal PPO pipeline.
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
for path in (ROOT, TRAINV3_PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ai.train_v2.classic_actions_v1 import decode_action  # noqa: E402
from ai.train_v2.model_mlx import load_checkpoint, save_checkpoint  # noqa: E402
from ai.train_v2.onnx_policy import OnnxActionPolicy  # noqa: E402
from ai.train_v2.policies import GreedyFacePolicy  # noqa: E402
from core.actions import ManaDrawAction  # noqa: E402
from train_v3.contracts import AssistModeV5, InfoModeV5  # noqa: E402
from train_v3.env_v5 import TrainV3ClassicEnv, TrainV3EnvConfig  # noqa: E402
from train_v3.mana_draw_head_v5 import mana_draw_legal_mask  # noqa: E402
from train_v3.v5_policy import create_v5_policy  # noqa: E402


DRAW_ACTION = -1
DEFAULT_V4_MAX = ROOT / "ai" / "models" / "extra-lr-v4-max.onnx"


@dataclass(frozen=True)
class PreBRecoveryConfig:
    base_checkpoint: Path
    v4_model: Path
    output_dir: Path
    games: int = 64
    anchor_games: int = 24
    max_steps: int = 180
    seed: int = 71410001
    search_candidates: int = 10
    search_depth_plies: int = 10
    min_score_margin: float = 8.0
    max_pairs: int = 4096
    epochs: int = 4
    batch_size: int = 128
    learning_rate: float = 1.0e-5
    ranking_margin: float = 0.5
    action_pair_coef: float = 1.0
    draw_bce_coef: float = 0.35
    recovery_policy_kl_coef: float = 2.0
    recovery_draw_kl_coef: float = 2.0
    anchor_policy_kl_coef: float = 4.0
    anchor_draw_kl_coef: float = 4.0
    min_pairs: int = 24
    greedy_face_fraction: float = 0.25
    hard_negative_only: bool = True
    save_dataset: bool = True
    dataset_path: Path | None = None


class _V5Policy:
    def __init__(self, checkpoint: Path):
        self.model = create_v5_policy(
            policy_kind="v5_split_encoder", hidden_dim=256, action_hidden_dim=128
        )
        self.loaded = load_checkpoint(str(checkpoint), self.model)

    def select(self, env: TrainV3ClassicEnv, player_id: int) -> int:
        import mlx.core as mx

        obs = env.observe(player_id).astype(np.float32, copy=False)
        features = env.action_features(player_id, include_preview=False).astype(np.float32, copy=False)
        mask = env.action_mask(player_id).astype(np.float32, copy=False)
        legal_draw = mana_draw_legal_mask(env.env._env.state, player_id)
        logits, _value, draw_logit = self.model(
            mx.array(obs[None, :]), mx.array(features[None, :, :])
        )
        mx.eval(logits, draw_logit)
        masked = np.where(mask.astype(bool), np.asarray(logits, dtype=np.float32)[0], -1.0e9)
        if legal_draw and float(np.asarray(draw_logit).reshape(-1)[0]) > 0.0:
            return DRAW_ACTION
        return int(np.argmax(masked))

    def ranked_actions(
        self,
        obs: np.ndarray,
        features: np.ndarray,
        mask: np.ndarray,
        limit: int,
    ) -> list[int]:
        import mlx.core as mx

        logits, _value, _draw = self.model(
            mx.array(obs[None, :].astype(np.float32, copy=False)),
            mx.array(features[None, :, :].astype(np.float32, copy=False)),
        )
        mx.eval(logits)
        scores = np.asarray(logits, dtype=np.float32)[0]
        legal = np.flatnonzero(mask == 1.0)
        order = sorted((int(action) for action in legal), key=lambda action: (-float(scores[action]), action))
        return order[: max(1, int(limit))]


def run_preB_recovery(config: PreBRecoveryConfig) -> dict[str, Any]:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = (
        load_counterfactual_dataset(config.dataset_path)
        if config.dataset_path is not None
        else collect_counterfactual_dataset(config)
    )
    if int(dataset["summary"]["pairs"]) < int(config.min_pairs):
        raise RuntimeError(
            f"counterfactual pairs={dataset['summary']['pairs']} below min_pairs={config.min_pairs}"
        )
    if config.save_dataset:
        _save_dataset(config.output_dir / "preB_counterfactual_dataset.npz", dataset)
    checkpoint_path, train_summary = train_counterfactual_recovery(config, dataset)
    result = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_summary": dataset["summary"],
        "train_summary": train_summary,
        "summary": {
            "status": "ok",
            "checkpoint_path": str(checkpoint_path),
            "pairs": int(dataset["summary"]["pairs"]),
            "action_pairs": int(dataset["summary"]["action_pairs"]),
            "draw_pairs": int(dataset["summary"]["draw_pairs"]),
            "preferred_draw_pairs": int(dataset["summary"]["preferred_draw_pairs"]),
            "preferred_trade_pairs": int(dataset["summary"]["preferred_trade_pairs"]),
            "final_loss": float(train_summary["final_loss"]),
        },
    }
    (config.output_dir / "preB_summary.json").write_text(
        json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def collect_counterfactual_dataset(config: PreBRecoveryConfig) -> dict[str, Any]:
    v5 = _V5Policy(config.base_checkpoint)
    v4 = OnnxActionPolicy(str(config.v4_model), mode="argmax", seed=config.seed, verify_mask=False)
    greedy_face = GreedyFacePolicy()
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
        "v5_states": 0,
        "gated_states": 0,
        "searched_states": 0,
        "candidate_evals": 0,
        "base_face_behind_states": 0,
        "v4_max_games": 0,
        "greedy_face_games": 0,
        "hard_negative_rejections": 0,
    }

    for game_idx in range(int(config.games)):
        seed = int(config.seed) + game_idx
        v5_player_id = 1 if game_idx % 2 == 0 else 2
        starting_player_id = 2 if v5_player_id == 1 else 1
        env = _new_mirror_env(seed=seed, starting_player_id=starting_player_id)
        # Deterministic stratification guarantees the requested lane share even
        # in short probes and keeps a resumed run reproducible by game index.
        greedy_before = math.floor(game_idx * float(config.greedy_face_fraction))
        greedy_after = math.floor((game_idx + 1) * float(config.greedy_face_fraction))
        use_greedy = greedy_after > greedy_before
        opponent: Any = greedy_face if use_greedy else v4
        lane_name = "greedy_face" if use_greedy else "v4_max"
        stats[f"{lane_name}_games"] += 1
        if hasattr(opponent, "reset"):
            opponent.reset(seed * 13 + (3 - v5_player_id))
        stats["games"] += 1
        for _step in range(int(config.max_steps)):
            current = env.current_player_id()
            if current == v5_player_id:
                obs = env.observe(current).astype(np.float32, copy=True)
                features = env.action_features(current, include_preview=False).astype(np.float32, copy=True)
                mask = env.action_mask(current).astype(np.float32, copy=True)
                draw_legal = bool(mana_draw_legal_mask(env.env._env.state, current))
                base_action = int(v5.select(env, current))
                stats["v5_states"] += 1
                reason = recovery_gate_reason(env, current, base_action)
                if reason:
                    stats["gated_states"] += 1
                    if reason & 8:
                        stats["base_face_behind_states"] += 1
                    teacher_action = None
                    if lane_name == "v4_max":
                        teacher_action = int(v4.select_action(env.env, current))
                    candidates = recovery_candidates(
                        env=env,
                        player_id=current,
                        base_action=base_action,
                        ranked=v5.ranked_actions(obs, features, mask, config.search_candidates),
                        draw_legal=draw_legal,
                        teacher_action=teacher_action,
                    )
                    scored: list[tuple[int, float]] = []
                    for candidate in candidates:
                        score = evaluate_candidate(
                            env=env,
                            candidate=candidate,
                            v5_player_id=v5_player_id,
                            v5=v5,
                            opponent=opponent,
                            depth_plies=config.search_depth_plies,
                        )
                        if math.isfinite(score):
                            scored.append((int(candidate), float(score)))
                    stats["candidate_evals"] += len(scored)
                    if scored:
                        stats["searched_states"] += 1
                        score_by_action = dict(scored)
                        base_score = score_by_action.get(base_action)
                        preferred, preferred_score = max(scored, key=lambda item: (item[1], -item[0]))
                        if (
                            base_score is not None
                            and preferred != base_action
                            and preferred_score >= base_score + float(config.min_score_margin)
                        ):
                            preferred_kind = action_kind(
                                env.env._env.state, current, preferred
                            )
                            base_kind = action_kind(
                                env.env._env.state, current, base_action
                            )
                            hard_negative = (
                                base_kind == "attack_face"
                                and preferred_kind != "attack_face"
                            ) or DRAW_ACTION in {preferred, base_action}
                            if config.hard_negative_only and not hard_negative:
                                stats["hard_negative_rejections"] += 1
                            else:
                                _append_pair(
                                    rows,
                                    obs=obs,
                                    features=features,
                                    mask=mask,
                                    draw_legal=draw_legal,
                                    preferred=preferred,
                                    base_action=base_action,
                                    margin=preferred_score - base_score,
                                    gate_reason=reason,
                                    seed=seed,
                                    state=env.env._env.state,
                                    player_id=current,
                                )
                                if len(rows["positive_actions"]) >= int(config.max_pairs):
                                    break
                action = base_action
            else:
                action = int(opponent.select_action(env.env, current))
            terminated, truncated = _step_action(env, action)
            if terminated or truncated:
                stats["terminal_games"] += 1
                break
        if len(rows["positive_actions"]) >= int(config.max_pairs):
            break
        if (game_idx + 1) % 16 == 0:
            print(
                f"PREB_COLLECT games={game_idx + 1}/{config.games} pairs={len(rows['positive_actions'])} "
                f"gated={stats['gated_states']} evals={stats['candidate_evals']}",
                flush=True,
            )

    _collect_anchor_states(config, v5=v5, v4=v4, anchors=anchors)
    if not rows["positive_actions"]:
        raise RuntimeError("counterfactual recovery dataset is empty")
    action_pair_mask = np.asarray(rows["action_pair_mask"], dtype=np.bool_)
    draw_mask = np.asarray(rows["draw_supervision_mask"], dtype=np.bool_)
    preferred_kinds = list(rows["preferred_action_kinds"])
    base_kinds = list(rows["base_action_kinds"])
    margins = np.asarray(rows["score_margins"], dtype=np.float32)
    summary = {
        "schema": "extra_lr_v5_preB_counterfactual_recovery_v1",
        "collection_mode": "model_benchmark_mirror_decks_v5_second",
        **{key: int(value) for key, value in stats.items()},
        "pairs": int(len(rows["positive_actions"])),
        "action_pairs": int(action_pair_mask.sum()),
        "draw_pairs": int(draw_mask.sum()),
        "preferred_draw_pairs": int(sum(kind == "mana_draw" for kind in preferred_kinds)),
        "preferred_trade_pairs": int(sum(kind == "attack_unit" for kind in preferred_kinds)),
        "base_face_pairs": int(sum(kind == "attack_face" for kind in base_kinds)),
        "anchor_states": int(len(anchors["anchor_observations"])),
        "avg_score_margin": float(margins.mean()),
        "min_score_margin": float(margins.min()),
        "max_score_margin": float(margins.max()),
        "mirror_decks": True,
        "v5_second_only": True,
        "reward_shaping_changed": False,
        "greedy_face_fraction": float(config.greedy_face_fraction),
        "hard_negative_only": bool(config.hard_negative_only),
    }
    return {
        **{key: _stack_dataset_values(key, value) for key, value in rows.items()},
        **{key: _stack_dataset_values(key, value) for key, value in anchors.items()},
        "summary": summary,
    }


def recovery_gate_reason(env: TrainV3ClassicEnv, player_id: int, base_action: int) -> int:
    state = env.env._env.state
    me = state.p1 if state.p1.user_id == int(player_id) else state.p2
    enemy = state.p2 if state.p1.user_id == int(player_id) else state.p1
    own_power = _board_power(me.board)
    enemy_power = _board_power(enemy.board)
    reason = 0
    if int(enemy.hero.hp) - int(me.hero.hp) >= 4:
        reason |= 1
    if enemy_power > own_power:
        reason |= 2
    if int(me.hero.hp) <= 20:
        reason |= 4
    if enemy_power > own_power and action_kind(state, player_id, base_action) == "attack_face":
        reason |= 8
    # A merely legal draw is not itself a recovery state.  Requiring an HP or
    # board crisis prevents the repair lane from teaching generic draw greed.
    if reason and mana_draw_legal_mask(state, player_id) and base_action != DRAW_ACTION:
        reason |= 16
    return reason


def recovery_candidates(
    *,
    env: TrainV3ClassicEnv,
    player_id: int,
    base_action: int,
    ranked: list[int],
    draw_legal: bool,
    teacher_action: int | None = None,
) -> list[int]:
    mask = env.action_mask(player_id)
    state = env.env._env.state
    selected = {int(base_action)}
    if teacher_action is not None and 0 <= int(teacher_action) < len(mask) and mask[int(teacher_action)] == 1.0:
        selected.add(int(teacher_action))
    selected.update(int(action) for action in ranked if 0 <= int(action) < len(mask) and mask[int(action)] == 1.0)
    for action in np.flatnonzero(mask == 1.0):
        if action_kind(state, player_id, int(action)) == "attack_unit":
            selected.add(int(action))
    if draw_legal:
        selected.add(DRAW_ACTION)
    return sorted(selected, key=lambda action: (action == DRAW_ACTION, action))


def evaluate_candidate(
    *,
    env: TrainV3ClassicEnv,
    candidate: int,
    v5_player_id: int,
    v5: _V5Policy,
    opponent: Any,
    depth_plies: int,
) -> float:
    sim = copy.deepcopy(env)
    if sim.current_player_id() != int(v5_player_id):
        return -math.inf
    terminated, truncated = _step_action(sim, int(candidate))
    for _ in range(max(0, int(depth_plies))):
        if terminated or truncated:
            break
        current = sim.current_player_id()
        action = v5.select(sim, current) if current == int(v5_player_id) else int(opponent.select_action(sim.env, current))
        terminated, truncated = _step_action(sim, int(action))
    return score_recovery_state(sim, v5_player_id)


def score_recovery_state(env: TrainV3ClassicEnv, v5_player_id: int) -> float:
    state = env.env._env.state
    enemy_id = 2 if int(v5_player_id) == 1 else 1
    me = state.p1 if state.p1.user_id == int(v5_player_id) else state.p2
    enemy = state.p1 if state.p1.user_id == enemy_id else state.p2
    winner = env.env.winner_id()
    terminal = 5000.0 if winner == int(v5_player_id) else -5000.0 if winner == enemy_id else 0.0
    hp_margin = float(me.hero.hp - enemy.hero.hp)
    board_margin = float(_board_power(me.board) - _board_power(enemy.board))
    hand_margin = float(len(me.hand) - len(enemy.hand))
    survival = -250.0 if int(me.hero.hp) <= 6 and int(enemy.hero.hp) > int(me.hero.hp) else 0.0
    return terminal + 12.0 * hp_margin + 1.5 * board_margin + 0.5 * hand_margin + survival


def train_counterfactual_recovery(
    config: PreBRecoveryConfig, dataset: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    # The migrated V2 interaction parameters are absent from old warm
    # checkpoints. Seed both constructions identically so their zero-gated
    # latent query/key initializations are reproducible and matched.
    mx.random.seed(int(config.seed) + 7000)
    model = create_v5_policy(policy_kind="v5_split_encoder", hidden_dim=256, action_hidden_dim=128)
    mx.random.seed(int(config.seed) + 7000)
    reference = create_v5_policy(policy_kind="v5_split_encoder", hidden_dim=256, action_hidden_dim=128)
    optimizer = optim.Adam(learning_rate=float(config.learning_rate))
    loaded = load_checkpoint(str(config.base_checkpoint), model, optimizer=None)
    load_checkpoint(str(config.base_checkpoint), reference, optimizer=None)
    n = int(dataset["observations"].shape[0])
    anchor_n = int(dataset["anchor_observations"].shape[0])
    rng = np.random.default_rng(int(config.seed) + 7001)
    epoch_metrics: list[dict[str, float]] = []
    trainable_prefixes = (
        "state_action_query_v2.",
        "state_action_key_v2.",
        "state_action_gate_v2.",
        "mana_draw_recovery_head.",
    )

    for epoch in range(int(config.epochs)):
        order = np.arange(n, dtype=np.int64)
        rng.shuffle(order)
        bucket: dict[str, list[float]] = {
            key: []
            for key in ("loss", "action_pair", "draw_bce", "recovery_policy_kl", "recovery_draw_kl", "anchor_policy_kl", "anchor_draw_kl", "action_accuracy", "draw_accuracy")
        }
        for start in range(0, n, int(config.batch_size)):
            idx = order[start : start + int(config.batch_size)]
            anchor_idx = rng.integers(0, anchor_n, size=max(1, len(idx)), dtype=np.int64)
            obs = mx.array(dataset["observations"][idx])
            features = mx.array(dataset["action_features"][idx])
            mask = mx.array(dataset["masks"][idx])
            positive = mx.array(dataset["positive_actions"][idx], dtype=mx.int32)
            negative = mx.array(dataset["negative_actions"][idx], dtype=mx.int32)
            action_rows = mx.array(dataset["action_pair_mask"][idx])
            draw_rows = mx.array(dataset["draw_supervision_mask"][idx])
            draw_targets = mx.array(dataset["draw_targets"][idx])
            draw_legal = mx.array(dataset["mana_draw_legal"][idx])
            anchor_obs = mx.array(dataset["anchor_observations"][anchor_idx])
            anchor_features = mx.array(dataset["anchor_action_features"][anchor_idx])
            anchor_mask = mx.array(dataset["anchor_masks"][anchor_idx])
            anchor_draw_legal = mx.array(dataset["anchor_mana_draw_legal"][anchor_idx])

            ref_logits, _ref_value, ref_draw = reference(obs, features)
            ref_anchor_logits, _ref_anchor_value, ref_anchor_draw = reference(anchor_obs, anchor_features)
            mx.eval(ref_logits, ref_draw, ref_anchor_logits, ref_anchor_draw)

            def loss_fn(current_model):
                logits, _value, draw_logit = current_model(obs, features)
                anchor_logits, _anchor_value, anchor_draw_logit = current_model(anchor_obs, anchor_features)
                valid = mask.astype(mx.bool_)
                masked = mx.where(valid, logits, mx.array(-1.0e9, dtype=mx.float32))
                row = mx.arange(positive.shape[0])
                diff = masked[row, positive] - masked[row, negative]
                pair_hinge = mx.maximum(mx.array(0.0), mx.array(float(config.ranking_margin)) - diff)
                action_count = mx.maximum(mx.sum(action_rows.astype(mx.float32)), mx.array(1.0))
                action_pair_loss = mx.sum(pair_hinge * action_rows.astype(mx.float32)) / action_count
                action_accuracy = mx.sum((diff > 0.0).astype(mx.float32) * action_rows.astype(mx.float32)) / action_count

                draw_prob = mx.clip(mx.sigmoid(draw_logit), 1.0e-7, 1.0 - 1.0e-7)
                draw_ce = -(draw_targets * mx.log(draw_prob) + (1.0 - draw_targets) * mx.log(1.0 - draw_prob))
                draw_count = mx.maximum(mx.sum(draw_rows.astype(mx.float32)), mx.array(1.0))
                draw_bce = mx.sum(draw_ce * draw_rows.astype(mx.float32)) / draw_count
                draw_accuracy = mx.sum(((draw_prob > 0.5) == (draw_targets > 0.5)).astype(mx.float32) * draw_rows.astype(mx.float32)) / draw_count

                recovery_policy_kl = _masked_ref_kl(ref_logits, logits, mask)
                recovery_draw_kl = _bernoulli_ref_kl(ref_draw, draw_logit, draw_legal)
                anchor_policy_kl = _masked_ref_kl(ref_anchor_logits, anchor_logits, anchor_mask)
                anchor_draw_kl = _bernoulli_ref_kl(ref_anchor_draw, anchor_draw_logit, anchor_draw_legal)
                total = (
                    float(config.action_pair_coef) * action_pair_loss
                    + float(config.draw_bce_coef) * draw_bce
                    + float(config.recovery_policy_kl_coef) * recovery_policy_kl
                    + float(config.recovery_draw_kl_coef) * recovery_draw_kl
                    + float(config.anchor_policy_kl_coef) * anchor_policy_kl
                    + float(config.anchor_draw_kl_coef) * anchor_draw_kl
                )
                return total, {
                    "action_pair": action_pair_loss,
                    "draw_bce": draw_bce,
                    "recovery_policy_kl": recovery_policy_kl,
                    "recovery_draw_kl": recovery_draw_kl,
                    "anchor_policy_kl": anchor_policy_kl,
                    "anchor_draw_kl": anchor_draw_kl,
                    "action_accuracy": action_accuracy,
                    "draw_accuracy": draw_accuracy,
                }

            value_and_grad = nn.value_and_grad(model, loss_fn)
            (loss_value, aux), grads = value_and_grad(model)
            grads = _zero_grads_except(grads, trainable_prefixes)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss_value, *aux.values())
            bucket["loss"].append(float(loss_value.item()))
            for key, value in aux.items():
                bucket[key].append(float(value.item()))
        metrics = {key: float(np.mean(values)) for key, values in bucket.items()}
        metrics["epoch"] = float(epoch + 1)
        epoch_metrics.append(metrics)
        print("PREB_EPOCH", json.dumps(metrics, sort_keys=True), flush=True)

    checkpoint = config.output_dir / f"extra_lr_v5_preB_counterfactual_{n}_pairs.npz"
    metadata = {
        "run_name": "preB_counterfactual_recovery",
        "phase": "preB",
        "source_checkpoint": str(config.base_checkpoint.resolve()),
        "source_metadata": loaded.get("metadata", {}),
        "config": _jsonable(asdict(config)),
        "dataset": dataset["summary"],
        "training": epoch_metrics,
        "objective_benchmark": "/ai/model_benchmark",
        "mirror_decks": True,
        "reward_shaping_changed": False,
        "loss": "counterfactual_action_pair_plus_draw_bce_plus_reference_kl",
        "trainable_parameter_prefixes": list(trainable_prefixes),
    }
    save_checkpoint(str(checkpoint), model, optimizer=optimizer, metadata=metadata)
    return checkpoint, {
        "epochs": int(config.epochs),
        "pairs": n,
        "anchor_states": anchor_n,
        "epoch_metrics": epoch_metrics,
        "final_loss": float(epoch_metrics[-1]["loss"]),
        "final_action_accuracy": float(epoch_metrics[-1]["action_accuracy"]),
        "final_draw_accuracy": float(epoch_metrics[-1]["draw_accuracy"]),
    }


def _append_pair(
    rows: dict[str, list[Any]],
    *,
    obs: np.ndarray,
    features: np.ndarray,
    mask: np.ndarray,
    draw_legal: bool,
    preferred: int,
    base_action: int,
    margin: float,
    gate_reason: int,
    seed: int,
    state: Any,
    player_id: int,
) -> None:
    action_pair = preferred != DRAW_ACTION and base_action != DRAW_ACTION
    draw_supervision = preferred == DRAW_ACTION or base_action == DRAW_ACTION
    rows["observations"].append(obs)
    rows["action_features"].append(features)
    rows["masks"].append(mask)
    rows["mana_draw_legal"].append(bool(draw_legal))
    rows["positive_actions"].append(int(preferred if preferred != DRAW_ACTION else 0))
    rows["negative_actions"].append(int(base_action if base_action != DRAW_ACTION else 0))
    rows["action_pair_mask"].append(bool(action_pair))
    rows["draw_supervision_mask"].append(bool(draw_supervision))
    rows["draw_targets"].append(float(preferred == DRAW_ACTION))
    rows["score_margins"].append(float(margin))
    rows["base_action_codes"].append(int(base_action))
    rows["preferred_action_codes"].append(int(preferred))
    rows["preferred_action_kinds"].append(action_kind(state, player_id, preferred))
    rows["base_action_kinds"].append(action_kind(state, player_id, base_action))
    rows["gate_reasons"].append(int(gate_reason))
    rows["seeds"].append(int(seed))


def _collect_anchor_states(
    config: PreBRecoveryConfig,
    *,
    v5: _V5Policy,
    v4: OnnxActionPolicy,
    anchors: dict[str, list[Any]],
) -> None:
    for game_idx in range(int(config.anchor_games)):
        seed = int(config.seed) + 1_000_000 + game_idx
        v5_player_id = 1 if game_idx % 2 == 0 else 2
        starting_player_id = 1 if game_idx % 4 < 2 else 2
        env = _new_mirror_env(seed=seed, starting_player_id=starting_player_id)
        v4.reset(seed * 17 + (3 - v5_player_id))
        for _step in range(int(config.max_steps)):
            current = env.current_player_id()
            if current == v5_player_id:
                anchors["anchor_observations"].append(env.observe(current).astype(np.float32, copy=True))
                anchors["anchor_action_features"].append(env.action_features(current, include_preview=False).astype(np.float32, copy=True))
                anchors["anchor_masks"].append(env.action_mask(current).astype(np.float32, copy=True))
                anchors["anchor_mana_draw_legal"].append(bool(mana_draw_legal_mask(env.env._env.state, current)))
                action = v5.select(env, current)
            else:
                action = int(v4.select_action(env.env, current))
            terminated, truncated = _step_action(env, int(action))
            if terminated or truncated:
                break


def _new_mirror_env(*, seed: int, starting_player_id: int) -> TrainV3ClassicEnv:
    env = TrainV3ClassicEnv(
        TrainV3EnvConfig(
            seed=int(seed),
            max_turns=120,
            verify_mask=False,
            placement_mode="append_only",
            include_legal_actions_in_info=False,
            info_mode=InfoModeV5(),
            assist_mode=AssistModeV5(),
            history_limit=20,
        )
    )
    levels = {int(card_id): 4 for card_id in env.env._cards_data}
    env.reset(
        p1_levels=levels,
        p2_levels=levels,
        p1_is_bot=True,
        p2_is_bot=True,
        starting_player_id=int(starting_player_id),
        seed=int(seed),
    )
    return env


def action_kind(state: Any, player_id: int, action: int) -> str:
    if int(action) == DRAW_ACTION:
        return "mana_draw"
    decoded = decode_action(state, int(player_id), int(action))
    if decoded is None:
        return "unknown"
    data = decoded.to_dict()
    kind = str(data.get("type") or "unknown")
    if kind == "attack":
        return "attack_face" if data.get("target_id") is None else "attack_unit"
    return kind


def _step_action(env: TrainV3ClassicEnv, action: int) -> tuple[bool, bool]:
    if int(action) == DRAW_ACTION:
        _obs, _reward, terminated, truncated, _info = env.step_core_action(ManaDrawAction())
    else:
        _obs, _reward, terminated, truncated, _info = env.step(int(action))
    return bool(terminated), bool(truncated)


def _board_power(board: Any) -> int:
    return int(sum(max(0, int(card.attack)) * max(0, int(card.hp)) for card in board))


def _masked_ref_kl(ref_logits: Any, logits: Any, mask: Any) -> Any:
    import mlx.core as mx

    valid = mask.astype(mx.bool_)
    neg = mx.array(-1.0e9, dtype=mx.float32)
    ref_masked = mx.where(valid, ref_logits, neg)
    current_masked = mx.where(valid, logits, neg)
    ref_log_probs = ref_masked - mx.logsumexp(ref_masked, axis=-1, keepdims=True)
    log_probs = current_masked - mx.logsumexp(current_masked, axis=-1, keepdims=True)
    ref_probs = mx.exp(ref_log_probs)
    terms = mx.where(valid, ref_probs * (ref_log_probs - log_probs), mx.array(0.0))
    return mx.mean(mx.sum(terms, axis=-1))


def _bernoulli_ref_kl(ref_logits: Any, logits: Any, legal_mask: Any) -> Any:
    import mlx.core as mx

    legal = legal_mask.astype(mx.float32)
    p = mx.clip(mx.sigmoid(ref_logits), 1.0e-7, 1.0 - 1.0e-7)
    q = mx.clip(mx.sigmoid(logits), 1.0e-7, 1.0 - 1.0e-7)
    terms = p * (mx.log(p) - mx.log(q)) + (1.0 - p) * (mx.log(1.0 - p) - mx.log(1.0 - q))
    denom = mx.maximum(mx.sum(legal), mx.array(1.0))
    return mx.sum(terms * legal) / denom


def _zero_grads_except(grads: Any, trainable_prefixes: tuple[str, ...]) -> Any:
    """Keep the warm policy byte-identical outside the new repair capacity."""
    import mlx.core as mx
    import mlx.nn as nn

    flat = nn.utils.tree_flatten(grads)
    return nn.utils.tree_unflatten(
        [
            (
                name,
                value if name.startswith(trainable_prefixes) else mx.zeros_like(value),
            )
            for name, value in flat
        ]
    )


def _stack_dataset_values(key: str, values: list[Any]) -> np.ndarray:
    if key in {"observations", "action_features", "masks", "anchor_observations", "anchor_action_features", "anchor_masks"}:
        return np.stack(values).astype(np.float32, copy=False)
    if key in {"mana_draw_legal", "action_pair_mask", "draw_supervision_mask", "anchor_mana_draw_legal"}:
        return np.asarray(values, dtype=np.bool_)
    if key in {"draw_targets", "score_margins"}:
        return np.asarray(values, dtype=np.float32)
    if key in {"preferred_action_kinds", "base_action_kinds"}:
        return np.asarray(values, dtype="U32")
    return np.asarray(values, dtype=np.int64)


def _save_dataset(path: Path, dataset: dict[str, Any]) -> None:
    np.savez_compressed(path, **{key: value for key, value in dataset.items() if key != "summary"})
    path.with_suffix(".summary.json").write_text(
        json.dumps(dataset["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_counterfactual_dataset(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"counterfactual dataset not found: {path}")
    summary_path = path.with_suffix(".summary.json")
    if not summary_path.exists():
        raise FileNotFoundError(f"counterfactual dataset summary not found: {summary_path}")
    loaded = np.load(path, allow_pickle=False)
    dataset = {key: loaded[key] for key in loaded.files}
    dataset["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    return dataset


def _validate_config(config: PreBRecoveryConfig) -> None:
    if not config.base_checkpoint.exists():
        raise FileNotFoundError(f"base checkpoint not found: {config.base_checkpoint}")
    if not config.v4_model.exists():
        raise FileNotFoundError(f"V4 model not found: {config.v4_model}")
    for name in ("games", "anchor_games", "max_steps", "search_candidates", "search_depth_plies", "max_pairs", "epochs", "batch_size", "min_pairs"):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("min_score_margin", "learning_rate", "ranking_margin", "action_pair_coef", "draw_bce_coef", "recovery_policy_kl_coef", "recovery_draw_kl_coef", "anchor_policy_kl_coef", "anchor_draw_kl_coef"):
        if float(getattr(config, name)) <= 0.0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 <= float(config.greedy_face_fraction) <= 1.0:
        raise ValueError("greedy_face_fraction must be between 0 and 1")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-model", type=Path, default=DEFAULT_V4_MAX)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--anchor-games", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=71410001)
    parser.add_argument("--search-candidates", type=int, default=10)
    parser.add_argument("--search-depth-plies", type=int, default=10)
    parser.add_argument("--min-score-margin", type=float, default=8.0)
    parser.add_argument("--max-pairs", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--ranking-margin", type=float, default=0.5)
    parser.add_argument("--action-pair-coef", type=float, default=1.0)
    parser.add_argument("--draw-bce-coef", type=float, default=0.35)
    parser.add_argument("--recovery-policy-kl-coef", type=float, default=2.0)
    parser.add_argument("--recovery-draw-kl-coef", type=float, default=2.0)
    parser.add_argument("--anchor-policy-kl-coef", type=float, default=4.0)
    parser.add_argument("--anchor-draw-kl-coef", type=float, default=4.0)
    parser.add_argument("--min-pairs", type=int, default=24)
    parser.add_argument("--greedy-face-fraction", type=float, default=0.25)
    parser.add_argument(
        "--include-all-counterfactuals",
        action="store_true",
        help="Keep generic action-order labels in addition to face/trade and draw hard negatives.",
    )
    parser.add_argument("--no-save-dataset", action="store_true")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Reuse an existing preB_counterfactual_dataset.npz instead of collecting games.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = PreBRecoveryConfig(
        base_checkpoint=args.base_checkpoint.resolve(),
        v4_model=args.v4_model.resolve(),
        output_dir=args.output_dir.resolve(),
        games=args.games,
        anchor_games=args.anchor_games,
        max_steps=args.max_steps,
        seed=args.seed,
        search_candidates=args.search_candidates,
        search_depth_plies=args.search_depth_plies,
        min_score_margin=args.min_score_margin,
        max_pairs=args.max_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ranking_margin=args.ranking_margin,
        action_pair_coef=args.action_pair_coef,
        draw_bce_coef=args.draw_bce_coef,
        recovery_policy_kl_coef=args.recovery_policy_kl_coef,
        recovery_draw_kl_coef=args.recovery_draw_kl_coef,
        anchor_policy_kl_coef=args.anchor_policy_kl_coef,
        anchor_draw_kl_coef=args.anchor_draw_kl_coef,
        min_pairs=args.min_pairs,
        greedy_face_fraction=args.greedy_face_fraction,
        hard_negative_only=not args.include_all_counterfactuals,
        save_dataset=not args.no_save_dataset,
        dataset_path=args.dataset_path.resolve() if args.dataset_path is not None else None,
    )
    result = run_preB_recovery(config)
    print("PREB_RESULT", json.dumps(result["summary"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

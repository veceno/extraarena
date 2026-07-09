#!/usr/bin/env python3
"""Run V5 ExtraLR Phase A as random-heavy Rust ArenaEnv PPO bootstrap.

This is the replacement for the failed first-phase semi-synthetic
LLM/V4Max-distillation idea. It trains directly in the Rust ArenaEnv against a
teacher-free random-heavy opponent mix, saving periodic MLX checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAINV3_PYTHON = ROOT / "TrainV3.5" / "python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAINV3_PYTHON) not in sys.path:
    sys.path.insert(0, str(TRAINV3_PYTHON))

from ai.train_v2.model_mlx import load_checkpoint, save_checkpoint  # noqa: E402
from train_v3.contracts import ACTION_FEATURE_DIM, OBS_V5_DIM  # noqa: E402
from train_v3.ppo_phaseA_config import (  # noqa: E402
    PHASE_A_RANDOM_BOOTSTRAP_TARGET_RANDOM_SCORE,
    build_phase_a_random_bootstrap_config,
)
from train_v3.league_v5 import parse_v5_opponent_mix  # noqa: E402
from train_v3.rust_live_self_play import LearnerCtxBatch, OpponentCtx, run_live_self_play_update  # noqa: E402
from train_v3.rust_policy import score_padded_legal_actions  # noqa: E402
from train_v3.v5_policy import create_v5_policy  # noqa: E402
from train_v3.warm_start_v5 import load_v4_max_into_v5, resolve_v4_max_npz_path  # noqa: E402


class MLXV5LearnerPolicy:
    """Adapter from V5 MLX policy to A4 live self-play learner protocol."""

    def __init__(
        self,
        model: Any,
        *,
        library_path: str | Path | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.model = model
        self.library_path = library_path
        self.rng = rng or np.random.default_rng()

    def select(self, ctx: LearnerCtxBatch) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        envs = np.asarray(ctx.env_indices, dtype=np.intp)
        obs = np.asarray(ctx.observation_v5, dtype=np.float32)[envs]
        counts, offsets, ids, features = _compact_legal_subset(
            envs=envs,
            legal_action_counts=ctx.legal_action_counts,
            legal_action_offsets=ctx.legal_action_offsets,
            legal_action_ids=ctx.legal_action_ids,
            legal_action_features=ctx.legal_action_features,
        )
        scores = score_padded_legal_actions(
            self.model,
            obs,
            counts,
            features,
            legal_action_offsets=offsets,
            legal_action_ids=ids,
            padding_backend="python",
            mask_invalid_logits=True,
            library_path=self.library_path,
        )
        if scores.mana_draw_logits is None:
            raise ValueError("V5 learner requires a mana_draw_head for live self-play")
        _mx_eval(scores.padded_logits, scores.values, scores.mana_draw_logits)
        logits = np.asarray(scores.padded_logits, dtype=np.float32)
        values = np.asarray(scores.values, dtype=np.float32)
        mana_draw_logits = np.asarray(scores.mana_draw_logits, dtype=np.float32)
        actions, log_probs, selected_local, mana_draw_flags = _select_from_joint_padded(
            logits=logits,
            counts=counts,
            ids=ids,
            mana_draw_logits=mana_draw_logits,
            mana_draw_legal=np.asarray(ctx.mana_draw_legal, dtype=np.bool_)[envs],
            rng=self.rng,
        )
        return actions, values, log_probs, selected_local, mana_draw_flags

    def argmax_select(self, ctx: OpponentCtx) -> int:
        action, _mana_draw = self.argmax_select_with_mana(ctx)
        return action

    def argmax_select_with_mana(self, ctx: OpponentCtx) -> tuple[int, bool]:
        """Return the deterministic best action from the joint legal action space.

        ``mana_draw`` remains a parallel FFI flag, but for policy selection and
        PPO it is a first-class legal option competing with the 601 candidates.
        """
        if ctx.legal_action_features is None:
            raise ValueError("V5 learner requires legal_action_features for opponent argmax")
        counts = np.asarray([int(ctx.legal_action_counts)], dtype=np.uintp)
        offsets = np.asarray([0], dtype=np.uintp)
        ids = np.asarray(ctx.legal_action_ids, dtype=np.uintp)
        obs = np.asarray(ctx.observation_v5, dtype=np.float32)[None, :]
        features = np.asarray(ctx.legal_action_features, dtype=np.float32)
        scores = score_padded_legal_actions(
            self.model,
            obs,
            counts,
            features,
            legal_action_offsets=offsets,
            legal_action_ids=ids,
            padding_backend="python",
            mask_invalid_logits=True,
            library_path=self.library_path,
        )
        if scores.mana_draw_logits is None:
            raise ValueError("V5 learner requires a mana_draw_head for live inference")
        _mx_eval(scores.padded_logits, scores.values, scores.mana_draw_logits)
        actions, _log_probs, _selected_local, mana_draw = _select_from_joint_padded(
            logits=np.asarray(scores.padded_logits, dtype=np.float32),
            counts=counts,
            ids=ids,
            mana_draw_logits=np.asarray(scores.mana_draw_logits, dtype=np.float32),
            mana_draw_legal=np.asarray([bool(ctx.mana_draw_legal)], dtype=np.bool_),
            rng=None,
        )
        return int(actions[0]), bool(mana_draw[0])


def _compact_legal_subset(
    *,
    envs: np.ndarray,
    legal_action_counts: np.ndarray,
    legal_action_offsets: np.ndarray,
    legal_action_ids: np.ndarray,
    legal_action_features: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if legal_action_features is None:
        raise ValueError("V5 learner requires legal_action_features")
    counts_full = np.asarray(legal_action_counts, dtype=np.intp)
    offsets_full = np.asarray(legal_action_offsets, dtype=np.intp)
    ids_full = np.asarray(legal_action_ids, dtype=np.uintp)
    features_full = np.asarray(legal_action_features, dtype=np.float32)
    counts: list[int] = []
    ids_parts: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    compact_offsets: list[int] = []
    cursor = 0
    for env in envs.tolist():
        count = int(counts_full[int(env)])
        offset = int(offsets_full[int(env)])
        if count <= 0:
            raise ValueError(f"env {env} has no legal actions")
        compact_offsets.append(cursor)
        counts.append(count)
        ids_parts.append(ids_full[offset : offset + count])
        feature_parts.append(features_full[offset : offset + count])
        cursor += count
    return (
        np.asarray(counts, dtype=np.uintp),
        np.asarray(compact_offsets, dtype=np.uintp),
        np.concatenate(ids_parts).astype(np.uintp, copy=False),
        np.concatenate(feature_parts).astype(np.float32, copy=False),
    )


def _select_from_joint_padded(
    *,
    logits: np.ndarray,
    counts: np.ndarray,
    ids: np.ndarray,
    mana_draw_logits: np.ndarray,
    mana_draw_legal: np.ndarray,
    rng: np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select from legal candidates plus legal mana-draw as ONE distribution.

    The Rust core keeps ``mana_draw`` outside the frozen 601-action codec, so
    the action id remains a legal candidate placeholder when mana-draw wins.
    Its log-probability is nevertheless the joint categorical probability used
    by the PPO update, keeping rollout and gradient semantics identical.
    """
    actions = np.empty(counts.shape[0], dtype=np.uintp)
    log_probs = np.empty(counts.shape[0], dtype=np.float32)
    selected_local = np.empty(counts.shape[0], dtype=np.int32)
    mana_draw_flags = np.zeros(counts.shape[0], dtype=np.bool_)
    md_logits = np.asarray(mana_draw_logits, dtype=np.float64).reshape((-1,))
    md_legal = np.asarray(mana_draw_legal, dtype=np.bool_).reshape((-1,))
    if md_logits.shape != (counts.shape[0],) or md_legal.shape != (counts.shape[0],):
        raise ValueError("mana-draw logits and legality must have one value per policy row")
    offset = 0
    for row, count_raw in enumerate(np.asarray(counts, dtype=np.intp).tolist()):
        count = int(count_raw)
        local_logits = np.asarray(logits[row, :count], dtype=np.float64)
        local_ids = np.asarray(ids[offset : offset + count], dtype=np.uintp)
        if count <= 0:
            raise ValueError(f"row {row} has no legal candidate actions")
        option_logits = local_logits
        if bool(md_legal[row]):
            option_logits = np.concatenate([local_logits, np.asarray([md_logits[row]])])
        shifted = option_logits - float(np.max(option_logits))
        denom = float(np.exp(shifted).sum())
        if not np.isfinite(denom) or denom <= 0.0:
            probs = np.full(option_logits.shape[0], 1.0 / float(option_logits.shape[0]), dtype=np.float64)
        else:
            probs = np.exp(shifted) / denom
        chosen = int(np.argmax(probs)) if rng is None else int(rng.choice(probs.shape[0], p=probs))
        # The candidate id is ignored by step_mana_draw when this is true, but
        # retain a valid candidate index for the compact legal-action tape.
        candidate_local = int(np.argmax(local_logits)) if chosen == count else chosen
        actions[row] = int(local_ids[candidate_local])
        selected_local[row] = candidate_local
        mana_draw_flags[row] = bool(md_legal[row] and chosen == count)
        log_probs[row] = float(np.log(max(float(probs[chosen]), 1.0e-12)))
        offset += count
    return actions, log_probs, selected_local, mana_draw_flags


def run(config: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.optimizers as optim

    output_dir = _resolve_output_dir(config.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model = create_v5_policy(policy_kind="v5_split_encoder", hidden_dim=256, action_hidden_dim=128)
    optimizer = optim.Adam(learning_rate=float(config.learning_rate))
    source: dict[str, Any] = {"kind": "fresh_init"}

    if config.resume_checkpoint is not None:
        loaded = load_checkpoint(str(config.resume_checkpoint), model, optimizer=optimizer)
        source = {"kind": "resume_checkpoint", "path": str(config.resume_checkpoint), "metadata": loaded.get("metadata", {})}
    elif bool(config.warm_start_v4) and not bool(config.no_warm_start):
        try:
            npz_path = resolve_v4_max_npz_path(config.v4_max_npz)
            report = load_v4_max_into_v5(model, npz_path=npz_path)
            source = {"kind": "partial_v4max_warm_start", "path": str(npz_path), "report": _jsonable(report)}
        except RuntimeError as exc:
            source = {"kind": "fresh_init", "warm_start_skip_reason": str(exc)}

    mx.eval(model.parameters(), optimizer.state)

    progress_path = output_dir / "progress.jsonl"
    config_snapshot = vars(config).copy()
    custom_opponent_mix = str(config.opponent_mix) if config.opponent_mix else None
    custom_opponent_mix_parsed = (
        parse_v5_opponent_mix(custom_opponent_mix) if custom_opponent_mix else None
    )
    curriculum_metadata: dict[str, Any] = {}
    phase_overrides: dict[str, Any] = {}
    if custom_opponent_mix is not None:
        curriculum_metadata["opponent_mix_override"] = custom_opponent_mix
        curriculum_metadata["opponent_mix_override_parsed"] = [
            [name, weight] for name, weight in custom_opponent_mix_parsed or []
        ]
        phase_overrides["opponent_mix"] = custom_opponent_mix
        phase_overrides["opponent_mix_spec"] = {
            name: weight for name, weight in custom_opponent_mix_parsed or []
        }

    phase_config = build_phase_a_random_bootstrap_config(
        run_name=str(config.run_name),
        env_count=int(config.env_count),
        steps_per_update=int(config.steps_per_update),
        epochs=int(config.epochs),
        minibatch_size=int(config.minibatch_size),
        checkpoint_dir=str(checkpoints_dir),
        checkpoint_every=int(config.checkpoint_every),
        metrics_path=str(progress_path),
        legal_row_pack_backend="python",
        seed=int(config.seed),
        curriculum_metadata=curriculum_metadata,
        **phase_overrides,
    )
    learner = MLXV5LearnerPolicy(
        model,
        library_path=config.library_path,
        rng=np.random.default_rng(int(config.seed) + 17),
    )
    run_meta = {
        "schema": "extra_lr_v5_phaseA_random_bootstrap_run_v1",
        "run_name": str(config.run_name),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "target_random_score": float(config.target_random_score),
        "source": source,
        "config": _jsonable(config_snapshot),
        "phase_config": _jsonable(asdict(phase_config)),
        "opponent_mix_override": custom_opponent_mix,
        "obs_dim": OBS_V5_DIM,
        "action_feature_dim": ACTION_FEATURE_DIM,
        "mana_draw_policy": "joint_legal_categorical_ppo",
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metrics_history: list[dict[str, Any]] = []
    total_states = 0
    last_checkpoint: Path | None = None
    for update in range(1, int(config.updates) + 1):
        update_seed = int(config.seed) + update - 1
        update_config = replace(phase_config, seed=update_seed)
        started = time.perf_counter()
        metrics = run_live_self_play_update(
            update_config,
            learner,
            seed=update_seed,
            library_path=config.library_path,
            model=model,
            optimizer=optimizer,
        )
        elapsed = time.perf_counter() - started
        states = int(metrics["ppo_batch"].actions.size)
        total_states += states
        row = _compact_metrics(metrics)
        row.update(
            {
                "update": int(update),
                "seed": int(update_seed),
                "states": int(states),
                "total_states": int(total_states),
                "elapsed_seconds": float(elapsed),
            }
        )
        metrics_history.append(row)
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
        print("PHASEA_RANDOM_BOOTSTRAP_UPDATE", json.dumps(_jsonable(row), sort_keys=True), flush=True)

        if int(config.checkpoint_every) > 0 and update % int(config.checkpoint_every) == 0:
            last_checkpoint = _save_checkpoint(
                model=model,
                optimizer=optimizer,
                path=checkpoints_dir / f"extra_lr_v5_phaseA_random_bootstrap_update_{update:05d}_{total_states}_states.npz",
                run_meta=run_meta,
                metrics_history=metrics_history,
                update=update,
                total_states=total_states,
                partial=True,
            )

    final_path = _save_checkpoint(
        model=model,
        optimizer=optimizer,
        path=output_dir / f"extra_lr_v5_phaseA_random_bootstrap_final_{total_states}_states.npz",
        run_meta=run_meta,
        metrics_history=metrics_history,
        update=int(config.updates),
        total_states=total_states,
        partial=False,
    )
    random_gate = _evaluate_random_field_gate(
        learner,
        config=phase_config,
        library_path=config.library_path,
        games=int(config.random_gate_games),
        seed=int(config.seed) + 90_000_000,
        max_steps=int(config.random_gate_max_steps),
    )
    gate_passed = (
        float(random_gate["score_rate"]) >= float(config.target_random_score)
        and int(random_gate["invalid_actions"]) == 0
        and int(random_gate["mana_draw_count"]) > 0
    )
    summary = {
        "status": "ok" if gate_passed else "random_gate_failed",
        "checkpoint_path": str(final_path),
        "last_periodic_checkpoint": str(last_checkpoint) if last_checkpoint is not None else None,
        "updates": int(config.updates),
        "states": int(total_states),
        "target_random_score": float(config.target_random_score),
        "progress_path": str(progress_path),
        "random_gate": random_gate,
        "random_gate_passed": gate_passed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASEA_RANDOM_BOOTSTRAP_RESULT", json.dumps(_jsonable(summary), sort_keys=True), flush=True)
    return summary


def _evaluate_random_field_gate(
    learner: MLXV5LearnerPolicy,
    *,
    config: Any,
    library_path: Path | None,
    games: int,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    """Side-stratified real-engine acceptance check against legal_random.

    This deliberately does not reuse PPO telemetry: it asks the deterministic
    deployed policy to play actual Rust ArenaEnv games and records every legal
    mana-draw decision, invalid candidate guard, and outcome.
    """
    from train_v3.rust_ffi import RustBatchWorker

    if games <= 0:
        raise ValueError("random gate games must be positive")
    wins = draws = losses = invalid_actions = 0
    mana_draw_count = eligible_turns = 0
    for game_idx in range(games):
        for candidate_actor in (1, 2):
            worker = RustBatchWorker.from_live(
                seed=int(seed) + game_idx * 2 + candidate_actor,
                env_count=1,
                max_turns=int(config.max_turns),
                library_path=library_path,
                action_features_dtype=config.action_features_dtype,
                action_features_mode=config.action_features_mode,
                observation_mode=config.observation_mode,
                action_mask_mode=config.action_mask_mode,
                terminal_observation_mode=config.terminal_observation_mode,
                diagnostic_mode="none",
            )
            try:
                outcome = "draw"
                for _step in range(max(1, max_steps)):
                    arrays = worker.arrays(copy=True)
                    actor = int(worker.current_actor_ids()[0])
                    counts = np.asarray(arrays["legal_action_counts"], dtype=np.intp)
                    offsets = np.asarray(arrays["legal_action_offsets"], dtype=np.intp)
                    legal_ids = np.asarray(arrays["legal_action_ids"], dtype=np.uintp)
                    legal_features = arrays.get("legal_action_features")
                    mana_legal = bool(worker.mana_draw_legal()[0])
                    mana_draw = False
                    if actor == candidate_actor:
                        eligible_turns += int(mana_legal)
                        start = int(offsets[0])
                        end = start + int(counts[0])
                        ctx = OpponentCtx(
                            env_idx=0,
                            actor_id=actor,
                            observation_v5=np.asarray(arrays["observation_v5"], dtype=np.float32)[0],
                            legal_action_ids=legal_ids[start:end],
                            legal_action_features=(
                                None
                                if legal_features is None
                                else np.asarray(legal_features, dtype=np.float32)[start:end]
                            ),
                            legal_action_counts=int(counts[0]),
                            mana_draw_legal=mana_legal,
                        )
                        action_id, mana_draw = learner.argmax_select_with_mana(ctx)
                        if not mana_draw and int(action_id) not in set(int(v) for v in legal_ids[start:end]):
                            invalid_actions += 1
                            action_id = int(legal_ids[start])
                        mana_draw_count += int(mana_draw)
                    else:
                        action_id = int(
                            worker.select_rule_actions(np.asarray([0], dtype=np.uint32))[0]
                        )
                    out = worker.step_mana_draw(
                        np.asarray([action_id], dtype=np.uintp),
                        np.asarray([mana_draw], dtype=np.bool_),
                        copy=True,
                    )
                    if bool(np.asarray(out["terminated"], dtype=np.bool_)[0]) or bool(worker.truncated()[0]):
                        hp = np.asarray(worker.hero_hp(), dtype=np.int32)[0]
                        if bool(worker.truncated()[0]) or (int(hp[0]) <= 0 and int(hp[2]) <= 0):
                            outcome = "draw"
                        elif (candidate_actor == 1 and int(hp[2]) <= 0) or (
                            candidate_actor == 2 and int(hp[0]) <= 0
                        ):
                            outcome = "win"
                        else:
                            outcome = "loss"
                        break
                if outcome == "win":
                    wins += 1
                elif outcome == "loss":
                    losses += 1
                else:
                    draws += 1
            finally:
                worker.close()
    total = wins + draws + losses
    return {
        "games": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_rate": (wins + 0.5 * draws) / max(total, 1),
        "invalid_actions": invalid_actions,
        "mana_draw_count": mana_draw_count,
        "mana_draw_eligible": eligible_turns,
        "mana_draw_rate": mana_draw_count / max(eligible_turns, 1),
        "seed": int(seed),
        "both_sides": True,
    }


def _save_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    path: Path,
    run_meta: dict[str, Any],
    metrics_history: list[dict[str, Any]],
    update: int,
    total_states: int,
    partial: bool,
) -> Path:
    import mlx.core as mx

    mx.eval(model.parameters(), optimizer.state)
    metadata = {
        "run_name": run_meta["run_name"],
        "model_name": "extra-lr-v5-adaptive",
        "phase": "phaseA_random_bootstrap",
        "policy_kind": "v5_split_encoder",
        "obs_dim": int(OBS_V5_DIM),
        "action_feature_dim": int(ACTION_FEATURE_DIM),
        "target_random_score": float(run_meta["target_random_score"]),
        "source": run_meta["source"],
        "config": run_meta["config"],
        "phase_config": run_meta["phase_config"],
        "completed_updates": int(update),
        "total_states": int(total_states),
        "partial_checkpoint": bool(partial),
        "last_metrics": metrics_history[-1] if metrics_history else {},
    }
    save_checkpoint(str(path), model, optimizer=optimizer, metadata=_jsonable(metadata))
    return path


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: value
        for key, value in metrics.items()
        if key not in {"rollout", "ppo_batch"}
    }
    compact["opponent_counts"] = _counts(compact.get("opponent_identities", []))
    compact["learner_actor_counts"] = _counts(compact.get("learner_actor_ids", []))
    return _jsonable(compact)


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in list(values or []):
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return result


def _resolve_output_dir(path: Path | None) -> Path:
    if path is not None:
        return path.resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return (ROOT / "TrainV3.5" / "runs" / f"phaseA_random_bootstrap_{stamp}").resolve()


def _mx_eval(*values: Any) -> None:
    import mlx.core as mx

    mx.eval(*values)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="phaseA_random_bootstrap")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--env-count", type=int, default=128)
    parser.add_argument("--steps-per-update", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=350500)
    parser.add_argument(
        "--opponent-mix",
        default=None,
        help=(
            "Optional canonical Phase-A opponent mix override, e.g. "
            "'random:1.0' for a pure random-focus continuation. Defaults to the "
            "random-heavy bootstrap mix."
        ),
    )
    parser.add_argument("--target-random-score", type=float, default=PHASE_A_RANDOM_BOOTSTRAP_TARGET_RANDOM_SCORE)
    parser.add_argument(
        "--random-gate-games",
        type=int,
        default=64,
        help="Games per side for the post-run field acceptance against legal_random.",
    )
    parser.add_argument("--random-gate-max-steps", type=int, default=240)
    parser.add_argument("--library-path", type=Path, default=_default_library_path())
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--v4-max-npz", type=Path, default=None)
    parser.add_argument(
        "--warm-start-v4",
        action="store_true",
        help="Opt in to the legacy partial V4Max warm-start. Phase A defaults to fresh teacher-free init.",
    )
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Deprecated compatibility flag; fresh init is already the Phase-A default.",
    )
    args = parser.parse_args(argv)
    if int(args.updates) <= 0:
        parser.error("--updates must be positive")
    if int(args.env_count) <= 0:
        parser.error("--env-count must be positive")
    if int(args.steps_per_update) <= 0:
        parser.error("--steps-per-update must be positive")
    if int(args.epochs) <= 0:
        parser.error("--epochs must be positive")
    if int(args.minibatch_size) <= 0:
        parser.error("--minibatch-size must be positive")
    if float(args.learning_rate) <= 0.0:
        parser.error("--learning-rate must be positive")
    if not 0.0 < float(args.target_random_score) <= 1.0:
        parser.error("--target-random-score must be in (0, 1]")
    if int(args.random_gate_games) <= 0 or int(args.random_gate_max_steps) <= 0:
        parser.error("random gate games and max steps must be positive")
    if args.opponent_mix:
        try:
            parse_v5_opponent_mix(str(args.opponent_mix))
        except ValueError as exc:
            parser.error(f"--opponent-mix is invalid: {exc}")
    return args


def _default_library_path() -> Path | None:
    env = os.environ.get("TRAINV3_CORE_LIB")
    if env:
        return Path(env)
    candidate = ROOT / "TrainV3.5" / "target" / "release" / "libtrainv3_core.dylib"
    return candidate if candidate.exists() else None


def main(argv: list[str] | None = None) -> int:
    summary = run(parse_args(argv))
    return 0 if bool(summary.get("random_gate_passed", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run V5 ExtraLR Block B long-form live league training.

This is the operational wrapper around the Block-B components: it continues from
the Phase-A random bootstrap checkpoint, trains in the Rust ArenaEnv with the
Block-B opponent mix, writes progress/checkpoints, and grows a self-snapshot
pool for the league.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
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
from run_phaseA_random_bootstrap import MLXV5LearnerPolicy  # noqa: E402
from train_v3.a_gate import GameResult, ManaDrawBaseline  # noqa: E402
from train_v3.block_b_league_driver import BlockBLeagueDriver, _merge_self_snapshot_split  # noqa: E402
from train_v3.block_b_opponent_mix import (  # noqa: E402
    build_block_b_opponent_mix,
    collapse_reweight_boost,
    parse_block_b_opponent_mix,
)
from train_v3.contracts import ACTION_FEATURE_DIM, OBS_V5_DIM  # noqa: E402
from train_v3.curriculum import CurriculumReweighter, extract_lane_outcomes  # noqa: E402
from train_v3.ppo_phaseA_config import build_phase_a_random_bootstrap_config  # noqa: E402
from train_v3.rust_live_self_play import (  # noqa: E402
    EndTurnOpponent,
    GreedyFaceOpponent,
    OpponentCtx,
    PolicyOpponent,
    RULE_DISPATCH,
    resolve_opponent_dispatch,
    run_live_self_play_update,
)
from train_v3.second_start_parity import BlockBGameResult, SecondStartParityLoop  # noqa: E402
from train_v3.snapshot_pool import SnapshotEntry, SnapshotPool  # noqa: E402
from train_v3.v4_orig_temp_spectrum import V4_ORIG_TEMP_IDENTITIES, V4TempSpectrumIdentity  # noqa: E402
from train_v3.v5_policy import create_v5_policy  # noqa: E402


class DynamicSelfSnapshotOpponent:
    """Self opponent that follows the latest league snapshot, with live fallback."""

    name = "self"

    def __init__(
        self,
        pool: SnapshotPool,
        live_policy: MLXV5LearnerPolicy,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.pool = pool
        self.live_policy = live_policy
        self.rng = rng or np.random.default_rng()
        self._loaded_policies: dict[str, MLXV5LearnerPolicy] = {}
        self._pinned_path_by_env: dict[int, str | None] = {}

    def select(self, env_idx: int, ctx: OpponentCtx) -> int:
        action, _draw = self.select_with_mana(env_idx, ctx)
        return int(action)

    def select_with_mana(self, env_idx: int, ctx: OpponentCtx) -> tuple[int, bool]:
        policy = self._snapshot_policy(env_idx)
        if policy is None:
            return self.live_policy.argmax_select_with_mana(ctx)
        return policy.argmax_select_with_mana(ctx)

    def _snapshot_policy(self, env_idx: int) -> MLXV5LearnerPolicy | None:
        if int(env_idx) not in self._pinned_path_by_env:
            entry = self._selected_entry()
            self._pinned_path_by_env[int(env_idx)] = None if entry is None else str(entry.path)
        path = self._pinned_path_by_env[int(env_idx)]
        if path is None:
            return None
        if path in self._loaded_policies:
            return self._loaded_policies[path]
        model = create_v5_policy(policy_kind="v5_split_encoder", hidden_dim=256, action_hidden_dim=128)
        load_checkpoint(path, model)
        self._loaded_policies[path] = MLXV5LearnerPolicy(model)
        return self._loaded_policies[path]

    def _selected_entry(self) -> SnapshotEntry | None:
        rolling = list(self.pool.rolling)
        pool_entries = ([self.pool.seed_anchor] if self.pool.seed_anchor is not None else []) + rolling
        if self.pool.best_ever is not None and (not pool_entries or float(self.rng.random()) < 0.5):
            return self.pool.best_ever
        if pool_entries:
            return pool_entries[int(self.rng.integers(0, len(pool_entries)))]
        return None

    def reset_session(self) -> None:
        """Release per-lane snapshot pins at an explicit league boundary.

        Persistent PPO sessions must keep the opponent fixed while an episode
        is in flight. Once the whole session rotates, lanes may bind to the
        newly promoted/rolling snapshot pool.
        """
        self._pinned_path_by_env.clear()


class LiveBlockBGameRunner:
    """Minimal operational side-stratified external bench runner."""

    def __init__(
        self,
        *,
        config: Any,
        learner: MLXV5LearnerPolicy,
        opponent_policies: dict[str, PolicyOpponent],
        library_path: Path | None,
        max_steps: int,
    ) -> None:
        self.config = config
        self.learner = learner
        self.opponent_policies = opponent_policies
        self.library_path = library_path
        self.max_steps = int(max_steps)

    def play(self, opponent_kind: str, *, seed: int, candidate_side: str) -> BlockBGameResult:
        from train_v3.rust_ffi import RustBatchWorker

        candidate_actor = 1 if str(candidate_side) == "p1" else 2
        identity = "self" if str(opponent_kind) == "best_ever" else str(opponent_kind)
        worker = RustBatchWorker.from_live(
            seed=int(seed),
            env_count=1,
            max_turns=int(self.config.max_turns),
            library_path=self.library_path,
            action_features_dtype=self.config.action_features_dtype,
            action_features_mode=self.config.action_features_mode,
            observation_mode=self.config.observation_mode,
            action_mask_mode=self.config.action_mask_mode,
            terminal_observation_mode=self.config.terminal_observation_mode,
            diagnostic_mode="none" if self.config.diagnostic_mode == "auto" else self.config.diagnostic_mode,
        )
        try:
            mana_draw_count = 0
            eligible_turns = 0
            outcome = "draw"
            for step in range(max(1, self.max_steps)):
                arrays = worker.arrays(copy=True)
                actor = int(worker.current_actor_ids()[0])
                counts = np.asarray(arrays["legal_action_counts"], dtype=np.intp)
                offsets = np.asarray(arrays["legal_action_offsets"], dtype=np.intp)
                legal_ids = np.asarray(arrays["legal_action_ids"], dtype=np.uintp)
                legal_features = arrays.get("legal_action_features")
                md_legal = bool(worker.mana_draw_legal()[0])
                mana_draw = False
                if actor == candidate_actor:
                    eligible_turns += int(md_legal)
                    ctx = OpponentCtx(
                        env_idx=0,
                        actor_id=actor,
                        observation_v5=np.asarray(arrays["observation_v5"], dtype=np.float32)[0],
                        legal_action_ids=legal_ids[int(offsets[0]) : int(offsets[0]) + int(counts[0])],
                        legal_action_features=(
                            None
                            if legal_features is None
                            else np.asarray(legal_features, dtype=np.float32)[
                                int(offsets[0]) : int(offsets[0]) + int(counts[0])
                            ]
                        ),
                        legal_action_counts=int(counts[0]),
                        mana_draw_legal=md_legal,
                    )
                    action_id, mana_draw = self.learner.argmax_select_with_mana(ctx)
                    mana_draw_count += int(mana_draw)
                else:
                    action_id, mana_draw = self._opponent_action(identity, worker, arrays, actor)
                out = worker.step_mana_draw(
                    np.asarray([action_id], dtype=np.uintp),
                    np.asarray([mana_draw], dtype=np.bool_),
                    copy=True,
                )
                terminated = bool(np.asarray(out["terminated"], dtype=np.bool_)[0])
                truncated = bool(worker.truncated()[0])
                if terminated or truncated:
                    hp = np.asarray(worker.hero_hp(), dtype=np.int32)[0]
                    if truncated:
                        outcome = "draw"
                    elif int(hp[0]) <= 0 and int(hp[2]) <= 0:
                        outcome = "draw"
                    elif int(hp[2]) <= 0:
                        outcome = "win" if candidate_actor == 1 else "loss"
                    elif int(hp[0]) <= 0:
                        outcome = "win" if candidate_actor == 2 else "loss"
                    else:
                        outcome = "draw"
                    break
            return BlockBGameResult(
                game=GameResult(
                    outcome=outcome,
                    mana_draw_count=mana_draw_count,
                    eligible_turns=max(eligible_turns, 0),
                    opponent=str(opponent_kind),
                ),
                candidate_side=str(candidate_side),
            )
        finally:
            worker.close()

    def _opponent_action(self, identity: str, worker: Any, arrays: dict[str, Any], actor: int) -> tuple[int, bool]:
        kind, code = resolve_opponent_dispatch(identity)
        if kind == RULE_DISPATCH:
            return int(worker.select_rule_actions(np.asarray([int(code)], dtype=np.uint32))[0]), False
        policy = self.opponent_policies[identity]
        counts = np.asarray(arrays["legal_action_counts"], dtype=np.intp)
        offsets = np.asarray(arrays["legal_action_offsets"], dtype=np.intp)
        legal_ids = np.asarray(arrays["legal_action_ids"], dtype=np.uintp)
        legal_features = arrays.get("legal_action_features")
        ctx = OpponentCtx(
            env_idx=0,
            actor_id=int(actor),
            observation_v5=np.asarray(arrays["observation_v5"], dtype=np.float32)[0],
            legal_action_ids=legal_ids[int(offsets[0]) : int(offsets[0]) + int(counts[0])],
            legal_action_features=(
                None
                if legal_features is None
                else np.asarray(legal_features, dtype=np.float32)[
                    int(offsets[0]) : int(offsets[0]) + int(counts[0])
                ]
            ),
            legal_action_counts=int(counts[0]),
            mana_draw_legal=bool(worker.mana_draw_legal()[0]),
        )
        select_with_mana = getattr(policy, "select_with_mana", None)
        if callable(select_with_mana):
            action, draw = select_with_mana(0, ctx)
            return int(action), bool(draw) and bool(ctx.mana_draw_legal)
        return int(policy.select(0, ctx)), False


class V4TempOnnxOpponent:
    """Packed-context V4-orig ONNX opponent with argmax/sample temperature."""

    def __init__(self, session: Any, identity: V4TempSpectrumIdentity, *, seed: int) -> None:
        self.session = session
        self.identity = identity
        self.name = identity.name
        self.rng = np.random.default_rng(int(seed))

    def select(self, env_idx: int, ctx: OpponentCtx) -> int:
        return self.select_batch([ctx])[0]

    def select_batch(self, contexts: list[OpponentCtx]) -> list[int]:
        """Select a batch of V4 actions in one ONNX Runtime invocation.

        Block B previously ran one CPU inference per environment, leaving most
        of the machine idle despite a large rollout batch.  The V4 ONNX graph
        accepts a dynamic leading batch dimension, so preserving per-row legal
        masks after one batched call is equivalent to calling ``select`` in a
        loop and substantially improves throughput.
        """
        if not contexts:
            return []
        features_batch = np.zeros((len(contexts), 601, ACTION_FEATURE_DIM), dtype=np.float32)
        observations = np.empty((len(contexts), 1456), dtype=np.float32)
        legal_ids_batch: list[np.ndarray] = []
        for row, ctx in enumerate(contexts):
            legal_ids = np.asarray(ctx.legal_action_ids, dtype=np.intp)
            legal_ids_batch.append(legal_ids)
            observations[row] = np.asarray(ctx.observation_v5, dtype=np.float32)[:1456]
            if legal_ids.size <= 0 or ctx.legal_action_features is None:
                continue
            features_batch[row, legal_ids] = np.asarray(ctx.legal_action_features, dtype=np.float32)
        logits_batch = self.session.run(
            ["logits", "value"],
            {"observation": observations, "action_features": features_batch},
        )[0].astype(np.float32, copy=False)
        selected: list[int] = []
        for row, legal_ids in enumerate(legal_ids_batch):
            if legal_ids.size <= 0 or contexts[row].legal_action_features is None:
                selected.append(0)
                continue
            mask = np.zeros(601, dtype=np.bool_)
            mask[legal_ids] = True
            masked = np.where(mask, logits_batch[row], -1.0e9)
            if self.identity.mode == "sample":
                scaled = masked / float(self.identity.temperature)
                shifted = scaled - float(np.max(scaled))
                probs = np.exp(shifted) * mask.astype(np.float32)
                denom = float(probs.sum())
                if not np.isfinite(denom) or denom <= 0.0:
                    selected.append(int(legal_ids[0]))
                else:
                    selected.append(int(self.rng.choice(len(probs), p=probs / denom)))
            else:
                selected.append(int(np.argmax(masked)))
        return selected

    def _select_scalar_reference(self, env_idx: int, ctx: OpponentCtx) -> int:
        """Reference implementation retained for test parity documentation."""
        legal_ids = np.asarray(ctx.legal_action_ids, dtype=np.intp)
        if legal_ids.size <= 0:
            return 0
        if ctx.legal_action_features is None:
            return int(legal_ids[0])
        features = np.asarray(ctx.legal_action_features, dtype=np.float32)
        full_features = np.zeros((601, ACTION_FEATURE_DIM), dtype=np.float32)
        full_features[legal_ids] = features
        obs = np.asarray(ctx.observation_v5, dtype=np.float32)[:1456]
        logits = self.session.run(
            ["logits", "value"],
            {
                "observation": obs[None, :],
                "action_features": full_features[None, :, :],
            },
        )[0][0].astype(np.float32, copy=False)
        mask = np.zeros(601, dtype=np.bool_)
        mask[legal_ids] = True
        masked = np.where(mask, logits, -1.0e9)
        if self.identity.mode == "sample":
            scaled = masked / float(self.identity.temperature)
            shifted = scaled - float(np.max(scaled))
            probs = np.exp(shifted) * mask.astype(np.float32)
            denom = float(probs.sum())
            if not np.isfinite(denom) or denom <= 0.0:
                return int(legal_ids[0])
            probs = probs / denom
            return int(self.rng.choice(len(probs), p=probs))
        return int(np.argmax(masked))


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.optimizers as optim

    out_dir = _resolve_output_dir(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = create_v5_policy(policy_kind="v5_split_encoder", hidden_dim=256, action_hidden_dim=128)
    optimizer = optim.Adam(learning_rate=float(args.learning_rate))
    loaded = load_checkpoint(str(args.source_checkpoint), model, optimizer=optimizer)
    mx.eval(model.parameters(), optimizer.state)

    progress_path = out_dir / "progress.jsonl"
    manifest_path = out_dir / "blockB_league_manifest.json"
    pool_path = out_dir / "snapshot_pool.json"
    opponent_mix_override = _parse_opponent_mix_override(args.opponent_mix)
    config = build_phase_a_random_bootstrap_config(
        run_name=str(args.run_name),
        env_count=int(args.env_count),
        steps_per_update=int(args.steps_per_update),
        epochs=int(args.epochs),
        minibatch_size=int(args.minibatch_size),
        turn_order_second_mover_reward_bonus=float(args.second_mover_reward_bonus),
        checkpoint_dir=str(ckpt_dir),
        checkpoint_every=int(args.checkpoint_every),
        metrics_path=str(progress_path),
        legal_row_pack_backend="python",
        seed=int(args.seed),
    )
    if args.second_start_weight is not None:
        config = replace(
            config,
            second_start_oversampling={
                "policy": "fixed_second_start_weight",
                "second_start_weight": float(args.second_start_weight),
            },
        )
    elif args.p2_start_weight is not None:
        config = replace(
            config,
            second_start_oversampling={
                "policy": "fixed_p2_weight",
                "p2_weight": float(args.p2_start_weight),
            },
        )
    elif str(args.side_sampling_policy) == "strict_balanced":
        config = replace(
            config,
            second_start_oversampling={
                "gap_threshold": 1.0,
                "base_weight": 0.5,
                "policy": "strict_balanced",
            },
        )
    elif str(args.side_sampling_policy) == "start_second":
        config = replace(
            config,
            second_start_oversampling={
                "gap_threshold": 1.0,
                "base_weight": 0.5,
                "policy": "start_second",
            },
        )
    config = replace(config, decisive_early_end=False)
    learner = MLXV5LearnerPolicy(
        model,
        library_path=args.library_path,
        rng=np.random.default_rng(int(args.seed) + 31),
    )
    pool = SnapshotPool(target_non_anchor_count=int(args.pool_size))
    # The accepted post-A policy is the immutable league anchor. Without this,
    # the first (possibly failing) B snapshot silently replaces the baseline.
    pool.set_seed_anchor(SnapshotEntry(
        update_number=0,
        h2h_vs_best=0.5,
        path=str(args.source_checkpoint.resolve()),
        p1_p2_gap=0.0,
        promotion_eligible=True,
        role="seed_anchor",
    ))
    curriculum = CurriculumReweighter(window_n=int(args.curriculum_window))
    parity = SecondStartParityLoop(window_n=int(args.parity_window))
    opponent_policies = _build_opponent_policies(args, pool=pool, learner=learner)
    run_meta = {
        "schema": "extra_lr_v5_blockB_league_run_v1",
        "run_name": str(args.run_name),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_metadata": loaded.get("metadata", {}),
        "config": _jsonable(vars(args)),
        "phase_config": _jsonable(asdict(config)),
        "obs_dim": OBS_V5_DIM,
        "action_feature_dim": ACTION_FEATURE_DIM,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    snapshots: list[dict[str, Any]] = []

    def checkpoint_namer(update_number: int) -> str:
        path = ckpt_dir / f"extra_lr_v5_blockB_league_update_{int(update_number):05d}.npz"
        _save_checkpoint(
            model=model,
            optimizer=optimizer,
            path=path,
            run_meta=run_meta,
            update_rows=[],
            update=int(update_number),
            partial=True,
        )
        snapshots.append({"update": int(update_number), "path": str(path)})
        return str(path)

    game_runner = LiveBlockBGameRunner(
        config=config,
        learner=learner,
        opponent_policies=opponent_policies,
        library_path=args.library_path,
        max_steps=int(args.eval_max_steps),
    )
    if args.mana_draw_baseline_count is None:
        # The old CLI fabricated a successful 1/2 baseline. Measure the actual
        # source checkpoint instead, so the collapse monitor and promotion gate
        # have auditable field evidence.
        reference = [
            game_runner.play(
                "random",
                seed=int(args.seed) * 10_000 + game_idx,
                candidate_side="p1" if game_idx % 2 == 0 else "p2",
            ).game
            for game_idx in range(8)
        ]
        baseline_count = sum(int(game.mana_draw_count) for game in reference)
        baseline_eligible = sum(int(game.eligible_turns) for game in reference)
        if baseline_eligible <= 0:
            raise RuntimeError("field mana-draw baseline has no eligible learner turns")
    else:
        baseline_count = int(args.mana_draw_baseline_count)
        baseline_eligible = int(args.mana_draw_baseline_eligible)
    mana_draw_baseline = ManaDrawBaseline(
        mana_draw_count=baseline_count,
        eligible_turns=baseline_eligible,
        rate=float(baseline_count) / float(baseline_eligible),
        hand_cap=4,
        mana_draw_base=2,
        valid=True,
    )
    run_meta["mana_draw_baseline"] = _jsonable(asdict(mana_draw_baseline))
    (out_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    driver = BlockBLeagueDriver(
        config,
        pool=pool,
        game_runner=game_runner,
        learner_policy=learner,
        opponent_policies_factory=lambda: opponent_policies,
        curriculum=curriculum,
        parity=parity,
        seed=int(args.seed),
        mana_draw_baseline=mana_draw_baseline,
        snapshot_cadence=int(args.checkpoint_every),
        # This operational command promises the requested long run. Plateau is
        # telemetry for the later C handoff, not permission to stop B early.
        k_snap=int(args.updates) // int(args.checkpoint_every) + 1,
        checkpoint_namer=checkpoint_namer,
        games_per_opponent_per_side=int(args.games_per_opponent_per_side),
        games_per_opponent_gauntlet=int(args.games_per_opponent_gauntlet),
        model=model,
        optimizer=optimizer,
        learning_rate_for_update=lambda update_number: _learning_rate_schedule(
            base_lr=float(args.learning_rate),
            update_number=int(update_number),
            total_updates=int(args.updates),
            warmup_updates=int(args.lr_warmup_updates),
            final_scale=float(args.lr_final_scale),
        ),
        steps_per_update=int(args.steps_per_update),
        opponent_mix_override=opponent_mix_override,
    )
    driver_manifest = driver.run(int(args.updates))
    update_rows = [_compact_metrics(row) for row in driver_manifest.update_metrics]
    with progress_path.open("w", encoding="utf-8") as f:
        for row in update_rows:
            f.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
            print("BLOCKB_LEAGUE_UPDATE", json.dumps(_jsonable(_console_row(row)), sort_keys=True), flush=True)
    pool_path.write_text(json.dumps(pool.to_manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    final_checkpoint = _save_checkpoint(
        model=model,
        optimizer=optimizer,
        path=out_dir / f"extra_lr_v5_blockB_league_final_update_{int(driver_manifest.n_updates_run):05d}.npz",
        run_meta=run_meta,
        update_rows=update_rows,
        update=int(driver_manifest.n_updates_run),
        partial=False,
    )
    best_checkpoint = Path(pool.best_ever.path) if pool.best_ever is not None else final_checkpoint
    summary = _write_manifest(
        manifest_path,
        run_meta,
        update_rows,
        snapshots,
        pool,
        best_checkpoint,
        status="ok",
    )
    summary["driver_manifest"] = driver_manifest.to_dict()
    manifest_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("BLOCKB_LEAGUE_RESULT", json.dumps(_jsonable(summary), sort_keys=True), flush=True)
    return summary


def _build_mix(
    pool: SnapshotPool,
    curriculum: CurriculumReweighter,
    *,
    opponent_mix_override: list[tuple[str, float]] | None = None,
) -> list[tuple[str, float]]:
    if opponent_mix_override is not None:
        return _normalize_mix(opponent_mix_override)
    mix = build_block_b_opponent_mix(pool, **collapse_reweight_boost(1.0))
    return _merge_self_snapshot_split(curriculum.reweight(mix, cap=0.25))


def _parse_opponent_mix_override(raw: str | None) -> list[tuple[str, float]] | None:
    if raw is None or not str(raw).strip():
        return None
    return _normalize_mix(_merge_self_snapshot_split(parse_block_b_opponent_mix(str(raw))))


def _normalize_mix(mix: list[tuple[str, float]]) -> list[tuple[str, float]]:
    rows = [(str(name), float(weight)) for name, weight in mix if float(weight) > 0.0]
    total = sum(weight for _name, weight in rows)
    if total <= 0.0:
        raise ValueError("opponent mix must contain at least one positive weight")
    return [(name, weight / total) for name, weight in rows]


def _build_opponent_policies(
    args: argparse.Namespace,
    *,
    pool: SnapshotPool,
    learner: MLXV5LearnerPolicy,
) -> dict[str, PolicyOpponent]:
    policies: dict[str, PolicyOpponent] = {
        "end_turn": EndTurnOpponent(),
        "greedy_face": GreedyFaceOpponent(),
        "self": DynamicSelfSnapshotOpponent(pool, learner),
    }
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(args.v4_onnx), providers=["CPUExecutionProvider"])
        for idx, identity in enumerate(V4_ORIG_TEMP_IDENTITIES):
            policies[identity.name] = V4TempOnnxOpponent(session, identity, seed=int(args.seed) + idx * 1009)
    except Exception as exc:
        if not bool(args.allow_missing_v4):
            raise
        print(f"BLOCKB_LEAGUE_WARN v4 temp opponents unavailable: {exc}", flush=True)
        for identity in V4_ORIG_TEMP_IDENTITIES:
            policies[identity.name] = EndTurnOpponent()
    return policies


def _save_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    path: Path,
    run_meta: dict[str, Any],
    update_rows: list[dict[str, Any]],
    update: int,
    partial: bool,
) -> Path:
    import mlx.core as mx

    mx.eval(model.parameters(), optimizer.state)
    metadata = {
        "run_name": run_meta["run_name"],
        "model_name": "extra-lr-v5-adaptive",
        "phase": "blockB_league",
        "policy_kind": "v5_split_encoder",
        "state_action_interaction": "gated_bilinear_query_cap01_v1",
        "obs_dim": int(OBS_V5_DIM),
        "action_feature_dim": int(ACTION_FEATURE_DIM),
        "source_checkpoint": run_meta["source_checkpoint"],
        "completed_updates": int(update),
        "partial_checkpoint": bool(partial),
        "last_metrics": update_rows[-1] if update_rows else {},
    }
    save_checkpoint(str(path), model, optimizer=optimizer, metadata=_jsonable(metadata))
    return path


def _write_manifest(
    path: Path,
    run_meta: dict[str, Any],
    update_rows: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    pool: SnapshotPool,
    best_checkpoint: Path,
    *,
    status: str,
) -> dict[str, Any]:
    summary = {
        "schema": "extra_lr_v5_blockB_league_manifest_v1",
        "status": status,
        "run_name": run_meta["run_name"],
        "source_checkpoint": run_meta["source_checkpoint"],
        "updates": len(update_rows),
        "best_checkpoint": str(best_checkpoint),
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
        "pool": pool.to_manifest(),
        "last_metrics": update_rows[-1] if update_rows else {},
    }
    path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    compact = {k: v for k, v in metrics.items() if k not in {"rollout", "ppo_batch"}}
    compact["opponent_counts"] = _counts(compact.get("opponent_identities", []))
    compact["learner_actor_counts"] = _counts(compact.get("learner_actor_ids", []))
    return _jsonable(compact)


def _console_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("update_metrics") or {}
    return {
        "update": row.get("update_number"),
        "entropy": metrics.get("entropy"),
        "loss": metrics.get("loss"),
        "approx_kl": metrics.get("approx_kl"),
        "clip_fraction": metrics.get("clip_fraction"),
        "opponent_counts": row.get("opponent_counts"),
    }


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in list(values or []):
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return result


def _learning_rate_schedule(
    *,
    base_lr: float,
    update_number: int,
    total_updates: int,
    warmup_updates: int,
    final_scale: float,
) -> float:
    update = max(1, int(update_number))
    total = max(1, int(total_updates))
    warmup = min(max(0, int(warmup_updates)), total)
    base = float(base_lr)
    final = base * float(final_scale)
    if warmup > 0 and update <= warmup:
        return base * float(update) / float(warmup)
    if total <= warmup + 1:
        return final
    progress = (float(update) - float(warmup)) / float(total - warmup)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final + (base - final) * cosine


def _resolve_output_dir(path: Path | None) -> Path:
    if path is not None:
        return path.resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return (ROOT / "TrainV3.5" / "runs" / f"blockB_league_{stamp}").resolve()


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


def _default_library_path() -> Path | None:
    env = os.environ.get("TRAINV3_CORE_LIB")
    if env:
        return Path(env)
    candidate = ROOT / "TrainV3.5" / "target" / "release" / "libtrainv3_core.dylib"
    return candidate if candidate.exists() else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="blockB_league")
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--env-count", type=int, default=128)
    parser.add_argument("--steps-per-update", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lr-warmup-updates", type=int, default=100)
    parser.add_argument("--lr-final-scale", type=float, default=0.1)
    parser.add_argument(
        "--second-mover-reward-bonus",
        type=float,
        default=0.0,
        help="Terminal-win-only bonus for games where the learner moved second.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=410500)
    parser.add_argument("--pool-size", type=int, default=6)
    parser.add_argument("--curriculum-window", type=int, default=32)
    parser.add_argument("--parity-window", type=int, default=64)
    parser.add_argument("--eval-max-steps", type=int, default=240)
    parser.add_argument("--games-per-opponent-per-side", type=int, default=1)
    parser.add_argument("--games-per-opponent-gauntlet", type=int, default=1)
    parser.add_argument(
        "--mana-draw-baseline-count",
        type=int,
        default=None,
        help="Measured reference baseline count; omit to measure the source checkpoint in the field runner.",
    )
    parser.add_argument(
        "--mana-draw-baseline-eligible",
        type=int,
        default=None,
        help="Measured reference eligible turns; required together with --mana-draw-baseline-count.",
    )
    parser.add_argument(
        "--side-sampling-policy",
        choices=["adaptive_oversample", "strict_balanced", "start_second"],
        default="adaptive_oversample",
    )
    parser.add_argument(
        "--p2-start-weight",
        type=float,
        default=None,
        help=(
            "Legacy fixed learner actor-2 seat share in (0, 1); this does NOT "
            "control initiative. Overrides --side-sampling-policy."
        ),
    )
    parser.add_argument(
        "--second-start-weight",
        type=float,
        default=None,
        help=(
            "Fixed fraction of episodes in which the learner moves second. "
            "Rebinds the learner actor after every episode reset and overrides "
            "--side-sampling-policy."
        ),
    )
    parser.add_argument(
        "--opponent-mix",
        default=None,
        help=(
            "Optional Block-B identity mix override, e.g. "
            "'v4-orig-argmax:0.6,v4-orig-t07:0.2,self:0.1,stall:0.1'. "
            "When omitted, use the default Block-B curriculum mix."
        ),
    )
    parser.add_argument("--v4-onnx", type=Path, default=ROOT / "ai" / "models" / "extra-lr-v4-max.onnx")
    parser.add_argument("--allow-missing-v4", action="store_true")
    parser.add_argument("--library-path", type=Path, default=_default_library_path())
    args = parser.parse_args(argv)
    if not args.source_checkpoint.exists():
        parser.error(f"--source-checkpoint not found: {args.source_checkpoint}")
    for name in ("updates", "env_count", "steps_per_update", "epochs", "minibatch_size", "checkpoint_every"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("eval_max_steps", "games_per_opponent_per_side", "games_per_opponent_gauntlet"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if (args.mana_draw_baseline_count is None) != (args.mana_draw_baseline_eligible is None):
        parser.error("mana-draw baseline count and eligible turns must be provided together")
    if args.mana_draw_baseline_eligible is not None and int(args.mana_draw_baseline_eligible) <= 0:
        parser.error("--mana-draw-baseline-eligible must be positive")
    if args.mana_draw_baseline_count is not None and int(args.mana_draw_baseline_count) < 0:
        parser.error("--mana-draw-baseline-count must be non-negative")
    if args.p2_start_weight is not None and not 0.0 < float(args.p2_start_weight) < 1.0:
        parser.error("--p2-start-weight must be between 0 and 1 (exclusive)")
    if args.second_start_weight is not None and not 0.0 < float(args.second_start_weight) < 1.0:
        parser.error("--second-start-weight must be between 0 and 1 (exclusive)")
    if args.p2_start_weight is not None and args.second_start_weight is not None:
        parser.error("--p2-start-weight and --second-start-weight are mutually exclusive")
    if (
        args.mana_draw_baseline_count is not None
        and int(args.mana_draw_baseline_count) > int(args.mana_draw_baseline_eligible)
    ):
        parser.error("mana-draw baseline count cannot exceed eligible turns")
    if float(args.learning_rate) <= 0.0:
        parser.error("--learning-rate must be positive")
    if int(args.lr_warmup_updates) < 0:
        parser.error("--lr-warmup-updates must be non-negative")
    if float(args.lr_final_scale) <= 0.0:
        parser.error("--lr-final-scale must be positive")
    if float(args.second_mover_reward_bonus) < 0.0:
        parser.error("--second-mover-reward-bonus must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

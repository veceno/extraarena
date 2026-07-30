#!/usr/bin/env python3
"""Run Extra-LR V5 Ultra as the canonical Block-D League-2 consolidation.

This wrapper deliberately records the provisional Phase-C override, runs the
episode-continuous Rust league, saves eight cadence snapshots by default, and
persists a bounded hard-state trajectory sample for later auxiliary training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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

import mlx.core as mx  # noqa: E402
import mlx.optimizers as optim  # noqa: E402

import run_blockB_league as blockb  # noqa: E402
from ai.train_v2.model_mlx import load_checkpoint, save_checkpoint  # noqa: E402
from run_phaseA_random_bootstrap import MLXV5LearnerPolicy  # noqa: E402
from train_v3.a_gate import ManaDrawBaseline  # noqa: E402
from train_v3.aux_inference import CARD_CATALOG  # noqa: E402
from train_v3.block_d_league_driver import BlockDLeagueDriver  # noqa: E402
from train_v3.contracts import ACTION_FEATURE_DIM, OBS_V5_DIM  # noqa: E402
from train_v3.curriculum import CurriculumReweighter  # noqa: E402
from train_v3.c_to_d_handoff import E1CandidateSet  # noqa: E402
from train_v3.ppo_phaseA_config import (  # noqa: E402
    build_phase_a_random_bootstrap_config,
)
from train_v3.second_start_parity import SecondStartParityLoop  # noqa: E402
from train_v3.snapshot_pool import SnapshotEntry, SnapshotPool  # noqa: E402
from train_v3.ultra_trajectory_sink import UltraTrajectorySink  # noqa: E402
from train_v3.v5_policy import create_v5_policy  # noqa: E402


DEFAULT_SOURCE = (
    ROOT
    / "TrainV3.5/runs/phase_c_main_u29250_h299_luna10_paddingfix_20260727"
    / "checkpoints/extra_lr_v5_phaseC_candidate_h299.npz"
)
DEFAULT_ANCHOR_DIR = (
    ROOT
    / "TrainV3.5/runs/blockB_from_phaseA_p2accepted100_parallel_20260714_210400"
    / "checkpoints"
)
DEFAULT_POST_B = DEFAULT_ANCHOR_DIR / "extra_lr_v5_blockB_league_update_29250.npz"
DEFAULT_POST_B_PEER = (
    DEFAULT_ANCHOR_DIR / "extra_lr_v5_blockB_league_update_18500.npz"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    return blockb._jsonable(value)


def _git_provenance() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "status_lines": status.splitlines(),
        "tracked_diff_sha256": _sha256_bytes(diff),
        "tracked_diff_bytes": len(diff),
    }


def _catalog_provenance() -> dict[str, Any]:
    payload = json.dumps(
        CARD_CATALOG,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "card_count": len(CARD_CATALOG),
        "sha256": _sha256_bytes(payload),
        "ruleset": "arena_v5_mana_draw_50_cards",
    }


def _source_file_hashes(paths: list[Path]) -> list[dict[str, Any]]:
    output = []
    for path in paths:
        if path.exists():
            output.append(
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    return output


def _save_ultra_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    path: Path,
    run_meta: dict[str, Any],
    update_rows: list[dict[str, Any]],
    update: int,
    partial: bool,
) -> Path:
    mx.eval(model.parameters(), optimizer.state)
    metadata = {
        "run_name": run_meta["run_name"],
        "model_name": "extra-lr-v5-ultra",
        "phase": "blockD_league2_consolidation",
        "policy_kind": "v5_split_encoder",
        "state_action_interaction": "gated_bilinear_query_cap01_v1",
        "obs_dim": int(OBS_V5_DIM),
        "action_feature_dim": int(ACTION_FEATURE_DIM),
        "source_checkpoint": run_meta["source_checkpoint"],
        "source_checkpoint_sha256": run_meta["source_checkpoint_sha256"],
        "completed_updates": int(update),
        "partial_checkpoint": bool(partial),
        "provisional_phase_c_override": bool(
            run_meta["provisional_phase_c_override"]["enabled"]
        ),
        "last_metrics": update_rows[-1] if update_rows else {},
    }
    save_checkpoint(
        str(path),
        model,
        optimizer=optimizer,
        metadata=_jsonable(metadata),
    )
    return path


def _measure_mana_draw_baseline(
    *,
    game_runner: Any,
    seed: int,
    games: int,
) -> ManaDrawBaseline:
    results = [
        game_runner.play(
            "random",
            seed=int(seed) * 10_000 + game_index,
            candidate_side="p1" if game_index % 2 == 0 else "p2",
        ).game
        for game_index in range(int(games))
    ]
    count = sum(int(game.mana_draw_count) for game in results)
    eligible = sum(int(game.eligible_turns) for game in results)
    if eligible <= 0:
        raise RuntimeError("field mana-draw baseline has no eligible learner turns")
    return ManaDrawBaseline(
        mana_draw_count=count,
        eligible_turns=eligible,
        rate=float(count) / float(eligible),
        hand_cap=4,
        mana_draw_base=2,
        valid=True,
    )


def _write_run_files_manifest(out_dir: Path) -> dict[str, Any]:
    excluded = {"run_files_manifest.json"}
    files = sorted(
        path
        for path in out_dir.rglob("*")
        if path.is_file() and path.name not in excluded
    )
    manifest = {
        "schema": "extra_lr_v5_ultra_run_files_manifest_v1",
        "files": [
            {
                "path": str(path.relative_to(out_dir)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    (out_dir / "run_files_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.output_dir.resolve()
    checkpoint_dir = out_dir / "checkpoints"
    battle_log_dir = out_dir / "battle_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    battle_log_dir.mkdir(parents=True, exist_ok=True)

    if not bool(args.allow_provisional_phase_c):
        raise RuntimeError(
            "Phase-C h299 is provisional (51.074% no-assist vs u29250, "
            "CI includes 50%); pass --allow-provisional-phase-c only for an "
            "explicitly authorized Block-D continuation"
        )

    model = create_v5_policy(
        policy_kind="v5_split_encoder",
        hidden_dim=256,
        action_hidden_dim=128,
    )
    optimizer = optim.Adam(learning_rate=float(args.learning_rate))
    loaded = load_checkpoint(
        str(args.source_checkpoint.resolve()),
        model,
        optimizer=optimizer,
    )
    mx.eval(model.parameters(), optimizer.state)
    learner = MLXV5LearnerPolicy(
        model,
        library_path=args.library_path,
        rng=np.random.default_rng(int(args.seed) + 31),
    )

    config = build_phase_a_random_bootstrap_config(
        run_name=str(args.run_name),
        env_count=int(args.env_count),
        steps_per_update=int(args.steps_per_update),
        epochs=int(args.epochs),
        minibatch_size=int(args.minibatch_size),
        turn_order_second_mover_reward_bonus=0.0,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_every=int(args.checkpoint_every),
        metrics_path=str(out_dir / "progress.jsonl"),
        legal_row_pack_backend="python",
        seed=int(args.seed),
    )
    config = replace(
        config,
        decisive_early_end=False,
        second_start_oversampling={
            "policy": "fixed_second_start_weight",
            "second_start_weight": 0.5,
        },
    )

    pool = SnapshotPool(target_non_anchor_count=int(args.pool_size))
    pool.set_seed_anchor(
        SnapshotEntry(
            update_number=0,
            h2h_vs_best=float(args.phase_c_h2h_score),
            path=str(args.source_checkpoint.resolve()),
            p1_p2_gap=0.0,
            promotion_eligible=True,
            role="seed_anchor",
        )
    )
    for anchor in args.post_b_anchors:
        pool.add_snapshot(
            SnapshotEntry(
                update_number=0,
                h2h_vs_best=0.0,
                path=str(anchor.resolve()),
                p1_p2_gap=0.0,
                promotion_eligible=True,
                role="rolling",
            )
        )
    e1_candidates = E1CandidateSet(
        post_c3_best_path=str(args.source_checkpoint.resolve()),
        post_b_path=str(args.post_b_fallback.resolve()),
    )
    curriculum = CurriculumReweighter(window_n=int(args.curriculum_window))
    parity = SecondStartParityLoop(window_n=int(args.parity_window))
    opponent_policies = blockb._build_opponent_policies(
        args,
        pool=pool,
        learner=learner,
    )
    game_runner = blockb.LiveBlockBGameRunner(
        config=config,
        learner=learner,
        opponent_policies=opponent_policies,
        library_path=args.library_path,
        max_steps=int(args.eval_max_steps),
    )
    mana_draw_baseline = _measure_mana_draw_baseline(
        game_runner=game_runner,
        seed=int(args.seed),
        games=int(args.mana_draw_baseline_games),
    )

    rust_library = (
        args.library_path.resolve()
        if args.library_path is not None
        else ROOT / "TrainV3.5/target/release/libtrainv3_core.dylib"
    )
    run_meta = {
        "schema": "extra_lr_v5_ultra_blockD_run_v1",
        "status": "running",
        "run_name": str(args.run_name),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint.resolve()),
        "source_metadata": loaded.get("metadata", {}),
        "post_b_anchors": [
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path.resolve()),
            }
            for path in args.post_b_anchors
        ],
        "provisional_phase_c_override": {
            "enabled": True,
            "authorized_by": "user_request_2026-07-27",
            "no_assist_h2h_vs_u29250": {
                "record": [259, 248, 5],
                "games": 512,
                "score_rate": float(args.phase_c_h2h_score),
                "ci95": [0.4857, 0.5357],
            },
            "reason": (
                "Proceed to Block D while retaining h299 and post-B anchors; "
                "auxiliary uplift is not treated as a no-assist promotion pass."
            ),
        },
        "phase": {
            "canonical": "Block D League-2 post-C consolidation",
            "product_alias": "Extra-LR V5 Ultra",
            "next": "Block E1 tournament",
        },
        "config": _jsonable(vars(args)),
        "phase_config": _jsonable(asdict(config)),
        "opponent_mix_contract": {
            "self_v5_target": float(args.self_share_target),
            "curriculum_off": True,
            "exit_mode": "fixed_schedule",
        },
        "information_contract": {
            "history_events": 20,
            "own_hand_identity_known": True,
            "own_deck_known": True,
            "enemy_hand_known": True,
            "enemy_deck_known": True,
            "enemy_deck_order_known": True,
            "draw_assist_enabled": False,
            "assist_profile_id": 0,
        },
        "catalog": _catalog_provenance(),
        "git": _git_provenance(),
        "rust_library": (
            {
                "path": str(rust_library),
                "sha256": _sha256(rust_library),
                "bytes": rust_library.stat().st_size,
            }
            if rust_library.exists()
            else None
        ),
        "source_files": _source_file_hashes(
            [
                Path(__file__),
                ROOT
                / "TrainV3.5/python/train_v3/block_d_league_driver.py",
                ROOT / "TrainV3.5/python/train_v3/rust_live_self_play.py",
                ROOT / "TrainV3.5/python/train_v3/ultra_trajectory_sink.py",
            ]
        ),
        "mana_draw_baseline": _jsonable(asdict(mana_draw_baseline)),
        "trajectory_contract": {
            "sampled_envs_per_update": int(args.log_envs),
            "learner_transitions_only": True,
            "opponent_actions_not_materialized": True,
            "accepted_actions_only": True,
            "authoritative_assembler_labels": False,
            "authoritative_cardoptimum_labels": False,
            "followup_required": "paired_matchups_and_counterfactual_draw_branches",
        },
    }
    run_meta_path = out_dir / "run_meta.json"
    run_meta_path.write_text(
        json.dumps(_jsonable(run_meta), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    trajectory_sink = UltraTrajectorySink(
        battle_log_dir,
        sampled_envs=int(args.log_envs),
    )
    progress_path = out_dir / "progress.jsonl"
    update_rows: list[dict[str, Any]] = []

    def learning_rate(update_number: int) -> float:
        return blockb._learning_rate_schedule(
            base_lr=float(args.learning_rate),
            update_number=int(update_number),
            total_updates=int(args.updates),
            warmup_updates=int(args.lr_warmup_updates),
            final_scale=float(args.lr_final_scale),
        )

    def trajectory_and_progress_sink(
        update_number: int,
        rollout: Any,
        metrics: dict[str, Any],
        session: Any,
    ) -> None:
        trajectory_sink(update_number, rollout, metrics, session)
        row = blockb._compact_metrics(metrics)
        row["update_number"] = int(update_number)
        row["learning_rate"] = float(learning_rate(update_number))
        row["opponent_counts"] = blockb._counts(
            row.get("opponent_identities", [])
        )
        row["learner_actor_counts"] = blockb._counts(
            row.get("learner_actor_ids", [])
        )
        update_rows.append(row)
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(_jsonable(row), sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        if (
            update_number == 1
            or update_number == int(args.updates)
            or update_number % int(args.console_every) == 0
        ):
            print(
                "BLOCKD_ULTRA_UPDATE",
                json.dumps(_jsonable(blockb._console_row(row)), sort_keys=True),
                flush=True,
            )

    snapshots: list[dict[str, Any]] = []

    def checkpoint_namer(update_number: int) -> str:
        path = (
            checkpoint_dir
            / f"extra_lr_v5_ultra_blockD_update_{int(update_number):05d}.npz"
        )
        _save_ultra_checkpoint(
            model=model,
            optimizer=optimizer,
            path=path,
            run_meta=run_meta,
            update_rows=update_rows,
            update=int(update_number),
            partial=True,
        )
        snapshots.append(
            {
                "update": int(update_number),
                "path": str(path),
                "sha256": _sha256(path),
            }
        )
        return str(path)

    driver = BlockDLeagueDriver(
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
        n_snap=min(5, max(1, int(args.updates) // int(args.checkpoint_every))),
        k_snap=int(args.updates) // int(args.checkpoint_every) + 1,
        checkpoint_namer=checkpoint_namer,
        games_per_opponent_per_side=int(args.games_per_opponent_per_side),
        games_per_opponent_gauntlet=int(args.games_per_opponent_gauntlet),
        model=model,
        optimizer=optimizer,
        learning_rate_for_update=learning_rate,
        trajectory_sink=trajectory_and_progress_sink,
        steps_per_update=int(args.steps_per_update),
        self_share_target=float(args.self_share_target),
        exit_mode="fixed_schedule",
        curriculum_off=True,
        e1_candidate_set=e1_candidates,
    )

    try:
        driver_manifest = driver.run(int(args.updates))
        if len(update_rows) != int(driver_manifest.n_updates_run):
            raise RuntimeError(
                "trajectory/progress accounting mismatch: "
                f"{len(update_rows)} != {driver_manifest.n_updates_run}"
            )
        trajectory_manifest = trajectory_sink.finalize()
        (out_dir / "snapshot_pool.json").write_text(
            json.dumps(pool.to_manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        final_checkpoint = _save_ultra_checkpoint(
            model=model,
            optimizer=optimizer,
            path=out_dir
            / (
                "extra_lr_v5_ultra_blockD_final_update_"
                f"{int(driver_manifest.n_updates_run):05d}.npz"
            ),
            run_meta=run_meta,
            update_rows=update_rows,
            update=int(driver_manifest.n_updates_run),
            partial=False,
        )
        best_checkpoint = (
            Path(pool.best_ever.path)
            if pool.best_ever is not None
            else final_checkpoint
        )
        summary = {
            "schema": "extra_lr_v5_ultra_blockD_manifest_v1",
            "status": "ok",
            "run_name": str(args.run_name),
            "source_checkpoint": str(args.source_checkpoint.resolve()),
            "source_checkpoint_sha256": run_meta["source_checkpoint_sha256"],
            "updates": len(update_rows),
            "best_checkpoint": str(best_checkpoint),
            "final_checkpoint": str(final_checkpoint),
            "snapshots": snapshots,
            "pool": pool.to_manifest(),
            "driver_manifest": driver_manifest.to_dict(),
            "trajectory_manifest": trajectory_manifest,
            "e1_candidates": driver_manifest.candidate_paths,
            "last_metrics": update_rows[-1] if update_rows else {},
        }
        manifest_path = out_dir / "blockD_ultra_manifest.json"
        manifest_path.write_text(
            json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run_meta["status"] = "ok"
        run_meta["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        run_meta_path.write_text(
            json.dumps(_jsonable(run_meta), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_run_files_manifest(out_dir)
        print(
            "BLOCKD_ULTRA_RESULT",
            json.dumps(_jsonable(summary), sort_keys=True),
            flush=True,
        )
        return summary
    except Exception as exc:
        try:
            trajectory_sink.finalize()
        except Exception:
            pass
        run_meta["status"] = "failed"
        run_meta["failed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        run_meta["error"] = f"{exc.__class__.__name__}: {exc}"
        run_meta_path.write_text(
            json.dumps(_jsonable(run_meta), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="extra_lr_v5_ultra_blockD")
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--post-b-anchors",
        type=Path,
        nargs="+",
        default=[DEFAULT_POST_B, DEFAULT_POST_B_PEER],
    )
    parser.add_argument(
        "--post-b-fallback",
        type=Path,
        default=DEFAULT_POST_B,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-provisional-phase-c", action="store_true")
    parser.add_argument("--phase-c-h2h-score", type=float, default=0.51074)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--env-count", type=int, default=128)
    parser.add_argument("--steps-per-update", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lr-warmup-updates", type=int, default=50)
    parser.add_argument("--lr-final-scale", type=float, default=0.2)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=52724001)
    parser.add_argument("--pool-size", type=int, default=6)
    parser.add_argument("--self-share-target", type=float, default=0.50)
    parser.add_argument("--curriculum-window", type=int, default=32)
    parser.add_argument("--parity-window", type=int, default=64)
    parser.add_argument("--eval-max-steps", type=int, default=240)
    parser.add_argument("--games-per-opponent-per-side", type=int, default=8)
    parser.add_argument("--games-per-opponent-gauntlet", type=int, default=4)
    parser.add_argument("--mana-draw-baseline-games", type=int, default=16)
    parser.add_argument("--log-envs", type=int, default=8)
    parser.add_argument("--console-every", type=int, default=10)
    parser.add_argument(
        "--v4-onnx",
        type=Path,
        default=ROOT / "ai/models/extra-lr-v4-max.onnx",
    )
    parser.add_argument("--allow-missing-v4", action="store_true")
    parser.add_argument(
        "--library-path",
        type=Path,
        default=blockb._default_library_path(),
    )
    args = parser.parse_args(argv)
    required_paths = [
        args.source_checkpoint,
        args.post_b_fallback,
        *args.post_b_anchors,
    ]
    for path in required_paths:
        if not path.exists():
            parser.error(f"required checkpoint not found: {path}")
    for name in (
        "updates",
        "env_count",
        "steps_per_update",
        "epochs",
        "minibatch_size",
        "checkpoint_every",
        "eval_max_steps",
        "mana_draw_baseline_games",
        "log_envs",
        "console_every",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if float(args.learning_rate) <= 0.0:
        parser.error("--learning-rate must be positive")
    if not 0.0 < float(args.lr_final_scale) <= 1.0:
        parser.error("--lr-final-scale must be in (0, 1]")
    if int(args.lr_warmup_updates) < 0:
        parser.error("--lr-warmup-updates must be non-negative")
    if not 0.0 <= float(args.self_share_target) <= 1.0:
        parser.error("--self-share-target must be in [0, 1]")
    return args


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

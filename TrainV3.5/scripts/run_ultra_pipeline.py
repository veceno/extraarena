#!/usr/bin/env python3
"""Detached-safe Block-D Ultra -> snapshot screen -> full bench -> aux chain."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "TrainV3.5/scripts/run_blockD_ultra.py"
POST = ROOT / "TrainV3.5/scripts/run_ultra_post_bench.py"


def _state(path: Path, status: str, **extra: object) -> None:
    payload = {
        "schema": "extra_lr_v5_ultra_orchestration_v1",
        "status": status,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **extra,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run(command: list[str], log: Path) -> None:
    with log.open("a", encoding="utf-8") as stream:
        stream.write("COMMAND " + json.dumps(command) + "\n")
        stream.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "orchestration_state.json"
    log_path = run_dir / "orchestration.log"
    _state(
        state_path,
        "training",
        run_dir=str(run_dir),
        updates=int(args.updates),
    )
    try:
        _run(
            [
                sys.executable,
                str(TRAIN),
                "--output-dir",
                str(run_dir),
                "--run-name",
                str(args.run_name),
                "--allow-provisional-phase-c",
                "--updates",
                str(int(args.updates)),
                "--env-count",
                str(int(args.env_count)),
                "--steps-per-update",
                str(int(args.steps_per_update)),
                "--checkpoint-every",
                str(int(args.checkpoint_every)),
                "--log-envs",
                str(int(args.log_envs)),
                "--console-every",
                str(int(args.console_every)),
            ],
            log_path,
        )
        _state(
            state_path,
            "post_benchmark",
            run_dir=str(run_dir),
            updates=int(args.updates),
        )
        post_command = [
            sys.executable,
            str(POST),
            "--run-dir",
            str(run_dir),
            "--screen-seeds",
            str(int(args.screen_seeds)),
            "--full-seeds",
            str(int(args.full_seeds)),
        ]
        if bool(args.generate_aux):
            post_command.extend(
                [
                    "--generate-aux",
                    "--aux-battles",
                    str(int(args.aux_battles)),
                    "--branch-repeats",
                    str(int(args.branch_repeats)),
                ]
            )
        _run(post_command, log_path)
        _state(
            state_path,
            "complete",
            run_dir=str(run_dir),
            updates=int(args.updates),
            result=str(
                run_dir
                / "post_training_benchmarks/post_bench_result.json"
            ),
        )
    except Exception as exc:
        _state(
            state_path,
            "failed",
            run_dir=str(run_dir),
            updates=int(args.updates),
            error=f"{exc.__class__.__name__}: {exc}",
        )
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="extra_lr_v5_ultra_blockD_2000")
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--env-count", type=int, default=128)
    parser.add_argument("--steps-per-update", type=int, default=24)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--log-envs", type=int, default=8)
    parser.add_argument("--console-every", type=int, default=10)
    parser.add_argument("--screen-seeds", type=int, default=16)
    parser.add_argument("--full-seeds", type=int, default=64)
    parser.add_argument("--generate-aux", action="store_true")
    parser.add_argument("--aux-battles", type=int, default=10_000)
    parser.add_argument("--branch-repeats", type=int, default=4)
    args = parser.parse_args(argv)
    for name in (
        "updates",
        "env_count",
        "steps_per_update",
        "checkpoint_every",
        "log_envs",
        "console_every",
        "screen_seeds",
        "full_seeds",
        "aux_battles",
        "branch_repeats",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

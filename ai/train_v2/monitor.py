"""
Training throughput monitor for TrainV2 PPO.
Reads metrics.jsonl from a run directory and prints a concise summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(path_or_run_dir: str) -> list[dict]:
    p = Path(path_or_run_dir)
    if p.is_file() and p.suffix == ".jsonl":
        jsonl_path = p
    else:
        jsonl_path = p / "metrics.jsonl"

    if not jsonl_path.is_file():
        return []

    records: list[dict] = []
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
    return records


def summarize_metrics(records: list[dict]) -> dict:
    train_records = [r for r in records if r.get("type") not in ("eval", "skipped_update")]
    eval_records = [r for r in records if r.get("type") == "eval"]
    skipped_records = [r for r in records if r.get("type") == "skipped_update"]

    last_update: int | None = None
    last_steps: int = 0
    last_loss: float | None = None
    last_entropy: float | None = None

    if train_records:
        last = train_records[-1]
        last_update = last.get("update")
        last_steps = last.get("steps", 0)
        last_loss = last.get("loss")
        last_entropy = last.get("entropy")

    steps_per_update: float = 0.0
    if train_records:
        steps_per_update = last_steps / len(train_records)

    best_eval: dict | None = None
    best_winrate = -1.0
    for rec in eval_records:
        wr = rec.get("winrate")
        if wr is not None and isinstance(wr, (int, float)) and wr > best_winrate:
            best_winrate = wr
            best_eval = {
                "opponent": rec.get("opponent", "?"),
                "winrate": wr,
                "update": rec.get("update", 0),
            }
    if best_eval is None and not eval_records:
        best_eval = None
    elif best_eval is None:
        best_eval = None

    return {
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "last_update": last_update,
        "last_steps": last_steps,
        "last_loss": last_loss,
        "last_entropy": last_entropy,
        "steps_per_update": steps_per_update,
        "best_eval": best_eval,
        "skipped_updates": len(skipped_records),
        "last_skipped_update": skipped_records[-1]["update"] if skipped_records else None,
    }


def recommended_commands(*, output_dir: str = "ai/train_v2/runs") -> dict[str, str]:
    return {
        "quick": (
            f"python3 -m ai.train_v2.experiment --preset m4_quick "
            f"--name quick_rl --output-dir {output_dir} --eval-games 8"
        ),
        "night": (
            f"python3 -m ai.train_v2.experiment --preset m4_night "
            f"--name night_rl --output-dir {output_dir} --eval-games 20"
        ),
        "resume": (
            f"python3 -m ai.train_v2.experiment --preset m4_quick "
            f"--resume-checkpoint <checkpoint.npz> "
            f"--name resume_rl --output-dir {output_dir}"
        ),
        "leaderboard": (
            f"python3 -m ai.train_v2.leaderboard "
            f"--paths {output_dir} --games 8 --seed 42 --max-steps 200"
        ),
    }


def _print_summary(summary: dict) -> None:
    print(f"Train records: {summary['train_records']} | Eval records: {summary['eval_records']}")
    if summary["last_update"] is not None:
        print(f"Last update: {summary['last_update']} | Steps: {summary['last_steps']}")
    if summary["last_loss"] is not None:
        print(f"Loss: {summary['last_loss']:.4f} | Entropy: {summary['last_entropy']:.2f}")
    print(f"Steps/update: {summary['steps_per_update']:.1f}")
    if summary["skipped_updates"] > 0:
        print(f"Skipped updates: {summary['skipped_updates']} | Last skipped: {summary['last_skipped_update']}")
    if summary["best_eval"]:
        be = summary["best_eval"]
        print(f"Best eval: {be['opponent']} wr={be['winrate']:.3f} at update={be['update']}")


def _main():
    parser = argparse.ArgumentParser(description="Monitor PPO training metrics")
    parser.add_argument("--run", default=None, help="Path to run directory")
    parser.add_argument("--metrics", default=None, help="Path to metrics.jsonl file")
    parser.add_argument("--commands", action="store_true", help="Print recommended M4 Pro commands")
    args = parser.parse_args()

    if args.commands:
        cmds = recommended_commands()
        for label, cmd in cmds.items():
            print(f"[{label}]\n{cmd}\n")
        return

    path = args.run or args.metrics
    if path is None:
        parser.print_help()
        return

    records = load_metrics(path)
    if not records:
        print("No metrics found.")
        return

    summary = summarize_metrics(records)
    _print_summary(summary)


if __name__ == "__main__":
    _main()

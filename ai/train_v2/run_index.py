from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ai.train_v2.report import load_run_report


def _is_run_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    return (
        (p / "config.json").is_file()
        or (p / "summary.json").is_file()
        or (p / "metrics.jsonl").is_file()
    )


def discover_runs(root: str) -> list[str]:
    p = Path(root)
    if not p.exists():
        return []

    candidates: list[Path] = []

    if _is_run_dir(p):
        candidates.append(p)
    elif p.is_dir():
        for child in p.iterdir():
            if _is_run_dir(child):
                candidates.append(child)

    return sorted([str(c.resolve()) for c in candidates])


def build_run_index(root: str) -> dict:
    rows: list[dict] = []

    for run_dir in discover_runs(root):
        report = load_run_report(run_dir)
        config = report.get("config") or {}
        summary = report.get("summary") or {}
        train = summary.get("train", {}) or {}
        eval_data = summary.get("eval", {}) or {}
        metrics_summary = report.get("metrics_summary", {}) or {}

        latest_ckpt = report.get("latest_checkpoint")
        latest_onnx = report.get("latest_onnx")

        row = {
            "run_dir": run_dir,
            "name": config.get("name") or Path(run_dir).name,
            "seed": config.get("seed") if config else None,
            "updates": train.get("updates"),
            "steps": train.get("steps"),
            "last_loss": train.get("last_loss"),
            "last_entropy": train.get("last_entropy"),
            "skipped_updates": metrics_summary.get("skipped_updates", 0),
            "latest_checkpoint": latest_ckpt,
            "latest_onnx": latest_onnx,
            "has_checkpoint": bool(latest_ckpt),
            "has_onnx": bool(latest_onnx),
            "wr_random": eval_data.get("random", {}).get("winrate"),
            "wr_end_turn": eval_data.get("end_turn", {}).get("winrate"),
            "wr_greedy_face": eval_data.get("greedy_face", {}).get("winrate"),
            "status": "ok" if (report.get("summary") is not None or bool(report.get("latest_checkpoint"))) else "partial",
        }
        rows.append(row)

    rows.sort(key=lambda r: r["run_dir"])

    return {
        "root": str(Path(root).resolve()),
        "runs": len(rows),
        "rows": rows,
    }


def save_run_index(index: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def _fmt(v, width: int, fmt: str | None = None) -> str:
    if v is None:
        return "-".rjust(width)
    if fmt:
        return f"{v:{fmt}}".rjust(width)
    return str(v).rjust(width)


def _print_compact_table(index: dict) -> None:
    rows = index.get("rows", [])
    print(f"runs: {len(rows)}")
    if not rows:
        return

    header = (
        f"{'name':28} "
        f"{'upd':>4} "
        f"{'steps':>8} "
        f"{'loss':>8} "
        f"{'wr_rnd':>7} "
        f"{'wr_grd':>7} "
        f"{'skip':>5} "
        f"{'onnx':>5}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        name = str(row.get("name") or Path(row["run_dir"]).name)[:28]
        upd = _fmt(row.get("updates"), 4)
        steps = _fmt(row.get("steps"), 8)
        loss = _fmt(row.get("last_loss"), 8, ".4f")
        wr_rnd = _fmt(row.get("wr_random"), 7, ".3f")
        wr_grd = _fmt(row.get("wr_greedy_face"), 7, ".3f")
        skip = _fmt(row.get("skipped_updates"), 5)
        onnx = "yes" if row.get("has_onnx") else "no"
        print(f"{name:28} {upd} {steps} {loss} {wr_rnd} {wr_grd} {skip} {onnx:>5}")


def _main():
    parser = argparse.ArgumentParser(description="Build TrainV2 run index")
    parser.add_argument("--root", required=True, help="Path to runs directory or a single run")
    parser.add_argument("--output", default=None, help="Path to save index JSON")
    args = parser.parse_args()

    index = build_run_index(args.root)

    if args.output:
        save_run_index(index, args.output)
        print(f"Index saved to {args.output}")
    else:
        _print_compact_table(index)


if __name__ == "__main__":
    _main()

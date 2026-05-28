from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ai.train_v2.monitor import load_metrics, summarize_metrics


def load_run_report(run_dir: str) -> dict:
    p = Path(run_dir)

    report: dict = {
        "run_dir": str(p.resolve()),
        "config": None,
        "summary": None,
        "metrics_summary": {},
        "checkpoints": [],
        "latest_checkpoint": None,
        "onnx_models": [],
        "latest_onnx": None,
        "leaderboard_row": None,
    }

    config_path = p / "config.json"
    if config_path.is_file():
        try:
            report["config"] = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    summary_path = p / "summary.json"
    if summary_path.is_file():
        try:
            report["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    records = load_metrics(str(p))
    if records:
        report["metrics_summary"] = summarize_metrics(records)

    ckpt_dir = p / "checkpoints"
    if ckpt_dir.is_dir():
        ckpts = sorted(ckpt_dir.glob("update_*.npz"))
        report["checkpoints"] = [str(c.resolve()) for c in ckpts]
        if ckpts:
            report["latest_checkpoint"] = str(ckpts[-1].resolve())

    exported_dir = p / "exported"
    if exported_dir.is_dir():
        onnxs = sorted(exported_dir.glob("*.onnx"))
        report["onnx_models"] = [str(o.resolve()) for o in onnxs]
        if onnxs:
            report["latest_onnx"] = str(onnxs[-1].resolve())

    leaderboard_path = p / "leaderboard.json"
    if leaderboard_path.is_file() and report["latest_onnx"]:
        try:
            lb = json.loads(leaderboard_path.read_text(encoding="utf-8"))
            rows = lb.get("rows", [])
            latest_onnx = report["latest_onnx"]
            latest_stem = Path(latest_onnx).stem
            for row in rows:
                if row.get("onnx_path") == latest_onnx:
                    report["leaderboard_row"] = row
                    break
                if row.get("model_name") == latest_stem:
                    report["leaderboard_row"] = row
                    break
        except (json.JSONDecodeError, OSError):
            pass

    return report


def format_report_markdown(report: dict) -> str:
    run_dir = report.get("run_dir", "n/a")
    ms = report.get("metrics_summary", {}) or {}
    summary = report.get("summary") or {}
    train = summary.get("train", {}) or {}
    eval_data = summary.get("eval", {}) or {}

    updates = train.get("updates")
    steps = train.get("steps")
    last_loss = train.get("last_loss")
    last_entropy = train.get("last_entropy")
    skipped = ms.get("skipped_updates")

    latest_ckpt = report.get("latest_checkpoint")
    latest_onnx = report.get("latest_onnx")

    wr_random = eval_data.get("random", {}).get("winrate")
    wr_end_turn = eval_data.get("end_turn", {}).get("winrate")
    wr_greedy = eval_data.get("greedy_face", {}).get("winrate")

    lb_row = report.get("leaderboard_row")

    def _val(v):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = [
        "# TrainV2 Run Report",
        "",
        f"Run: `{run_dir}`",
        "",
        "## Training",
        f"- Updates: {_val(updates)}",
        f"- Steps: {_val(steps)}",
        f"- Last loss: {_val(last_loss)}",
        f"- Last entropy: {_val(last_entropy)}",
        f"- Skipped updates: {_val(skipped)}",
        "",
        "## Artifacts",
        f"- Latest checkpoint: {latest_ckpt or 'n/a'}",
        f"- Latest ONNX: {latest_onnx or 'n/a'}",
        "",
        "## Eval",
        f"- random winrate: {_val(wr_random)}",
        f"- end_turn winrate: {_val(wr_end_turn)}",
        f"- greedy_face winrate: {_val(wr_greedy)}",
        "",
    ]

    if lb_row:
        lines.extend([
            "## Leaderboard",
            f"- Rank: {lb_row.get('rank', 'n/a')}",
            f"- Score: {_val(lb_row.get('score'))}",
            f"- Parity mismatches: {lb_row.get('parity_mismatches', 'n/a')}",
            "",
        ])
    else:
        lines.extend([
            "## Leaderboard",
            "- n/a",
            "",
        ])

    return "\n".join(lines)


def save_report(report: dict, output_path: str, *, markdown: bool = False) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if markdown:
            f.write(format_report_markdown(report))
        else:
            json.dump(report, f, indent=2, ensure_ascii=False)


def _main():
    parser = argparse.ArgumentParser(description="Generate TrainV2 run report")
    parser.add_argument("--run", required=True, help="Path to run directory")
    parser.add_argument("--output", default=None, help="Path to save report")
    parser.add_argument("--markdown", action="store_true", help="Output markdown instead of JSON")
    args = parser.parse_args()

    report = load_run_report(args.run)

    if args.output:
        save_report(report, args.output, markdown=args.markdown)
        print(f"Report saved to {args.output}")
    else:
        if args.markdown:
            print(format_report_markdown(report))
        else:
            print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()

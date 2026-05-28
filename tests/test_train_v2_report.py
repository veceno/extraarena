import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from ai.train_v2.report import load_run_report, format_report_markdown, save_report


def test_load_run_report_empty_dir(tmp_path):
    report = load_run_report(str(tmp_path))
    assert report["run_dir"] == str(tmp_path.resolve())
    assert report["config"] is None
    assert report["summary"] is None
    assert report["metrics_summary"] == {}
    assert report["checkpoints"] == []
    assert report["latest_checkpoint"] is None
    assert report["onnx_models"] == []
    assert report["latest_onnx"] is None
    assert report["leaderboard_row"] is None


def test_load_run_report_with_files(tmp_path):
    run = tmp_path / "run_demo"
    run.mkdir()

    config = {"name": "demo", "seed": 42}
    summary = {
        "run_dir": str(run),
        "config": config,
        "train": {"updates": 3, "steps": 120, "last_loss": 0.5, "last_entropy": 1.2},
        "eval": {
            "random": {"winrate": 0.75},
            "end_turn": {"winrate": 0.60},
            "greedy_face": {"winrate": 0.45},
        },
    }
    (run / "config.json").write_text(json.dumps(config))
    (run / "summary.json").write_text(json.dumps(summary))

    # metrics.jsonl
    metrics = [
        {"type": "train", "update": 1, "steps": 40, "loss": 0.6, "entropy": 1.3},
        {"type": "eval", "update": 1, "opponent": "random", "winrate": 0.70},
        {"type": "skipped_update", "update": 2},
    ]
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics))

    ckpt_dir = run / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "update_0001.npz").write_bytes(b"")
    (ckpt_dir / "update_0002.npz").write_bytes(b"")

    exported_dir = run / "exported"
    exported_dir.mkdir()
    (exported_dir / "update_0002.onnx").write_bytes(b"")

    leaderboard = {
        "rows": [
            {
                "onnx_path": str(exported_dir / "update_0002.onnx"),
                "model_name": "update_0002",
                "score": 1.0,
                "wr_random": 0.75,
            }
        ]
    }
    (run / "leaderboard.json").write_text(json.dumps(leaderboard))

    report = load_run_report(str(run))
    assert report["config"] == config
    assert report["summary"] == summary
    assert report["metrics_summary"]["train_records"] == 1
    assert report["metrics_summary"]["skipped_updates"] == 1
    assert len(report["checkpoints"]) == 2
    assert report["latest_checkpoint"] == str((ckpt_dir / "update_0002.npz").resolve())
    assert len(report["onnx_models"]) == 1
    assert report["latest_onnx"] == str((exported_dir / "update_0002.onnx").resolve())
    assert report["leaderboard_row"] is not None
    assert report["leaderboard_row"]["score"] == 1.0


def test_format_report_markdown():
    report = {
        "run_dir": "/tmp/run",
        "metrics_summary": {
            "last_update": 5,
            "last_steps": 200,
            "last_loss": 0.1234,
            "last_entropy": 0.98,
            "skipped_updates": 2,
        },
        "summary": {
            "train": {"updates": 5, "steps": 200, "last_loss": 0.1234, "last_entropy": 0.98},
            "eval": {
                "random": {"winrate": 0.8},
                "end_turn": {"winrate": 0.6},
                "greedy_face": {"winrate": 0.4},
            },
        },
        "latest_checkpoint": "/tmp/run/ckpt.npz",
        "latest_onnx": "/tmp/run/model.onnx",
        "leaderboard_row": None,
    }
    md = format_report_markdown(report)
    assert "# TrainV2 Run Report" in md
    assert "`/tmp/run`" in md
    assert "Updates: 5" in md
    assert "Last loss: 0.1234" in md
    assert "random winrate: 0.8000" in md
    assert "end_turn winrate: 0.6000" in md
    assert "greedy_face winrate: 0.4000" in md


def test_save_report_json(tmp_path):
    report = {"run_dir": str(tmp_path), "dummy": True}
    out = tmp_path / "report.json"
    save_report(report, str(out), markdown=False)
    assert out.is_file()
    loaded = json.loads(out.read_text())
    assert loaded["dummy"] is True


def test_save_report_markdown(tmp_path):
    report = {
        "run_dir": str(tmp_path),
        "metrics_summary": {},
        "summary": {"train": {}, "eval": {}},
        "latest_checkpoint": None,
        "latest_onnx": None,
    }
    out = tmp_path / "report.md"
    save_report(report, str(out), markdown=True)
    assert out.is_file()
    text = out.read_text()
    assert "# TrainV2 Run Report" in text


def test_report_cli_smoke(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai.train_v2.report",
            "--run",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Run:" in proc.stdout or "run_dir" in proc.stdout

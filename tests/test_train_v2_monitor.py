"""
Tests for training monitor (Task 14).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from ai.train_v2.monitor import (
    load_metrics,
    summarize_metrics,
    recommended_commands,
)


class TestLoadMetrics:
    def test_load_metrics_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"update": 1, "loss": 0.5}) + "\n")
            f.write(json.dumps({"update": 2, "loss": 0.3}) + "\n")
            tmp_path = f.name

        try:
            records = load_metrics(tmp_path)
            assert len(records) == 2
            assert records[0]["update"] == 1
            assert records[1]["loss"] == 0.3
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_metrics_from_run_dir(self):
        run_dir = tempfile.mkdtemp(prefix="_t14_run_")
        try:
            metrics_path = Path(run_dir) / "metrics.jsonl"
            metrics_path.write_text(
                json.dumps({"update": 1, "loss": 0.1}) + "\n"
            )
            records = load_metrics(run_dir)
            assert len(records) == 1
            assert records[0]["loss"] == 0.1
        finally:
            import shutil
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_load_metrics_ignores_bad_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"update": 1, "loss": 0.5}) + "\n")
            f.write("not valid json!!!\n")
            f.write(json.dumps({"update": 2, "loss": 0.3}) + "\n")
            f.write("\n")
            f.write("  \n")
            tmp_path = f.name

        try:
            records = load_metrics(tmp_path)
            assert len(records) == 2
            assert records[0]["update"] == 1
            assert records[1]["update"] == 2
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestSummarizeMetrics:
    def test_summarize_metrics(self):
        records = [
            {"update": 1, "steps": 500, "loss": 0.5, "entropy": 2.0},
            {"update": 2, "steps": 1000, "loss": 0.3, "entropy": 1.8},
            {"type": "eval", "update": 2, "opponent": "random", "winrate": 0.6, "games": 8},
            {"type": "eval", "update": 2, "opponent": "end_turn", "winrate": 0.4, "games": 8},
        ]
        summary = summarize_metrics(records)
        assert summary["train_records"] == 2
        assert summary["eval_records"] == 2
        assert summary["last_update"] == 2
        assert summary["last_steps"] == 1000
        assert summary["last_loss"] == 0.3
        assert summary["last_entropy"] == 1.8
        assert summary["steps_per_update"] == 500.0
        assert summary["best_eval"] is not None
        assert summary["best_eval"]["opponent"] == "random"
        assert summary["best_eval"]["winrate"] == 0.6
        assert summary["best_eval"]["update"] == 2

    def test_summarize_metrics_empty(self):
        summary = summarize_metrics([])
        assert summary["train_records"] == 0
        assert summary["eval_records"] == 0
        assert summary["last_update"] is None
        assert summary["last_steps"] == 0
        assert summary["last_loss"] is None
        assert summary["last_entropy"] is None
        assert summary["steps_per_update"] == 0.0
        assert summary["best_eval"] is None

    def test_summarize_metrics_eval_only(self):
        records = [
            {"type": "eval", "update": 1, "opponent": "random", "winrate": 0.5, "games": 4},
        ]
        summary = summarize_metrics(records)
        assert summary["train_records"] == 0
        assert summary["eval_records"] == 1
        assert summary["last_update"] is None
        assert summary["steps_per_update"] == 0.0
        assert summary["best_eval"] is not None
        assert summary["best_eval"]["winrate"] == 0.5


class TestRecommendedCommands:
    def test_recommended_commands_contains_presets(self):
        cmds = recommended_commands(output_dir="ai/train_v2/runs")
        assert set(cmds.keys()) == {"quick", "night", "resume", "leaderboard"}
        assert "m4_quick" in cmds["quick"]
        assert "m4_night" in cmds["night"]
        assert "checkpoint.npz" in cmds["resume"]
        assert "leaderboard" in cmds["leaderboard"]


class TestMonitorCLI:
    def test_monitor_cli_commands_smoke(self):
        proc = subprocess.run(
            [sys.executable, "-m", "ai.train_v2.monitor", "--commands"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, f"CLI failed:\n{proc.stderr}"
        assert "quick" in proc.stdout
        assert "night" in proc.stdout
        assert "resume" in proc.stdout
        assert "leaderboard" in proc.stdout

    def test_monitor_cli_no_args_shows_help(self):
        proc = subprocess.run(
            [sys.executable, "-m", "ai.train_v2.monitor"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0
        assert "usage:" in proc.stdout.lower() or "usage:" in proc.stderr.lower()

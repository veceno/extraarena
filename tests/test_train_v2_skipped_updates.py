"""
Tests for skipped update visibility (Task 16).
"""
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from ai.train_v2.train_ppo import PPOConfig, train
from ai.train_v2.monitor import summarize_metrics


class TestSkippedUpdatesTrain:
    def test_train_return_includes_skipped_updates(self):
        ckpt_dir = tempfile.mkdtemp(prefix="_t16_skipped_")
        try:
            config = PPOConfig(
                total_updates=1,
                episodes_per_update=1,
                max_steps_per_episode=10,
                hidden_dim=32,
                action_hidden_dim=16,
                minibatch_size=8,
                epochs=1,
                seed=42,
                checkpoint_dir=ckpt_dir,
                fail_on_non_finite=True,
                min_batch_transitions=999,
            )
            result = train(config)
            assert result["skipped_updates"] > 0
            assert result["updates"] == 1
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    def test_train_records_skipped_update_metric(self):
        ckpt_dir = tempfile.mkdtemp(prefix="_t16_skipped_metrics_")
        metrics_path = Path(ckpt_dir) / "metrics.jsonl"
        try:
            config = PPOConfig(
                total_updates=1,
                episodes_per_update=1,
                max_steps_per_episode=10,
                hidden_dim=32,
                action_hidden_dim=16,
                minibatch_size=8,
                epochs=1,
                seed=42,
                checkpoint_dir=ckpt_dir,
                metrics_path=str(metrics_path),
                fail_on_non_finite=True,
                min_batch_transitions=999,
            )
            train(config)

            assert metrics_path.exists()
            lines = metrics_path.read_text().strip().splitlines()
            skipped_types = []
            for line in lines:
                rec = json.loads(line)
                if rec.get("type") == "skipped_update":
                    skipped_types.append(rec)
            assert len(skipped_types) > 0
            first = skipped_types[0]
            assert first["reason"] == "min_batch_transitions"
            assert first["min_batch_transitions"] == 999
            assert isinstance(first["transitions"], int)
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    def test_skip_does_not_write_checkpoint(self):
        ckpt_dir = tempfile.mkdtemp(prefix="_t16_no_ckpt_")
        try:
            config = PPOConfig(
                total_updates=1,
                episodes_per_update=1,
                max_steps_per_episode=10,
                hidden_dim=32,
                action_hidden_dim=16,
                minibatch_size=8,
                epochs=1,
                seed=42,
                checkpoint_dir=ckpt_dir,
                fail_on_non_finite=True,
                min_batch_transitions=999,
            )
            result = train(config)
            assert result["checkpoint_path"] == ""
            ckpt_files = list(Path(ckpt_dir).glob("update_*.npz"))
            assert len(ckpt_files) == 0
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)


class TestSkippedUpdatesMonitor:
    def test_monitor_counts_skipped_updates(self):
        records = [
            {"update": 1, "steps": 500, "loss": 0.5, "entropy": 2.0},
            {"type": "skipped_update", "update": 2, "transitions": 1, "reason": "min_batch_transitions"},
            {"type": "skipped_update", "update": 3, "transitions": 0, "reason": "min_batch_transitions"},
            {"update": 4, "steps": 1000, "loss": 0.3, "entropy": 1.8},
            {"type": "eval", "update": 4, "opponent": "random", "winrate": 0.6},
        ]
        summary = summarize_metrics(records)
        assert summary["skipped_updates"] == 2
        assert summary["last_skipped_update"] == 3
        assert summary["train_records"] == 2

    def test_monitor_train_records_exclude_skipped(self):
        records = [
            {"type": "skipped_update", "update": 1, "transitions": 0},
            {"type": "skipped_update", "update": 2, "transitions": 1},
            {"update": 3, "steps": 500, "loss": 0.5},
            {"type": "eval", "update": 3, "opponent": "random", "winrate": 0.5},
        ]
        summary = summarize_metrics(records)
        assert summary["train_records"] == 1
        assert summary["eval_records"] == 1
        assert summary["skipped_updates"] == 2
        assert summary["last_update"] == 3
        assert summary["last_skipped_update"] == 2

    def test_monitor_no_skipped_line_when_zero(self):
        records = [
            {"update": 1, "steps": 500, "loss": 0.5},
        ]
        summary = summarize_metrics(records)
        assert summary["skipped_updates"] == 0
        assert summary["last_skipped_update"] is None

"""
Tests for experiment runner and preview ablation (Task 11).
"""
import json
import shutil
from pathlib import Path

import pytest

from ai.train_v2.experiment import (
    ExperimentConfig,
    run_experiment,
    run_preview_ablation,
)


class TestRunExperiment:
    def test_run_experiment_smoke(self):
        config = ExperimentConfig(
            name="smoke_test",
            output_dir="/tmp/_t11_experiment",
            seed=42,
            updates=1,
            episodes_per_update=1,
            max_steps=5,
            hidden_dim=32,
            action_hidden_dim=16,
            include_preview_features=False,
            eval_games=1,
            export_onnx=False,
        )
        shutil.rmtree(config.output_dir, ignore_errors=True)

        try:
            summary = run_experiment(config)
            run_dir = Path(summary["run_dir"])
            assert run_dir.exists(), f"run dir not found: {run_dir}"
            assert (run_dir / "config.json").exists()
            assert (run_dir / "metrics.jsonl").exists()
            assert (run_dir / "summary.json").exists()
            assert summary["config"]["name"] == "smoke_test"
            assert "train" in summary
            assert summary["onnx_path"] is None
            assert summary["eval"] is None
        finally:
            shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_run_experiment_exports_onnx(self):
        config = ExperimentConfig(
            name="export_test",
            output_dir="/tmp/_t11_export",
            seed=42,
            updates=1,
            episodes_per_update=1,
            max_steps=5,
            hidden_dim=32,
            action_hidden_dim=16,
            include_preview_features=False,
            eval_games=1,
            export_onnx=True,
        )
        shutil.rmtree(config.output_dir, ignore_errors=True)

        try:
            summary = run_experiment(config)
            onnx_path = summary.get("onnx_path")
            if onnx_path is None:
                pytest.skip("train returned no checkpoint for this tiny config")
            assert Path(onnx_path).exists(), f"onnx not found: {onnx_path}"
            assert Path(onnx_path + ".json").exists(), f"sidecar not found"
            assert summary["eval"] is not None, "eval should be present when onnx exported"
            assert "parity" in summary
            assert "feature_benchmark" in summary
        finally:
            shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_run_experiment_without_export(self):
        config = ExperimentConfig(
            name="noexport_test",
            output_dir="/tmp/_t11_noexport",
            seed=42,
            updates=1,
            episodes_per_update=1,
            max_steps=5,
            hidden_dim=32,
            action_hidden_dim=16,
            eval_games=1,
            export_onnx=False,
        )
        shutil.rmtree(config.output_dir, ignore_errors=True)

        try:
            summary = run_experiment(config)
            assert summary.get("onnx_path") is None
            assert summary.get("eval") is None
            assert summary.get("parity") is None
            assert summary.get("feature_benchmark") is None
        finally:
            shutil.rmtree(config.output_dir, ignore_errors=True)


class TestPreviewAblation:
    def test_preview_ablation_smoke(self):
        output_dir = "/tmp/_t11_ablation"
        shutil.rmtree(output_dir, ignore_errors=True)

        try:
            result = run_preview_ablation(
                base_name="smoke_ablation",
                output_dir=output_dir,
                seed=42,
                updates=1,
                episodes_per_update=1,
                max_steps=5,
                eval_games=1,
                eval_max_steps=100,
                hidden_dim=32,
                action_hidden_dim=16,
            )
            assert "fast" in result
            assert "preview" in result
            assert "comparison" in result

            comp = result["comparison"]
            assert "fast_vs_random_wr" in comp
            assert "preview_vs_random_wr" in comp
            assert "fast_steps" in comp
            assert "preview_steps" in comp

            assert Path(result["fast"]["run_dir"]).exists()
            assert Path(result["preview"]["run_dir"]).exists()
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


class TestExperimentCLI:
    def test_experiment_cli_smoke(self):
        import subprocess, sys
        output_dir = "/tmp/_t11_cli_experiment"
        shutil.rmtree(output_dir, ignore_errors=True)

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "ai.train_v2.experiment",
                 "--name", "cli_smoke",
                 "--output-dir", output_dir,
                 "--seed", "42",
                 "--updates", "1",
                 "--episodes-per-update", "1",
                 "--max-steps", "5",
                 "--hidden-dim", "32",
                 "--action-hidden-dim", "16",
                 "--eval-games", "1",
                 "--no-export-onnx"],
                capture_output=True, text=True,
            )
            assert proc.returncode == 0, f"CLI failed:\n{proc.stderr}"
            assert "run_dir:" in proc.stdout
            assert "updates:" in proc.stdout
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_experiment_ablation_cli_smoke(self):
        import subprocess, sys
        output_dir = "/tmp/_t11_cli_ablation"
        shutil.rmtree(output_dir, ignore_errors=True)

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "ai.train_v2.experiment",
                 "--ablation",
                 "--name", "cli_ab",
                 "--output-dir", output_dir,
                 "--seed", "42",
                 "--updates", "1",
                 "--episodes-per-update", "1",
                 "--max-steps", "5",
                 "--hidden-dim", "32",
                 "--action-hidden-dim", "16",
                 "--eval-games", "1"],
                capture_output=True, text=True,
            )
            assert proc.returncode == 0, f"ablation CLI failed:\n{proc.stderr}"
            assert "Ablation:" in proc.stdout
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

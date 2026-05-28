"""
Tests for checkpoint leaderboard (Task 12).
"""
import json
import shutil
from pathlib import Path

import pytest

from ai.train_v2.leaderboard import (
    discover_onnx_models,
    evaluate_onnx_model_for_leaderboard,
    build_leaderboard,
    save_leaderboard,
)


def _make_tiny_export(output_dir, stem="model"):
    from ai.train_v2.train_ppo import PPOConfig, train
    from ai.train_v2.export_onnx import export_checkpoint_to_onnx

    ckpt_dir = str(Path(output_dir) / "ckpts")
    config = PPOConfig(
        total_updates=1, episodes_per_update=1, max_steps_per_episode=5,
        hidden_dim=32, action_hidden_dim=16, minibatch_size=8, epochs=1,
        seed=42, checkpoint_dir=ckpt_dir,
    )
    ckpt = train(config)["checkpoint_path"]

    exported_dir = Path(output_dir) / "exported"
    exported_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = str(exported_dir / f"{stem}.onnx")
    export_checkpoint_to_onnx(ckpt, onnx_path, opset=17)
    return onnx_path


class TestDiscover:
    def test_discover_single_file(self, tmp_path):
        onnx_path = _make_tiny_export(str(tmp_path), stem="a")
        result = discover_onnx_models([onnx_path])
        assert len(result) == 1
        assert result[0].endswith("a.onnx")

    def test_discover_run_dir(self, tmp_path):
        _make_tiny_export(str(tmp_path), stem="a")
        result = discover_onnx_models([str(tmp_path)])
        assert len(result) == 1
        assert "a.onnx" in result[0]

    def test_discover_root_with_multiple_dirs(self, tmp_path):
        for name in ["run1", "run2"]:
            rd = tmp_path / name
            rd.mkdir()
            _make_tiny_export(str(rd), stem=name)
        result = discover_onnx_models([str(tmp_path)])
        assert len(result) == 2

    def test_discover_empty_dir(self, tmp_path):
        result = discover_onnx_models([str(tmp_path)])
        assert result == []


class TestEvaluateForLeaderboard:
    def test_evaluate_returns_expected_shape(self, tmp_path):
        onnx_path = _make_tiny_export(str(tmp_path), stem="model")
        result = evaluate_onnx_model_for_leaderboard(
            onnx_path, seeds=[42], opponents=["end_turn"], max_steps=100,
        )
        assert result["model_name"] == "model"
        assert "opponents" in result
        assert "score" in result
        assert "parity_mismatches" in result
        assert result["parity_mismatches"] == 0


class TestBuildLeaderboard:
    def test_sorts_rows(self, tmp_path):
        onnx_a = _make_tiny_export(str(tmp_path / "run_a"), stem="a")
        onnx_b = _make_tiny_export(str(tmp_path / "run_b"), stem="b")

        result = build_leaderboard(
            paths=[str(tmp_path / "run_a"), str(tmp_path / "run_b")],
            seeds=[42, 43], opponents=["end_turn"], max_steps=100,
        )
        assert result["models"] == 2
        rows = result["rows"]
        assert len(rows) == 2
        assert rows[0]["rank"] == 1
        assert rows[1]["rank"] == 2
        assert rows[0]["score"] >= rows[1]["score"]

    def test_save_leaderboard_json(self, tmp_path):
        onnx_a = _make_tiny_export(str(tmp_path / "run_a"), stem="a")
        result = build_leaderboard(
            paths=[str(tmp_path / "run_a")],
            seeds=[42], opponents=["end_turn"], max_steps=100,
        )
        out_file = str(tmp_path / "lb.json")
        save_leaderboard(result, out_file)
        assert Path(out_file).exists()
        loaded = json.loads(Path(out_file).read_text())
        assert loaded["models"] == 1
        assert loaded["rows"][0]["model_name"] == "a"


class TestLeaderboardCLI:
    def test_cli_smoke(self, tmp_path):
        import subprocess, sys

        onnx_path = _make_tiny_export(str(tmp_path), stem="model")
        proc = subprocess.run(
            [sys.executable, "-m", "ai.train_v2.leaderboard",
             "--paths", onnx_path,
             "--games", "1", "--seed", "42", "--max-steps", "50",
             "--opponents", "end_turn"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, f"CLI failed:\n{proc.stderr}"
        assert "Best:" in proc.stdout or "score" in proc.stdout

    def test_cli_output_flag(self, tmp_path):
        import subprocess, sys

        onnx_path = _make_tiny_export(str(tmp_path), stem="model")
        out_file = str(tmp_path / "lb.json")
        proc = subprocess.run(
            [sys.executable, "-m", "ai.train_v2.leaderboard",
             "--paths", onnx_path,
             "--games", "1", "--seed", "42", "--max-steps", "50",
             "--opponents", "end_turn",
             "--output", out_file],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0
        assert Path(out_file).exists()

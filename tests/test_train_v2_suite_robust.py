import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from ai.train_v2.run_index import build_run_index
from ai.train_v2.report import format_report_markdown
from ai.train_v2.suite import run_suite, SuiteConfig, promote_best_candidate


def test_suite_continue_on_error_records_failure(tmp_path):
    suite_root = tmp_path / "robust"
    suite_root.mkdir()

    call_count = 0
    def _mock_run_experiment(cfg):
        nonlocal call_count
        call_count += 1
        run_dir = Path(cfg.output_dir) / f"{cfg.name}_ts{call_count}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if call_count == 1:
            raise ValueError("boom")
        return {
            "run_dir": str(run_dir),
            "config": {"name": cfg.name, "seed": cfg.seed},
            "train": {"updates": 1, "steps": 5},
            "checkpoint_path": str(run_dir / "ckpt.npz"),
            "onnx_path": None,
            "eval": None,
            "parity": None,
            "feature_benchmark": None,
        }

    config = SuiteConfig(
        name="robust",
        output_dir=str(suite_root),
        seeds=[1, 2],
        presets=["smoke"],
        updates=1,
        max_steps=5,
        eval_games=1,
        export_onnx=False,
        continue_on_error=True,
    )

    with patch("ai.train_v2.suite.run_experiment", side_effect=_mock_run_experiment):
        result = run_suite(config)

    assert result["health"]["total_runs"] == 2
    assert result["health"]["ok_runs"] == 1
    assert result["health"]["failed_runs"] == 1
    assert len(result["run_statuses"]) == 2
    assert result["run_statuses"][0]["status"] == "error"
    assert result["run_statuses"][1]["status"] == "ok"
    assert result["run_statuses"][0]["run_dir"] is None
    assert "expected_run_name" in result["run_statuses"][0]
    assert "traceback" in result["run_statuses"][0]


def test_suite_fail_fast_raises(tmp_path):
    suite_root = tmp_path / "failfast"
    suite_root.mkdir()

    def _mock_run_experiment(cfg):
        raise ValueError("boom")

    config = SuiteConfig(
        name="failfast",
        output_dir=str(suite_root),
        seeds=[1],
        presets=["smoke"],
        updates=1,
        max_steps=5,
        eval_games=1,
        export_onnx=False,
        continue_on_error=False,
    )

    with patch("ai.train_v2.suite.run_experiment", side_effect=_mock_run_experiment):
        with pytest.raises(ValueError, match="boom"):
            run_suite(config)


def test_suite_health_summary():
    config = SuiteConfig(
        name="health",
        output_dir="/tmp/_t23_health",
        seeds=[1],
        presets=["smoke"],
        updates=1,
        max_steps=5,
        eval_games=1,
        export_onnx=False,
    )
    shutil.rmtree(config.output_dir, ignore_errors=True)
    try:
        result = run_suite(config)
        h = result["health"]
        assert h["total_runs"] == 1
        assert h["ok_runs"] == 1
        assert h["failed_runs"] == 0
        assert "runs_with_checkpoint" in h
        assert "runs_with_onnx" in h
        assert "skipped_updates_total" in h
        assert result["run_statuses"][0]["status"] == "ok"
    finally:
        shutil.rmtree(config.output_dir, ignore_errors=True)


def test_promote_best_candidate(tmp_path):
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()

    run_dir = suite_dir / "run1"
    run_dir.mkdir()
    exported = run_dir / "exported"
    exported.mkdir()
    onnx = exported / "model.onnx"
    onnx.write_bytes(b"")
    (exported / "model.onnx.json").write_text('{"meta": true}')
    (run_dir / "report.md").write_text("# report")
    (suite_dir / "leaderboard.json").write_text('{"rows": []}')

    best_row = {
        "onnx_path": str(onnx),
        "model_name": "model",
        "score": 1.23,
    }

    result = promote_best_candidate(
        best_row=best_row,
        suite_dir=str(suite_dir),
    )

    assert result is not None
    cdir = Path(result["candidate_dir"])
    assert cdir.exists()
    assert Path(result["candidate_onnx"]).exists()
    assert Path(result["candidate_meta"]).exists()
    assert (cdir / "model.onnx").exists()
    assert (cdir / "model.onnx.json").exists()
    assert (cdir / "report.md").exists()
    assert (cdir / "leaderboard.json").exists()
    assert (cdir / "candidate.json").exists()
    meta = json.loads((cdir / "candidate.json").read_text())
    assert meta["model_name"] == "model"
    assert meta["score"] == 1.23
    assert meta["source_onnx"] == str(onnx)


def test_suite_with_export_promotes_candidate(tmp_path):
    suite_root = tmp_path / "promote_suite"
    suite_root.mkdir()

    def _mock_run_experiment(cfg):
        run_dir = Path(cfg.output_dir) / f"{cfg.name}_ts"
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir()
        exported = run_dir / "exported"
        exported.mkdir()
        onnx = exported / "model.onnx"
        onnx.write_bytes(b"")
        return {
            "run_dir": str(run_dir),
            "config": {"name": cfg.name, "seed": cfg.seed},
            "train": {"updates": 1, "steps": 5, "last_loss": 0.1, "last_entropy": 1.0},
            "checkpoint_path": str(ckpt_dir / "update_0001.npz"),
            "onnx_path": str(onnx),
        }

    def _mock_build_leaderboard(paths, seeds, opponents, max_steps):
        suite_dir = Path(paths[0])
        onnxx = list(suite_dir.rglob("*.onnx"))
        if not onnxx:
            return {"models": 0, "rows": [], "best": None}
        onnx_path = str(onnxx[0])
        return {
            "models": 1,
            "rows": [
                {"rank": 1, "model_name": "model", "score": 1.0, "onnx_path": onnx_path}
            ],
            "best": {"model_name": "model", "score": 1.0, "onnx_path": onnx_path},
        }

    config = SuiteConfig(
        name="promote",
        output_dir=str(suite_root),
        seeds=[1],
        presets=["smoke"],
        updates=1,
        max_steps=5,
        eval_games=1,
        export_onnx=True,
        build_leaderboard=True,
        leaderboard_games=1,
        promote_best=True,
    )

    with patch("ai.train_v2.suite.run_experiment", side_effect=_mock_run_experiment):
        with patch("ai.train_v2.suite.build_leaderboard", side_effect=_mock_build_leaderboard):
            result = run_suite(config)

    assert result["candidate"] is not None
    assert Path(result["candidate"]["candidate_dir"]).exists()
    assert Path(result["candidate"]["candidate_onnx"]).exists()
    assert Path(result["candidate"]["candidate_meta"]).exists()


def test_run_index_name_fallback_and_status(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    run = root / "partial_run"
    run.mkdir()
    (run / "metrics.jsonl").write_text("")
    index = build_run_index(str(root))
    assert index["runs"] == 1
    row = index["rows"][0]
    assert row["name"] == "partial_run"
    assert row["status"] == "partial"


def test_report_includes_leaderboard_section():
    report = {
        "run_dir": "/tmp/run",
        "metrics_summary": {},
        "summary": {"train": {}, "eval": {}},
        "latest_checkpoint": None,
        "latest_onnx": None,
        "leaderboard_row": {"rank": 1, "score": 0.95, "parity_mismatches": 0},
    }
    md = format_report_markdown(report)
    assert "## Leaderboard" in md
    assert "Rank: 1" in md
    assert "Score: 0.9500" in md
    assert "Parity mismatches: 0" in md


def test_suite_cli_no_promote_smoke(tmp_path):
    output_dir = tmp_path / "cli_suite"
    output_dir.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai.train_v2.suite",
            "--name", "cli_no_promote",
            "--output-dir", str(output_dir),
            "--presets", "smoke",
            "--seeds", "1",
            "--updates", "1",
            "--max-steps", "5",
            "--eval-games", "1",
            "--no-export-onnx",
            "--no-promote-best",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Suite dir" in proc.stdout

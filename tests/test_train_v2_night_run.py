import json
import subprocess
import sys
from pathlib import Path

import pytest

from ai.train_v2.night_run import (
    NightRunConfig,
    preflight_night_run,
    build_night_experiment_config,
    run_night,
)
from ai.train_v2.train_ppo import TRAIN_PRESETS


def test_preflight_night_run_ok(tmp_path):
    config = NightRunConfig(
        output_dir=str(tmp_path / "runs"),
        preset="smoke",
    )
    pf = preflight_night_run(config)
    assert pf["ok"] is True
    assert len(pf["errors"]) == 0
    assert pf["preset"] is not None
    assert pf["estimated_transitions"] > 0
    assert pf["estimated_action_feature_mb"] > 0


def test_preflight_unknown_preset(tmp_path):
    config = NightRunConfig(
        output_dir=str(tmp_path / "runs"),
        preset="nonexistent_preset_xyz",
    )
    pf = preflight_night_run(config)
    assert pf["ok"] is False
    assert "Unknown preset" in pf["errors"][0]
    assert pf["estimated_transitions"] == 0


def test_preflight_preview_warning(tmp_path):
    config = NightRunConfig(
        output_dir=str(tmp_path / "runs"),
        preset="smoke",
        include_preview_features=True,
    )
    pf = preflight_night_run(config)
    assert pf["ok"] is True
    assert any("preview" in w.lower() for w in pf["warnings"])


def test_preflight_export_warning(tmp_path):
    config = NightRunConfig(
        output_dir=str(tmp_path / "runs"),
        preset="smoke",
        export_onnx=True,
    )
    pf = preflight_night_run(config)
    assert pf["ok"] is True
    assert any("export" in w.lower() for w in pf["warnings"])


def test_build_night_experiment_config():
    config = NightRunConfig(
        name="night_v1",
        preset="m4_quick",
        seed=123,
        include_preview_features=True,
        export_onnx=True,
        eval_games=8,
        eval_max_steps=100,
    )
    exp = build_night_experiment_config(config)
    assert exp.name == "night_v1"
    assert exp.preset == "m4_quick"
    assert exp.seed == 123
    assert exp.include_preview_features is True
    assert exp.export_onnx is True
    assert exp.eval_games == 8
    assert exp.eval_max_steps == 100


def test_run_night_dry_run(tmp_path):
    config = NightRunConfig(
        output_dir=str(tmp_path / "runs"),
        preset="smoke",
        dry_run=True,
    )
    result = run_night(config)
    assert result["version"] == "train_v2_night_run_v2"
    assert result["dry_run"] is True
    assert "preflight" in result
    assert "planned_experiment_config" in result
    assert result["preflight"]["ok"] is True


def test_run_night_dry_run_unknown_preset_raises(tmp_path):
    config = NightRunConfig(
        output_dir=str(tmp_path / "runs"),
        preset="nonexistent",
        dry_run=True,
    )
    with pytest.raises(RuntimeError, match="preflight failed"):
        run_night(config)


def test_night_run_cli_dry_run(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai.train_v2.night_run",
            "--preset",
            "smoke",
            "--output-dir",
            str(tmp_path / "runs"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "preflight: OK" in proc.stdout or "OK (dry run)" in proc.stdout


def test_night_run_cli_json(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai.train_v2.night_run",
            "--preset",
            "smoke",
            "--output-dir",
            str(tmp_path / "runs"),
            "--dry-run",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    parsed = json.loads(proc.stdout)
    assert parsed["dry_run"] is True
    assert parsed["preflight"]["ok"] is True
    assert "planned_experiment_config" in parsed


def test_night_run_cli_help():
    proc = subprocess.run(
        [sys.executable, "-m", "ai.train_v2.night_run", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "usage:" in proc.stdout


def test_docs_exist():
    repo_root = Path(__file__).parent.parent
    legacy_doc = repo_root / "docs" / "TRAIN_V2_NIGHT_RUN.md"
    doc = repo_root / "docs" / "TrainV3.5" / "TRAINING_GUIDE.md"
    assert not legacy_doc.exists()
    assert doc.exists()
    text = doc.read_text()
    assert "Block B" in text
    assert "snapshot" in text.lower()
    assert "smoke" in text.lower()

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ai.train_v2 import TRAIN_V2_ARTIFACT_VERSIONS
from ai.train_v2.operator_smoke import (
    build_synthetic_operator_tree,
    check_operator_contracts,
    run_operator_smoke,
)


def test_build_synthetic_operator_tree(tmp_path):
    root = tmp_path / "smoke"
    paths = build_synthetic_operator_tree(str(root))
    assert Path(paths["root"]).exists()
    assert (Path(paths["run_dir"]) / "config.json").exists()
    assert (Path(paths["run_dir"]) / "summary.json").exists()
    assert (Path(paths["run_dir"]) / "exported" / "demo.onnx").exists()
    assert (Path(paths["candidate_dir"]) / "candidate.json").exists()
    assert (Path(paths["candidate_dir"]) / "demo.onnx").exists()
    assert (Path(paths["candidate_dir"]) / "profile_overlay.json").exists()
    assert (Path(paths["shadow_pack_dir"]) / "manifest.json").exists()
    assert (Path(paths["acceptance_dir"]) / "acceptance_gate.json").exists()
    assert (Path(paths["bundle_dir"]) / "release_manifest.json").exists()
    assert (Path(paths["bundle_dir"]) / "README.md").exists()


def test_check_operator_contracts(tmp_path):
    root = tmp_path / "smoke"
    build_synthetic_operator_tree(str(root))
    result = check_operator_contracts(str(root))
    assert result["ok"] is True
    checks = {c["name"]: c for c in result["checks"]}
    assert checks["run_index"]["status"] == "pass"
    assert checks["profile_registry"]["status"] == "pass"
    assert checks["gate_inputs"]["status"] == "pass"
    assert checks["gate_eval"]["status"] == "pass"
    assert checks["panel_data"]["status"] == "pass"
    assert checks["version_profile_overlay"]["status"] == "pass"
    assert checks["version_shadow_evidence"]["status"] == "pass"
    assert checks["version_acceptance_gate"]["status"] == "pass"
    assert checks["version_release_bundle"]["status"] == "pass"
    assert checks["no_production_touch"]["status"] == "pass"


def test_run_operator_smoke(tmp_path):
    root = tmp_path / "smoke"
    result = run_operator_smoke(str(root))
    assert result["contracts"]["ok"] is True
    assert "tree" in result
    assert "root" in result


def test_operator_smoke_cli_json(tmp_path):
    root = tmp_path / "smoke_cli"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai.train_v2.operator_smoke",
            "--root",
            str(root),
            "--json",
            "--keep",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    parsed = json.loads(proc.stdout)
    assert parsed["contracts"]["ok"] is True


def test_operator_smoke_cli_human(tmp_path):
    root = tmp_path / "smoke_human"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai.train_v2.operator_smoke",
            "--root",
            str(root),
            "--keep",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "PASS" in proc.stdout

def test_artifact_versions_present():
    assert "profile_registry" in TRAIN_V2_ARTIFACT_VERSIONS
    assert "profile_overlay" in TRAIN_V2_ARTIFACT_VERSIONS
    assert "shadow_evidence" in TRAIN_V2_ARTIFACT_VERSIONS
    assert "acceptance_gate" in TRAIN_V2_ARTIFACT_VERSIONS
    assert "release_bundle" in TRAIN_V2_ARTIFACT_VERSIONS
    assert "panel_snapshot" in TRAIN_V2_ARTIFACT_VERSIONS


@pytest.mark.parametrize("module", [
    "ai.train_v2.operator",
    "ai.train_v2.web_panel",
    "ai.train_v2.release_bundle",
    "ai.train_v2.acceptance_gate",
    "ai.train_v2.shadow_report",
    "ai.train_v2.profile_registry",
    "ai.train_v2.candidate_profile",
])
def test_cli_help_smoke(module):
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"{module} --help failed: {proc.stderr}"
    assert "usage:" in proc.stdout


def test_docs_exist():
    repo_root = Path(__file__).parent.parent
    legacy_operator_doc = repo_root / "docs" / "TRAIN_V2_OPERATOR.md"
    artifacts_doc = repo_root / "docs" / "TRAIN_V2_ARTIFACTS.md"
    assert not legacy_operator_doc.exists()
    assert artifacts_doc.exists()
    art_text = artifacts_doc.read_text()
    assert "panel" in art_text.lower()
    assert "snapshot" in art_text.lower()
    assert "doctor" in art_text.lower()
    assert "artifact" in art_text.lower()
    assert "version" in art_text.lower()
    assert "synthetic" in art_text.lower() or "inference" in art_text.lower()

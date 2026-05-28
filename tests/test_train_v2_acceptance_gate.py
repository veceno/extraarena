import json
import subprocess
import sys
from pathlib import Path

import pytest

from ai.train_v2.acceptance_gate import (
    GateThresholds,
    load_gate_inputs,
    evaluate_acceptance_gate,
    format_gate_markdown,
    write_gate_report,
)


def _make_good_inputs():
    return {
        "candidate": {"model_name": "m1", "score": 1.5},
        "profile": {"source": {"model_name": "m1"}},
        "overlay": {"version": "train_v2_profile_overlay_v1", "profiles": {}},
        "shadow_manifest": {"summary": {"steps": 100}},
        "shadow_summary": {
            "steps": 100,
            "match_rate": 0.9,
            "mismatches": 0,
            "overlay_invalid_actions": 0,
            "played_invalid_actions": 0,
            "legacy_invalid_actions": 0,
            "overlay_latency_ms_p95": 10.0,
        },
        "leaderboard": {
            "best": {
                "model_name": "m1",
                "score": 1.5,
                "wr_random": 0.5,
                "wr_end_turn": 0.9,
                "brain_invalid_total": 0,
                "parity_mismatches": 0,
            },
            "rows": [
                {
                    "model_name": "m1",
                    "score": 1.5,
                    "wr_random": 0.5,
                    "wr_end_turn": 0.9,
                    "brain_invalid_total": 0,
                    "parity_mismatches": 0,
                }
            ],
        },
        "paths": {},
        "missing": [],
    }


def test_load_gate_inputs_from_candidate_dir(tmp_path):
    cdir = tmp_path / "candidate"
    cdir.mkdir()
    (cdir / "candidate.json").write_text(json.dumps({"model_name": "m1", "score": 1.0}))
    (cdir / "candidate_profile.json").write_text(json.dumps({"source": {"model_name": "m1"}}))
    (cdir / "leaderboard.json").write_text(json.dumps({
        "best": {
            "model_name": "m1", "score": 1.0,
            "wr_random": 0.5, "wr_end_turn": 0.9,
            "brain_invalid_total": 0, "parity_mismatches": 0,
        },
        "rows": [
            {
                "model_name": "m1", "score": 1.0,
                "wr_random": 0.5, "wr_end_turn": 0.9,
                "brain_invalid_total": 0, "parity_mismatches": 0,
            }
        ],
    }))

    se = cdir / "shadow_evidence" / "pack1"
    se.mkdir(parents=True)
    (se / "manifest.json").write_text(json.dumps({"summary": {"steps": 100}}))
    (se / "shadow_summary.json").write_text(json.dumps({
        "steps": 100,
        "match_rate": 0.9,
        "mismatches": 0,
        "overlay_invalid_actions": 0,
        "played_invalid_actions": 0,
        "legacy_invalid_actions": 0,
        "overlay_latency_ms_p95": 10.0,
    }))

    inputs = load_gate_inputs(candidate_dir=str(cdir))
    assert inputs["candidate"] is not None
    assert inputs["profile"] is not None
    assert inputs["leaderboard"] is not None
    assert inputs["shadow_manifest"] is not None
    assert inputs["shadow_summary"] is not None
    assert inputs["missing"] == []


def test_load_gate_inputs_missing_safe(tmp_path):
    cdir = tmp_path / "candidate"
    cdir.mkdir()
    (cdir / "candidate.json").write_text(json.dumps({"model_name": "m1"}))
    inputs = load_gate_inputs(candidate_dir=str(cdir))
    assert inputs["candidate"] is not None
    assert "profile" in inputs["missing"]
    assert "shadow_manifest" in inputs["missing"]
    assert "shadow_summary" in inputs["missing"]
    assert "leaderboard" in inputs["missing"]


def test_evaluate_acceptance_gate_pass():
    inputs = _make_good_inputs()
    result = evaluate_acceptance_gate(inputs)
    assert result["status"] == "pass"
    assert result["score"] > 0
    assert result["version"] == "train_v2_acceptance_gate_v1"
    assert any(c["name"] == "artifact_count" and c["status"] == "pass" for c in result["checks"])


def test_evaluate_acceptance_gate_parity_from_leaderboard_not_shadow():
    """Parity mismatches must come from leaderboard row, not shadow divergence."""
    inputs = _make_good_inputs()
    # Leaderboard says parity is clean
    inputs["leaderboard"]["best"]["parity_mismatches"] = 0
    inputs["leaderboard"]["rows"][0]["parity_mismatches"] = 0
    # Shadow intentionally diverges from legacy
    inputs["shadow_summary"]["mismatches"] = 5
    result = evaluate_acceptance_gate(inputs)
    assert result["status"] == "pass"
    parity_check = next(c for c in result["checks"] if c["name"] == "parity_mismatches")
    assert parity_check["status"] == "pass"
    assert parity_check["value"] == 0


def test_evaluate_acceptance_gate_warn_missing_shadow():
    inputs = _make_good_inputs()
    inputs["shadow_summary"] = None
    inputs["shadow_manifest"] = None
    inputs["missing"] = ["shadow_summary", "shadow_manifest"]
    result = evaluate_acceptance_gate(inputs)
    assert result["status"] == "warn"
    assert any(c["name"] == "shadow_steps" and c["status"] == "warn" for c in result["checks"])


def test_evaluate_acceptance_gate_fail_latency():
    inputs = _make_good_inputs()
    inputs["shadow_summary"]["overlay_latency_ms_p95"] = 100.0
    result = evaluate_acceptance_gate(inputs)
    assert result["status"] == "fail"
    assert any(c["name"] == "overlay_latency_p95" and c["status"] == "fail" for c in result["checks"])


def test_evaluate_acceptance_gate_fail_invalid_actions():
    inputs = _make_good_inputs()
    inputs["shadow_summary"]["overlay_invalid_actions"] = 5
    result = evaluate_acceptance_gate(inputs)
    assert result["status"] == "fail"
    assert any(c["name"] == "overlay_invalid_actions" and c["status"] == "fail" for c in result["checks"])


def test_format_gate_markdown():
    result = evaluate_acceptance_gate(_make_good_inputs())
    md = format_gate_markdown(result)
    assert "## Verdict" in md
    assert "## Summary" in md
    assert "## Checks" in md
    assert "PASS" in md or "WARN" in md or "FAIL" in md


def test_write_gate_report(tmp_path):
    result = evaluate_acceptance_gate(_make_good_inputs())
    report = write_gate_report(result, str(tmp_path / "gate"))
    assert Path(report["result_path"]).exists()
    assert Path(report["markdown_path"]).exists()
    assert report["status"] == "pass"


class TestCLI:
    def _make_candidate_dir(self, tmp_path):
        cdir = tmp_path / "candidate"
        cdir.mkdir()
        (cdir / "candidate.json").write_text(json.dumps({"model_name": "m1", "score": 1.0}))
        (cdir / "candidate_profile.json").write_text(json.dumps({"source": {"model_name": "m1"}}))
        (cdir / "leaderboard.json").write_text(json.dumps({
            "best": {
                "model_name": "m1", "score": 1.0,
                "wr_random": 0.5, "wr_end_turn": 0.9,
                "brain_invalid_total": 0, "parity_mismatches": 0,
            },
            "rows": [
                {
                    "model_name": "m1", "score": 1.0,
                    "wr_random": 0.5, "wr_end_turn": 0.9,
                    "brain_invalid_total": 0, "parity_mismatches": 0,
                }
            ],
        }))
        se = cdir / "shadow_evidence" / "pack1"
        se.mkdir(parents=True)
        (se / "manifest.json").write_text(json.dumps({"summary": {"steps": 100}}))
        (se / "shadow_summary.json").write_text(json.dumps({
            "steps": 100,
            "match_rate": 0.9,
            "mismatches": 0,
            "overlay_invalid_actions": 0,
            "played_invalid_actions": 0,
            "legacy_invalid_actions": 0,
            "overlay_latency_ms_p95": 10.0,
        }))
        return cdir

    def test_acceptance_gate_cli_smoke(self, tmp_path):
        cdir = self._make_candidate_dir(tmp_path)
        out = tmp_path / "gate"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "ai.train_v2.acceptance_gate",
                "--candidate-dir",
                str(cdir),
                "--output-dir",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "Acceptance:" in proc.stdout
        assert (out / "acceptance_gate.json").exists()
        assert (out / "acceptance_gate.md").exists()

    def test_acceptance_gate_cli_json(self, tmp_path):
        cdir = self._make_candidate_dir(tmp_path)
        out = tmp_path / "gate"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "ai.train_v2.acceptance_gate",
                "--candidate-dir",
                str(cdir),
                "--output-dir",
                str(out),
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        parsed = json.loads(proc.stdout)
        assert "status" in parsed
        assert "result_path" in parsed
        assert "markdown_path" in parsed

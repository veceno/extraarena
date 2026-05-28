import json
import subprocess
import sys
from pathlib import Path

import pytest

from ai.train_v2.shadow_report import (
    summarize_shadow_result,
    extract_shadow_mismatches,
    format_shadow_markdown,
    write_shadow_evidence_pack,
    attach_shadow_pack_to_candidate,
)


FAKE_SHADOW_RESULT = {
    "episodes": 1,
    "steps": 2,
    "matches": 1,
    "mismatches": 1,
    "match_rate": 0.5,
    "legacy_invalid_actions": 0,
    "overlay_invalid_actions": 0,
    "played_invalid_actions": 0,
    "legacy_latency_ms_p50": 1.0,
    "legacy_latency_ms_p95": 2.0,
    "overlay_latency_ms_p50": 8.0,
    "overlay_latency_ms_p95": 12.0,
    "episodes_detail": [
        {
            "seed": 42,
            "summary": {"steps": 2},
            "decisions": [
                {
                    "step": 0,
                    "player_id": 1,
                    "match": True,
                    "legacy_action_id": 0,
                    "overlay_action_id": 0,
                    "played_action_id": 0,
                    "legacy": {"type": "end_turn"},
                    "overlay": {"type": "end_turn"},
                    "played": {"type": "end_turn"},
                },
                {
                    "step": 1,
                    "player_id": 2,
                    "match": False,
                    "legacy_action_id": 545,
                    "overlay_action_id": 0,
                    "played_action_id": 545,
                    "legacy": {"type": "attack"},
                    "overlay": {"type": "end_turn"},
                    "played": {"type": "attack"},
                },
            ],
        }
    ],
}


def test_summarize_shadow_result():
    summary = summarize_shadow_result(FAKE_SHADOW_RESULT)
    assert summary["episodes"] == 1
    assert summary["steps"] == 2
    assert summary["matches"] == 1
    assert summary["mismatches"] == 1
    assert summary["match_rate"] == 0.5
    assert summary["mismatch_rate"] == 0.5
    assert summary["legacy_invalid_actions"] == 0
    assert summary["overlay_invalid_actions"] == 0
    assert summary["played_invalid_actions"] == 0
    assert summary["legacy_latency_ms_p50"] == 1.0
    assert summary["legacy_latency_ms_p95"] == 2.0
    assert summary["overlay_latency_ms_p50"] == 8.0
    assert summary["overlay_latency_ms_p95"] == 12.0


def test_summarize_shadow_result_defaults():
    summary = summarize_shadow_result({})
    assert summary["steps"] == 0
    assert summary["mismatch_rate"] == 0.0
    assert summary["legacy_latency_ms_p50"] == 0.0


def test_extract_shadow_mismatches():
    rows = extract_shadow_mismatches(FAKE_SHADOW_RESULT)
    assert len(rows) == 1
    row = rows[0]
    assert row["episode_index"] == 0
    assert row["seed"] == 42
    assert row["step"] == 1
    assert row["player_id"] == 2
    assert row["legacy_action_id"] == 545
    assert row["overlay_action_id"] == 0
    assert row["played_action_id"] == 545
    assert row["legacy_type"] == "attack"
    assert row["overlay_type"] == "end_turn"
    assert row["played_type"] == "attack"


def test_extract_shadow_mismatches_limit_zero():
    assert extract_shadow_mismatches(FAKE_SHADOW_RESULT, limit=0) == []


def test_extract_shadow_mismatches_no_episodes():
    assert extract_shadow_mismatches({"steps": 0}) == []


def test_format_shadow_markdown():
    md = format_shadow_markdown(FAKE_SHADOW_RESULT)
    assert "## Summary" in md
    assert "## Latency" in md
    assert "attack" in md
    assert "end_turn" in md
    assert "| Episode | Step | Player | Legacy | Overlay | Played |" in md


def test_format_shadow_markdown_no_mismatches():
    result = dict(FAKE_SHADOW_RESULT)
    result["mismatches"] = 0
    result["matches"] = 2
    result["match_rate"] = 1.0
    result["episodes_detail"] = [
        {
            "seed": 42,
            "summary": {"steps": 2},
            "decisions": [
                {
                    "step": 0,
                    "player_id": 1,
                    "match": True,
                    "legacy": {"type": "end_turn"},
                    "overlay": {"type": "end_turn"},
                    "played": {"type": "end_turn"},
                },
                {
                    "step": 1,
                    "player_id": 2,
                    "match": True,
                    "legacy": {"type": "end_turn"},
                    "overlay": {"type": "end_turn"},
                    "played": {"type": "end_turn"},
                },
            ],
        }
    ]
    md = format_shadow_markdown(result)
    assert "No mismatches recorded." in md
    assert "## Top Mismatches" in md


def test_write_shadow_evidence_pack(tmp_path):
    out = tmp_path / "pack"
    manifest = write_shadow_evidence_pack(
        FAKE_SHADOW_RESULT,
        str(out),
        overlay_path="overlay.json",
        candidate_profile_path="profile.json",
        candidate_dir="candidate",
    )

    assert (out / "shadow_result.json").exists()
    assert (out / "shadow_summary.json").exists()
    assert (out / "shadow_summary.md").exists()
    assert (out / "shadow_mismatches.json").exists()
    assert (out / "manifest.json").exists()

    assert manifest["version"] == "train_v2_shadow_evidence_v1"
    assert manifest["overlay_path"] == "overlay.json"
    assert manifest["candidate_profile_path"] == "profile.json"
    assert manifest["candidate_dir"] == "candidate"
    assert "summary" in manifest
    assert "artifacts" in manifest
    assert "manifest_path" in manifest

    loaded_manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert loaded_manifest["version"] == manifest["version"]
    assert Path(loaded_manifest["artifacts"]["shadow_result"]).is_absolute()


def test_attach_shadow_pack_to_candidate(tmp_path):
    pack = tmp_path / "shadow_pack"
    pack.mkdir()
    (pack / "shadow_summary.md").write_text("# md")
    (pack / "manifest.json").write_text("{}")

    candidate = tmp_path / "candidate"
    candidate.mkdir()

    info = attach_shadow_pack_to_candidate(str(candidate), str(pack))
    attached = Path(info["attached_dir"])
    assert attached.exists()
    assert (attached / "shadow_summary.md").read_text() == "# md"
    assert (attached / "manifest.json").exists()
    assert info["candidate_dir"] == str(candidate.resolve())


def test_attach_shadow_pack_to_candidate_collision(tmp_path):
    pack = tmp_path / "shadow_pack"
    pack.mkdir()
    (pack / "a.txt").write_text("a")

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    existing = candidate / "shadow_evidence" / "shadow_pack"
    existing.mkdir(parents=True)
    (existing / "a.txt").write_text("old")

    info = attach_shadow_pack_to_candidate(str(candidate), str(pack))
    attached = Path(info["attached_dir"])
    assert attached.name.startswith("shadow_pack_")
    assert (attached / "a.txt").read_text() == "a"


class TestCLI:
    def test_shadow_report_cli_input_mode(self, tmp_path):
        inp = tmp_path / "result.json"
        inp.write_text(json.dumps(FAKE_SHADOW_RESULT, indent=2))
        out = tmp_path / "pack"

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "ai.train_v2.shadow_report",
                "--input",
                str(inp),
                "--output-dir",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "Shadow evidence pack:" in proc.stdout
        assert (out / "manifest.json").exists()

    def test_shadow_report_cli_json(self, tmp_path):
        inp = tmp_path / "result.json"
        inp.write_text(json.dumps(FAKE_SHADOW_RESULT, indent=2))
        out = tmp_path / "pack"

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "ai.train_v2.shadow_report",
                "--input",
                str(inp),
                "--output-dir",
                str(out),
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        parsed = json.loads(proc.stdout)
        assert parsed["version"] == "train_v2_shadow_evidence_v1"
        assert "summary" in parsed
        assert "artifacts" in parsed

    def test_shadow_report_cli_attach(self, tmp_path):
        inp = tmp_path / "result.json"
        inp.write_text(json.dumps(FAKE_SHADOW_RESULT, indent=2))
        out = tmp_path / "pack"
        candidate = tmp_path / "candidate"
        candidate.mkdir()

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "ai.train_v2.shadow_report",
                "--input",
                str(inp),
                "--output-dir",
                str(out),
                "--candidate-dir",
                str(candidate),
                "--attach",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "Attached:" in proc.stdout
        attached = candidate / "shadow_evidence" / "pack"
        assert attached.exists()
        assert (attached / "manifest.json").exists()

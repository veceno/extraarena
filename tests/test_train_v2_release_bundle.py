import json
import subprocess
import sys
from pathlib import Path

import pytest

from ai.train_v2.release_bundle import (
    ReleaseBundleConfig,
    discover_release_inputs,
    sha256_file,
    collect_file_manifest,
    build_release_bundle,
    format_release_readme,
)


def _make_candidate_dir(tmp_path):
    cdir = tmp_path / "candidate"
    cdir.mkdir()
    candidate = {
        "model_name": "update_0003",
        "score": 1.23,
        "candidate_onnx": str(cdir / "model.onnx"),
    }
    (cdir / "candidate.json").write_text(json.dumps(candidate))
    (cdir / "model.onnx").write_bytes(b"fake_onnx_data")
    (cdir / "model.onnx.json").write_text(json.dumps({"opset": 17}))
    (cdir / "candidate_profile.json").write_text(json.dumps({"format": "train_v2_classic_v1"}))
    (cdir / "profile_overlay.json").write_text(
        json.dumps({"version": "train_v2_profile_overlay_v1", "profiles": {}})
    )

    se = cdir / "shadow_evidence" / "pack1"
    se.mkdir(parents=True)
    (se / "manifest.json").write_text(json.dumps({"version": "train_v2_shadow_evidence_v1"}))
    (se / "shadow_summary.json").write_text(json.dumps({
        "steps": 100,
        "match_rate": 0.9,
        "overlay_latency_ms_p95": 10.0,
        "overlay_invalid_actions": 0,
    }))
    (se / "shadow_result.json").write_text("{}")
    (se / "shadow_summary.md").write_text("# md")
    (se / "shadow_mismatches.json").write_text("[]")

    ag = cdir / "acceptance_gate"
    ag.mkdir()
    (ag / "acceptance_gate.json").write_text(json.dumps({
        "status": "pass",
        "score": 8.5,
        "version": "train_v2_acceptance_gate_v1",
    }))
    (ag / "acceptance_gate.md").write_text("# gate")

    return cdir


def test_discover_release_inputs(tmp_path):
    cdir = _make_candidate_dir(tmp_path)
    config = ReleaseBundleConfig(
        candidate_dir=str(cdir),
        output_dir=str(tmp_path / "out"),
    )
    inputs = discover_release_inputs(config)
    assert inputs["candidate_json"] is not None
    assert inputs["onnx"] is not None
    assert inputs["onnx_sidecar"] is not None
    assert inputs["profile"] is not None
    assert inputs["overlay"] is not None
    assert inputs["shadow_pack"] is not None
    assert inputs["acceptance_dir"] is not None
    assert "onnx_sidecar" not in inputs["missing"]


def test_discover_release_inputs_missing_optional(tmp_path):
    cdir = tmp_path / "candidate"
    cdir.mkdir()
    candidate = {
        "model_name": "m1",
        "candidate_onnx": str(cdir / "model.onnx"),
    }
    (cdir / "candidate.json").write_text(json.dumps(candidate))
    (cdir / "model.onnx").write_bytes(b"data")

    config = ReleaseBundleConfig(
        candidate_dir=str(cdir),
        output_dir=str(tmp_path / "out"),
    )
    inputs = discover_release_inputs(config)
    assert inputs["candidate_json"] is not None
    assert inputs["onnx"] is not None
    assert inputs["profile"] is None
    assert inputs["overlay"] is None
    assert inputs["shadow_pack"] is None
    assert inputs["acceptance_dir"] is None
    assert "profile" in inputs["missing"]
    assert "overlay" in inputs["missing"]
    assert "shadow_pack" in inputs["missing"]
    assert "acceptance_dir" in inputs["missing"]


def test_discover_release_inputs_missing_candidate_raises_on_build(tmp_path):
    cdir = tmp_path / "candidate"
    cdir.mkdir()
    config = ReleaseBundleConfig(
        candidate_dir=str(cdir),
        output_dir=str(tmp_path / "out"),
    )
    with pytest.raises(FileNotFoundError):
        build_release_bundle(config)


def test_sha256_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello")
    h1 = sha256_file(str(f))
    h2 = sha256_file(str(f))
    assert h1 == h2
    assert len(h1) == 64


def test_collect_file_manifest(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b").mkdir(parents=True)
    (tmp_path / "b" / "c.txt").write_text("c")
    manifest = collect_file_manifest(str(tmp_path))
    paths = {m["path"] for m in manifest}
    assert "a.txt" in paths
    assert "b/c.txt" in paths
    assert all("sha256" in m and "size" in m for m in manifest)
    assert not any(m["path"] == "release_manifest.json" for m in manifest)


def test_build_release_bundle_smoke(tmp_path):
    cdir = _make_candidate_dir(tmp_path)
    out = tmp_path / "out"
    config = ReleaseBundleConfig(
        candidate_dir=str(cdir),
        output_dir=str(out),
        name="test_bundle",
    )
    result = build_release_bundle(config)
    bundle_dir = Path(result["bundle_dir"])
    assert bundle_dir.exists()
    assert (bundle_dir / "README.md").exists()
    assert (bundle_dir / "release_manifest.json").exists()
    assert (bundle_dir / "model" / "model.onnx").exists()
    assert (bundle_dir / "candidate" / "candidate.json").exists()
    assert (bundle_dir / "profile" / "candidate_profile.json").exists()
    assert (bundle_dir / "profile" / "profile_overlay.json").exists()
    assert (bundle_dir / "shadow_evidence" / "manifest.json").exists()
    assert (bundle_dir / "acceptance_gate" / "acceptance_gate.json").exists()


def test_build_release_bundle_without_shadow(tmp_path):
    cdir = _make_candidate_dir(tmp_path)
    config = ReleaseBundleConfig(
        candidate_dir=str(cdir),
        output_dir=str(tmp_path / "out"),
        name="no_shadow",
        include_shadow=False,
    )
    result = build_release_bundle(config)
    bundle_dir = Path(result["bundle_dir"])
    assert not (bundle_dir / "shadow_evidence").exists()


def test_build_release_bundle_without_acceptance(tmp_path):
    cdir = _make_candidate_dir(tmp_path)
    config = ReleaseBundleConfig(
        candidate_dir=str(cdir),
        output_dir=str(tmp_path / "out"),
        name="no_acc",
        include_acceptance=False,
    )
    result = build_release_bundle(config)
    bundle_dir = Path(result["bundle_dir"])
    assert not (bundle_dir / "acceptance_gate").exists()


def test_build_release_bundle_with_archive(tmp_path):
    cdir = _make_candidate_dir(tmp_path)
    config = ReleaseBundleConfig(
        candidate_dir=str(cdir),
        output_dir=str(tmp_path / "out"),
        name="archived",
        create_archive=True,
    )
    result = build_release_bundle(config)
    assert result["archive_path"] is not None
    assert Path(result["archive_path"]).exists()
    # verify tar.gz contains top-level bundle dir
    import tarfile
    with tarfile.open(result["archive_path"], "r:gz") as tf:
        names = tf.getnames()
        top = {n.split("/", 1)[0] for n in names if n}
        assert "archived" in top or any("archived" in t for t in top)


def test_build_release_bundle_archive_uses_unique_bundle_name(tmp_path):
    cdir = _make_candidate_dir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "fixed_name").mkdir()
    config = ReleaseBundleConfig(
        candidate_dir=str(cdir),
        output_dir=str(out),
        name="fixed_name",
        create_archive=True,
    )
    result = build_release_bundle(config)
    assert Path(result["bundle_dir"]).name == "fixed_name_1"
    assert Path(result["archive_path"]).name == "fixed_name_1.tar.gz"
    assert Path(result["archive_path"]).exists()
    import tarfile
    with tarfile.open(result["archive_path"], "r:gz") as tf:
        names = tf.getnames()
        top = {n.split("/", 1)[0] for n in names if n}
        assert "fixed_name_1" in top


def test_build_release_bundle_unique_dir(tmp_path):
    cdir = _make_candidate_dir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "dup").mkdir()
    config = ReleaseBundleConfig(
        candidate_dir=str(cdir),
        output_dir=str(out),
        name="dup",
    )
    result = build_release_bundle(config)
    assert Path(result["bundle_dir"]).name == "dup_1"


def test_format_release_readme():
    inputs = {
        "candidate_json": "/tmp/candidate.json",
        "onnx": "/tmp/model.onnx",
        "profile": "/tmp/profile.json",
        "overlay": "/tmp/overlay.json",
        "shadow_pack": "/tmp/shadow",
        "acceptance_dir": "/tmp/gate",
        "missing": ["onnx_sidecar"],
    }
    acceptance = {"status": "pass", "score": 8.5}
    readme = format_release_readme({"model_name": "m1"}, inputs, acceptance=acceptance)
    assert "## Verdict" in readme
    assert "## Contents" in readme
    assert "## Safety" in readme
    assert "m1" in readme
    assert "PASS" in readme
    assert "8.5" in readme


def test_release_bundle_manifest_excludes_itself(tmp_path):
    cdir = _make_candidate_dir(tmp_path)
    config = ReleaseBundleConfig(
        candidate_dir=str(cdir),
        output_dir=str(tmp_path / "out"),
        name="self_test",
    )
    result = build_release_bundle(config)
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    files = [f["path"] for f in manifest["files"]]
    assert "release_manifest.json" not in files


class TestCLI:
    def _make_candidate_dir(self, tmp_path):
        return _make_candidate_dir(tmp_path)

    def test_release_bundle_cli_smoke(self, tmp_path):
        cdir = self._make_candidate_dir(tmp_path)
        out = tmp_path / "out"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "ai.train_v2.release_bundle",
                "--candidate-dir",
                str(cdir),
                "--output-dir",
                str(out),
                "--name",
                "cli_bundle",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "Release bundle:" in proc.stdout
        assert (out / "cli_bundle" / "release_manifest.json").exists()

    def test_release_bundle_cli_json(self, tmp_path):
        cdir = self._make_candidate_dir(tmp_path)
        out = tmp_path / "out"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "ai.train_v2.release_bundle",
                "--candidate-dir",
                str(cdir),
                "--output-dir",
                str(out),
                "--name",
                "json_bundle",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        parsed = json.loads(proc.stdout)
        assert "bundle_dir" in parsed
        assert "manifest_path" in parsed
        assert "status" in parsed

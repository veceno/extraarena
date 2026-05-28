import json
import subprocess
import sys
import threading
import time
from http.server import HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

import pytest

from ai.train_v2.web_panel import (
    WebPanelConfig,
    collect_panel_data,
    read_safe_text_file,
    _is_path_allowed,
    make_handler,
    run_web_panel,
)
from ai.train_v2.operator import (
    write_panel_snapshot,
    run_doctor,
)


def test_collect_panel_data_lifecycle(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    run = runs_dir / "demo_20260519_010203"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"name": "demo", "seed": 42}))
    (run / "summary.json").write_text(json.dumps({
        "train": {"updates": 10, "steps": 12800, "last_loss": 0.1234, "last_entropy": 1.1},
        "eval": {
            "random": {"winrate": 0.75},
            "end_turn": {"winrate": 0.6},
            "greedy_face": {"winrate": 0.45},
        },
    }))
    (run / "metrics.jsonl").write_text("")

    releases_dir = tmp_path / "releases"
    releases_dir.mkdir()
    bundle = releases_dir / "update_0003_20260520_010203"
    bundle.mkdir()
    (bundle / "release_manifest.json").write_text(json.dumps({
        "version": "train_v2_release_bundle_v1",
        "model_name": "update_0003",
        "created_at": "2026-05-20T01:02:03",
        "files": [{"path": "README.md", "size": 100, "sha256": "abc"}],
        "missing": [],
    }))

    ag = releases_dir / "acceptance_gate"
    ag.mkdir()
    (ag / "acceptance_gate.json").write_text(json.dumps({
        "status": "pass",
        "score": 8.5,
        "version": "train_v2_acceptance_gate_v1",
    }))

    data = collect_panel_data(runs_dir=str(runs_dir), releases_dir=str(releases_dir))
    lc = data["lifecycle"]
    assert lc["runs"] == 1
    assert lc["profiles_ok"] == 0  # no candidate_profile.json in run
    assert lc["profiles_errors"] == 0
    assert lc["release_bundles"] == 1
    assert lc["acceptance_pass"] == 1
    assert lc["acceptance_warn"] == 0
    assert lc["acceptance_fail"] == 0
    assert lc["shadow_packs"] == 0
    assert lc["latest_release"] is not None
    assert lc["latest_acceptance"] is not None
    assert lc["best_profile"] is None


def test_collect_panel_data_artifacts(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    run = runs_dir / "demo"
    run.mkdir()
    (run / "config.json").write_text("{}")
    (run / "summary.json").write_text("{}")
    (run / "metrics.jsonl").write_text("")
    (run / "candidate_profile.json").write_text(json.dumps({
        "difficulty": "train_v2_candidate",
        "profile": {
            "model_path": "model.onnx",
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
        },
        "source": {"model_name": "m1", "score": 1.5},
    }))

    releases_dir = tmp_path / "releases"
    releases_dir.mkdir()
    bundle = releases_dir / "b1"
    bundle.mkdir()
    (bundle / "release_manifest.json").write_text(json.dumps({
        "version": "train_v2_release_bundle_v1",
        "model_name": "m1",
        "created_at": "2026-05-20T01:02:03",
        "files": [],
        "missing": [],
    }))

    ag = releases_dir / "ag1"
    ag.mkdir()
    (ag / "acceptance_gate.json").write_text(json.dumps({"status": "warn", "score": 6.0}))

    data = collect_panel_data(runs_dir=str(runs_dir), releases_dir=str(releases_dir))
    artifacts = data["artifacts"]
    kinds = {a["kind"] for a in artifacts}
    assert "release_bundle" in kinds
    assert "acceptance_gate" in kinds
    assert "profile" in kinds
    # No shadow packs, no leaderboard
    assert "shadow_pack" not in kinds
    assert "leaderboard" not in kinds


def test_api_artifact_endpoint(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    f = runs_dir / "report.md"
    f.write_text("# Hello")

    config = WebPanelConfig(runs_dir=str(runs_dir), releases_dir=None, host="127.0.0.1")
    port, server = _start_test_server(config)
    try:
        url = f"http://127.0.0.1:{port}/api/artifact?path={quote(str(f))}"
        res = urlopen(url, timeout=5)
        data = json.loads(res.read().decode("utf-8"))
        assert data["path"] == str(f)
        assert data["content_type"] == "text/plain; charset=utf-8"
        assert data["content"] == "# Hello"
    finally:
        if server:
            server.shutdown()


def test_api_snapshot_endpoint(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    config = WebPanelConfig(runs_dir=str(runs_dir), releases_dir=None, host="127.0.0.1")
    port, server = _start_test_server(config)
    try:
        url = f"http://127.0.0.1:{port}/api/snapshot"
        res = urlopen(url, timeout=5)
        data = json.loads(res.read().decode("utf-8"))
        assert "generated_at" in data
        assert "summary" in data
        assert data["summary"]["version"] == "train_v2_web_panel_v1"
    finally:
        if server:
            server.shutdown()


def test_operator_snapshot_writes_file(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    out = tmp_path / "snapshot.json"
    result = write_panel_snapshot(
        runs_dir=str(runs_dir),
        releases_dir=None,
        output_path=str(out),
    )
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["version"] == "train_v2_panel_snapshot_v1"
    assert "generated_at" in loaded
    assert "data" in loaded


def test_operator_doctor(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    result = run_doctor(runs_dir=str(runs_dir), releases_dir=None)
    assert result["runs_dir_exists"] is True
    assert result["releases_dir_exists"] is None
    assert result["run_count"] == 0
    assert result["profile_count"] == 0
    assert result["release_count"] == 0
    assert "No runs found" in result["issues"]
    assert "No profiles found" in result["issues"]


def test_operator_cli_snapshot(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    out = tmp_path / "snap.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai.train_v2.operator",
            "snapshot",
            "--runs-dir",
            str(runs_dir),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()


def test_operator_cli_doctor_json(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai.train_v2.operator",
            "doctor",
            "--runs-dir",
            str(runs_dir),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    parsed = json.loads(proc.stdout)
    assert "runs_dir_exists" in parsed
    assert "issues" in parsed


def test_frontend_contains_new_tabs():
    html = (Path(__file__).parent.parent / "ai" / "train_v2" / "web_panel_static" / "index.html").read_text()
    assert "Artifacts" in html
    assert "Doctor" in html
    js = (Path(__file__).parent.parent / "ai" / "train_v2" / "web_panel_static" / "app.js").read_text()
    assert "renderArtifacts" in js
    assert "renderDoctor" in js
    assert "localStorage" in js


def test_static_assets_compile_smoke():
    css = (Path(__file__).parent.parent / "ai" / "train_v2" / "web_panel_static" / "style.css").read_text()
    assert len(css) > 0
    assert ".btn" in css
    assert ".detail-header" in css


# Helpers (reused from test_train_v2_web_panel.py)
def _find_free_port():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_test_server(config):
    port = _find_free_port()
    config.port = port
    server = None

    def serve():
        nonlocal server
        handler = make_handler(config)
        server = HTTPServer((config.host, config.port), handler)
        server.serve_forever()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    time.sleep(0.3)
    return config.port, server

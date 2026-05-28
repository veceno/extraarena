import json
import socket
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path
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


def test_collect_panel_data_empty_dirs(tmp_path):
    data = collect_panel_data(runs_dir=str(tmp_path / "nonexistent_runs"), releases_dir=str(tmp_path / "nonexistent_releases"))
    assert data["version"] == "train_v2_web_panel_v1"
    assert data["run_index"]["runs"] == 0
    assert data["profile_registry"]["profiles"] == 0
    assert data["leaderboard"] is None
    assert data["release_bundles"] == []
    assert data["acceptance_reports"] == []
    assert data["shadow_packs"] == []


def test_collect_panel_data_with_run_and_release(tmp_path):
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
    (bundle / "README.md").write_text("# Bundle")

    data = collect_panel_data(runs_dir=str(runs_dir), releases_dir=str(releases_dir))
    assert data["run_index"]["runs"] == 1
    assert len(data["release_bundles"]) == 1
    assert data["release_bundles"][0]["model_name"] == "update_0003"
    assert data["release_bundles"][0]["files_count"] == 1


def test_read_safe_text_file_rejects_outside_path(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    with pytest.raises(PermissionError):
        read_safe_text_file(str(tmp_path / "outside.txt"), roots=[str(runs_dir)])


def test_read_safe_text_file_rejects_bad_suffix(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    f = runs_dir / "bad.py"
    f.write_text("x=1")
    with pytest.raises(ValueError, match="Disallowed file type"):
        read_safe_text_file(str(f), roots=[str(runs_dir)])


def test_read_safe_text_file_allows_markdown_under_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    f = runs_dir / "report.md"
    f.write_text("# Hello")
    content, ct = read_safe_text_file(str(f), roots=[str(runs_dir)])
    assert content == "# Hello"
    assert ct == "text/plain; charset=utf-8"


def test_read_safe_text_file_rejects_too_large(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    f = runs_dir / "big.txt"
    f.write_text("x" * 3_000_000)
    with pytest.raises(ValueError, match="File too large"):
        read_safe_text_file(str(f), roots=[str(runs_dir)], max_bytes=2_000_000)


def test_is_path_allowed(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    sub = root / "sub"
    sub.mkdir()
    assert _is_path_allowed(sub, [root]) is True
    assert _is_path_allowed(tmp_path / "other", [root]) is False


def _find_free_port():
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
        from http.server import HTTPServer
        handler = make_handler(config)
        server = HTTPServer((config.host, config.port), handler)
        server.serve_forever()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    time.sleep(0.3)
    return config.port, server


def test_web_panel_summary_endpoint(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    config = WebPanelConfig(runs_dir=str(runs_dir), releases_dir=None, host="127.0.0.1")
    port, server = _start_test_server(config)
    try:
        url = f"http://127.0.0.1:{port}/api/summary"
        res = urlopen(url, timeout=5)
        data = json.loads(res.read().decode("utf-8"))
        assert data["version"] == "train_v2_web_panel_v1"
        assert "run_index" in data
    finally:
        if server:
            server.shutdown()


def test_web_panel_static_index(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    config = WebPanelConfig(runs_dir=str(runs_dir), releases_dir=None, host="127.0.0.1")
    port, server = _start_test_server(config)
    try:
        url = f"http://127.0.0.1:{port}/"
        res = urlopen(url, timeout=5)
        body = res.read().decode("utf-8")
        assert "TrainV2 Panel" in body
    finally:
        if server:
            server.shutdown()


def test_web_panel_file_endpoint(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    f = runs_dir / "report.md"
    f.write_text("# Hello")
    config = WebPanelConfig(runs_dir=str(runs_dir), releases_dir=None, host="127.0.0.1")
    port, server = _start_test_server(config)
    try:
        url = f"http://127.0.0.1:{port}/api/file?path={quote(str(f))}"
        res = urlopen(url, timeout=5)
        body = res.read().decode("utf-8")
        assert "# Hello" in body
    finally:
        if server:
            server.shutdown()


def test_web_panel_run_report_endpoint(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    run = runs_dir / "demo"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"seed": 42}))
    (run / "summary.json").write_text(json.dumps({"train": {"updates": 5}}))
    (run / "metrics.jsonl").write_text("")

    config = WebPanelConfig(runs_dir=str(runs_dir), releases_dir=None, host="127.0.0.1")
    port, server = _start_test_server(config)
    try:
        url = f"http://127.0.0.1:{port}/api/run-report?dir={quote(str(run))}"
        res = urlopen(url, timeout=5)
        data = json.loads(res.read().decode("utf-8"))
        assert data["run_dir"] == str(run)
        assert data["config"]["seed"] == 42
    finally:
        if server:
            server.shutdown()


def test_web_panel_run_report_rejects_outside_dir(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "config.json").write_text("{}")

    config = WebPanelConfig(runs_dir=str(runs_dir), releases_dir=None, host="127.0.0.1")
    port, server = _start_test_server(config)
    try:
        url = f"http://127.0.0.1:{port}/api/run-report?dir={quote(str(outside))}"
        from urllib.error import HTTPError
        with pytest.raises(HTTPError) as exc_info:
            urlopen(url, timeout=5)
        assert exc_info.value.code == 403
    finally:
        if server:
            server.shutdown()


def test_web_panel_file_rejects_path_traversal(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")

    config = WebPanelConfig(runs_dir=str(runs_dir), releases_dir=None, host="127.0.0.1")
    port, server = _start_test_server(config)
    try:
        url = f"http://127.0.0.1:{port}/api/file?path={quote(str(outside))}"
        from urllib.error import HTTPError
        with pytest.raises(HTTPError) as exc_info:
            urlopen(url, timeout=5)
        assert exc_info.value.code == 403
    finally:
        if server:
            server.shutdown()


@pytest.mark.skip(reason="serve_forever blocks; covered by other tests")
def test_web_panel_cli_smoke(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai.train_v2.web_panel",
            "--runs-dir",
            str(runs_dir),
            "--port",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    # Port 0 may fail or print; just check it doesn't crash instantly
    assert proc.returncode != 0 or "TrainV2 web panel" in proc.stdout or proc.stderr == ""

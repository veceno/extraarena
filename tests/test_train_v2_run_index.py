import json
import subprocess
import sys
from pathlib import Path

from ai.train_v2.run_index import discover_runs, build_run_index, save_run_index


def test_discover_runs_empty_nonexistent():
    result = discover_runs("/tmp/nonexistent_run_index_path_12345")
    assert result == []


def test_discover_runs_root_is_run(tmp_path):
    run = tmp_path / "demo_20260519_010203"
    run.mkdir()
    (run / "config.json").write_text("{}")
    result = discover_runs(str(run))
    assert result == [str(run.resolve())]


def test_discover_runs_children(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()

    run1 = root / "demo_20260519_010203"
    run1.mkdir()
    (run1 / "summary.json").write_text("{}")

    run2 = root / "demo_20260519_020304"
    run2.mkdir()
    (run2 / "metrics.jsonl").write_text("")

    not_run = root / "checkpoints"
    not_run.mkdir()
    (not_run / "update_0001.npz").write_bytes(b"")

    result = discover_runs(str(root))
    assert len(result) == 2
    assert str(run1.resolve()) in result
    assert str(run2.resolve()) in result


def test_build_run_index_with_files(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()

    run = root / "demo_20260519_010203"
    run.mkdir()

    config = {"name": "demo", "seed": 42}
    summary = {
        "train": {"updates": 10, "steps": 12800, "last_loss": 0.1234, "last_entropy": 1.1},
        "eval": {
            "random": {"winrate": 0.75},
            "end_turn": {"winrate": 0.6},
            "greedy_face": {"winrate": 0.45},
        },
    }
    (run / "config.json").write_text(json.dumps(config))
    (run / "summary.json").write_text(json.dumps(summary))

    metrics = [
        {"type": "train", "update": 1, "steps": 40, "loss": 0.6, "entropy": 1.3},
        {"type": "skipped_update", "update": 2},
    ]
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics))

    ckpt_dir = run / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "update_0001.npz").write_bytes(b"")

    exported_dir = run / "exported"
    exported_dir.mkdir()
    (exported_dir / "model.onnx").write_bytes(b"")

    index = build_run_index(str(root))
    assert index["root"] == str(root.resolve())
    assert index["runs"] == 1
    row = index["rows"][0]
    assert row["name"] == "demo"
    assert row["seed"] == 42
    assert row["updates"] == 10
    assert row["steps"] == 12800
    assert row["last_loss"] == 0.1234
    assert row["last_entropy"] == 1.1
    assert row["skipped_updates"] == 1
    assert row["has_checkpoint"] is True
    assert row["has_onnx"] is True
    assert row["wr_random"] == 0.75
    assert row["wr_end_turn"] == 0.6
    assert row["wr_greedy_face"] == 0.45


def test_build_run_index_partial_run(tmp_path):
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
    assert row["seed"] is None
    assert row["updates"] is None
    assert row["steps"] is None
    assert row["last_loss"] is None
    assert row["last_entropy"] is None
    assert row["skipped_updates"] == 0
    assert row["has_checkpoint"] is False
    assert row["has_onnx"] is False
    assert row["wr_random"] is None
    assert row["wr_end_turn"] is None
    assert row["wr_greedy_face"] is None


def test_save_run_index(tmp_path):
    index = {
        "root": str(tmp_path),
        "runs": 0,
        "rows": [],
    }
    out = tmp_path / "index.json"
    save_run_index(index, str(out))
    assert out.is_file()
    loaded = json.loads(out.read_text())
    assert loaded["runs"] == 0


def test_run_index_cli_smoke(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai.train_v2.run_index",
            "--root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "runs: 0" in proc.stdout


def test_run_index_cli_with_output(tmp_path):
    out = tmp_path / "index.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai.train_v2.run_index",
            "--root",
            str(tmp_path),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
    loaded = json.loads(out.read_text())
    assert loaded["runs"] == 0
    assert "root" in loaded

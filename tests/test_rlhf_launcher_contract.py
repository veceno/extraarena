from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "rlhf_env" / "start_rlhf_env.sh"


def test_rlhf_launcher_exposes_dataset_plane_without_polluting_stdio():
    subprocess.run(
        ["bash", "-n", str(LAUNCHER)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    help_result = subprocess.run(
        [str(LAUNCHER), "help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "--datasets-dir DIR" in help_result.stdout
    assert "--enable-production-datasets" in help_result.stdout
    assert "--python FILE" in help_result.stdout
    assert 'DATASETS_DIR="${RLHF_DATASETS_DIR:-datasets}"' in source
    assert 'PYTHON_BIN="${RLHF_PYTHON:-}"' in source
    assert 'PY="$PYTHON_BIN"' in source
    assert 'case "${RLHF_ENABLE_PRODUCTION_DATASETS:-0}" in' in source
    assert "ENABLE_PRODUCTION_DATASETS=1" in source
    assert '--datasets-dir "$DATASETS_DIR"' in source
    assert "--enable-production-datasets" in source
    assert "asyncpg, dotenv" in source
    # Launcher diagnostics must go to stderr: stdout belongs exclusively to
    # the JSON-RPC server once the `mcp` command execs it.
    assert 'log() {' in source
    assert '"$*" >&2' in source

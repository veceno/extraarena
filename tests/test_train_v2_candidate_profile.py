import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from ai.train_v2.candidate_profile import (
    load_candidate,
    build_train_v2_profile,
    write_candidate_profile,
    format_profile_snippet,
    validate_candidate_profile,
)


@pytest.fixture
def fake_candidate(tmp_path):
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    candidate = {
        "candidate_onnx": str(candidate_dir / "model.onnx"),
        "source_onnx": str(candidate_dir / "source.onnx"),
        "model_name": "model",
        "score": 1.0,
        "source_run_dir": str(tmp_path / "run"),
    }
    (candidate_dir / "candidate.json").write_text(json.dumps(candidate))
    return str(candidate_dir), candidate


def test_load_candidate_from_json(fake_candidate):
    cdir, expected = fake_candidate
    loaded = load_candidate(str(Path(cdir) / "candidate.json"))
    assert loaded["model_name"] == "model"


def test_load_candidate_from_dir(fake_candidate):
    cdir, expected = fake_candidate
    loaded = load_candidate(cdir)
    assert loaded["model_name"] == "model"


def test_build_profile_defaults(fake_candidate):
    cdir, candidate = fake_candidate
    pack = build_train_v2_profile(candidate)
    assert pack["difficulty"] == "train_v2_candidate"
    assert pack["profile"]["format"] == "train_v2_classic_v1"
    assert pack["profile"]["obs_dim"] == 1456
    assert pack["profile"]["action_feature_dim"] == 171
    assert pack["profile"]["max_candidate_actions"] == 601
    assert pack["profile"]["selection"] == "argmax"
    assert pack["profile"]["model_path"] == candidate["candidate_onnx"]


def test_build_profile_relative_to(fake_candidate):
    cdir, candidate = fake_candidate
    pack = build_train_v2_profile(candidate, relative_to=cdir)
    assert pack["profile"]["model_path"] == "model.onnx"


def test_write_candidate_profile(fake_candidate):
    cdir, candidate = fake_candidate
    pack = write_candidate_profile(cdir)
    assert "profile_path" in pack
    out = Path(pack["profile_path"])
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["difficulty"] == "train_v2_candidate"
    assert loaded["profile_path"] == str(out)


def test_format_profile_snippet(fake_candidate):
    cdir, candidate = fake_candidate
    pack = build_train_v2_profile(candidate)
    snippet = format_profile_snippet(pack)
    assert "train_v2_classic_v1" in snippet
    assert "model_path" in snippet
    assert "Opt-in TrainV2 profile artifact" in snippet


def test_load_candidate_missing():
    with pytest.raises(FileNotFoundError):
        load_candidate("/tmp/nonexistent_candidate_12345")


def test_build_profile_missing_onnx():
    with pytest.raises(ValueError, match="missing"):
        build_train_v2_profile({})


def test_validate_candidate_profile_missing_model():
    pack = build_train_v2_profile({"candidate_onnx": "/tmp/nonexistent_model.onnx"})
    pack["profile_path"] = "/tmp/nonexistent_profile.json"
    result = validate_candidate_profile(pack)
    assert result["ok"] is False
    assert "not found" in result["error"]


def test_validate_candidate_profile_ok():
    from ai.train_v2.train_ppo import PPOConfig, train
    from ai.train_v2.export_onnx import export_checkpoint_to_onnx

    tmp = Path("/tmp/_t24_validate")
    tmp.mkdir(exist_ok=True)
    try:
        ckpt_dir = str(tmp / "ckpts")
        config = PPOConfig(
            total_updates=1, episodes_per_update=1, max_steps_per_episode=5,
            hidden_dim=32, action_hidden_dim=16, minibatch_size=8, epochs=1,
            seed=42, checkpoint_dir=ckpt_dir,
        )
        result = train(config)
        onnx_path = str(tmp / "model.onnx")
        export_checkpoint_to_onnx(result["checkpoint_path"], onnx_path, opset=17)

        candidate = {
            "candidate_onnx": onnx_path,
            "source_onnx": onnx_path,
            "model_name": "model",
            "score": 1.0,
            "source_run_dir": str(tmp),
        }
        (tmp / "candidate.json").write_text(json.dumps(candidate))
        pack = write_candidate_profile(str(tmp / "candidate.json"))
        result = validate_candidate_profile(pack)
        assert result["ok"] is True
        assert result["parity"] is not None
        assert result["eval"] is not None
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_candidate_profile_cli_smoke(tmp_path):
    candidate_dir = tmp_path / "cand"
    candidate_dir.mkdir()
    candidate = {
        "candidate_onnx": str(candidate_dir / "model.onnx"),
        "source_onnx": str(candidate_dir / "source.onnx"),
        "model_name": "model",
        "score": 1.0,
        "source_run_dir": str(tmp_path),
    }
    (candidate_dir / "candidate.json").write_text(json.dumps(candidate))

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai.train_v2.candidate_profile",
            "--candidate",
            str(candidate_dir),
            "--output",
            str(candidate_dir / "profile.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (candidate_dir / "profile.json").exists()


def test_candidate_profile_cli_validate_smoke():
    from ai.train_v2.train_ppo import PPOConfig, train
    from ai.train_v2.export_onnx import export_checkpoint_to_onnx

    tmp = Path("/tmp/_t24_cli_validate")
    tmp.mkdir(exist_ok=True)
    try:
        ckpt_dir = str(tmp / "ckpts")
        config = PPOConfig(
            total_updates=1, episodes_per_update=1, max_steps_per_episode=5,
            hidden_dim=32, action_hidden_dim=16, minibatch_size=8, epochs=1,
            seed=42, checkpoint_dir=ckpt_dir,
        )
        result = train(config)
        onnx_path = str(tmp / "model.onnx")
        export_checkpoint_to_onnx(result["checkpoint_path"], onnx_path, opset=17)

        candidate = {
            "candidate_onnx": onnx_path,
            "source_onnx": onnx_path,
            "model_name": "model",
            "score": 1.0,
            "source_run_dir": str(tmp),
        }
        (tmp / "candidate.json").write_text(json.dumps(candidate))

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "ai.train_v2.candidate_profile",
                "--candidate",
                str(tmp / "candidate.json"),
                "--validate",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "ok=" in proc.stdout
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

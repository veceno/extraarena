import json
import subprocess
import sys
from pathlib import Path

import pytest

from ai.train_v2.profile_registry import (
    discover_candidate_profiles,
    load_profile_pack,
    resolve_profile_model_path,
    build_profile_registry,
    select_profile,
    write_profile_overlay,
    load_profile_overlay,
    validate_profile_overlay,
)


def _make_profile_pack(
    *,
    model_path: str = "/tmp/model.onnx",
    score: float = 1.0,
    model_name: str = "model",
    difficulty: str = "train_v2_candidate",
    selection: str = "argmax",
    created_at: str = "2026-01-01 00:00:00",
) -> dict:
    return {
        "difficulty": difficulty,
        "profile": {
            "model_path": model_path,
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": [1.0, 1.0],
            "selection": selection,
        },
        "source": {
            "candidate_onnx": model_path,
            "source_onnx": model_path,
            "model_name": model_name,
            "score": score,
            "source_run_dir": "/tmp/run",
        },
        "created_at": created_at,
        "notes": "test",
    }


class TestDiscovery:
    def test_discover_candidate_profiles_direct_file(self, tmp_path):
        profile = tmp_path / "candidate_profile.json"
        profile.write_text(json.dumps(_make_profile_pack()))
        result = discover_candidate_profiles([str(profile)])
        assert result == [str(profile.resolve())]

    def test_discover_candidate_profiles_from_dir(self, tmp_path):
        cdir = tmp_path / "candidate"
        cdir.mkdir()
        profile = cdir / "candidate_profile.json"
        profile.write_text(json.dumps(_make_profile_pack()))
        result = discover_candidate_profiles([str(cdir)])
        assert result == [str(profile.resolve())]

    def test_discover_candidate_profiles_recursive(self, tmp_path):
        suite = tmp_path / "suite"
        suite.mkdir()
        c1 = suite / "cand1"
        c1.mkdir()
        p1 = c1 / "candidate_profile.json"
        p1.write_text(json.dumps(_make_profile_pack()))

        c2 = suite / "nested" / "cand2"
        c2.mkdir(parents=True)
        p2 = c2 / "candidate_profile.json"
        p2.write_text(json.dumps(_make_profile_pack()))

        result = discover_candidate_profiles([str(suite)])
        assert len(result) == 2
        assert str(p1.resolve()) in result
        assert str(p2.resolve()) in result


class TestLoadAndValidate:
    def test_load_profile_pack_validates_schema(self, tmp_path):
        p = tmp_path / "pack.json"
        pack = _make_profile_pack()
        p.write_text(json.dumps(pack))
        loaded = load_profile_pack(str(p))
        assert loaded["difficulty"] == "train_v2_candidate"
        assert "_profile_path" in loaded
        assert loaded["_profile_path"] == str(p.resolve())

    def test_load_profile_pack_rejects_bad_format(self, tmp_path):
        p = tmp_path / "bad.json"
        pack = _make_profile_pack()
        pack["profile"]["format"] = "legacy_v0"
        p.write_text(json.dumps(pack))
        with pytest.raises(ValueError, match="Unsupported format"):
            load_profile_pack(str(p))

    def test_load_profile_pack_rejects_missing_model_path(self, tmp_path):
        p = tmp_path / "bad.json"
        pack = _make_profile_pack()
        del pack["profile"]["model_path"]
        p.write_text(json.dumps(pack))
        with pytest.raises(ValueError, match="model_path"):
            load_profile_pack(str(p))

    def test_resolve_profile_model_path_absolute(self, tmp_path):
        pack = _make_profile_pack(model_path="/tmp/model.onnx")
        pack["_profile_path"] = str(tmp_path / "pack.json")
        assert resolve_profile_model_path(pack) == "/tmp/model.onnx"

    def test_resolve_profile_model_path_relative(self, tmp_path):
        pack = _make_profile_pack(model_path="model.onnx")
        profile_path = tmp_path / "pack.json"
        pack["_profile_path"] = str(profile_path)
        resolved = resolve_profile_model_path(pack)
        assert resolved == str((tmp_path / "model.onnx").resolve())


class TestRegistry:
    def test_build_profile_registry_sorts_best_first(self, tmp_path):
        c1 = tmp_path / "c1"
        c1.mkdir()
        p1 = _make_profile_pack(score=2.0, model_name="best_model", model_path=str(c1 / "best.onnx"))
        (c1 / "candidate_profile.json").write_text(json.dumps(p1))
        (c1 / "best.onnx").write_bytes(b"")

        c2 = tmp_path / "c2"
        c2.mkdir()
        p2 = _make_profile_pack(score=1.0, model_name="worse_model", model_path=str(c2 / "worse.onnx"))
        (c2 / "candidate_profile.json").write_text(json.dumps(p2))
        (c2 / "worse.onnx").write_bytes(b"")

        registry = build_profile_registry([str(tmp_path)])
        assert registry["ok"] == 2
        assert registry["errors"] == 0
        assert registry["best"]["model_name"] == "best_model"
        assert registry["rows"][0]["model_name"] == "best_model"
        assert registry["rows"][1]["model_name"] == "worse_model"

    def test_build_profile_registry_keeps_error_rows(self, tmp_path):
        c1 = tmp_path / "c1"
        c1.mkdir()
        (c1 / "candidate_profile.json").write_text("{not json")

        c2 = tmp_path / "c2"
        c2.mkdir()
        p2 = _make_profile_pack(score=1.0, model_path=str(c2 / "ok.onnx"))
        (c2 / "candidate_profile.json").write_text(json.dumps(p2))
        (c2 / "ok.onnx").write_bytes(b"")

        registry = build_profile_registry([str(tmp_path)])
        assert registry["ok"] == 1
        assert registry["errors"] == 1
        assert registry["rows"][0]["status"] == "ok"
        assert registry["rows"][1]["status"] == "error"

    def test_select_profile_best(self, tmp_path):
        c1 = tmp_path / "c1"
        c1.mkdir()
        p1 = _make_profile_pack(score=2.0, model_path=str(c1 / "a.onnx"))
        (c1 / "candidate_profile.json").write_text(json.dumps(p1))
        (c1 / "a.onnx").write_bytes(b"")

        c2 = tmp_path / "c2"
        c2.mkdir()
        p2 = _make_profile_pack(score=1.0, model_path=str(c2 / "b.onnx"))
        (c2 / "candidate_profile.json").write_text(json.dumps(p2))
        (c2 / "b.onnx").write_bytes(b"")

        registry = build_profile_registry([str(tmp_path)])
        best = select_profile(registry, selector="best")
        assert best is not None
        assert best["score"] == 2.0

    def test_select_profile_by_model_name(self, tmp_path):
        c1 = tmp_path / "c1"
        c1.mkdir()
        p1 = _make_profile_pack(score=1.0, model_name="target", model_path=str(c1 / "a.onnx"))
        (c1 / "candidate_profile.json").write_text(json.dumps(p1))
        (c1 / "a.onnx").write_bytes(b"")

        registry = build_profile_registry([str(tmp_path)])
        selected = select_profile(registry, selector="target")
        assert selected is not None
        assert selected["model_name"] == "target"

    def test_select_profile_missing(self, tmp_path):
        c1 = tmp_path / "c1"
        c1.mkdir()
        p1 = _make_profile_pack(score=1.0, model_name="x", model_path=str(c1 / "a.onnx"))
        (c1 / "candidate_profile.json").write_text(json.dumps(p1))
        (c1 / "a.onnx").write_bytes(b"")

        registry = build_profile_registry([str(tmp_path)])
        selected = select_profile(registry, selector="nonexistent")
        assert selected is None

    def test_select_profile_require_onnx(self, tmp_path):
        c1 = tmp_path / "c1"
        c1.mkdir()
        p1 = _make_profile_pack(score=1.0, model_name="x", model_path=str(c1 / "a.onnx"))
        (c1 / "candidate_profile.json").write_text(json.dumps(p1))
        # do NOT create a.onnx

        registry = build_profile_registry([str(tmp_path)])
        selected = select_profile(registry, selector="best", require_onnx=True)
        assert selected is None


class TestOverlay:
    def test_write_and_load_profile_overlay(self, tmp_path):
        pack = _make_profile_pack(model_path=str(tmp_path / "model.onnx"))
        pack["_profile_path"] = str(tmp_path / "pack.json")

        out = tmp_path / "overlay.json"
        overlay = write_profile_overlay(pack, str(out), difficulty="custom_diff")
        assert out.exists()
        assert overlay["version"] == "train_v2_profile_overlay_v1"
        assert "custom_diff" in overlay["profiles"]
        assert overlay["profiles"]["custom_diff"]["difficulty"] == "custom_diff"

        loaded = load_profile_overlay(str(out))
        assert loaded["version"] == "train_v2_profile_overlay_v1"
        assert "custom_diff" in loaded["profiles"]

    def test_write_profile_overlay_relative_to(self, tmp_path):
        pack = _make_profile_pack(model_path=str(tmp_path / "subdir" / "model.onnx"))
        pack["_profile_path"] = str(tmp_path / "pack.json")
        out = tmp_path / "overlay.json"
        overlay = write_profile_overlay(pack, str(out), relative_to=str(tmp_path))
        loaded = load_profile_overlay(str(out))
        assert loaded["profiles"]["train_v2_candidate"]["model_path"] == "subdir/model.onnx"


class TestValidation:
    def test_validate_profile_overlay_smoke(self):
        from ai.train_v2.train_ppo import PPOConfig, train
        from ai.train_v2.export_onnx import export_checkpoint_to_onnx

        tmp = Path("/tmp/_t25_validate")
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

            pack = _make_profile_pack(model_path=onnx_path, model_name="model")
            pack["_profile_path"] = str(tmp / "pack.json")
            overlay = write_profile_overlay(pack, str(tmp / "overlay.json"))

            validation = validate_profile_overlay(overlay["overlay_path"], games=1, max_steps=50)
            assert isinstance(validation["ok"], bool)
            assert len(validation["profiles"]) == 1
            assert validation["profiles"][0]["difficulty"] == "train_v2_candidate"
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


class TestCLI:
    def test_profile_registry_cli_smoke(self, tmp_path):
        c1 = tmp_path / "c1"
        c1.mkdir()
        p1 = _make_profile_pack(score=1.0, model_name="m1", model_path=str(c1 / "a.onnx"))
        (c1 / "candidate_profile.json").write_text(json.dumps(p1))
        (c1 / "a.onnx").write_bytes(b"")

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "ai.train_v2.profile_registry",
                "--paths",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "profiles:" in proc.stdout
        assert "m1" in proc.stdout

    def test_profile_registry_cli_write_overlay(self, tmp_path):
        c1 = tmp_path / "c1"
        c1.mkdir()
        p1 = _make_profile_pack(score=1.0, model_name="m1", model_path=str(c1 / "a.onnx"))
        (c1 / "candidate_profile.json").write_text(json.dumps(p1))
        (c1 / "a.onnx").write_bytes(b"")
        out = tmp_path / "overlay.json"

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "ai.train_v2.profile_registry",
                "--paths",
                str(tmp_path),
                "--write-overlay",
                str(out),
                "--select",
                "m1",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["version"] == "train_v2_profile_overlay_v1"

    def test_profile_registry_cli_validate_smoke(self):
        from ai.train_v2.train_ppo import PPOConfig, train
        from ai.train_v2.export_onnx import export_checkpoint_to_onnx

        tmp = Path("/tmp/_t25_cli_validate")
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

            c1 = tmp / "c1"
            c1.mkdir()
            p1 = _make_profile_pack(model_path=onnx_path, model_name="m1")
            p1["_profile_path"] = str(c1 / "pack.json")
            (c1 / "candidate_profile.json").write_text(json.dumps(p1))

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ai.train_v2.profile_registry",
                    "--paths",
                    str(tmp),
                    "--write-overlay",
                    str(tmp / "overlay.json"),
                    "--validate",
                ],
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, proc.stderr
            assert "Validation" in proc.stdout or "ok=" in proc.stdout
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

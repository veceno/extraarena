"""Block B — custom model by path+adapter in series.

Покрывает:
  - _resolve_policy_spec пробрасывает path+kind (flat и nested форма) в build_policy.
  - _safe_model_path: repo-relative путь не удваивается (regression для
    `ai/models/foo.onnx` → `models_dir/ai/models/foo.onnx`); bare-имя → models_dir;
    path-traversal блокируется; absolute-outside блокируется.
  - серия с nested p2_model {name,path,kind:'random'} строится и играется
    (baseline-kind не грузит onnx).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from rlhf_env.tests._v5_helpers import create_match, make_manager

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FAKE_ONNX = str(_REPO_ROOT / "ai" / "models" / "fake_v5.onnx")


def test_resolve_policy_spec_flat_path_kind(tmp_path):
    mgr = make_manager(tmp_path)
    spec = mgr._resolve_policy_spec(
        {"p2_model": "myCustom", "p2_model_path": _FAKE_ONNX, "p2_model_kind": "v5"},
        "p2", seed=42,
    )
    assert spec["name"] == "myCustom"
    assert spec["path"].endswith("fake_v5.onnx")
    assert spec["kind"] == "v5"


def test_resolve_policy_spec_nested_dict(tmp_path):
    mgr = make_manager(tmp_path)
    spec = mgr._resolve_policy_spec(
        {"p2_model": {"name": "snap", "path": _FAKE_ONNX, "kind": "v5"}}, "p2", seed=1,
    )
    assert spec["name"] == "snap"
    assert spec["path"].endswith("fake_v5.onnx")
    assert spec["kind"] == "v5"


def test_resolve_policy_spec_defaults_to_random(tmp_path):
    mgr = make_manager(tmp_path)
    spec = mgr._resolve_policy_spec({}, "p2", seed=1)
    assert spec["name"] == "random"
    assert "path" not in spec


def test_safe_model_path_repo_relative_not_doubled(tmp_path):
    mgr = make_manager(tmp_path)
    p = mgr._safe_model_path("ai/models/fake_v5.onnx")
    assert p.count("ai/models") == 1
    assert p.endswith("ai/models/fake_v5.onnx")


def test_safe_model_path_bare_filename_under_models_dir(tmp_path):
    mgr = make_manager(tmp_path)
    p = mgr._safe_model_path("fake_v5.onnx")
    assert p.endswith("ai/models/fake_v5.onnx")


def test_safe_model_path_traversal_blocked(tmp_path):
    mgr = make_manager(tmp_path)
    with pytest.raises(ValueError):
        mgr._safe_model_path("../../../etc/passwd")


def test_safe_model_path_absolute_outside_blocked(tmp_path):
    mgr = make_manager(tmp_path)
    with pytest.raises(ValueError):
        mgr._safe_model_path("/etc/passwd")


def test_safe_model_path_absolute_under_repo_ok(tmp_path):
    mgr = make_manager(tmp_path)
    p = mgr._safe_model_path(str((_REPO_ROOT / "ai" / "models" / "fake_v5.onnx").resolve()))
    assert p.endswith("fake_v5.onnx")


def test_series_nested_p2_model_random_kind_plays(tmp_path):
    """nested p2_model с baseline-kind строится и доигрывается (onnx не грузится)."""
    mgr = make_manager(tmp_path)
    match, engine, runner = create_match(
        mgr, p1_actor_type="rl", p1_model="random",
        p2_model="myrand", p2_model_path=_FAKE_ONNX, p2_model_kind="random",
        starting_player="p1",
    )
    # p2 — baseline random (kind=random), path проигнорирован фабрикой baseline.
    assert match.bot_policy.kind == "random"
    asyncio.run(runner.run_auto())
    assert engine.is_ended is True
    man = json.loads((tmp_path / "sessions" / match.group_id / "manifest.json").read_text("utf-8"))
    assert man["battles_results"][0]["battle_tag"] == "rl-vs-bot"
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from ai.train_v2.profile_actions import (
    profile_action_pipeline,
    _apply_stable_defaults,
)


def test_profile_action_pipeline_smoke():
    result = profile_action_pipeline(episodes=1, steps_per_episode=3, seed=1)

    expected = {
        "episodes",
        "steps",
        "avg_legal_actions",
        "mask_ms_p50",
        "mask_ms_p95",
        "features_full_ms_p50",
        "features_full_ms_p95",
        "features_fast_ms_p50",
        "features_fast_ms_p95",
        "preview_overhead_ms_p50",
        "fast_speedup",
        "action_mode",
        "warmup_steps",
    }
    assert expected.issubset(result.keys())
    assert result["episodes"] == 1
    assert result["steps"] > 0
    assert result["action_mode"] == "first"
    assert result["warmup_steps"] == 0


def test_profile_action_pipeline_values_finite():
    result = profile_action_pipeline(episodes=1, steps_per_episode=5, seed=2)
    for key, value in result.items():
        if isinstance(value, (int, float)):
            assert np.isfinite(value), f"{key} is not finite: {value}"


def test_fast_not_slower_than_full_smoke():
    result = profile_action_pipeline(episodes=1, steps_per_episode=5, seed=3)
    assert result["features_fast_ms_p50"] <= result["features_full_ms_p50"] * 1.5


def test_profile_action_pipeline_random_mode():
    result = profile_action_pipeline(episodes=1, steps_per_episode=5, seed=4, action_mode="random")
    assert result["action_mode"] == "random"
    assert result["steps"] > 0
    for key, value in result.items():
        if isinstance(value, (int, float)):
            assert np.isfinite(value), f"{key} is not finite: {value}"


def test_profile_action_pipeline_greedy_face_mode():
    result = profile_action_pipeline(episodes=1, steps_per_episode=5, seed=5, action_mode="greedy_face")
    assert result["action_mode"] == "greedy_face"
    assert result["steps"] > 0


def test_profile_action_pipeline_bad_mode():
    try:
        profile_action_pipeline(episodes=1, steps_per_episode=3, seed=6, action_mode="unknown")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_profile_actions_cli_smoke():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai.train_v2.profile_actions",
            "--episodes",
            "1",
            "--steps",
            "3",
            "--seed",
            "4",
            "--action-mode",
            "random",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "features" in proc.stdout


def test_profile_action_pipeline_warmup():
    result = profile_action_pipeline(episodes=1, steps_per_episode=3, seed=1, warmup_steps=2)
    assert result["steps"] > 0
    assert result["steps"] <= 3
    assert result["warmup_steps"] == 2


def test_profile_actions_cli_json():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai.train_v2.profile_actions",
            "--episodes",
            "1",
            "--steps",
            "3",
            "--seed",
            "4",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    parsed = json.loads(proc.stdout)
    assert "steps" in parsed
    assert "features_full_ms_p50" in parsed


def test_profile_actions_cli_output_file(tmp_path):
    out_path = tmp_path / "profile.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai.train_v2.profile_actions",
            "--episodes",
            "1",
            "--steps",
            "3",
            "--seed",
            "5",
            "--output",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_path.exists()
    with open(out_path, "r", encoding="utf-8") as f:
        parsed = json.load(f)
    assert "steps" in parsed
    assert "features_full_ms_p50" in parsed


def test_apply_stable_defaults_helper():
    args = SimpleNamespace(episodes=1, steps=3, warmup_steps=0)
    _apply_stable_defaults(args, stable_episodes=5, stable_steps=20, stable_warmup=2)
    assert args.episodes == 5
    assert args.steps == 20
    assert args.warmup_steps == 2

    args2 = SimpleNamespace(episodes=100, steps=200, warmup_steps=50)
    _apply_stable_defaults(args2, stable_episodes=5, stable_steps=20, stable_warmup=2)
    assert args2.episodes == 100
    assert args2.steps == 200
    assert args2.warmup_steps == 50

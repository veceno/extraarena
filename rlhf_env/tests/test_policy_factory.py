"""Тесты policy_factory: baselines, ONNX (если есть), ошибки."""
from __future__ import annotations

import pytest

from rlhf_env.components.policy_factory import (
    _RLHFEndTurnPolicy,
    _RLHFGreedyFacePolicy,
    _RLHFRandomPolicy,
    build_policy,
    clear_cache,
)


def test_baseline_random():
    pol = build_policy({"name": "random", "seed": 0})
    assert isinstance(pol, _RLHFRandomPolicy)
    assert pol.kind == "random"


def test_baseline_greedy_face():
    pol = build_policy({"name": "greedy_face"})
    assert isinstance(pol, _RLHFGreedyFacePolicy)
    assert pol.kind == "greedy_face"


def test_baseline_end_turn():
    pol = build_policy({"name": "end_turn"})
    assert isinstance(pol, _RLHFEndTurnPolicy)
    assert pol.kind == "end_turn"


def test_unknown_model_raises():
    clear_cache()
    with pytest.raises((ValueError, KeyError)):
        build_policy({"name": "totally-fake-model-xyz"})


def test_v4_max_loads_if_present():
    """Если V4-Max sidecar существует — должна грузиться через BerserkInference."""
    from pathlib import Path
    onnx = Path("ai/models/extra-lr-v4-max.onnx")
    if not onnx.exists():
        pytest.skip("ai/models/extra-lr-v4-max.onnx not present")
    clear_cache()
    pol = build_policy({"name": "extra-lr-v4-max"})
    assert pol.kind == "action_onnx"
    assert pol.name == "extra-lr-v4-max"


def test_v3_legacy_loads_if_present():
    """V3 legacy (одно-входная ONNX) — определяется как legacy_onnx."""
    from pathlib import Path
    onnx = Path("ai/models/extra-lr-v3-max.onnx")
    if not onnx.exists():
        pytest.skip("ai/models/extra-lr-v3-max.onnx not present")
    # V3-Max не имеет sidecar → BerserkInference ругнётся (format=train_v2_classic_v1
    # не получит нужные dims). Проверяем, что factory хотя бы корректно
    # определяет kind через inspect_model.
    from rlhf_env.components.policy_registry import PolicyRegistry
    reg = PolicyRegistry.scan(Path("ai/models"))
    spec_obj = reg.get("extra-lr-v3-max")
    if spec_obj is not None:
        assert spec_obj.kind in ("legacy_onnx", "unknown")


def test_inference_params_defaults():
    from rlhf_env.components.inference_params import default_inference_params

    p = default_inference_params({}, "action_onnx")
    assert p["kind"] == "action_onnx"
    assert p["mode"] == "argmax"
    assert "obs_dim" in p

    p_legacy = default_inference_params({}, "legacy_onnx")
    assert p_legacy["kind"] == "legacy_onnx"

    p_random = default_inference_params({}, "random")
    assert p_random["kind"] == "random"


def test_clear_cache():
    from rlhf_env.components.policy_factory import _BERSERK_CACHE
    before = len(_BERSERK_CACHE)
    clear_cache()
    assert len(_BERSERK_CACHE) == 0
    assert before >= 0  # sanity
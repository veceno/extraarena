"""Block A — modular adapter registry (tracked layer B) unit tests.

Покрывает:
  - baseline build (random/greedy_face/end_turn) → kind+name attrs + select_action int.
  - v5 stub: зарезервированный слот, select_action → NotImplementedError (шаблон Block 0).
  - layer A absent: auto-detect по path → ValueError (не ModuleNotFoundError).
  - register(custom adapter) переопределяет stub без правки if/elif.
  - register_detector срабатывает (LIFO) раньше fallback.
  - legacy_onnx без layer A → ValueError с понятным сообщением.
  - unknown kind → ValueError.

Без полных игр — fast unit tests.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rlhf_env.components.policy_adapters import (
    AdapterRegistry,
    V5StubAdapter,
    default_registry,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FAKE_ONNX = str(_REPO_ROOT / "ai" / "models" / "fake_v5.onnx")


class _FakeEngine:
    """Минимальный engine для select_action baseline-политик."""

    def __init__(self, legal):
        self._legal = legal

    def get_legal_actions(self, player_id):
        return list(self._legal)


def test_baseline_build_random():
    p = default_registry().build({"name": "random", "seed": 7})
    assert p.kind == "random"
    assert p.name == "random"
    assert p.model_path is None and p.weights_hash is None
    idx = p.select_action(_FakeEngine([1, 2, 3]), 1)
    assert isinstance(idx, int) and 0 <= idx < 3


def test_baseline_build_end_turn():
    from core.actions import EndTurnAction
    p = default_registry().build({"name": "end_turn"})
    assert p.kind == "end_turn"
    idx = p.select_action(_FakeEngine([EndTurnAction()]), 1)
    assert idx == 0


def test_baseline_build_greedy_face():
    p = default_registry().build({"name": "greedy_face"})
    assert p.kind == "greedy_face"
    # нет легальных → 0
    assert p.select_action(_FakeEngine([]), 1) == 0


def test_v5_stub_raises_on_select():
    # C1: the registry 'v5' slot now routes to the real V5RlhfAdapter, so the
    # stub is exercised via its OWN constructor (the retained test double) —
    # NOT via default_registry().build, which would build the real adapter and
    # try to load the (nonexistent) ONNX. V5StubAdapter stays importable for
    # tests that want stub behaviour.
    stub = V5StubAdapter({"name": "v5x", "path": "/x.onnx"})
    assert stub.kind == "v5"
    with pytest.raises(NotImplementedError):
        stub.select_action(_FakeEngine([1]), 1)


def test_v5_stub_is_template_class():
    # прямой конструктор — тоже шаблон (не падает на construct, только на select).
    stub = V5StubAdapter({"name": "my-v5"})
    assert stub.kind == "v5"
    with pytest.raises(NotImplementedError):
        stub.select_action(None, 1)


def test_layer_a_missing_auto_detect_raises_valueerror():
    # path к реальному onnx, kind не задан → auto-detect → layer A absent → ValueError
    with pytest.raises(ValueError):
        default_registry().build({"name": "foo", "path": _FAKE_ONNX})


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        default_registry().build({"name": "foo", "path": _FAKE_ONNX, "kind": "unknown"})


def test_legacy_onnx_without_layer_a_raises_valueerror():
    # legacy_onnx factory пытается import ai.model_benchmark (gitignored layer A).
    # Тест верифицирует путь БЕЗ layer A (ясная ValueError, не ModuleNotFoundError).
    # Если layer A доступен (напр. symlink-ом в worktree для V3/V4-max игр) —
    # factory доходит до onnx-загрузки и путь «без layer A» неактуален → skip.
    try:
        import ai.model_benchmark.policies  # noqa: F401
    except Exception:
        pass  # layer A отсутствует → тестируемый путь активен
    else:
        pytest.skip("ai.model_benchmark (layer A) present — no-layer-A path N/A")
    with pytest.raises(ValueError):
        default_registry().build({"name": "foo", "kind": "legacy_onnx", "path": _FAKE_ONNX})


def test_register_custom_adapter_overrides_stub():
    """register(kind, factory) — V5 добавляется без правки if/elif."""
    reg = AdapterRegistry()

    class MyV5Adapter:
        kind = "v5"
        name = "my-v5"
        model_path = weights_hash = weights_version = None

        def __init__(self, spec):
            self.name = spec.get("name", "my-v5")

        def select_action(self, engine, player_id):
            return 42

    reg.register("v5", lambda spec, r: MyV5Adapter(spec))
    p = reg.build({"name": "v5x", "kind": "v5", "path": "/x.onnx"})
    assert isinstance(p, MyV5Adapter)
    assert p.select_action(_FakeEngine([1, 2, 3]), 1) == 42


def test_register_detector_lifo_priority():
    """register_detector prepend (LIFO) — пользовательский детектор раньше fallback."""
    reg = AdapterRegistry()
    seen = []

    def my_detector(path, sidecar, name=None):
        seen.append(path)
        return "mykind" if path and path.endswith(".onnx") else None

    reg.register_detector(my_detector)
    # mykind не зарегистрирован как factory — build дойдёт до resolve('mykind') → KeyError,
    # но detect_kind должен вернуть 'mykind' (детектор сработал).
    assert reg.detect_kind("/x.onnx", None, name="foo") == "mykind"
    assert seen == ["/x.onnx"]


def test_default_registry_has_builtin_kinds():
    kinds = set(default_registry().kinds())
    for k in ("random", "greedy_face", "end_turn", "legacy_onnx", "action_onnx", "v4", "v5"):
        assert k in kinds, f"missing builtin kind {k!r}"


def test_build_policy_delegates_to_registry():
    """policy_factory.build_policy — тонкая обёртка над default_registry().build."""
    from rlhf_env.components.policy_factory import build_policy
    p = build_policy({"name": "random", "seed": 3})
    assert p.kind == "random"

def test_sidecar_detector_v4_without_layer_a(tmp_path):
    """V4 ONNX определяется как action_onnx по sidecar — без gitignored inspector.

    Воспроизводит prod-сценарий: scan V4 ONNX без layer A (ai.model_benchmark
    отсутствует) — sidecar несёт inference-контракт.
    """
    import json
    from rlhf_env.components.policy_adapters import default_registry, _sidecar_kind_detector

    sidecar = {
        "model_version": "classic_action_conditioned_onnx_v1",
        "inputs": ["observation", "action_features"],
        "outputs": ["logits", "value"],
        "obs_dim": 1456,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
    }
    # детектор напрямую
    assert _sidecar_kind_detector("/abs/path/extra-lr-v4-max.onnx", sidecar, "v4") == "action_onnx"
    # через реестр (sidecar приоритетнее layer-A fallback)
    assert default_registry().detect_kind("/abs/path/extra-lr-v4-max.onnx", sidecar, name="v4") == "action_onnx"

    # legacy по model_version classic_* без action_conditioned
    legacy_sc = {"model_version": "classic_legacy_v1", "inputs": ["observation"], "outputs": ["logits"]}
    assert _sidecar_kind_detector("/x.onnx", legacy_sc, "v3") == "legacy_onnx"

    # нет sidecar / нет path → None (не валит, передаёт дальше по цепочке)
    assert _sidecar_kind_detector(None, sidecar, "x") is None
    assert _sidecar_kind_detector("/x.onnx", {}, "x") is None


def test_v5_sidecar_detected_before_broader_v4_detector():
    from rlhf_env.components.policy_adapters import (
        _sidecar_kind_detector,
        _v5_sidecar_kind_detector,
        default_registry,
    )

    sidecar = {
        "model_version": "v5_split_encoder_onnx_v1",
        "inputs": ["observation", "action_features"],
        "outputs": ["logits", "value", "mana_draw_logit"],
        "obs_dim": 7128,
        "action_feature_dim": 171,
        "mana_draw_head": True,
        "format": "v5",
    }
    # This documents the ambiguity: the old detector alone calls it V4.
    assert _sidecar_kind_detector("/x.onnx", sidecar, "v5") == "action_onnx"
    assert _v5_sidecar_kind_detector("/x.onnx", sidecar, "v5") == "v5"
    assert default_registry().detect_kind("/x.onnx", sidecar, name="v5") == "v5"


def test_scan_directory_derives_v4_kind_from_sidecar(tmp_path):
    """PolicyRegistry.scan_directory сканирует V4 ONNX по sidecar без layer A."""
    import json
    from rlhf_env.components.policy_registry import PolicyRegistry

    onnx = tmp_path / "extra-lr-v4-max.onnx"
    onnx.write_bytes(b"not a real onnx model")
    onnx.with_suffix(".onnx.json").write_text(
        json.dumps(
            {
                "model_version": "classic_action_conditioned_onnx_v1",
                "inputs": ["observation", "action_features"],
                "outputs": ["logits", "value"],
                "obs_dim": 1456,
                "action_feature_dim": 171,
                "max_candidate_actions": 601,
            }
        ),
        encoding="utf-8",
    )

    reg = PolicyRegistry.scan(tmp_path)
    spec = reg.get("extra-lr-v4-max")
    assert spec is not None
    assert spec.kind == "action_onnx"

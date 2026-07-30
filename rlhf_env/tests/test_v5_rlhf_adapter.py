"""C1 — V5RlhfAdapter unit tests (synthetic: fake inference + real engine).

Covers:
  1. factory 2-arg contract: _factory_v5_real(spec, registry) -> V5RlhfAdapter;
     default_registry().resolve('v5') is _factory_v5_real (register swap); build
     via default_registry returns V5RlhfAdapter, NOT V5StubAdapter.
  2. argmax over legal-masked candidates -> legal_action_index (end-to-end on a
     real engine with a fake deterministic inference callable).
  3. D11 omniscient: select_action builds obs with InfoModeV5(enemy_hand_known=True,
     enemy_deck_known=True) EXPLICIT (spy on encode_observation_v5).
  4. append_only: encode_action_features + build_action_mask called with
     placement_mode='append_only'.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from rlhf_env.components.policy_adapters import (
    AdapterRegistry,
    V5StubAdapter,
    default_registry,
    _register_builtins,
)
from rlhf_env.components.v5_rlhf_adapter import V5RlhfAdapter, _factory_v5_real

# Real-game fixtures (in-process, no HTTP/socket).
from rlhf_env.tests._v5_helpers import create_match, make_manager


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _fake_inference(logits=None):
    """Deterministic inference callable returning a V5 3-tuple."""
    if logits is None:
        # Increasing logits -> argmax picks the highest masked-in candidate id.
        logits = np.arange(601, dtype=np.float32)
    logits = np.asarray(logits, dtype=np.float32)

    def _infer(obs, action_features):
        return logits, 0.0, 0.0

    return _infer


# ----------------------------------------------------------------------------
# 1. factory 2-arg contract
# ----------------------------------------------------------------------------

def test_factory_v5_real_two_arg_returns_adapter():
    spec = {"name": "v5x", "path": "/x.onnx", "inference": _fake_inference()}
    adapter = _factory_v5_real(spec, None)
    assert isinstance(adapter, V5RlhfAdapter)
    assert adapter.kind == "v5"
    assert adapter.name == "v5x"
    assert adapter.model_path == "/x.onnx"
    # callable as (spec, registry) with registry a dummy object too.
    adapter2 = _factory_v5_real(spec, object())
    assert isinstance(adapter2, V5RlhfAdapter)


def test_default_registry_v5_slot_is_real_factory():
    """The register swap: default_registry().resolve('v5') is _factory_v5_real,
    NOT the stub factory."""
    reg = default_registry()
    assert reg.resolve("v5") is _factory_v5_real


def test_default_registry_build_v5_returns_real_adapter_not_stub():
    spec = {"name": "v5x", "kind": "v5", "path": "/x.onnx", "inference": _fake_inference()}
    p = default_registry().build(spec)
    assert isinstance(p, V5RlhfAdapter)
    assert not isinstance(p, V5StubAdapter)


def test_real_onnx_wrapper_accepts_batched_scalar_heads(monkeypatch):
    """Production ONNX exports return value and mana heads as shape [1, 1]."""
    class FakeSession:
        def __init__(self, path, providers):
            assert providers == ["CPUExecutionProvider"]

        def run(self, output_names, feed):
            assert output_names == ["logits", "value", "mana_draw_logit"]
            return [
                np.zeros((1, 601), dtype=np.float32),
                np.asarray([[0.25]], dtype=np.float32),
                np.asarray([[-0.75]], dtype=np.float32),
            ]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        types.SimpleNamespace(InferenceSession=FakeSession),
    )
    infer = V5RlhfAdapter._build_onnx_inference("fake.onnx")
    logits, value, mana_draw_logit = infer(
        np.zeros(7128, dtype=np.float32),
        np.zeros((601, 171), dtype=np.float32),
    )
    assert logits.shape == (601,)
    assert value == pytest.approx(0.25)
    assert mana_draw_logit == pytest.approx(-0.75)


def test_fresh_registry_resolves_registered_real_factory():
    """A fresh AdapterRegistry registering the real factory resolves to it."""
    reg = AdapterRegistry()
    reg.register("v5", _factory_v5_real)
    spec = {"name": "v5x", "kind": "v5", "path": "/x.onnx", "inference": _fake_inference()}
    p = reg.build(spec)
    assert isinstance(p, V5RlhfAdapter)


# ----------------------------------------------------------------------------
# 2. argmax -> legal-index (end-to-end on a real engine)
# ----------------------------------------------------------------------------

def test_argmax_maps_to_legal_index(tmp_path):
    """Fake inference returns increasing logits; the adapter must pick the
    highest masked-in candidate and map it to the matching legal action index."""
    mgr = make_manager(tmp_path)
    match, engine, _runner = create_match(mgr, p2_model="random", starting_player="p1")
    arena = engine._arena
    pid = engine.human_user_id

    legal = arena.get_legal_actions(pid)
    assert legal  # p1 starts

    adapter = V5RlhfAdapter({"name": "v5-test", "inference": _fake_inference()})
    idx = adapter.select_action(arena, pid)
    assert isinstance(idx, int)
    assert 0 <= idx < len(legal)

    # Independently compute the expected index: argmax over legal-masked
    # candidates, decode, match against legal (mirrors the adapter).
    from ai.train_v2.classic_actions_v1 import build_action_mask, decode_action
    state = arena.state
    mask = build_action_mask(state, pid, verify_mask=False, placement_mode="append_only")
    masked = np.where(mask.astype(bool), np.arange(601, dtype=np.float32), -np.inf)
    expected_candidate = int(np.argmax(masked))
    base = decode_action(state, pid, expected_candidate)
    expected_idx = None
    if base is not None:
        for i, a in enumerate(legal):
            if a == base:
                expected_idx = i
                break
    if expected_idx is None:
        expected_idx = len(legal) - 1
    assert idx == expected_idx


def test_argmax_peak_at_end_turn(tmp_path):
    """Peak at candidate 0 (end_turn, always masked-in) -> returned idx points
    at an EndTurnAction."""
    from core.actions import EndTurnAction

    mgr = make_manager(tmp_path)
    match, engine, _runner = create_match(mgr, p2_model="random", starting_player="p1")
    arena = engine._arena
    pid = engine.human_user_id
    legal = arena.get_legal_actions(pid)

    logits = np.full(601, -1.0, dtype=np.float32)
    logits[0] = 5.0
    adapter = V5RlhfAdapter({"name": "v5-test", "inference": _fake_inference(logits)})
    idx = adapter.select_action(arena, pid)
    assert 0 <= idx < len(legal)
    assert isinstance(legal[idx], EndTurnAction)


def test_select_action_empty_legal_fails_closed(tmp_path):
    adapter = V5RlhfAdapter({"name": "v5-test", "inference": _fake_inference()})

    class _EmptyEngine:
        state = None

        def get_legal_actions(self, player_id):
            return []

    with pytest.raises(
        RuntimeError,
        match=r"^v5_policy_failure:empty_legal_actions$",
    ):
        adapter.select_action(_EmptyEngine(), 1)


def test_non_finite_logits_fail_closed_with_stable_code(tmp_path):
    mgr = make_manager(tmp_path)
    _, engine, _ = create_match(
        mgr,
        p2_model="random",
        starting_player="p1",
    )
    adapter = V5RlhfAdapter(
        {
            "name": "v5-test",
            "inference": _fake_inference(
                np.full(601, np.nan, dtype=np.float32)
            ),
        }
    )

    with pytest.raises(
        RuntimeError,
        match=r"^v5_policy_failure:non_finite_logits$",
    ):
        adapter.select_action(engine._arena, engine.human_user_id)


def test_unexpected_inference_error_does_not_reflect_secret(tmp_path):
    mgr = make_manager(tmp_path)
    _, engine, _ = create_match(
        mgr,
        p2_model="random",
        starting_player="p1",
    )

    def broken_inference(_obs, _action_features):
        raise RuntimeError(
            "postgresql://alice:SUPERSECRET@example.invalid/prod"
        )

    adapter = V5RlhfAdapter(
        {"name": "v5-test", "inference": broken_inference}
    )
    with pytest.raises(RuntimeError) as caught:
        adapter.select_action(engine._arena, engine.human_user_id)

    assert str(caught.value) == "v5_policy_failure:unexpected_failure"
    assert "SUPERSECRET" not in str(caught.value)


@pytest.mark.parametrize(("bad_logits", "failure_code"), [
    (np.full(601, np.nan, dtype=np.float32), "non_finite_logits"),
    (np.zeros(600, dtype=np.float32), "invalid_output_contract"),
])
def test_v5_rejects_invalid_logits_instead_of_silent_end_turn(
    tmp_path,
    bad_logits,
    failure_code,
):
    mgr = make_manager(tmp_path)
    _match, engine, _runner = create_match(mgr, p2_model="random", starting_player="p1")
    adapter = V5RlhfAdapter({"name": "broken-v5", "inference": _fake_inference(bad_logits)})
    with pytest.raises(RuntimeError, match=f"v5_policy_failure:{failure_code}"):
        adapter.select_action(engine._arena, engine.human_user_id)


def test_v5_rejects_non_finite_mana_draw_logit(tmp_path):
    mgr = make_manager(tmp_path)
    _match, engine, _runner = create_match(mgr, p2_model="random", starting_player="p1")

    def broken_draw(obs, action_features):
        return np.zeros(601, dtype=np.float32), 0.0, np.nan

    adapter = V5RlhfAdapter({"name": "broken-v5", "inference": broken_draw})
    with pytest.raises(RuntimeError, match="v5_policy_failure:non_finite_mana_logit"):
        adapter.select_action(engine._arena, engine.human_user_id)


# ----------------------------------------------------------------------------
# 3. D11 omniscient InfoModeV5 explicit
# ----------------------------------------------------------------------------

def test_v5_uses_training_self_visible_info_mode(tmp_path, monkeypatch):
    """C2 inference must use the same private-information contract as A/B."""
    from train_v3.contracts import OBS_V5_DIM
    import train_v3.obs_v5 as obs_v5_mod

    captured = {}

    def _spy(state, player_id, *, info_mode=None, assist_mode=None, history_events=None):
        captured["info_mode"] = info_mode
        captured["assist_mode"] = assist_mode
        captured["history_events"] = history_events
        return np.zeros(OBS_V5_DIM, dtype=np.float32)

    monkeypatch.setattr(obs_v5_mod, "encode_observation_v5", _spy)

    mgr = make_manager(tmp_path)
    match, engine, _runner = create_match(mgr, p2_model="random", starting_player="p1")
    arena = engine._arena
    pid = engine.human_user_id

    adapter = V5RlhfAdapter({"name": "v5-test", "inference": _fake_inference()})
    adapter.select_action(arena, pid)

    im = captured["info_mode"]
    assert im is not None
    assert im.enemy_hand_known is True
    assert im.enemy_deck_known is True
    # The global default is also omniscient; keep an explicit adapter setting so
    # an unrelated default change cannot silently create deploy/train drift.
    from train_v3.contracts import InfoModeV5
    assert InfoModeV5().enemy_hand_known is True
    assert InfoModeV5().enemy_deck_known is True


def test_history_events_reads_dedicated_v5_ring(tmp_path, monkeypatch):
    """select_action must use the Phase-C ``v5_history_events`` contract.

    Native ``state.history`` and UI ``state.action_history`` remain separate
    compatibility logs and must never be substituted into the model input.
    """
    from train_v3.contracts import OBS_V5_DIM
    import train_v3.obs_v5 as obs_v5_mod

    captured = {}

    def _spy(state, player_id, *, info_mode=None, assist_mode=None, history_events=None):
        captured["history_events"] = history_events
        return np.zeros(OBS_V5_DIM, dtype=np.float32)

    monkeypatch.setattr(obs_v5_mod, "encode_observation_v5", _spy)

    mgr = make_manager(tmp_path)
    match, engine, _runner = create_match(mgr, p2_model="random", starting_player="p1")
    arena = engine._arena
    pid = engine.human_user_id

    # Keep all three histories populated so selecting the wrong one is visible.
    hist_event = {
        "actor_id": pid, "action_type": "play_card", "action_id": 7,
        "enemy_hero_hp_delta": 0.0, "own_hero_hp_delta": 0.0,
        "my_board_count_delta": 1.0, "enemy_board_count_delta": 0.0,
        "turn_number": 1, "board_power_delta": 3.0,
    }
    arena.state.v5_history_events.append(dict(hist_event))
    arena.state.history.append({"type": "end_turn"})
    arena.state.action_history.append(("play_card", "p1 plays card 7"))

    adapter = V5RlhfAdapter({"name": "v5-test", "inference": _fake_inference()})
    adapter.select_action(arena, pid)

    he = captured["history_events"]
    assert he == [hist_event], (
        "adapter must pass list(state.v5_history_events) to encode_observation_v5; "
        f"got {he!r}"
    )
    assert len(he) == 1 and isinstance(he[0], dict)


# ----------------------------------------------------------------------------
# 4. append_only action_features + legal mask
# ----------------------------------------------------------------------------

def test_append_only_placement_mode_passed(tmp_path, monkeypatch):
    """encode_action_features + build_action_mask must be called with
    placement_mode='append_only'."""
    import ai.train_v2.classic_actions_v1 as classic_mod
    from train_v3.contracts import OBS_V5_DIM

    real_encode_af = classic_mod.encode_action_features
    real_build_mask = classic_mod.build_action_mask
    recorded = {}

    def _spy_encode_af(state, player_id, **kwargs):
        recorded["af_placement_mode"] = kwargs.get("placement_mode")
        recorded["af_include_preview"] = kwargs.get("include_preview")
        recorded["af_verify_mask"] = kwargs.get("verify_mask")
        return real_encode_af(state, player_id, **kwargs)

    def _spy_build_mask(state, player_id, **kwargs):
        recorded["mask_placement_mode"] = kwargs.get("placement_mode")
        recorded["mask_verify_mask"] = kwargs.get("verify_mask")
        return real_build_mask(state, player_id, **kwargs)

    monkeypatch.setattr(classic_mod, "encode_action_features", _spy_encode_af)
    monkeypatch.setattr(classic_mod, "build_action_mask", _spy_build_mask)

    mgr = make_manager(tmp_path)
    match, engine, _runner = create_match(mgr, p2_model="random", starting_player="p1")
    arena = engine._arena
    pid = engine.human_user_id

    adapter = V5RlhfAdapter({"name": "v5-test", "inference": _fake_inference()})
    adapter.select_action(arena, pid)

    assert recorded["af_placement_mode"] == "append_only"
    assert recorded["mask_placement_mode"] == "append_only"
    assert recorded["af_verify_mask"] is False
    assert recorded["mask_verify_mask"] is False
    assert recorded["af_include_preview"] is False

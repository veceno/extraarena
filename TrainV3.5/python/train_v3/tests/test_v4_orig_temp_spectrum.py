"""Block B component B2 -- ``v4_orig_temp_spectrum.py`` tests (TRACKED, PURE-PYTHON).

Six tests for ``TrainV3.5/python/train_v3/v4_orig_temp_spectrum.py`` -- the V4-orig
temperature spectrum (3 identities from 1 frozen V4 ONNX: argmax / t07 / t12).

PURE-PYTHON: NO real ONNX, NO real onnxruntime, NO real V4 npz. The adapter math is
exercised via a ``_FakeOnnxSession`` with a KNOWN logits vector (entropy is a pure
function of temperature on fixed logits -- source-vs-source, not self-referential
fixture regen, ``BLOCK_B_PLAN.md:704-710``). The real-path factory test is skip-gated
on ``V4OnnxUnavailableError`` (worktree has no V4 ONNX npz / no onnxruntime).

Tests (``BLOCK_B_PLAN.md:305-312``):
  (1) ``test_three_identities_from_one_model`` -- one ``_FakeOnnxSession`` shared by
      three identities with temps 0 / 0.7 / 1.2.
  (2) ``test_higher_temperature_more_random`` -- entropy(argmax)=0 < entropy(t07) <
      entropy(t12) on a fixed logit set (entropy strictly increases with temperature).
  (3) ``test_argmax_identity_is_deterministic`` -- temp=0 / argmax yields the argmax
      legal id (42, the highest-logit legal id) on every call.
  (4) ``test_weights_frozen`` -- the 0.40 / 0.20 / 0.15 weights are the spectrum config.
  (5) ``test_skip_if_no_v4_onnx`` -- skip-gate when the V4 ONNX is absent.
  (6) ``test_adapter_picks_legal_and_matches_onnx_math`` -- the adapter aid is in
      ``ctx.legal_action_ids`` AND equals ``argmax(mlogits restricted to legal)`` for
      the argmax identity (``onnx_policy.py:90/:101`` reproduced); for the sample
      identities the aid is drawn from the temperature-scaled legal distribution
      (assert it is legal and that the distribution is temperature-dependent).

Run: ``PYTHONPATH=.:TrainV3.5/python python3 -m pytest \
TrainV3.5/python/train_v3/tests/test_v4_orig_temp_spectrum.py -q``
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from train_v3.rust_live_self_play import OpponentCtx
from train_v3.v4_orig_temp_spectrum import (
    V4OnnxUnavailableError,
    V4_ORIG_ARGMAX,
    V4_ORIG_T07,
    V4_ORIG_T12,
    V4_ORIG_TEMP_ALIASES,
    V4_ORIG_TEMP_IDENTITIES,
    V4_ORIG_TEMP_WEIGHTS,
    V4TempSpectrumIdentity,
    build_v4_temp_spectrum_opponents,
    make_v4_temp_select_fn,
    mana_draw_collapse_monitor_hook,
    register_mana_draw_collapse_monitor,
)


# -----------------------------------------------------------------------------
# Fakes -- a KNOWN-logit ONNX session + a packed OpponentCtx (no env, no onnxruntime).
# -----------------------------------------------------------------------------
class _FakeOnnxSession:
    """A fake ``onnxruntime.InferenceSession`` returning a known logits vector.

    ``run(["logits","value"], feeds)`` returns ``[logits_batch, value_batch]`` where
    ``logits_batch`` is ``(1, 601)`` -- the same shape ``OnnxActionPolicy`` reads at
    ``onnx_policy.py:88`` (``outputs[0][0]``). The logits vector is fixed so entropy
    is a pure function of temperature (source-vs-source, ``BLOCK_B_PLAN.md:707``).
    """

    def __init__(self, logits_601: np.ndarray) -> None:
        self._logits = np.asarray(logits_601, dtype=np.float32)
        assert self._logits.shape == (601,), self._logits.shape
        self.run_count = 0

    def run(self, output_names: list[str], feeds: dict[str, Any]) -> list[np.ndarray]:
        self.run_count += 1
        logits_batch = self._logits[np.newaxis, :].astype(np.float32)
        value_batch = np.array([0.0], dtype=np.float32)
        return [logits_batch, value_batch]


# Known logits over the 601-candidate space. Legal ids = [10, 42, 77]; logits[42]=3.0
# is the highest among legal (argmax -> 42), logits[77]=2.0 second, logits[10]=1.0
# third; all other candidates 0.0 (masked to -1e9 by the adapter, so they never win).
_KNOWN_LOGITS = np.zeros(601, dtype=np.float32)
_KNOWN_LOGITS[10] = 1.0
_KNOWN_LOGITS[42] = 3.0
_KNOWN_LOGITS[77] = 2.0
_LEGAL_IDS = np.array([10, 42, 77], dtype=np.intp)
_OBS = np.zeros(73, dtype=np.float32)  # observation_v5 shape is irrelevant to the fake
_LEGAL_FEATURES = np.zeros((3, 171), dtype=np.float32)


def _make_ctx() -> OpponentCtx:
    return OpponentCtx(
        env_idx=0,
        actor_id=2,
        observation_v5=_OBS,
        legal_action_ids=_LEGAL_IDS,
        legal_action_features=_LEGAL_FEATURES,
        legal_action_counts=3,
        mana_draw_legal=True,
    )


def _shannon_entropy(counts: dict[int, int]) -> float:
    """Empirical Shannon entropy (nats) over the sampled action-id distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * np.log(p)
    return float(ent)


# =============================================================================
# (1) three identities from one model
# =============================================================================
def test_three_identities_from_one_model() -> None:
    session = _FakeOnnxSession(_KNOWN_LOGITS)
    opponents = build_v4_temp_spectrum_opponents(session)

    assert set(opponents.keys()) == {
        "v4-orig-argmax",
        "v4-orig-t07",
        "v4-orig-t12",
    }
    # All three share the SAME session (one frozen V4 ONNX -> three identities).
    # The factory built three select_fns over the single injected session; verify
    # each opponent is wired and carries its frozen identity.
    for name, opp in opponents.items():
        assert opp.name == name
        assert opp.wired is True
        assert opp.identity is not None
        assert isinstance(opp.identity, V4TempSpectrumIdentity)
    # Temperatures per D-B6 (t07=0.7, t12=1.2; argmax temp=0 label).
    assert opponents["v4-orig-argmax"].identity.mode == "argmax"
    assert opponents["v4-orig-argmax"].identity.temperature == 0.0
    assert opponents["v4-orig-t07"].identity.mode == "sample"
    assert opponents["v4-orig-t07"].identity.temperature == 0.7
    assert opponents["v4-orig-t12"].identity.mode == "sample"
    assert opponents["v4-orig-t12"].identity.temperature == 1.2


# =============================================================================
# (2) higher temperature -> more random (entropy strictly increases with temp)
# =============================================================================
def test_higher_temperature_more_random() -> None:
    session = _FakeOnnxSession(_KNOWN_LOGITS)
    argmax_fn = make_v4_temp_select_fn(session, V4_ORIG_ARGMAX)
    t07_fn = make_v4_temp_select_fn(session, V4_ORIG_T07)
    t12_fn = make_v4_temp_select_fn(session, V4_ORIG_T12)
    ctx = _make_ctx()

    n = 4000

    np.random.seed(0)
    argmax_counts: dict[int, int] = {}
    for _ in range(n):
        aid = argmax_fn(ctx)
        argmax_counts[aid] = argmax_counts.get(aid, 0) + 1

    np.random.seed(1)
    t07_counts: dict[int, int] = {}
    for _ in range(n):
        aid = t07_fn(ctx)
        t07_counts[aid] = t07_counts.get(aid, 0) + 1

    np.random.seed(2)
    t12_counts: dict[int, int] = {}
    for _ in range(n):
        aid = t12_fn(ctx)
        t12_counts[aid] = t12_counts.get(aid, 0) + 1

    e_argmax = _shannon_entropy(argmax_counts)
    e_t07 = _shannon_entropy(t07_counts)
    e_t12 = _shannon_entropy(t12_counts)

    # argmax is deterministic -> entropy 0.
    assert e_argmax == pytest.approx(0.0, abs=1e-12)
    # t07 samples the temperature-scaled legal distribution -> non-trivial entropy.
    assert e_t07 > 0.0
    # Higher temperature -> flatter distribution -> strictly higher entropy.
    assert e_t07 < e_t12, (e_argmax, e_t07, e_t12)


# =============================================================================
# (3) argmax identity is deterministic (temp=0 / argmax -> argmax legal id)
# =============================================================================
def test_argmax_identity_is_deterministic() -> None:
    session = _FakeOnnxSession(_KNOWN_LOGITS)
    argmax_fn = make_v4_temp_select_fn(session, V4_ORIG_ARGMAX)
    ctx = _make_ctx()

    # logits[42]=3.0 is the highest among legal [10,42,77] -> argmax = 42.
    for _ in range(100):
        assert argmax_fn(ctx) == 42


# =============================================================================
# (4) weights frozen (0.40 / 0.20 / 0.15)
# =============================================================================
def test_weights_frozen() -> None:
    assert V4_ORIG_TEMP_WEIGHTS == {
        "v4-orig-argmax": 0.40,
        "v4-orig-t07": 0.20,
        "v4-orig-t12": 0.15,
    }
    # The frozen weights live on the identity dataclass too.
    assert V4_ORIG_ARGMAX.weight == 0.40
    assert V4_ORIG_T07.weight == 0.20
    assert V4_ORIG_T12.weight == 0.15
    # Identity tuple carries all three in canonical order.
    assert tuple(ident.name for ident in V4_ORIG_TEMP_IDENTITIES) == (
        "v4-orig-argmax",
        "v4-orig-t07",
        "v4-orig-t12",
    )
    # Alias map is identity-to-canonical (B3 resolves the league parse side).
    assert V4_ORIG_TEMP_ALIASES == {
        "v4-orig-argmax": "v4-orig-argmax",
        "v4-orig-t07": "v4-orig-t07",
        "v4-orig-t12": "v4-orig-t12",
    }


# =============================================================================
# (5) skip-gate when the V4 ONNX is absent
# =============================================================================
def test_skip_if_no_v4_onnx(tmp_path) -> None:
    missing = tmp_path / "definitely_not_present_v4.onnx"
    try:
        build_v4_temp_spectrum_opponents(str(missing))
    except V4OnnxUnavailableError as exc:
        pytest.skip(f"V4 ONNX unavailable (worktree skip-gate): {exc}")
    # If build succeeded (a real onnxruntime + model somehow present), the three
    # identities must still be built -- not a skip, just a no-op pass-through.
    opponents = build_v4_temp_spectrum_opponents(str(missing))  # type: ignore[unreachable]
    assert set(opponents.keys()) == {
        "v4-orig-argmax",
        "v4-orig-t07",
        "v4-orig-t12",
    }


# =============================================================================
# (6) adapter picks legal AND matches OnnxActionPolicy math (source-vs-source)
# =============================================================================
def test_adapter_picks_legal_and_matches_onnx_math() -> None:
    session = _FakeOnnxSession(_KNOWN_LOGITS)
    argmax_fn = make_v4_temp_select_fn(session, V4_ORIG_ARGMAX)
    t07_fn = make_v4_temp_select_fn(session, V4_ORIG_T07)
    t12_fn = make_v4_temp_select_fn(session, V4_ORIG_T12)
    ctx = _make_ctx()
    legal_set = set(int(x) for x in ctx.legal_action_ids)

    # --- argmax identity: aid == argmax(mlogits restricted to legal) ---
    # Reproduce the oracle math (onnx_policy.py:90/:101) directly from the known
    # logits: mlogits = where(legal, logits, -1e9); aid = argmax(mlogits).
    mask = np.zeros(601, dtype=bool)
    mask[ctx.legal_action_ids] = True
    mlogits = np.where(mask, _KNOWN_LOGITS, -1e9).astype(np.float32)
    oracle_argmax = int(np.argmax(mlogits))
    assert oracle_argmax == 42  # sanity: highest-logit legal id

    aid = argmax_fn(ctx)
    assert aid in legal_set
    assert aid == oracle_argmax, (aid, oracle_argmax)

    # --- sample identities: aid drawn from the temperature-scaled legal dist ---
    # The aid MUST be legal; the distribution MUST be temperature-dependent
    # (t12 flatter than t07 -> higher entropy, exercised in test (2)). Here we
    # assert legality + that the empirical distribution differs from argmax-only.
    np.random.seed(7)
    t07_samples = [t07_fn(ctx) for _ in range(800)]
    np.random.seed(8)
    t12_samples = [t12_fn(ctx) for _ in range(800)]

    assert all(a in legal_set for a in t07_samples)
    assert all(a in legal_set for a in t12_samples)

    # The sample identities are NOT deterministic argmax (they spread over >=2 ids).
    assert len(set(t07_samples)) >= 2
    assert len(set(t12_samples)) >= 2

    # t12 is flatter than t07: the dominant-id share is lower under t12.
    t07_dom = max(np.bincount(t07_samples, minlength=601).tolist()) / len(t07_samples)
    t12_dom = max(np.bincount(t12_samples, minlength=601).tolist()) / len(t12_samples)
    assert t12_dom < t07_dom, (t07_dom, t12_dom)


# =============================================================================
# (7) Q5 mana_draw-collapse monitor hook is exposed (placeholder; B3/B4 wires)
# =============================================================================
def test_mana_draw_collapse_monitor_hook_exposed() -> None:
    # B2 only EXPOSES the hook; it does NOT implement the monitor. Verify the
    # registration + call-site plumbing without asserting any collapse logic.
    assert mana_draw_collapse_monitor_hook() is None

    seen: list[tuple[str, int]] = []

    def monitor(identity_name: str, ctx: OpponentCtx, aid: int) -> None:
        seen.append((identity_name, aid))

    register_mana_draw_collapse_monitor(monitor)
    try:
        assert mana_draw_collapse_monitor_hook() is monitor
        session = _FakeOnnxSession(_KNOWN_LOGITS)
        argmax_fn = make_v4_temp_select_fn(session, V4_ORIG_ARGMAX)
        ctx = _make_ctx()
        aid = argmax_fn(ctx)
        assert seen == [("v4-orig-argmax", aid)]
    finally:
        register_mana_draw_collapse_monitor(None)
    assert mana_draw_collapse_monitor_hook() is None


# =============================================================================
# (8) adapter guards: None legal_action_features + empty legal ids
# =============================================================================
def test_adapter_requires_legal_action_features() -> None:
    session = _FakeOnnxSession(_KNOWN_LOGITS)
    argmax_fn = make_v4_temp_select_fn(session, V4_ORIG_ARGMAX)
    ctx = OpponentCtx(
        env_idx=0,
        actor_id=2,
        observation_v5=_OBS,
        legal_action_ids=_LEGAL_IDS,
        legal_action_features=None,  # adapter cannot reproduce the forward
        legal_action_counts=3,
        mana_draw_legal=True,
    )
    with pytest.raises(ValueError, match="legal_action_features"):
        argmax_fn(ctx)


def test_adapter_rejects_empty_legal_ids() -> None:
    session = _FakeOnnxSession(_KNOWN_LOGITS)
    argmax_fn = make_v4_temp_select_fn(session, V4_ORIG_ARGMAX)
    ctx = OpponentCtx(
        env_idx=0,
        actor_id=2,
        observation_v5=_OBS,
        legal_action_ids=np.array([], dtype=np.intp),
        legal_action_features=np.zeros((0, 171), dtype=np.float32),
        legal_action_counts=0,
        mana_draw_legal=False,
    )
    with pytest.raises(ValueError, match="no legal actions"):
        argmax_fn(ctx)


def test_sample_mode_requires_positive_temperature() -> None:
    # Mirrors OnnxActionPolicy.__init__ validation (onnx_policy.py:30-31).
    session = _FakeOnnxSession(_KNOWN_LOGITS)
    bad = V4TempSpectrumIdentity("bad", mode="sample", temperature=0.0, weight=0.1)
    with pytest.raises(ValueError, match="temperature must be > 0"):
        make_v4_temp_select_fn(session, bad)
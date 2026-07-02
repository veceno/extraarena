"""Synthetic tests for the E5 ship component (``e1_ship.py``) + the prod wiring
(``ai/bot_brain.py`` V5 branch + ``infrastructure/config.py`` V5 profile + tier
retargets).

ALL collaborators are SYNTHETIC: fake ``ort.InferenceSession`` stubs + a fake
``onnx_export_fn`` + a real ``ReleaseBundleConfig`` with a tmp ``candidate_dir``
+ a fake ``winner_report``. NO real MLX/Rust/ONNX is touched -- the V5 ONNX
export + release bundle + detector registration + prod wiring verification are
exercised via pure-python fakes that mirror the real contract shapes.

The tests assert the SPECIFIC E5 load-bearing invariants:
  * the V5 detector returns ``"v5"`` for a V5 sidecar (explicit model_version
    OR structural obs_dim+mana_draw_head+format fingerprint) and ``None`` for a
    V4 sidecar (delegates to the LIFO chain);
  * the V5 detector is registered AHEAD of ``_sidecar_kind_detector`` (LIFO
    insert-at-0);
  * ``ship_v5_winner`` exports + bundles + registers + verifies, asserts
    ``winner_report.passed()`` (NO-SHIP gate), and populates ``ShipResult``;
  * the prod ``_get_action_v5`` path wires the mana_draw parallel binary head
    (takes mana_draw when legal + logit higher; skips when illegal OR logit
    lower);
  * the ONNX fallback guard (``_assert_v5_logits_finite_legal``) raises
    ``RuntimeError`` on NaN logits OR no-legal-candidate (NOT a silent
    ``_legal_fallback``) -- the ``RuntimeError`` propagates out of
    ``_get_action_v5``;
  * the V4 path (``_validate_train_v2_contract`` +
    ``_get_action_train_v2_classic``) is byte-unchanged (regression guard);
  * the vendored ``ai.train_v2.obs_v5`` / ``mana_draw_head_v5`` are
    byte-faithful to the ``train_v3`` originals (codec-sync invariant);
  * ``ai/bot_brain.py`` imports ONLY ``ai.train_v2.*`` + ``core.*`` (ZERO
    ``train_v3`` / ``rlhf_env`` imports on the live hot path).
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

# Ensure the train_v3 package is importable when run from the worktree root via
# `python -m pytest` (PYTHONPATH is set by the runner; this is a belt-and-braces
# fallback so the file is robust to direct invocation from the tests/ dir).
_TV3 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _TV3 not in sys.path:
    sys.path.insert(0, _TV3)


# ---------------------------------------------------------------------------
# GameState builders (mirror test_train_v2_mana_draw_head.py / test_bot_brain
# conventions; pure-python, NO MLX/Rust/ONNX).
# ---------------------------------------------------------------------------

from core.actions import ManaDrawAction, EndTurnAction, PlayCardAction, AttackAction  # noqa: E402
from core.engine import ArenaEnvironment, HAND_CAP, MANA_DRAW_BASE  # noqa: E402
from core.state import (  # noqa: E402
    CardInstance,
    CardType,
    GameState,
    GameStatus,
    PlayerState,
)


def _hero(user_id: int, hp: int = 30) -> CardInstance:
    return CardInstance(
        instance_id=uuid4(),
        card_id=0,
        name=f"hero_{user_id}",
        card_type=CardType.HERO,
        mana_cost=0,
        attack=0,
        hp=hp,
        max_hp=hp,
        level=1,
        is_ready=True,
    )


def _card(card_id: int, mana_cost: int = 1) -> CardInstance:
    return CardInstance(
        instance_id=uuid4(),
        card_id=card_id,
        name=f"card_{card_id}",
        card_type=CardType.WARRIOR,
        mana_cost=mana_cost,
        attack=1,
        hp=1,
        max_hp=1,
        level=1,
        is_ready=True,
    )


def _player(
    user_id: int,
    *,
    hand_size: int = 0,
    mana: int = 10,
    mana_draw_count_this_turn: int = 0,
) -> PlayerState:
    return PlayerState(
        user_id=user_id,
        hero=_hero(user_id),
        mana=mana,
        max_mana=max(mana, 10),
        mana_draw_count_this_turn=mana_draw_count_this_turn,
        hand=[_card(100 + i) for i in range(hand_size)],
    )


def _state(
    *,
    me_hand_size: int = 2,
    me_mana: int = 10,
    me_count: int = 0,
    me_user_id: int = 1,
    enemy_user_id: int = 2,
    turn_owner=None,
    status=GameStatus.ONGOING,
) -> GameState:
    p1 = _player(
        me_user_id,
        hand_size=me_hand_size,
        mana=me_mana,
        mana_draw_count_this_turn=me_count,
    )
    p2 = _player(enemy_user_id)
    st = GameState(
        p1=p1,
        p2=p2,
        current_turn_owner_id=turn_owner if turn_owner is not None else me_user_id,
        turn_number=1,
    )
    st.status = status
    return st


def _state_with_legal_actions(
    *,
    me_hand_size: int = 2,
    me_mana: int = 10,
    me_count: int = 0,
):
    """Build a real GameState + the engine legal-action list (oracle)."""
    st = _state(
        me_hand_size=me_hand_size,
        me_mana=me_mana,
        me_count=me_count,
    )
    env = ArenaEnvironment(st, apply_start_effects=False)
    legal = env.get_legal_actions(1)
    return st, legal


# ---------------------------------------------------------------------------
# Fake ONNX InferenceSession stubs (mirror the prod session.run IO contract).
# ---------------------------------------------------------------------------

class _FakeInput:
    def __init__(self, name: str, shape):
        self.name = name
        self.shape = shape


class _FakeOutput:
    def __init__(self, name: str, shape):
        self.name = name
        self.shape = shape


class FakeV5Session:
    """Fake ONNX session returning the V5 3-tuple (logits, value, mana_draw_logit).

    ``logits_fn`` is a callable returning the [1, 601] logits array;
    ``mana_draw_logit`` is the scalar mana_draw logit. ``value`` defaults to 0.
    """

    def __init__(self, logits_fn, mana_draw_logit=0.0, value=None):
        self._logits_fn = logits_fn
        self._mana_draw_logit = float(mana_draw_logit)
        self._value = value if value is not None else np.zeros((1, 1), dtype=np.float32)
        self.run_count = 0

    def get_inputs(self):
        return [
            _FakeInput("observation", [1, 7128]),
            _FakeInput("action_features", [1, 601, 171]),
        ]

    def get_outputs(self):
        return [
            _FakeOutput("logits", [1, 601]),
            _FakeOutput("value", [1, 1]),
            _FakeOutput("mana_draw_logit", [1, 1]),
        ]

    def run(self, output_names, input_feed):
        self.run_count += 1
        logits = np.asarray(self._logits_fn(), dtype=np.float32).reshape(1, 601)
        md = np.array([[self._mana_draw_logit]], dtype=np.float32)
        out_map = {"logits": logits, "value": self._value, "mana_draw_logit": md}
        return [out_map[n] for n in output_names]


class FakeV4Session:
    """Fake ONNX session returning the V4 2-tuple (logits, value)."""

    def __init__(self, logits_fn, value=None):
        self._logits_fn = logits_fn
        self._value = value if value is not None else np.zeros((1, 1), dtype=np.float32)
        self.run_count = 0

    def get_inputs(self):
        return [
            _FakeInput("observation", [1, 1456]),
            _FakeInput("action_features", [1, 601, 171]),
        ]

    def get_outputs(self):
        return [
            _FakeOutput("logits", [1, 601]),
            _FakeOutput("value", [1, 1]),
        ]

    def run(self, output_names, input_feed):
        self.run_count += 1
        logits = np.asarray(self._logits_fn(), dtype=np.float32).reshape(1, 601)
        out_map = {"logits": logits, "value": self._value}
        return [out_map[n] for n in output_names]


def _v5_profile_dict(session: FakeV5Session) -> dict:
    """A minimal V5 session-dict (as stored by BerserkInference.__init__)."""
    return {
        "session": session,
        "format": "v5",
        "obs_dim": 7128,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "include_preview_features": False,
        "temperature_range": (1.0, 1.0),
        "selection": "argmax",
        "placement_mode": "append_only",
        "verify_mask": False,
        "action_codec": "classic_actions_v1",
        "observation_codec": "classic_obs_v1",
        "input_names": ["observation", "action_features"],
        "output_names": ["logits", "value", "mana_draw_logit"],
        "mana_draw_head": True,
    }


def _v4_profile_dict(session: FakeV4Session) -> dict:
    """A minimal V4 session-dict (as stored by BerserkInference.__init__)."""
    return {
        "session": session,
        "format": "train_v2_classic_v1",
        "obs_dim": 1456,
        "action_feature_dim": 171,
        "max_candidate_actions": 601,
        "include_preview_features": False,
        "temperature_range": (1.0, 1.0),
        "selection": "argmax",
        "placement_mode": "append_only",
        "verify_mask": False,
        "action_codec": "classic_actions_v1",
        "observation_codec": "classic_obs_v1",
        "input_names": ["observation", "action_features"],
        "output_names": ["logits", "value"],
    }


# ============================================================================
# V5 sidecar kind detector
# ============================================================================

class TestV5Detector:
    def test_v5_detector_returns_v5_for_v5_sidecar(self):
        from train_v3.e1_ship import v5_detector

        sidecar = {"model_version": "v5_split_encoder_onnx_v1", "obs_dim": 7128}
        assert v5_detector("path.onnx", sidecar, "name") == "v5"

    def test_v5_detector_returns_v5_via_obs_dim_and_mana_draw(self):
        from train_v3.e1_ship import v5_detector

        sidecar = {"obs_dim": 7128, "mana_draw_head": True, "format": "v5"}
        assert v5_detector("path.onnx", sidecar, "name") == "v5"

    def test_v5_detector_returns_none_for_v4_sidecar(self):
        from train_v3.e1_ship import v5_detector
        from rlhf_env.components.policy_adapters import _sidecar_kind_detector

        v4_sidecar = {
            "model_version": "classic_action_conditioned_onnx_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "inputs": ["observation", "action_features"],
            "outputs": ["logits", "value"],
        }
        # v5_detector returns None for a V4 sidecar (delegates to the LIFO chain)
        assert v5_detector("path.onnx", v4_sidecar, "name") is None
        # The V4 _sidecar_kind_detector returns "action_onnx" for the V4 sidecar
        assert _sidecar_kind_detector("path.onnx", v4_sidecar, "name") == "action_onnx"

    def test_v5_detector_registered_ahead_of_v4_detector_lifo(self):
        from rlhf_env.components.policy_adapters import (
            AdapterRegistry,
            _register_builtins,
            _sidecar_kind_detector,
        )
        from train_v3.e1_ship import register_v5_kind_detector, v5_detector

        # Build a FRESH registry (not the module-level singleton) so this test
        # is hermetic -- register_v5_kind_detector() with no arg uses the
        # singleton, which other tests may have already populated. We pass a
        # fresh registry explicitly.
        reg = AdapterRegistry()
        _register_builtins(reg)
        register_v5_kind_detector(reg)
        # v5_detector at index 0 (LIFO head), ahead of _sidecar_kind_detector
        assert reg._detectors[0] is v5_detector
        assert _sidecar_kind_detector in reg._detectors
        assert reg._detectors.index(v5_detector) < reg._detectors.index(_sidecar_kind_detector)

        # LIFO load-bearing invariant: the FULL registry detects V5 for a V5
        # sidecar AND action_onnx for a V4 sidecar (V5 first, V4 falls through).
        v5_sidecar = {
            "model_version": "v5_split_encoder_onnx_v1",
            "obs_dim": 7128,
            "mana_draw_head": True,
            "format": "v5",
            "inputs": ["observation", "action_features"],
            "outputs": ["logits", "value", "mana_draw_logit"],
        }
        v4_sidecar = {
            "model_version": "classic_action_conditioned_onnx_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "inputs": ["observation", "action_features"],
            "outputs": ["logits", "value"],
        }
        assert reg.detect_kind("v5.onnx", v5_sidecar) == "v5"
        assert reg.detect_kind("v4.onnx", v4_sidecar) == "action_onnx"


# ============================================================================
# ship_v5_winner
# ============================================================================

class _FakeWinnerReport:
    """Fake E1CandidateReport -- has candidate_path + passed()."""

    def __init__(self, candidate_path: str, passed: bool = True):
        self.candidate_path = candidate_path
        self._passed = passed

    def passed(self) -> bool:
        return self._passed


class TestShipV5Winner:
    def _make_bundle_config(self, tmp_path, candidate_dir=None):
        from ai.train_v2.release_bundle import ReleaseBundleConfig

        cdir = candidate_dir or str(tmp_path / "candidate")
        os.makedirs(cdir, exist_ok=True)
        # write a candidate.json so build_release_bundle finds the candidate
        with open(os.path.join(cdir, "candidate.json"), "w") as f:
            json.dump({"model_name": "extra-lr-v5-max", "score": 0.9}, f)
        # write a profile overlay so build_release_bundle finds the profile
        with open(os.path.join(cdir, "candidate_profile.json"), "w") as f:
            json.dump({"profile": "v5"}, f)
        # write an overlay so build_release_bundle finds the overlay
        with open(os.path.join(cdir, "profile_overlay.json"), "w") as f:
            json.dump({"version": "train_v2_profile_overlay_v1"}, f)
        # acceptance gate dir
        acc_dir = os.path.join(cdir, "acceptance_gate")
        os.makedirs(acc_dir, exist_ok=True)
        with open(os.path.join(acc_dir, "acceptance_gate.json"), "w") as f:
            json.dump({"status": "pass", "score": 0.9}, f)

        return ReleaseBundleConfig(
            candidate_dir=cdir,
            output_dir=str(tmp_path / "output"),
            name="extra-lr-v5-max-test",
            include_shadow=False,
            include_acceptance=True,
            create_archive=False,
        )

    def test_ship_v5_winner_exports_bundles_registers(self, tmp_path):
        from train_v3.e1_ship import ShipResult, register_v5_kind_detector, ship_v5_winner

        bundle_config = self._make_bundle_config(tmp_path)
        candidate_dir = bundle_config.candidate_dir

        export_calls = []

        def fake_onnx_export_fn(checkpoint_path, onnx_output_path):
            export_calls.append((checkpoint_path, onnx_output_path))
            # write a dummy ONNX + sidecar into the candidate dir
            with open(onnx_output_path, "wb") as f:
                f.write(b"dummy onnx bytes")
            sidecar = {
                "model_version": "v5_split_encoder_onnx_v1",
                "obs_dim": 7128,
                "action_feature_dim": 171,
                "max_candidate_actions": 601,
                "mana_draw_head": True,
                "format": "v5",
                "inputs": ["observation", "action_features"],
                "outputs": ["logits", "value", "mana_draw_logit"],
            }
            with open(onnx_output_path + ".json", "w") as f:
                json.dump(sidecar, f)
            return onnx_output_path

        winner_report = _FakeWinnerReport(candidate_path=str(tmp_path / "winner.npz"), passed=True)

        result = ship_v5_winner(
            winner_report,
            onnx_export_fn=fake_onnx_export_fn,
            bundle_config=bundle_config,
        )

        # onnx_export_fn was called with winner_report.candidate_path
        assert len(export_calls) == 1
        assert export_calls[0][0] == winner_report.candidate_path
        assert export_calls[0][1].endswith("extra-lr-v5-max.onnx")

        # ShipResult populated
        assert isinstance(result, ShipResult)
        assert result.marker == "extra-lr-v5-max"
        assert result.prod_profile_key == "extra-lr-v5-max"
        assert os.path.exists(result.onnx_path)
        assert os.path.exists(result.manifest_path)
        assert result.fallback_guard_verified is True
        assert len(result.trophy_tiers_retargeted) == 4

        # V5 detector is registered (in the singleton registry)
        reg = register_v5_kind_detector()
        from train_v3.e1_ship import v5_detector
        assert reg._detectors[0] is v5_detector

    def test_ship_v5_winner_refuses_non_passing_winner(self, tmp_path):
        from train_v3.e1_ship import ship_v5_winner

        bundle_config = self._make_bundle_config(tmp_path)

        def fake_onnx_export_fn(cp, op):
            with open(op, "wb") as f:
                f.write(b"x")

        # passed() False -> raises
        winner_report = _FakeWinnerReport(candidate_path="x.npz", passed=False)
        with pytest.raises(RuntimeError, match="NO-SHIP"):
            ship_v5_winner(
                winner_report,
                onnx_export_fn=fake_onnx_export_fn,
                bundle_config=bundle_config,
            )

        # winner_report None -> raises
        with pytest.raises(RuntimeError, match="NO-SHIP"):
            ship_v5_winner(
                None,
                onnx_export_fn=fake_onnx_export_fn,
                bundle_config=bundle_config,
            )


# ============================================================================
# bot_brain V5 branch (additive; V4 unchanged)
# ============================================================================

class TestBotBrainV5Branch:
    def test_bot_brain_v5_branch_accepts_v5_profile(self, tmp_path, monkeypatch):
        """A V5 profile loads (format v5 accepted, no skip) alongside a V4 profile."""
        from ai.bot_brain import BerserkInference

        model_path = tmp_path / "v5.onnx"
        model_path.write_bytes(b"fake v5 onnx")

        class FakeSession:
            def __init__(self, *_a, **_k):
                pass

            def get_inputs(self):
                return [
                    _FakeInput("observation", [1, 7128]),
                    _FakeInput("action_features", [1, 601, 171]),
                ]

            def get_outputs(self):
                return [
                    _FakeOutput("logits", [1, 601]),
                    _FakeOutput("value", [1, 1]),
                    _FakeOutput("mana_draw_logit", [1, 1]),
                ]

        monkeypatch.setattr("ai.bot_brain.ort.InferenceSession", FakeSession)

        v5_profile = {
            "model_path": str(model_path),
            "format": "v5",
            "obs_dim": 7128,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": (1.0, 1.0),
            "selection": "argmax",
            "placement_mode": "append_only",
            "verify_mask": False,
            "mana_draw_head": True,
        }
        brain = BerserkInference(profiles={"v5-test": v5_profile})
        assert "v5-test" in brain.sessions
        assert brain.sessions["v5-test"]["format"] == "v5"
        assert brain.sessions["v5-test"]["mana_draw_head"] is True
        assert brain.sessions["v5-test"]["obs_dim"] == 7128

    def test_bot_brain_v4_and_v5_coexist(self, tmp_path, monkeypatch):
        """V4 + V5 profiles coexist (additive, both load)."""
        from ai.bot_brain import BerserkInference

        v4_path = tmp_path / "v4.onnx"
        v4_path.write_bytes(b"fake v4")
        v5_path = tmp_path / "v5.onnx"
        v5_path.write_bytes(b"fake v5")

        class FakeSession:
            def __init__(self, *_a, **_k):
                self._shape_cache = None

            def get_inputs(self):
                # detect V4 vs V5 by which model was loaded -- use the path
                # passed via the _last_path attribute set below
                if self._shape_cache == "v5":
                    return [
                        _FakeInput("observation", [1, 7128]),
                        _FakeInput("action_features", [1, 601, 171]),
                    ]
                return [
                    _FakeInput("observation", [1, 1456]),
                    _FakeInput("action_features", [1, 601, 171]),
                ]

            def get_outputs(self):
                if self._shape_cache == "v5":
                    return [
                        _FakeOutput("logits", [1, 601]),
                        _FakeOutput("value", [1, 1]),
                        _FakeOutput("mana_draw_logit", [1, 1]),
                    ]
                return [
                    _FakeOutput("logits", [1, 601]),
                    _FakeOutput("value", [1, 1]),
                ]

        sessions_by_path = {}

        def fake_init(model_path_str, *_a, **_k):
            sess = FakeSession()
            if os.path.basename(model_path_str).startswith("v5"):
                sess._shape_cache = "v5"
            else:
                sess._shape_cache = "v4"
            sessions_by_path[model_path_str] = sess
            return sess

        monkeypatch.setattr("ai.bot_brain.ort.InferenceSession", fake_init)

        v4_profile = {
            "model_path": str(v4_path),
            "format": "train_v2_classic_v1",
            "obs_dim": 1456,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": (1.0, 1.0),
            "selection": "argmax",
            "placement_mode": "append_only",
            "verify_mask": False,
        }
        v5_profile = {
            "model_path": str(v5_path),
            "format": "v5",
            "obs_dim": 7128,
            "action_feature_dim": 171,
            "max_candidate_actions": 601,
            "temperature_range": (1.0, 1.0),
            "selection": "argmax",
            "placement_mode": "append_only",
            "verify_mask": False,
            "mana_draw_head": True,
        }
        brain = BerserkInference(profiles={"v4-test": v4_profile, "v5-test": v5_profile})
        assert "v4-test" in brain.sessions
        assert "v5-test" in brain.sessions
        assert brain.sessions["v4-test"]["format"] == "train_v2_classic_v1"
        assert brain.sessions["v5-test"]["format"] == "v5"
        assert brain.sessions["v5-test"]["mana_draw_head"] is True


# ============================================================================
# _get_action_v5 mana_draw wiring + fallback guard
# ============================================================================

class TestGetActionV5:
    def _brain_with_v5_session(self, session):
        from ai.bot_brain import BerserkInference

        brain = BerserkInference.__new__(BerserkInference)
        brain.sessions = {"v5-test": _v5_profile_dict(session)}
        return brain

    def test_get_action_v5_takes_mana_draw_when_legal_and_logit_higher(self):
        """mana_draw_logit=5.0 (high) + a clear best 601 candidate at 1.0,
        on a state where mana_draw is legal -> returns the ManaDrawAction index."""
        # mana_draw is legal when hand < HAND_CAP(4) and mana >= cost(2).
        # hand_size=2 < 4, mana=10 >= 2 -> legal.
        st, legal = _state_with_legal_actions(me_hand_size=2, me_mana=10, me_count=0)
        # confirm a ManaDrawAction is in legal (the oracle)
        assert any(isinstance(a, ManaDrawAction) for a in legal)

        def logits_fn():
            logits = np.full(601, -1.0, dtype=np.float32)
            logits[0] = 1.0  # best 601 candidate at index 0 (EndTurnAction)
            return logits

        session = FakeV5Session(logits_fn=logits_fn, mana_draw_logit=5.0)
        brain = self._brain_with_v5_session(session)
        idx = brain._get_action_v5(st, 1, legal, "v5-test", brain.sessions["v5-test"])
        # should be the index of ManaDrawAction in legal
        md_indices = [i for i, a in enumerate(legal) if isinstance(a, ManaDrawAction)]
        assert idx in md_indices, f"expected mana_draw index {md_indices}, got {idx}"

    def test_get_action_v5_skips_mana_draw_when_illegal(self):
        """mana_draw_logit=5.0 but mana_draw illegal (mana < cost) -> returns a
        601-candidate index (NOT the mana_draw)."""
        # hand_size=2 < 4 OK, but mana=0 < cost(2) -> mana_draw illegal.
        st, legal = _state_with_legal_actions(me_hand_size=2, me_mana=0, me_count=0)
        # mana_draw should NOT be in legal (oracle confirms)
        assert not any(isinstance(a, ManaDrawAction) for a in legal)

        def logits_fn():
            logits = np.full(601, -1.0, dtype=np.float32)
            logits[0] = 1.0  # best 601 candidate
            return logits

        session = FakeV5Session(logits_fn=logits_fn, mana_draw_logit=5.0)
        brain = self._brain_with_v5_session(session)
        idx = brain._get_action_v5(st, 1, legal, "v5-test", brain.sessions["v5-test"])
        # mana_draw is NOT in legal, so idx must be a valid legal index (the
        # decoded 601-candidate match). It should NOT be out of range.
        assert 0 <= idx < len(legal)
        # the returned action should NOT be a ManaDrawAction (none are legal)
        assert not isinstance(legal[idx], ManaDrawAction)

    def test_get_action_v5_skips_mana_draw_when_logit_lower(self):
        """mana_draw legal but mana_draw_logit=0.1 < best_candidate_logit=1.0
        -> returns a 601-candidate (NOT the mana_draw)."""
        st, legal = _state_with_legal_actions(me_hand_size=2, me_mana=10, me_count=0)
        assert any(isinstance(a, ManaDrawAction) for a in legal)

        def logits_fn():
            logits = np.full(601, -1.0, dtype=np.float32)
            logits[0] = 1.0  # best 601 candidate at 1.0
            return logits

        # mana_draw_logit=0.1 < 1.0 -> select_includes_mana_draw returns False
        session = FakeV5Session(logits_fn=logits_fn, mana_draw_logit=0.1)
        brain = self._brain_with_v5_session(session)
        idx = brain._get_action_v5(st, 1, legal, "v5-test", brain.sessions["v5-test"])
        assert 0 <= idx < len(legal)
        # should NOT be the mana_draw (logit lower)
        assert not isinstance(legal[idx], ManaDrawAction)

    def test_get_action_v5_fallback_guard_nan_raises(self):
        """NaN logits -> _assert_v5_logits_finite_legal raises RuntimeError
        (NOT _legal_fallback). The RuntimeError MUST propagate."""
        st, legal = _state_with_legal_actions(me_hand_size=2, me_mana=10, me_count=0)
        assert len(legal) > 0

        def logits_fn():
            logits = np.full(601, 1.0, dtype=np.float32)
            logits[0] = np.nan  # NaN
            return logits

        session = FakeV5Session(logits_fn=logits_fn, mana_draw_logit=0.0)
        brain = self._brain_with_v5_session(session)
        with pytest.raises(RuntimeError, match="non-finite"):
            brain._get_action_v5(st, 1, legal, "v5-test", brain.sessions["v5-test"])

    def test_get_action_v5_fallback_guard_no_legal_raises(self):
        """All-False mask -> no finite masked-in candidate -> RuntimeError."""
        # Build a state where the action mask is all-False. The easiest way is
        # a state where the player has no legal candidate actions in the 601
        # space -- e.g. a game-over state (status != ONGOING). But
        # build_action_mask on a game-over state may still produce some
        # candidates. Instead, we patch the mask to all-False.
        st, legal = _state_with_legal_actions(me_hand_size=2, me_mana=10, me_count=0)
        assert len(legal) > 0

        def logits_fn():
            return np.zeros(601, dtype=np.float32)

        session = FakeV5Session(logits_fn=logits_fn, mana_draw_logit=0.0)
        brain = self._brain_with_v5_session(session)

        # Patch build_action_mask to return all-False mask.
        import ai.train_v2.classic_actions_v1 as codec

        original_build = codec.build_action_mask

        def fake_build_mask(*a, **kw):
            return np.zeros(601, dtype=np.float32)

        codec.build_action_mask = fake_build_mask
        try:
            with pytest.raises(RuntimeError, match="no finite masked-in candidate"):
                brain._get_action_v5(st, 1, legal, "v5-test", brain.sessions["v5-test"])
        finally:
            codec.build_action_mask = original_build


# ============================================================================
# V4 path unchanged (regression guard)
# ============================================================================

class TestV4PathUnchanged:
    def test_v4_path_unchanged_regression(self):
        """A V4 profile through BerserkInference.get_action returns the V4-decoded
        action (regression guard, V4 path byte-unchanged)."""
        from ai.bot_brain import BerserkInference

        st, legal = _state_with_legal_actions(me_hand_size=2, me_mana=10, me_count=0)
        assert len(legal) > 0

        def logits_fn():
            logits = np.full(601, -1.0, dtype=np.float32)
            logits[0] = 1.0  # best 601 candidate at index 0 (EndTurnAction)
            return logits

        session = FakeV4Session(logits_fn=logits_fn)
        brain = BerserkInference.__new__(BerserkInference)
        brain.sessions = {"v4-test": _v4_profile_dict(session)}
        idx = brain._get_action_train_v2_classic(st, 1, legal, "v4-test", brain.sessions["v4-test"])
        assert 0 <= idx < len(legal)

    def test_validate_train_v2_contract_still_exists(self):
        """_validate_train_v2_contract and _get_action_train_v2_classic still
        exist (signature check)."""
        from ai.bot_brain import BerserkInference

        assert hasattr(BerserkInference, "_validate_train_v2_contract")
        assert hasattr(BerserkInference, "_get_action_train_v2_classic")
        assert hasattr(BerserkInference, "_validate_v5_contract")
        assert hasattr(BerserkInference, "_get_action_v5")


# ============================================================================
# config V5 profile + tier retargets
# ============================================================================

class TestConfigV5Profile:
    def test_config_v5_profile_added_and_top_tiers_retargeted(self):
        from infrastructure.config import (
            BOT_MODEL_PROFILES,
            BOT_STRENGTH_TIERS,
            BOT_DIFFICULTY_PROFILES,
        )

        assert "extra-lr-v5-max" in BOT_MODEL_PROFILES
        v5 = BOT_MODEL_PROFILES["extra-lr-v5-max"]
        assert v5["obs_dim"] == 7128
        assert v5["mana_draw_head"] is True
        assert v5["format"] == "v5"

        top = ("tier_hard_4500", "tier_hard_plus_6000", "tier_max_minus_7500", "tier_max_9000")
        tier_by_key = {t["key"]: t for t in BOT_STRENGTH_TIERS}
        for k in top:
            assert tier_by_key[k]["brain_profile"] == "extra-lr-v5-max", k

        # 8 non-top tiers stay extra-lr-v4-* (regression guard)
        nontop = [t["key"] for t in BOT_STRENGTH_TIERS if t["key"] not in top]
        assert len(nontop) == 8
        for k in nontop:
            assert tier_by_key[k]["brain_profile"].startswith("extra-lr-v4-"), k

        # BOT_DIFFICULTY_PROFILES derivation propagated to obs_dim=7128 for top
        for k in top:
            assert BOT_DIFFICULTY_PROFILES[k]["obs_dim"] == 7128, k
        for k in nontop:
            assert BOT_DIFFICULTY_PROFILES[k]["obs_dim"] == 1456, k


# ============================================================================
# Vendored obs_v5 / mana_draw_head_v5 byte-faithful to train_v3 (codec-sync)
# ============================================================================

class TestVendoredByteFaithful:
    def test_vendored_obs_v5_byte_faithful_to_train_v3(self):
        """The vendored ai.train_v2.obs_v5 produces output IDENTICAL to
        train_v3.obs_v5 on the same GameState+player_id (codec-sync invariant)."""
        from ai.train_v2.obs_v5 import encode_observation_v5 as prod_encode
        from ai.train_v2.v5_contracts import OBS_V5_DIM
        from train_v3.obs_v5 import encode_observation_v5 as train_encode

        # OBS_V5_DIM constant matches
        assert OBS_V5_DIM == 7128

        st = _state(me_hand_size=2, me_mana=10, me_count=0)
        prod_out = prod_encode(st, 1)
        train_out = train_encode(st, 1)
        assert prod_out.shape == (7128,)
        assert train_out.shape == (7128,)
        assert np.array_equal(prod_out, train_out)

    def test_vendored_mana_draw_head_v5_agrees_with_train_v3(self):
        """ai.train_v2.mana_draw_head_v5.mana_draw_legal_mask agrees with
        train_v3.mana_draw_head_v5 on a constructed state."""
        from ai.train_v2.mana_draw_head_v5 import mana_draw_legal_mask as prod_mask
        from train_v3.mana_draw_head_v5 import mana_draw_legal_mask as train_mask

        # legal: hand=2 < 4, mana=10 >= cost=2
        st_legal = _state(me_hand_size=2, me_mana=10, me_count=0)
        assert prod_mask(st_legal, 1) == train_mask(st_legal, 1) is True

        # illegal: mana=0 < cost=2
        st_illegal = _state(me_hand_size=2, me_mana=0, me_count=0)
        assert prod_mask(st_illegal, 1) == train_mask(st_illegal, 1) is False

        # illegal: hand=4 >= HAND_CAP
        st_full = _state(me_hand_size=HAND_CAP, me_mana=10, me_count=0)
        assert prod_mask(st_full, 1) == train_mask(st_full, 1) is False


# ============================================================================
# No train_v3 / rlhf_env import in bot_brain (live hot path purity)
# ============================================================================

class TestNoProdTrainImport:
    def test_no_train_v3_or_rlhf_env_import_in_bot_brain(self):
        """ai/bot_brain.py must import ONLY ai.train_v2.* + core.* on the live
        hot path -- ZERO train_v3 / rlhf_env imports."""
        # __file__ = TrainV3.5/python/train_v3/tests/test_e1_ship.py
        # worktree root = dirname^5 from __file__
        worktree_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(__file__)
                    )
                )
            )
        )
        bot_brain_path = os.path.join(worktree_root, "ai", "bot_brain.py")
        with open(bot_brain_path, "r", encoding="utf-8") as f:
            src = f.read()
        forbidden = ["import train_v3", "from train_v3", "import rlhf_env", "from rlhf_env"]
        for pat in forbidden:
            assert pat not in src, f"forbidden import pattern {pat!r} found in ai/bot_brain.py"
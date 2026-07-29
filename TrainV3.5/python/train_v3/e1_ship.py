"""V5-family Block E1 SHIP component -- export + bundle + register + verify.

This is the TRAINING-SIDE ship component (lives in ``TrainV3.5/python/train_v3``,
NOT in prod ``ai/``). It consumes the E3 tournament winner report, exports the
winning checkpoint to ONNX via the E1 exporter, builds the release bundle via
the REUSED-AS-IS ``ai.train_v2.release_bundle.build_release_bundle``, registers
the V5 sidecar kind detector into the rlhf_env adapter registry (LIFO, ahead
of the V4 ``_sidecar_kind_detector``), and verifies that the complete
four-stage production progression (V4 Micro -> V5 Lite -> V5 -> V5 Ultra) is
in place. It does NOT mutate prod files -- those edits are committed source changes in
``ai/bot_brain.py`` + ``infrastructure/config.py``; ``ship_v5_winner`` only
EXPORTS + BUNDLES + REGISTERS + VERIFIES.

Vendoring decision (codec-sync invariant)
-----------------------------------------

The V5 live-path encoder set (``obs_v5`` + ``v5_contracts`` +
``mana_draw_head_v5`` + ``v5_inference_guard``) is VENDORED into
``ai/train_v2/`` as the prod-side live-path copies so the prod bot brain
(``ai/bot_brain.py``) imports ONLY ``ai.train_v2.*`` + ``core.*`` (ZERO
``train_v3`` / ``rlhf_env`` imports on the live hot path). The
``TrainV3.5/python/train_v3`` copies stay for training/experiments. The
codec-sync invariant guards byte-faithfulness between the vendored prod copies
and the train_v3 originals via the E5 test
``test_vendored_obs_v5_byte_faithful_to_train_v3`` -- the one relative-import
rewrite (``from .contracts import ...`` -> ``from ai.train_v2.v5_contracts
import ...`` in ``obs_v5.py``) is the ONLY intentional divergence; the encoder
logic is byte-identical so prod inference matches training-time encoding.

LIFO V5-detector load-bearing routing
--------------------------------------

The V5 sidecar ``model_version`` is ``"v5_split_encoder_onnx_v1"`` -- DISTINCT
from the V4 ``"classic_action_conditioned_onnx_v1"``, BUT the V4
``_sidecar_kind_detector`` (``policy_adapters.py:219-248``) matches a sidecar
via the ``inputs``/``action_feature_dim`` OR-branches (``:240-241``) which a V5
sidecar ALSO satisfies (it has ``inputs=[observation, action_features]`` +
``action_feature_dim=171``). A distinct ``model_version`` alone does NOT
prevent V4 misclassification -- the V5 detector MUST run FIRST (LIFO insert-at-
0) and return ``"v5"`` for any V5 sidecar. ``register_v5_kind_detector`` inserts
the V5 detector at the head of the registry's detector list, AHEAD of
``_sidecar_kind_detector``, so a V5 sidecar is routed to the V5 adapter factory
(``_factory_v5_real``, already registered at ``policy_adapters.py:411``) and
NEVER falls through to the V4 ``action_onnx`` factory. The V5 detector does NOT
re-register the factory (the slot is taken); it only registers the DETECTOR.

Ship gate
---------

``ship_v5_winner`` asserts ``winner_report.passed()`` is True (the E3
threshold-table gate). A non-passing winner (or ``None`` report) raises --
NO-SHIP. Ship is GATED on E2 (parity + fallback-guard) green + E3 (winner
passes the threshold table); this component is the last gate before the
release bundle is built.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from rlhf_env.components.policy_adapters import default_registry

# Re-exported so E5 tests can import the detector without reaching into the
# registry internals (the test asserts LIFO ordering via ``registry._detectors``).
__all__ = [
    "ShipResult",
    "register_v5_kind_detector",
    "ship_v5_winner",
    "v5_detector",
]

# ---------------------------------------------------------------------------
# V5 sidecar kind detector (LIFO, ahead of _sidecar_kind_detector)
# ---------------------------------------------------------------------------

_V5_MODEL_VERSION = "v5_split_encoder_onnx_v1"
_V5_OBS_DIM = 7128


def v5_detector(
    path: Optional[str], sidecar: dict[str, Any], name: Optional[str]
) -> Optional[str]:
    """V5 sidecar kind detector -- returns ``"v5"`` for a V5 sidecar, else None.

    A sidecar is V5 iff EITHER:
      (a) ``model_version == "v5_split_encoder_onnx_v1"`` (the explicit V5
          export marker written by ``export_v5_checkpoint_to_onnx``), OR
      (b) ``obs_dim == 7128`` AND ``mana_draw_head`` is truthy AND
          ``format == "v5"`` (the structural V5 fingerprint, independent of the
          explicit marker -- guards against a sidecar missing the
          ``model_version`` key).

    Returns ``None`` for any non-V5 sidecar so the LIFO chain proceeds to
    ``_sidecar_kind_detector`` (which returns ``"action_onnx"`` for V4 sidecars
    via the ``model_version == classic_action_conditioned_onnx_v1`` OR
    ``inputs``/``action_feature_dim`` OR-branches). A V4 sidecar lacks the V5
    keys (``mana_draw_head``, ``format=="v5"``, ``obs_dim==7128``) so it falls
    through cleanly.
    """
    if not sidecar:
        return None
    model_version = str(sidecar.get("model_version") or "")
    if model_version == _V5_MODEL_VERSION:
        return "v5"
    obs_dim = sidecar.get("obs_dim")
    mana_draw_head = sidecar.get("mana_draw_head")
    fmt = sidecar.get("format")
    if (
        obs_dim == _V5_OBS_DIM
        and bool(mana_draw_head) is True
        and str(fmt) == "v5"
    ):
        return "v5"
    return None


def register_v5_kind_detector(registry=None):
    """Register the V5 sidecar detector at the head of the LIFO detector chain.

    If ``registry`` is None, uses ``policy_adapters.default_registry()`` (the
    lazy singleton). The V5 detector is inserted at index 0 via
    ``register_detector`` (LIFO), so it sits AHEAD of
    ``_sidecar_kind_detector`` (the V4 detector, registered LAST at
    ``policy_adapters.py:416`` so it was head of LIFO before E5). The V5
    factory (``_factory_v5_real``) is ALREADY registered at
    ``policy_adapters.py:411`` -- this function does NOT re-register the
    factory, only the DETECTOR.

    Returns the registry (so callers/tests can inspect ``_detectors`` ordering).
    """
    if registry is None:
        registry = default_registry()
    # Idempotent: register_detector inserts at index 0 unconditionally, so a
    # bare call would prepend a DUPLICATE v5_detector on every ship op
    # (correctness is preserved -- first match wins -- but the list grows).
    # Guard by identity so repeated register_v5_kind_detector()/ship_v5_winner
    # calls do not accumulate redundant detector copies.
    if v5_detector not in registry._detectors:
        registry.register_detector(v5_detector)
    return registry


# ---------------------------------------------------------------------------
# ShipResult (frozen)
# ---------------------------------------------------------------------------

_SHIPPED_PROFILE_KEY = "extra-lr-v5"
_PRODUCTION_PROFILE_KEYS = (
    "extra-lr-v4-micro",
    "extra-lr-v5-lite",
    "extra-lr-v5",
    "extra-lr-v5-ultra",
)
_EXPECTED_TIER_PROGRESSION = (
    ("tier_lite_0000", 0, 99, "extra-lr-v4-micro"),
    ("tier_easy_0100", 100, 299, "extra-lr-v4-micro"),
    ("tier_easy_plus_0300", 300, 599, "extra-lr-v5-lite"),
    ("tier_easy_plus_0600", 600, 999, "extra-lr-v5-lite"),
    ("tier_medium_minus_1000", 1000, 1199, "extra-lr-v5-lite"),
    ("tier_medium_1200", 1200, 1999, "extra-lr-v5"),
    ("tier_medium_plus_2000", 2000, 2999, "extra-lr-v5"),
    ("tier_hard_minus_3000", 3000, 4499, "extra-lr-v5"),
    ("tier_hard_4500", 4500, 5999, "extra-lr-v5-ultra"),
    ("tier_hard_plus_6000", 6000, 7499, "extra-lr-v5-ultra"),
    ("tier_max_minus_7500", 7500, 8999, "extra-lr-v5-ultra"),
    ("tier_max_9000", 9000, 1_000_000_000, "extra-lr-v5-ultra"),
)
_PROGRESSION_TIER_KEYS = tuple(row[0] for row in _EXPECTED_TIER_PROGRESSION)


@dataclass(frozen=True)
class ShipResult:
    """Frozen result of ``ship_v5_winner`` -- the ship artifact manifest."""

    winner_path: str
    onnx_path: str
    sidecar_path: str
    bundle_dir: str
    manifest_path: str
    marker: str = _SHIPPED_PROFILE_KEY
    prod_profile_key: str = _SHIPPED_PROFILE_KEY
    production_profiles_verified: tuple = _PRODUCTION_PROFILE_KEYS
    trophy_tiers_retargeted: tuple = _PROGRESSION_TIER_KEYS
    fallback_guard_verified: bool = False


# ---------------------------------------------------------------------------
# ship_v5_winner -- the ship entry point
# ---------------------------------------------------------------------------

def ship_v5_winner(
    winner_report: Any,
    *,
    onnx_export_fn: Callable[[str, str], str],
    bundle_config: Any,
) -> ShipResult:
    """Ship the E3 tournament winner to the production V5-family bundle.

    GATED on E3: asserts ``winner_report`` is not None and
    ``winner_report.passed()`` is True (the E3 threshold-table gate). A
    non-passing winner raises -- NO-SHIP.

    Steps:
      (a) NO-SHIP guard -- assert winner_report.passed().
      (b) Export the winning checkpoint to ONNX via ``onnx_export_fn`` (E1
          ``export_v5_checkpoint_to_onnx``) into ``bundle_config.candidate_dir``
          as ``extra-lr-v5.onnx``. V5 Ultra intentionally shares this policy
          artifact and adds its Assembler/CardOptimum assist stack.
          ``onnx_export_fn`` writes the ONNX 3-tuple
          + the ``.onnx.json`` sidecar into the candidate dir.
      (c) Build the release bundle via ``release_bundle.build_release_bundle``
          (REUSED AS-IS, format-agnostic -- first *.onnx + .onnx.json sidecar
          + candidate.json).
      (d) Register the V5 sidecar kind detector (LIFO, ahead of V4).
      (e) Verify the prod wiring is in place: the registry contains exactly
          V4 Micro + V5 Lite + V5 + V5 Ultra; every trophy tier follows the
          four-stage progression; and ``BOT_DIFFICULTY_PROFILES`` carries the
          expected V4/V5 observation contract for every derived tier.
      (f) Return ``ShipResult`` populated.

    Does NOT mutate ``ai/bot_brain.py`` or ``infrastructure/config.py`` at call
    time -- those are committed source edits; ``ship_v5_winner`` only
    EXPORTS + BUNDLES + REGISTERS + VERIFIES.
    """
    # (a) NO-SHIP guard -- ship is GATED on E3 winner passing the threshold table.
    if winner_report is None:
        raise RuntimeError("ship_v5_winner: winner_report is None -- NO-SHIP (E3 gate)")
    if not winner_report.passed():
        raise RuntimeError(
            "ship_v5_winner: winner_report.passed() is False -- NO-SHIP (E3 gate)"
        )

    winner_path = winner_report.candidate_path

    # (b) Export the winning checkpoint to ONNX into the candidate dir.
    onnx_output_path = os.path.join(bundle_config.candidate_dir, "extra-lr-v5.onnx")
    onnx_export_fn(winner_path, onnx_output_path)

    # (c) Build the release bundle (REUSED AS-IS -- format-agnostic).
    from ai.train_v2.release_bundle import build_release_bundle

    bundle_result = build_release_bundle(bundle_config)
    bundle_dir = bundle_result["bundle_dir"]
    manifest_path = bundle_result["manifest_path"]

    # (d) Register the V5 sidecar kind detector (LIFO, ahead of V4).
    register_v5_kind_detector()

    # (e) Verify the prod wiring is in place (committed source edits).
    from infrastructure.config import (
        BOT_MODEL_PROFILES,
        BOT_STRENGTH_TIERS,
        BOT_DIFFICULTY_PROFILES,
    )

    actual_profile_keys = tuple(BOT_MODEL_PROFILES)
    if actual_profile_keys != _PRODUCTION_PROFILE_KEYS:
        raise RuntimeError(
            "ship_v5_winner: prod wiring incomplete -- BOT_MODEL_PROFILES must "
            f"contain exactly {_PRODUCTION_PROFILE_KEYS}, got {actual_profile_keys}"
        )

    v4_profile = BOT_MODEL_PROFILES["extra-lr-v4-micro"]
    if (
        v4_profile.get("obs_dim") != 1456
        or v4_profile.get("format") != "train_v2_classic_v1"
    ):
        raise RuntimeError(
            "ship_v5_winner: prod wiring incomplete -- 'extra-lr-v4-micro' "
            f"profile has wrong contract {v4_profile}"
        )

    for profile_key in _PRODUCTION_PROFILE_KEYS[1:]:
        profile = BOT_MODEL_PROFILES[profile_key]
        if (
            profile.get("obs_dim") != 7128
            or profile.get("format") != "v5"
            or not profile.get("mana_draw_head")
        ):
            raise RuntimeError(
                "ship_v5_winner: prod wiring incomplete -- "
                f"{profile_key!r} profile has wrong V5 contract {profile}"
            )

    # Ultra is a composite profile: it shares the shipped V5 policy artifact
    # and is distinguished by the two match-scoped assists.
    v5_profile = BOT_MODEL_PROFILES["extra-lr-v5"]
    ultra_profile = BOT_MODEL_PROFILES["extra-lr-v5-ultra"]
    if ultra_profile.get("model_path") != v5_profile.get("model_path"):
        raise RuntimeError(
            "ship_v5_winner: prod wiring incomplete -- V5 Ultra must share "
            "the ExtraLR V5 policy artifact"
        )
    if not (
        ultra_profile.get("assembler_enabled")
        and ultra_profile.get("cardoptimum_enabled")
    ):
        raise RuntimeError(
            "ship_v5_winner: prod wiring incomplete -- V5 Ultra requires "
            "Assembler V1 and CardOptimum V1"
        )

    actual_tiers = tuple(
        (
            tier.get("key"),
            tier.get("min_trophies"),
            tier.get("max_trophies"),
            tier.get("brain_profile"),
        )
        for tier in BOT_STRENGTH_TIERS
    )
    if actual_tiers != _EXPECTED_TIER_PROGRESSION:
        raise RuntimeError(
            "ship_v5_winner: prod wiring incomplete -- BOT_STRENGTH_TIERS "
            "does not match the four-stage V4 Micro/V5 progression"
        )

    # BOT_DIFFICULTY_PROFILES derives automatically from BOT_MODEL_PROFILES keyed
    # by tier brain_profile. Verify every tier retained the expected contract.
    for tier_key, _min_trophies, _max_trophies, profile_key in _EXPECTED_TIER_PROGRESSION:
        resolved = BOT_DIFFICULTY_PROFILES.get(tier_key)
        if resolved is None:
            raise RuntimeError(
                f"ship_v5_winner: prod wiring incomplete -- {tier_key} missing "
                "from BOT_DIFFICULTY_PROFILES (derivation broken)"
            )
        expected_obs_dim = 1456 if profile_key == "extra-lr-v4-micro" else 7128
        expected_format = (
            "train_v2_classic_v1"
            if profile_key == "extra-lr-v4-micro"
            else "v5"
        )
        if (
            resolved.get("obs_dim") != expected_obs_dim
            or resolved.get("format") != expected_format
        ):
            raise RuntimeError(
                "ship_v5_winner: prod wiring incomplete -- "
                f"{tier_key} derived contract is "
                f"obs_dim={resolved.get('obs_dim')}, "
                f"format={resolved.get('format')!r}; expected "
                f"obs_dim={expected_obs_dim}, format={expected_format!r}"
            )

    # The ONNX fallback guard (SPEC :174) is the last-resort prod safety -- it
    # is wired into the prod ``_get_action_v5`` path via the vendored
    # ``ai.train_v2.v5_inference_guard``. Verify the vendored guard imports
    # (the guard is the load-bearing last line of defense against a malformed
    # V5 ONNX producing NaN/garbage logits).
    from ai.train_v2.v5_inference_guard import _assert_v5_logits_finite_legal  # noqa: F401

    fallback_guard_verified = True

    # (f) Return ShipResult populated.
    sidecar_path = onnx_output_path + ".json"
    return ShipResult(
        winner_path=winner_path,
        onnx_path=onnx_output_path,
        sidecar_path=sidecar_path,
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        marker=_SHIPPED_PROFILE_KEY,
        prod_profile_key=_SHIPPED_PROFILE_KEY,
        production_profiles_verified=_PRODUCTION_PROFILE_KEYS,
        trophy_tiers_retargeted=_PROGRESSION_TIER_KEYS,
        fallback_guard_verified=fallback_guard_verified,
    )
